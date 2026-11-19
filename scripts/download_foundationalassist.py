#!/usr/bin/env python3
"""Download the gated FoundationalASSIST release without weakening its access gate.

The script only performs authenticated GET requests against the authoritative
ASSISTments Hugging Face repository. It never logs in, requests access, accepts
terms, or accepts a token as a command-line argument.
"""

from __future__ import print_function

import argparse
import csv
import datetime
import hashlib
import json
import os
import re
import shutil
import sys
import tempfile
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Sequence, Tuple
from urllib.error import HTTPError, URLError
from urllib.parse import quote, urlsplit
from urllib.request import (
    HTTPRedirectHandler,
    Request,
    build_opener,
)


REPO_ID = "ASSISTments/FoundationalASSIST"
REPO_URL = "https://huggingface.co/datasets/" + REPO_ID
HUB_BASE_URL = "https://huggingface.co"
DEFAULT_DESTINATION = (
    Path(__file__).resolve().parents[1] / "data" / "foundationalassist" / "raw" / "release_20261108"
)
MANIFEST_NAME = "download_manifest.json"

# Keep this allowlist intentionally small. In particular, it excludes the
# multi-gigabyte benchmark result archive.
SELECTED_FILES = (
    "README.md",
    "Code/clean_utils.py",
    "Code/cleantext.py",
    "Data/Interactions.csv",
    "Data/Problems.csv",
    "Data/Skills.csv",
)

# Each tuple is a group of accepted aliases; at least one member must exist.
REQUIRED_COLUMN_GROUPS = {
    "Data/Interactions.csv": (
        ("problem_id",),
        ("hint_count",),
        ("answer_text",),
        ("saw_answer",),
        ("discrete_score",),
        ("end_time",),
        ("user_id", "user_xid"),
    ),
    "Data/Problems.csv": (
        ("Problem Set Id",),
        ("Problem Part",),
        ("Problem Type",),
        ("Answer Type", "Answer Types"),
        ("Problem Body",),
        ("Fill-in Options",),
        ("Fill-in Answers",),
        ("Multiple Choice Options",),
        ("Multiple Choice Answers",),
        ("problem_id",),
    ),
    "Data/Skills.csv": (
        ("problem_id",),
        ("skill_id",),
        ("node_code",),
        ("node_name",),
    ),
}

# The release README documents the interaction scale. Requiring a broad
# fraction of that pinned value detects the known roughly-700k-row LFS loss
# while allowing the authors to add/remove rows and update the documented count.
MIN_DOCUMENTED_INTERACTION_RATIO = 0.80
DOWNLOAD_CHUNK_BYTES = 1024 * 1024
MAX_METADATA_BYTES = 4 * 1024 * 1024
MAX_CSV_FIELD_BYTES = 16 * 1024 * 1024
FULL_GIT_SHA_RE = re.compile(r"^[0-9a-f]{40}$", re.IGNORECASE)
DOCUMENTED_INTERACTION_PATTERNS = (
    re.compile(
        r"Interactions\s+consists\s+of\s+([\d,]+)\s+unique\s+instances",
        re.IGNORECASE,
    ),
    re.compile(
        r"Total\s+Student\s+Interactions\s*(?:\||:)?\s*([\d,]+)",
        re.IGNORECASE,
    ),
)


class FoundationalAssistError(RuntimeError):
    """Base exception for safe, user-facing downloader failures."""


class AuthenticationError(FoundationalAssistError):
    """No pre-existing Hugging Face authentication was found."""


class AccessError(FoundationalAssistError):
    """Authentication exists, but repository access is not approved."""


class DownloadError(FoundationalAssistError):
    """A remote file could not be downloaded safely."""


class ValidationError(FoundationalAssistError):
    """Downloaded files do not match the expected release contract."""


def _same_origin(first_url: str, second_url: str) -> bool:
    first = urlsplit(first_url)
    second = urlsplit(second_url)
    return (
        first.scheme.lower(),
        first.hostname,
        first.port,
    ) == (
        second.scheme.lower(),
        second.hostname,
        second.port,
    )


class _SafeRedirectHandler(HTTPRedirectHandler):
    """Never forward a bearer token to a different origin."""

    def redirect_request(
        self,
        request: Request,
        file_pointer,
        code: int,
        message: str,
        headers,
        new_url: str,
    ):
        old_parts = urlsplit(request.full_url)
        new_parts = urlsplit(new_url)
        if old_parts.scheme.lower() == "https" and new_parts.scheme.lower() != "https":
            raise DownloadError("Refusing an HTTPS-to-non-HTTPS redirect.")

        redirected = super().redirect_request(
            request, file_pointer, code, message, headers, new_url
        )
        if redirected is None:
            return None

        # HTTPRedirectHandler may copy regular headers. Remove authorization
        # from both header stores, then restore it only for the same origin.
        for header_store in (redirected.headers, redirected.unredirected_hdrs):
            for name in list(header_store):
                if name.lower() == "authorization":
                    del header_store[name]

        authorization = request.get_header("Authorization")
        if authorization and _same_origin(request.full_url, new_url):
            redirected.add_unredirected_header("Authorization", authorization)
        return redirected


def _token_file_candidates() -> Iterable[Path]:
    explicit_path = os.environ.get("HF_TOKEN_PATH")
    if explicit_path:
        yield Path(explicit_path).expanduser()

    hf_home = os.environ.get("HF_HOME")
    if hf_home:
        yield Path(hf_home).expanduser() / "token"

    xdg_cache_home = os.environ.get("XDG_CACHE_HOME")
    if xdg_cache_home:
        yield Path(xdg_cache_home).expanduser() / "huggingface" / "token"

    user_home = Path.home()
    yield user_home / ".cache" / "huggingface" / "token"
    yield user_home / ".huggingface" / "token"


def load_existing_hf_token() -> str:
    """Read existing Hugging Face auth without initiating a login flow."""

    for variable_name in ("HF_TOKEN", "HUGGING_FACE_HUB_TOKEN"):
        token = os.environ.get(variable_name, "").strip()
        if token:
            return token

    # Prefer the hub library's standard token resolver when it is already
    # installed, but do not make it a dependency.
    try:
        from huggingface_hub import get_token  # type: ignore

        token = (get_token() or "").strip()
        if token:
            return token
    except (ImportError, AttributeError):
        pass

    try:
        from huggingface_hub import HfFolder  # type: ignore

        token = (HfFolder.get_token() or "").strip()
        if token:
            return token
    except (ImportError, AttributeError):
        pass

    seen_paths = set()
    for token_path in _token_file_candidates():
        normalized_path = str(token_path)
        if normalized_path in seen_paths:
            continue
        seen_paths.add(normalized_path)
        try:
            token = token_path.read_text(encoding="utf-8").strip()
        except (FileNotFoundError, IsADirectoryError, PermissionError, OSError):
            continue
        if token:
            return token

    raise AuthenticationError(
        "No pre-existing Hugging Face authentication was found. Request and "
        "receive access at {0}, then authenticate through Hugging Face's "
        "standard login flow. This downloader does not log in or accept the "
        "dataset terms.".format(REPO_URL)
    )


class HubClient:
    """Minimal authenticated Hugging Face Hub reader."""

    def __init__(
        self,
        token: str,
        timeout_seconds: float = 60.0,
        base_url: str = HUB_BASE_URL,
    ):
        if not token:
            raise AuthenticationError("An existing Hugging Face token is required.")
        self._token = token
        self._timeout_seconds = timeout_seconds
        self._base_url = base_url.rstrip("/")
        self._opener = build_opener(_SafeRedirectHandler())

    def _request(self, url: str) -> Request:
        request = Request(url, method="GET")
        request.add_header(
            "User-Agent",
            "pykt-foundationalassist-downloader/1.0",
        )
        # Unredirected headers are not automatically copied to a CDN request.
        request.add_unredirected_header("Authorization", "Bearer " + self._token)
        return request

    def _open(self, url: str, purpose: str):
        try:
            return self._opener.open(
                self._request(url),
                timeout=self._timeout_seconds,
            )
        except HTTPError as error:
            if error.code in (401, 403):
                raise AccessError(
                    "Hugging Face denied access while {0} (HTTP {1}). Confirm "
                    "that this account has been approved for {2}; the downloader "
                    "will not submit or accept access terms.".format(
                        purpose, error.code, REPO_URL
                    )
                )
            raise DownloadError(
                "Hugging Face returned HTTP {0} while {1}.".format(
                    error.code, purpose
                )
            )
        except URLError as error:
            reason = getattr(error, "reason", "network error")
            raise DownloadError(
                "Network failure while {0}: {1}".format(purpose, reason)
            )

    def resolve_revision(self, requested_revision: str) -> str:
        if not requested_revision:
            raise DownloadError("Revision cannot be empty.")
        endpoint = "{0}/api/datasets/{1}/revision/{2}".format(
            self._base_url,
            quote(REPO_ID, safe="/"),
            quote(requested_revision, safe=""),
        )
        with self._open(endpoint, "resolving revision " + requested_revision) as response:
            payload = response.read(MAX_METADATA_BYTES + 1)
        if len(payload) > MAX_METADATA_BYTES:
            raise DownloadError("Repository metadata response was unexpectedly large.")
        try:
            metadata = json.loads(payload.decode("utf-8"))
        except (UnicodeDecodeError, ValueError):
            raise DownloadError("Repository metadata was not valid UTF-8 JSON.")

        resolved_revision = str(metadata.get("sha", "")).lower()
        if not FULL_GIT_SHA_RE.match(resolved_revision):
            raise DownloadError(
                "Hugging Face did not return an immutable 40-character revision."
            )
        return resolved_revision

    def download_file(self, repo_path: str, revision: str, target: Path) -> Dict:
        if repo_path not in SELECTED_FILES:
            raise DownloadError("Refusing a path outside the fixed allowlist.")

        endpoint = "{0}/datasets/{1}/resolve/{2}/{3}?download=true".format(
            self._base_url,
            quote(REPO_ID, safe="/"),
            quote(revision, safe=""),
            quote(repo_path, safe="/"),
        )
        target.parent.mkdir(parents=True, exist_ok=True)
        temporary_handle = tempfile.NamedTemporaryFile(
            mode="wb",
            prefix="." + target.name + ".",
            suffix=".part",
            dir=str(target.parent),
            delete=False,
        )
        temporary_path = Path(temporary_handle.name)
        digest = hashlib.sha256()
        received_bytes = 0
        expected_bytes = None
        etag = None

        try:
            with temporary_handle:
                with self._open(endpoint, "downloading " + repo_path) as response:
                    content_length = response.headers.get("Content-Length")
                    if content_length:
                        try:
                            expected_bytes = int(content_length)
                        except ValueError:
                            raise DownloadError(
                                "Invalid Content-Length for " + repo_path
                            )
                    response_revision = response.headers.get("X-Repo-Commit")
                    if (
                        response_revision
                        and response_revision.lower() != revision.lower()
                    ):
                        raise DownloadError(
                            "Hugging Face served {0} from an unexpected revision.".format(
                                repo_path
                            )
                        )
                    etag = response.headers.get("ETag")
                    while True:
                        chunk = response.read(DOWNLOAD_CHUNK_BYTES)
                        if not chunk:
                            break
                        temporary_handle.write(chunk)
                        digest.update(chunk)
                        received_bytes += len(chunk)

                temporary_handle.flush()
                os.fsync(temporary_handle.fileno())

            if expected_bytes is not None and received_bytes != expected_bytes:
                raise DownloadError(
                    "{0} ended at {1} bytes; HTTP declared {2} bytes.".format(
                        repo_path, received_bytes, expected_bytes
                    )
                )
            if received_bytes == 0:
                raise DownloadError(repo_path + " downloaded as an empty file.")

            os.chmod(str(temporary_path), 0o600)
            os.replace(str(temporary_path), str(target))
            _fsync_directory(target.parent)
        except Exception:
            _unlink_if_present(temporary_path)
            raise

        result = {
            "bytes": received_bytes,
            "sha256": digest.hexdigest(),
        }
        if etag:
            result["etag"] = etag
        return result


def _unlink_if_present(path: Path) -> None:
    try:
        path.unlink()
    except FileNotFoundError:
        pass


def _fsync_directory(directory: Path) -> None:
    if not hasattr(os, "O_DIRECTORY"):
        return
    descriptor = os.open(str(directory), os.O_RDONLY | os.O_DIRECTORY)
    try:
        os.fsync(descriptor)
    finally:
        os.close(descriptor)


def _missing_column_description(
    columns: Sequence[str],
    required_groups: Sequence[Tuple[str, ...]],
) -> List[str]:
    available = set(columns)
    missing = []
    for alternatives in required_groups:
        if not any(name in available for name in alternatives):
            if len(alternatives) == 1:
                missing.append(alternatives[0])
            else:
                missing.append("one of {" + ", ".join(alternatives) + "}")
    return missing


def inspect_csv(
    csv_path: Path,
    required_groups: Sequence[Tuple[str, ...]],
) -> Dict:
    """Validate a CSV header and count logical records without loading it."""

    csv.field_size_limit(MAX_CSV_FIELD_BYTES)
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_handle:
        reader = csv.reader(csv_handle)
        try:
            raw_columns = next(reader)
        except StopIteration:
            raise ValidationError(str(csv_path) + " is empty.")

        columns = [column.strip() for column in raw_columns]
        if not columns or all(not column for column in columns):
            raise ValidationError(str(csv_path) + " has no CSV header.")
        if len(columns) != len(set(columns)):
            raise ValidationError(str(csv_path) + " has duplicate CSV columns.")

        missing = _missing_column_description(columns, required_groups)
        if missing:
            raise ValidationError(
                "{0} is missing required columns: {1}".format(
                    csv_path.name, ", ".join(missing)
                )
            )

        row_count = 0
        for record_number, row in enumerate(reader, start=2):
            if not row or all(value == "" for value in row):
                continue
            if len(row) != len(columns):
                raise ValidationError(
                    "{0} has {1} fields at CSV record {2}; expected {3}.".format(
                        csv_path.name, len(row), record_number, len(columns)
                    )
                )
            row_count += 1

    if row_count == 0:
        raise ValidationError(csv_path.name + " contains no data records.")
    return {
        "columns": columns,
        "rows": row_count,
    }


def extract_documented_interaction_count(readme_text: str) -> int:
    for pattern in DOCUMENTED_INTERACTION_PATTERNS:
        match = pattern.search(readme_text)
        if match:
            return int(match.group(1).replace(",", ""))
    raise ValidationError(
        "The pinned README no longer exposes a numeric interaction count, so "
        "the known LFS truncation cannot be ruled out automatically."
    )


def validate_staged_snapshot(staging_root: Path) -> Dict:
    readme_path = staging_root / "README.md"
    try:
        readme_text = readme_path.read_text(encoding="utf-8")
    except (UnicodeDecodeError, OSError) as error:
        raise ValidationError("README.md could not be read: " + str(error))

    csv_results = {}
    for repo_path, required_groups in REQUIRED_COLUMN_GROUPS.items():
        csv_results[repo_path] = inspect_csv(
            staging_root / repo_path,
            required_groups,
        )

    documented_rows = extract_documented_interaction_count(readme_text)
    observed_rows = csv_results["Data/Interactions.csv"]["rows"]
    observed_ratio = observed_rows / float(documented_rows)
    if observed_ratio < MIN_DOCUMENTED_INTERACTION_RATIO:
        raise ValidationError(
            "Interactions.csv contains {0} rows, only {1:.1%} of the {2} rows "
            "documented by the same pinned README. This is consistent with the "
            "known roughly-700k-row Git LFS truncation; redownload the current "
            "official revision.".format(
                observed_rows, observed_ratio, documented_rows
            )
        )

    interaction_columns = csv_results["Data/Interactions.csv"]["columns"]
    student_id_column = (
        "user_id" if "user_id" in interaction_columns else "user_xid"
    )
    return {
        "csv": csv_results,
        "documented_interaction_rows": documented_rows,
        "observed_interaction_rows": observed_rows,
        "observed_to_documented_ratio": round(observed_ratio, 6),
        "minimum_documented_ratio": MIN_DOCUMENTED_INTERACTION_RATIO,
        "student_id_column": student_id_column,
        "known_lfs_truncation_check": "passed",
    }


def _utc_timestamp() -> str:
    timestamp = datetime.datetime.now(datetime.timezone.utc)
    return timestamp.replace(microsecond=0).isoformat().replace("+00:00", "Z")


def _write_json_atomically(target: Path, payload: Dict) -> None:
    target.parent.mkdir(parents=True, exist_ok=True)
    temporary_handle = tempfile.NamedTemporaryFile(
        mode="w",
        encoding="utf-8",
        prefix="." + target.name + ".",
        suffix=".part",
        dir=str(target.parent),
        delete=False,
    )
    temporary_path = Path(temporary_handle.name)
    try:
        with temporary_handle:
            json.dump(payload, temporary_handle, indent=2, sort_keys=True)
            temporary_handle.write("\n")
            temporary_handle.flush()
            os.fsync(temporary_handle.fileno())
        os.chmod(str(temporary_path), 0o600)
        os.replace(str(temporary_path), str(target))
        _fsync_directory(target.parent)
    except Exception:
        _unlink_if_present(temporary_path)
        raise


def _preflight_destination(destination: Path) -> None:
    """Require a new/empty destination so a download never overwrites data."""
    if destination.is_symlink():
        raise DownloadError("Refusing to publish into symlink: " + str(destination))
    if not destination.exists():
        return
    if not destination.is_dir():
        raise DownloadError("Destination is not a directory: " + str(destination))
    try:
        next(destination.iterdir())
    except StopIteration:
        destination.rmdir()
        return
    raise DownloadError(
        "Destination is not empty; refusing to overwrite an existing snapshot: "
        + str(destination)
    )


def download_snapshot(
    client: HubClient,
    destination: Path,
    requested_revision: str = "main",
    progress=None,
) -> Dict:
    """Download, validate, then publish the selected immutable snapshot."""

    destination = destination.expanduser().resolve()
    destination.parent.mkdir(parents=True, exist_ok=True)
    _preflight_destination(destination)

    resolved_revision = client.resolve_revision(requested_revision)
    if progress:
        progress(
            "Resolved {0!r} to {1}.".format(
                requested_revision, resolved_revision
            )
        )

    staging_directory = Path(
        tempfile.mkdtemp(
            prefix="." + destination.name + ".download-",
            dir=str(destination.parent),
        )
    )
    file_metadata = {}
    try:
        for repo_path in SELECTED_FILES:
            if progress:
                progress("Downloading " + repo_path + "...")
            file_metadata[repo_path] = client.download_file(
                repo_path,
                resolved_revision,
                staging_directory / repo_path,
            )

        validation = validate_staged_snapshot(staging_directory)
        for repo_path, csv_metadata in validation["csv"].items():
            file_metadata[repo_path].update(csv_metadata)

        manifest = {
            "schema_version": 1,
            "created_at_utc": _utc_timestamp(),
            "source": {
                "repo_id": REPO_ID,
                "repo_url": REPO_URL,
                "requested_revision": requested_revision,
                "resolved_revision": resolved_revision,
                "access": "pre-existing-approved-hugging-face-authentication",
            },
            "selected_files": list(SELECTED_FILES),
            "files": file_metadata,
            "validation": validation,
        }
        _write_json_atomically(staging_directory / MANIFEST_NAME, manifest)

        # Recheck after network I/O in case another process populated the path.
        _preflight_destination(destination)
        os.replace(str(staging_directory), str(destination))
        _fsync_directory(destination.parent)
    finally:
        shutil.rmtree(str(staging_directory), ignore_errors=True)

    return manifest


def build_argument_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Download the approved, gated FoundationalASSIST CSV release from "
            "the official ASSISTments Hugging Face repository."
        )
    )
    parser.add_argument(
        "--destination",
        type=Path,
        default=DEFAULT_DESTINATION,
        help=(
            "Dataset root (default: {0}). Official CSVs are placed in its Data/ "
            "subdirectory.".format(DEFAULT_DESTINATION)
        ),
    )
    parser.add_argument(
        "--revision",
        default="main",
        help=(
            "Hub branch, tag, or commit to resolve and pin before downloading "
            "(default: main)."
        ),
    )
    parser.add_argument(
        "--timeout",
        type=float,
        default=60.0,
        help="Per-request timeout in seconds (default: 60).",
    )
    return parser


def main(argv: Optional[Sequence[str]] = None) -> int:
    arguments = build_argument_parser().parse_args(argv)
    if arguments.timeout <= 0:
        print("error: --timeout must be greater than zero.", file=sys.stderr)
        return 2

    try:
        token = load_existing_hf_token()
        client = HubClient(token, timeout_seconds=arguments.timeout)
        manifest = download_snapshot(
            client,
            arguments.destination,
            requested_revision=arguments.revision,
            progress=print,
        )
    except FoundationalAssistError as error:
        print("error: " + str(error), file=sys.stderr)
        return 1
    except KeyboardInterrupt:
        print("error: download interrupted.", file=sys.stderr)
        return 130

    destination = arguments.destination.expanduser().resolve()
    print(
        "Validated {0} interactions from revision {1}.".format(
            manifest["validation"]["observed_interaction_rows"],
            manifest["source"]["resolved_revision"],
        )
    )
    print("Files: " + str(destination))
    print("Manifest: " + str(destination / MANIFEST_NAME))
    return 0


if __name__ == "__main__":
    sys.exit(main())
