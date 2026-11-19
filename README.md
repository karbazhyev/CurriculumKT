# CurriculumKT

CurriculumKT studies knowledge tracing with embedding-derived question-to-concept
taxonomies, matched original-label baselines, and mappings back to the original
curriculum. The source includes data preparation, OpenAI embedding pipelines,
clustering, model adapters, training controllers, and evaluation utilities.

This repository publishes source code only. Datasets, dataset samples, embeddings,
checkpoints, prediction exports, experiment results, runtime snapshots, local
research notes, environments, and credentials stay outside Git. The `.gitignore`
allows only the reviewed source locations and the supporting files listed below.

- `scripts/`: preparation, training, evaluation, storage, and verification code.
- `tests/`: offline contract tests using synthetic inputs.
- `research/debates/20261107_map_back_results/inspect_saved_structure.py`:
  an authored checkpoint and mapping diagnostic utility.
- `requirements-embeddings.in`: the retained, retired local-encoder dependency
  input; this is not a dependency specification for the current OpenAI pipeline.

The scripts retain the existing research workflow and its local paths. Training
and evaluation require separately prepared data, frozen protocol files, the
appropriate pyKT sources, and the established Python environment. Those artifacts
are not included here. Publishing these scripts does not launch any workload.

The offline contract tests require NumPy, scikit-learn, threadpoolctl, and the
OpenAI Python SDK. In an environment containing those dependencies, run:

```sh
python -B -m unittest discover -s tests -p 'test_*.py'
```
