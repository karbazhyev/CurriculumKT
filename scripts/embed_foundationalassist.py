#!/usr/bin/env python3
"""Disabled MPNet entry point; retained to stop stale resume commands safely.

The project uses the original OpenAI API embedding method. No API request,
model loading, or embedding generation is implemented by this retired command.
Historical source: results/foundationalassist_20261108/superseded_local_encoder/.
"""
import sys

if __name__ == "__main__":
    print("MPNet embedding runner retired: use the original OpenAI API method "
          "(text-embedding-3-small/768 and text-embedding-3-large/1536). "
          "See research/FOUNDATIONALASSIST_PLAN.md. No work was launched.", file=sys.stderr)
    raise SystemExit(2)
