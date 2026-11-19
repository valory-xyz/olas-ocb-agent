#!/bin/sh
# The full gate, also run in CI: lint, format, and tests with 100% line coverage.
# Run from anywhere: scripts/check.sh (it uses the checkout's .venv)
set -e
cd "$(dirname "$0")/.."
.venv/bin/ruff check .
.venv/bin/ruff format --check .
.venv/bin/python -m pytest -q -p no:cacheprovider -n 4 --dist loadfile --durations=20
