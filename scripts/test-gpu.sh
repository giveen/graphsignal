#!/bin/bash
# Run the test suites on a remote GPU machine: rsync the repo there, then run
# the Python suite and the native GPU tests.
#
# Usage:
#   scripts/test-gpu.sh [options] [pytest target args...]
#
# Options:
#   --host HOST      remote host (default: $GRAPHSIGNAL_GPU_HOST or dev-sglang-spark-01)
#   --setup          run scripts/init-novenv.sh on the remote first (full dep install)
#   --python-only    run only the Python suite
#   --native-only    run only the native tests
#
# Examples:
#   scripts/test-gpu.sh
#   scripts/test-gpu.sh --setup
#   scripts/test-gpu.sh --python-only test/recorders/test_shm_recorder.py

set -euo pipefail

HOST="${GRAPHSIGNAL_GPU_HOST:-dev-sglang-spark-01}"
REMOTE_DIR="${GRAPHSIGNAL_GPU_DIR:-/tmp/gsrepo}"
SETUP=0
RUN_PYTHON=1
RUN_NATIVE=1
PYTEST_ARGS=()

while [[ $# -gt 0 ]]; do
  case "$1" in
    --host) HOST="$2"; shift 2 ;;
    --setup) SETUP=1; shift ;;
    --python-only) RUN_NATIVE=0; shift ;;
    --native-only) RUN_PYTHON=0; shift ;;
    *) PYTEST_ARGS+=("$1"); shift ;;
  esac
done

cd "$(dirname "$0")/.."

echo "==> rsync -> $HOST:$REMOTE_DIR"
rsync -a --delete \
  --exclude .git --exclude .venv --exclude venv --exclude build --exclude dist \
  --exclude __pycache__ --exclude .pytest_cache \
  ./ "$HOST:$REMOTE_DIR/"

if [[ $SETUP -eq 1 ]]; then
  echo "==> remote setup (init-novenv.sh)"
  ssh "$HOST" "cd $REMOTE_DIR && bash scripts/init-novenv.sh"
fi

if [[ $RUN_PYTHON -eq 1 ]]; then
  echo "==> python suite"
  ssh "$HOST" "cd $REMOTE_DIR \
    && PIP_BREAK_SYSTEM_PACKAGES=1 python3 -m pip install -q -e . --no-deps \
    && python3 -m scripts.test_local ${PYTEST_ARGS[*]:-}"
fi

if [[ $RUN_NATIVE -eq 1 ]]; then
  echo "==> native tests"
  ssh "$HOST" "cd $REMOTE_DIR && bash scripts/test-gpu-native.sh"
fi

echo "==> all done"
