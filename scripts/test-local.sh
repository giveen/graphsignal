#!/bin/bash

set -e

cd "$(dirname "$0")/.."
poetry run python -m scripts.test_local "$@"
