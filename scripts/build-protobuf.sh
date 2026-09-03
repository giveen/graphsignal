#!/bin/bash
#
# Regenerates graphsignal/proto/signals_pb2.py from proto/signals.proto.
#
# THE GENERATOR IS PINNED, AND THAT PIN IS A USER-VISIBLE DEPENDENCY FLOOR.
# protoc stamps its own version into the generated file as the "gencode"
# version, and at import time google.protobuf refuses to load gencode NEWER
# than the installed runtime. So whatever protoc runs here decides the minimum
# `protobuf` a user must have — and the profiler is installed into containers
# somebody else built, commonly with `--no-deps` so it cannot upgrade anything.
# An unpinned `protoc` from PATH silently raised that floor to 7.36 once and
# made the profiler unimportable in every image shipping protobuf 6.x.
#
# protoc 27.2 stamps gencode 5.27, the lowest that still emits the version
# check at all (it was added in 5.27) — so the generated file loads on any
# runtime from 5.27 up, which is 6.x and 7.x included. Keep this in step with
# the `protobuf` floor in pyproject.toml. If an image ever turns up pinned
# below 5.27, protoc <= 26 emits no check and loads anywhere, at the cost of
# the guard.
#
# uv runs it ephemerally: nothing is installed into the project environment,
# and the version above is the whole toolchain.

set -e

cd "$(dirname "$0")/.."

uv run --quiet --with 'grpcio-tools==1.66.2' \
    python -m grpc_tools.protoc \
    --proto_path=./proto \
    --python_out=./graphsignal/proto \
    ./proto/signals.proto

echo "generated graphsignal/proto/signals_pb2.py with $(uv run --quiet --with 'grpcio-tools==1.66.2' python -m grpc_tools.protoc --version)"
