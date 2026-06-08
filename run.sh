#!/usr/bin/env sh
set -eu

FEATURES=""

if [ -n "${LIBTORCH:-}" ]; then
    export LD_LIBRARY_PATH="$LIBTORCH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif [ -n "${LIBTORCH_USE_PYTORCH:-}" ]; then
    FEATURES=""
else
    FEATURES="--features download-libtorch"
fi

export LIBTORCH_BYPASS_VERSION_CHECK="${LIBTORCH_BYPASS_VERSION_CHECK:-1}"
export BIREFNET_MODELS="${BIREFNET_MODELS:-birefnet-base|BiRefNet Base|models/birefnet-base.ts;birefnet-lite|BiRefNet Lite|models/birefnet-lite.ts;birefnet-hr|BiRefNet HR|models/birefnet-hr.ts}"
export BIND_ADDR="${BIND_ADDR:-0.0.0.0:3000}"

cargo build --release $FEATURES

if [ -z "${LIBTORCH:-}" ]; then
    LIBTORCH="$(find target/release/build -path '*/out/libtorch/libtorch' -type d | head -n 1)"
    if [ -z "$LIBTORCH" ]; then
        echo "libtorch not found after build" >&2
        exit 1
    fi
    export LIBTORCH
    export LD_LIBRARY_PATH="$LIBTORCH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
fi

exec target/release/birefnet
