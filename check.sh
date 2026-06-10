#!/usr/bin/env sh
set -eu

FEATURES=""
SCRIPT_DIR="$(CDPATH= cd -- "$(dirname -- "$0")" && pwd)"
PROJECT_ROOT="$(dirname "$SCRIPT_DIR")"

if [ -d "$PROJECT_ROOT/.venv" ] && [ -z "${VIRTUAL_ENV:-}" ]; then
    export VIRTUAL_ENV="$PROJECT_ROOT/.venv"
fi

if [ -n "${VIRTUAL_ENV:-}" ] && [ -d "$VIRTUAL_ENV/bin" ]; then
    export PATH="$VIRTUAL_ENV/bin:$PATH"
    export LIBTORCH_USE_PYTORCH="${LIBTORCH_USE_PYTORCH:-1}"
fi

if [ -n "${LIBTORCH:-}" ]; then
    export LD_LIBRARY_PATH="$LIBTORCH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif [ -n "${LIBTORCH_USE_PYTORCH:-}" ]; then
    FEATURES=""
else
    FEATURES="--features download-libtorch"
fi

export LIBTORCH_BYPASS_VERSION_CHECK="${LIBTORCH_BYPASS_VERSION_CHECK:-1}"

cargo check $FEATURES
cargo test $FEATURES
