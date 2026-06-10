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
    if [ -x "$VIRTUAL_ENV/bin/python" ]; then
        PYTHON_LIB_INFO="$("$VIRTUAL_ENV/bin/python" - <<'PY'
import sysconfig
libdir = sysconfig.get_config_var("LIBDIR") or ""
ldlibrary = sysconfig.get_config_var("LDLIBRARY") or ""
purelib = sysconfig.get_paths().get("purelib", "")
torch_lib = f"{purelib}/torch/lib" if purelib else ""
print(libdir)
print(ldlibrary)
print(torch_lib)
PY
)"
        PYTHON_LIBDIR="$(printf '%s\n' "$PYTHON_LIB_INFO" | sed -n '1p')"
        PYTHON_LDLIBRARY="$(printf '%s\n' "$PYTHON_LIB_INFO" | sed -n '2p')"
        TORCH_LIBDIR="$(printf '%s\n' "$PYTHON_LIB_INFO" | sed -n '3p')"

        if [ -n "$PYTHON_LIBDIR" ] && [ -n "$PYTHON_LDLIBRARY" ] && [ -f "$PYTHON_LIBDIR/$PYTHON_LDLIBRARY" ]; then
            export PYTHON_LIBRARY_PATH="${PYTHON_LIBRARY_PATH:-$PYTHON_LIBDIR/$PYTHON_LDLIBRARY}"
            export LD_LIBRARY_PATH="$PYTHON_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi

        if [ -n "$TORCH_LIBDIR" ] && [ -d "$TORCH_LIBDIR" ]; then
            export LD_LIBRARY_PATH="$TORCH_LIBDIR${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
        fi
    fi
fi

if [ -n "${LIBTORCH:-}" ]; then
    export LD_LIBRARY_PATH="$LIBTORCH/lib${LD_LIBRARY_PATH:+:$LD_LIBRARY_PATH}"
elif [ -n "${LIBTORCH_USE_PYTORCH:-}" ]; then
    FEATURES=""
else
    FEATURES="--features download-libtorch"
fi

export LIBTORCH_BYPASS_VERSION_CHECK="${LIBTORCH_BYPASS_VERSION_CHECK:-1}"
export BIREFNET_MODELS="${BIREFNET_MODELS:-birefnet-base|BiRefNet Base|models/birefnet-base.ts;birefnet-lite|BiRefNet Lite|models/birefnet-lite.ts;birefnet-hr|BiRefNet HR|models/birefnet-hr.ts}"
export BIND_ADDR="${BIND_ADDR:-127.0.0.1:3000}"

cargo build --release $FEATURES

if [ -n "${LIBTORCH_USE_PYTORCH:-}" ]; then
    exec target/release/birefnet
fi

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
