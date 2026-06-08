#!/usr/bin/env sh
set -eu

MODEL_DIR="${BIREFNET_MODEL_DIR:-/app/models}"
mkdir -p "$MODEL_DIR"

append_model_spec() {
    id="$1"
    label="$2"
    path="$3"

    if [ -z "${BIREFNET_MODELS:-}" ]; then
        BIREFNET_MODELS="$id|$label|$path"
    else
        BIREFNET_MODELS="$BIREFNET_MODELS;$id|$label|$path"
    fi
}

download_model() {
    id="$1"
    label="$2"
    url="$3"
    filename="${4:-$id.ts}"
    target="$MODEL_DIR/$filename"

    if [ ! -s "$target" ]; then
        tmp="$target.tmp"
        echo "Downloading $label model to $target"
        curl -fL --retry 3 --retry-delay 2 "$url" -o "$tmp"
        mv "$tmp" "$target"
    fi

    append_model_spec "$id" "$label" "$target"
}

if [ -n "${BIREFNET_MODEL_URLS:-}" ]; then
    OLD_IFS="$IFS"
    IFS=";"
    for item in $BIREFNET_MODEL_URLS; do
        IFS="$OLD_IFS"
        id="$(printf '%s' "$item" | cut -d '|' -f 1)"
        label="$(printf '%s' "$item" | cut -d '|' -f 2)"
        url="$(printf '%s' "$item" | cut -d '|' -f 3)"
        filename="$(printf '%s' "$item" | cut -d '|' -f 4)"
        if [ -z "$id" ] || [ -z "$label" ] || [ -z "$url" ]; then
            echo "Invalid BIREFNET_MODEL_URLS item: $item" >&2
            echo "Expected: id|label|url[|filename];id2|label2|url2[|filename2]" >&2
            exit 1
        fi
        download_model "$id" "$label" "$url" "$filename"
        IFS=";"
    done
    IFS="$OLD_IFS"
fi

if [ -z "${BIREFNET_MODELS:-}" ]; then
    if [ -s "$MODEL_DIR/birefnet-lite.ts" ]; then
        append_model_spec "birefnet-lite" "BiRefNet Lite" "$MODEL_DIR/birefnet-lite.ts"
    fi
    if [ -s "$MODEL_DIR/birefnet-base.ts" ]; then
        append_model_spec "birefnet-base" "BiRefNet Base" "$MODEL_DIR/birefnet-base.ts"
    fi
    if [ -s "$MODEL_DIR/birefnet-hr.ts" ]; then
        append_model_spec "birefnet-hr" "BiRefNet HR" "$MODEL_DIR/birefnet-hr.ts"
    fi
fi

if [ -z "${BIREFNET_MODELS:-}" ]; then
    echo "No BiRefNet TorchScript model configured." >&2
    echo "Mount .ts files in $MODEL_DIR or set BIREFNET_MODEL_URLS." >&2
    exit 1
fi

export BIREFNET_MODELS
exec "$@"
