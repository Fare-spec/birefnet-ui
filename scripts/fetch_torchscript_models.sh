#!/usr/bin/env sh
set -eu

MODEL_DIR="${MODEL_DIR:-${1:-./models}}"
MODEL_BASE_URL="${BIREFNET_MODEL_BASE_URL:-}"

if [ -z "$MODEL_BASE_URL" ]; then
    echo "Missing BIREFNET_MODEL_BASE_URL." >&2
    echo "Example: BIREFNET_MODEL_BASE_URL=https://your-host.example/models ./scripts/fetch_torchscript_models.sh" >&2
    exit 1
fi

mkdir -p "$MODEL_DIR"

validate_torchscript() {
    path="$1"
    magic="$(od -An -tx1 -N4 "$path" 2>/dev/null | tr -d ' \n')"
    [ "$magic" = "504b0304" ]
}

download_one() {
    filename="$1"
    target="$MODEL_DIR/$filename"
    url="${MODEL_BASE_URL%/}/$filename"

    if [ -s "$target" ] && validate_torchscript "$target"; then
        echo "Skipping $filename: already present"
        return 0
    fi

    rm -f "$target" "$target.tmp"
    echo "Downloading $url"
    wget -O "$target.tmp" "$url"

    if ! validate_torchscript "$target.tmp"; then
        rm -f "$target.tmp"
        echo "Invalid TorchScript file downloaded for $filename" >&2
        exit 1
    fi

    mv "$target.tmp" "$target"
}

download_one "birefnet-lite.ts"
download_one "birefnet-base.ts"
download_one "birefnet-hr.ts"
