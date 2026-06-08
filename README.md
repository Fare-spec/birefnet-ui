# BiRefNet UI

Rust web UI and HTTP API for batch background removal with BiRefNet TorchScript models.

The application is designed for Docker deployment. The final image contains the Rust binary plus the Python `torch` / `torchvision` runtime needed by the exported BiRefNet TorchScript models. It does not ship user uploads or BiRefNet model files.

## Features

- Upload one or many images.
- Select one of several BiRefNet models from the UI or API.
- Before/after preview for every image.
- Transparent, white, black, or custom image background.
- Parallel batch processing.
- Original output dimensions preserved.
- PNG output with lossless encoding.
- Per-image download and download-all ZIP.
- Delete one processed image without losing the other processed outputs.
- Delete all selected/processed images.
- Source EXIF/GPS/device metadata is not copied into output PNG files.
- ZIP entry timestamps are normalized to `1980-01-01 00:00:00`.
- No user image is written to disk by the application.
- Input/intermediate buffers are wiped when they leave the processing path.
- The UI blocks uploads from remote plain HTTP origins. Use HTTPS in production.

## How It Works

The backend is an Axum HTTP server written in Rust. It loads one or more BiRefNet TorchScript `.ts` models with `tch-rs`, which uses libtorch under the hood.

At startup:

1. The server reads `BIREFNET_MODELS` or `BIREFNET_TORCHSCRIPT_PATH`.
2. Each configured TorchScript model is loaded into memory.
3. `/models` exposes the model list to the UI.
4. `/ui` serves the browser interface.
5. `/ui/process` and `/birefnet/remove-background` process multipart image uploads.

For each image:

1. The image is decoded in memory.
2. A `1024x1024` copy is used only for model inference.
3. The predicted mask is resized back to the original image size.
4. The original pixels are reused with the generated alpha mask.
5. The result is optionally composited onto a selected background.
6. A fresh PNG is encoded from RGBA pixels, without copying source metadata.

When several images are selected in the UI, the browser sends several requests in parallel. The server-side batch API also uses Rayon for multi-image processing.

## Models

Do not commit `.ts` model files to Git. They are large and should be treated as deployment artifacts.

The official upstream BiRefNet repositories currently provide Hugging Face source weights, not ready-to-use TorchScript artifacts for this Rust server:

| App model id | Official source repo | Source weight URL | Runtime TorchScript path |
| --- | --- | --- | --- |
| `birefnet-lite` | `ZhengPeng7/BiRefNet_lite` | `https://huggingface.co/ZhengPeng7/BiRefNet_lite/resolve/main/model.safetensors` | `/app/models/birefnet-lite.ts` |
| `birefnet-base` | `ZhengPeng7/BiRefNet` | `https://huggingface.co/ZhengPeng7/BiRefNet/resolve/main/model.safetensors` | `/app/models/birefnet-base.ts` |
| `birefnet-hr` | `ZhengPeng7/BiRefNet_HR` | `https://huggingface.co/ZhengPeng7/BiRefNet_HR/resolve/main/model.safetensors` | `/app/models/birefnet-hr.ts` |

Important: `BIREFNET_MODEL_URLS` must point to exported TorchScript `.ts` files, not to the upstream `.safetensors` files. The `.safetensors` URLs above are the official sources to export from, or to audit the upstream model, but the Rust runtime cannot load them directly.

License note: the upstream BiRefNet GitHub repository is MIT licensed, and the main `ZhengPeng7/BiRefNet` plus `ZhengPeng7/BiRefNet_HR` Hugging Face pages show `License: mit`. The `BiRefNet_lite` page is part of the same official BiRefNet model family, but its crawled Hugging Face page does not display a license badge. For commercial use, keep a copy of the upstream license notice and verify the exact artifact you deploy.

### Export TorchScript Models

There is no official public URL for ready-to-use BiRefNet TorchScript `.ts` artifacts. The repo provides a separate exporter image that uses Python during export. The runtime image also includes Python `torchvision` because current BiRefNet TorchScript exports still reference custom `torchvision` ops such as `deform_conv2d`.

You should not store these `.ts` files as normal Git-tracked files in the repository itself. GitHub enforces a 100 MiB single-object limit for regular Git objects, and these model files are much larger. If you want simple `wget` downloads, host the exported `.ts` files outside normal Git history, for example as release assets or on object storage.

The exporter image is published by GitHub Actions as:

```text
ghcr.io/fare-spec/birefnet-ui:model-exporter-main
```

Export the three default models into the same Docker volume used by the app:

```bash
docker run --rm \
  -v birefnet-models:/app/models \
  -v birefnet-export-cache:/cache \
  ghcr.io/fare-spec/birefnet-ui:model-exporter-main
```

This writes:

```text
/app/models/birefnet-lite.ts
/app/models/birefnet-base.ts
/app/models/birefnet-hr.ts
```

Exporting downloads the official Hugging Face `.safetensors` source weights and traces them to TorchScript. It can take time and several GB of disk space/RAM, especially for Base and HR.

To force regeneration:

```bash
docker run --rm \
  -v birefnet-models:/app/models \
  -v birefnet-export-cache:/cache \
  ghcr.io/fare-spec/birefnet-ui:model-exporter-main --force
```

If you are developing locally and want to build the exporter yourself:

```bash
docker build -f Dockerfile.models -t birefnet-model-exporter .
```

If you already host exported `.ts` files somewhere, there is also a simple downloader script:

```bash
BIREFNET_MODEL_BASE_URL=https://your-host.example/models \
./scripts/fetch_torchscript_models.sh
```

Supported model provisioning modes:

- Mount TorchScript model files into `/app/models`.
- Set `BIREFNET_MODEL_URLS` so the container downloads models at startup.
- Set `BIREFNET_MODELS` manually with exact local paths.

`BIREFNET_MODEL_URLS` format:

```text
id|label|url[|filename];id2|label2|url2[|filename2]
```

Use `BIREFNET_MODEL_URLS` only when you have hosted your own exported `.ts` artifacts.

`BIREFNET_MODELS` format:

```text
id|label|path;id2|label2|path2
```

Example:

```bash
BIREFNET_MODELS='birefnet-lite|BiRefNet Lite|/app/models/birefnet-lite.ts;birefnet-base|BiRefNet Base|/app/models/birefnet-base.ts'
```

If files are mounted in `/app/models`, these standard filenames are detected automatically:

```text
birefnet-base.ts
birefnet-lite.ts
birefnet-hr.ts
```

## Docker

Build from this directory:

```bash
docker build -t birefnet-ui .
```

Run with local model files:

```bash
docker run --rm \
  -p 3000:3000 \
  -v "$PWD/models:/app/models:ro" \
  birefnet-ui
```

Run with TorchScript model download at container startup after hosting your own `.ts` artifacts:

```bash
docker run --rm \
  -p 3000:3000 \
  -v birefnet-models:/app/models \
  -e 'BIREFNET_MODEL_URLS=birefnet-lite|BiRefNet Lite|https://your-domain.tld/birefnet-lite.ts|birefnet-lite.ts' \
  birefnet-ui
```

Run with explicit mounted paths:

```bash
docker run --rm \
  -p 3000:3000 \
  -v "$PWD/private-models:/models:ro" \
  -e 'BIREFNET_MODELS=birefnet-lite|BiRefNet Lite|/models/lite.ts;birefnet-base|BiRefNet Base|/models/base.ts' \
  birefnet-ui
```

Open:

```text
http://127.0.0.1:3000/ui
```

### Docker Compose

With model files mounted from a local `./models` directory:

```yaml
services:
  birefnet-ui:
    image: ghcr.io/fare-spec/birefnet-ui:main
    container_name: birefnet-ui
    restart: unless-stopped
    ports:
      - "127.0.0.1:3000:3000"
    volumes:
      - ./models:/app/models:ro
```

Start it:

```bash
docker compose up -d
```

Default Compose setup with the three TorchScript models mounted from a local `./models` directory:

```yaml
services:
  birefnet-ui:
    image: ghcr.io/fare-spec/birefnet-ui:main
    container_name: birefnet-ui
    restart: unless-stopped
    ports:
      - "127.0.0.1:3000:3000"
    environment:
      BIREFNET_MODELS: >-
        birefnet-lite|BiRefNet Lite|/app/models/birefnet-lite.ts;
        birefnet-base|BiRefNet Base|/app/models/birefnet-base.ts;
        birefnet-hr|BiRefNet HR|/app/models/birefnet-hr.ts
    volumes:
      - ./models:/app/models:ro
```

The `127.0.0.1:3000:3000` binding is intentional for remote servers: expose the app through an HTTPS reverse proxy instead of publishing plain HTTP directly to the internet.

Compose setup using Docker volumes and the exporter:

```yaml
services:
  export-models:
    build:
      context: .
      dockerfile: Dockerfile.models
    profiles: ["export"]
    volumes:
      - birefnet-models:/app/models
      - birefnet-export-cache:/cache

  birefnet-ui:
    image: ghcr.io/fare-spec/birefnet-ui:main
    container_name: birefnet-ui
    restart: unless-stopped
    ports:
      - "127.0.0.1:3000:3000"
    volumes:
      - birefnet-models:/app/models:ro

volumes:
  birefnet-models:
  birefnet-export-cache:
```

Run the exporter once:

```bash
docker compose --profile export run --rm export-models
```

Then start the app:

```bash
docker compose up -d birefnet-ui
```

### What The Docker Image Contains

The Dockerfile is multi-stage:

- Builder stage: `python:3.13-slim`
- Runtime stage: `python:3.13-slim`

During the build, Cargo compiles the Rust app against the Python-installed PyTorch runtime:

```bash
cargo build --release
```

The final runtime image receives:

- `/usr/local/bin/birefnet`
- `docker-entrypoint.sh`
- Python `torch`
- Python `torchvision`
- minimal Debian runtime libraries

The final image does not include:

- model files
- local `target/`
- local `models/`

Alpine is intentionally not used because the PyTorch / torchvision runtime stack for these models targets glibc, while Alpine uses musl.

### Docker Entrypoint

`docker-entrypoint.sh` runs before the Rust binary.

It does the following:

1. Creates `BIREFNET_MODEL_DIR`, defaulting to `/app/models`.
2. Downloads models declared in `BIREFNET_MODEL_URLS` if they are missing.
3. Builds `BIREFNET_MODELS` automatically from downloaded files.
4. Detects standard mounted model filenames if `BIREFNET_MODELS` is still empty.
5. Refuses to start if no model is configured.
6. Starts the Rust server.

This keeps the image small enough to publish normally while keeping heavy models outside Git and outside the image.

### Troubleshooting Model Files

If the container exits with:

```text
PytorchStreamReader failed reading zip archive: failed finding central directory
```

then the file exists, but it is not a TorchScript archive. The most common cause is downloading Hugging Face `model.safetensors` and naming it `birefnet-*.ts`.

Check the first bytes of the files:

```bash
docker run --rm \
  -v birefnet-models:/app/models \
  alpine sh -c 'for f in /app/models/*.ts; do printf "%s " "$f"; od -An -tx1 -N4 "$f"; done'
```

Valid TorchScript files should start with zip magic:

```text
50 4b 03 04
```

If they do not, remove the bad volume files and replace them with exported TorchScript files:

```bash
docker run --rm \
  -v birefnet-models:/app/models \
  alpine sh -c 'rm -f /app/models/birefnet-*.ts'
```

The upstream `.safetensors` URLs listed in the Models section are source weights. They must be exported to TorchScript before this Rust runtime can load them.

## Remote Server Deployment

For a remote server, run the container behind an HTTPS reverse proxy. The UI allows:

- `https://...`
- `http://localhost`
- `http://127.0.0.1`

It blocks uploads from remote `http://...` origins because those uploads would not be protected by TLS.

Example with a reverse proxy:

```text
Internet -> HTTPS reverse proxy -> http://127.0.0.1:3000 inside Docker host
```

Container command:

```bash
docker run -d \
  --name birefnet-ui \
  --restart unless-stopped \
  -p 127.0.0.1:3000:3000 \
  -v /srv/birefnet-models:/app/models:ro \
  -e 'BIREFNET_MODELS=birefnet-lite|BiRefNet Lite|/app/models/birefnet-lite.ts;birefnet-base|BiRefNet Base|/app/models/birefnet-base.ts;birefnet-hr|BiRefNet HR|/app/models/birefnet-hr.ts' \
  ghcr.io/fare-spec/birefnet-ui:main
```

Then configure your reverse proxy to forward HTTPS traffic to `127.0.0.1:3000`.

## NixOS Auto Setup

On NixOS, you can let systemd run the model exporter once, then start the UI container. This does not require cloning the repository on the server.

Example `configuration.nix`:

```nix
{ pkgs, ... }:

{
  virtualisation.docker.enable = true;

  systemd.services.birefnet-model-export = {
    description = "Export BiRefNet TorchScript models into a Docker volume";
    after = [ "docker.service" "network-online.target" ];
    wants = [ "network-online.target" ];
    requires = [ "docker.service" ];
    wantedBy = [ "multi-user.target" ];

    serviceConfig = {
      Type = "oneshot";
      RemainAfterExit = true;
    };

    path = [ pkgs.docker pkgs.coreutils ];

    script = ''
      set -eu

      docker volume create birefnet-models >/dev/null
      docker volume create birefnet-export-cache >/dev/null

      if docker run --rm \
        -v birefnet-models:/app/models \
        alpine sh -c 'test -s /app/models/birefnet-lite.ts -a -s /app/models/birefnet-base.ts -a -s /app/models/birefnet-hr.ts'; then
        echo "BiRefNet TorchScript models already exist."
        exit 0
      fi

      docker run --rm \
        -v birefnet-models:/app/models \
        alpine sh -c 'rm -f /app/models/birefnet-*.ts'

      docker run --rm \
        -v birefnet-models:/app/models \
        -v birefnet-export-cache:/cache \
        ghcr.io/fare-spec/birefnet-ui:model-exporter-main
    '';
  };

  systemd.services.birefnet-ui = {
    description = "BiRefNet UI Docker container";
    after = [ "docker.service" "birefnet-model-export.service" ];
    requires = [ "docker.service" "birefnet-model-export.service" ];
    wantedBy = [ "multi-user.target" ];

    path = [ pkgs.docker pkgs.coreutils ];

    preStart = ''
      docker rm -f birefnet-ui 2>/dev/null || true
      docker pull ghcr.io/fare-spec/birefnet-ui:main
    '';

    script = ''
      exec docker run --rm \
        --name birefnet-ui \
        -p 127.0.0.1:3000:3000 \
        -v birefnet-models:/app/models:ro \
        ghcr.io/fare-spec/birefnet-ui:main
    '';

    serviceConfig = {
      ExecStop = "${pkgs.docker}/bin/docker stop birefnet-ui";
      Restart = "always";
      RestartSec = "10s";
    };
  };
}
```

Apply it:

```bash
sudo nixos-rebuild switch
```

Watch the exporter logs:

```bash
journalctl -u birefnet-model-export.service -f
```

Watch the app logs:

```bash
journalctl -u birefnet-ui.service -f
```

If you need to regenerate the models:

```bash
sudo systemctl stop birefnet-ui.service
docker run --rm -v birefnet-models:/app/models alpine sh -c 'rm -f /app/models/birefnet-*.ts'
sudo systemctl restart birefnet-model-export.service
sudo systemctl restart birefnet-ui.service
```

## GitHub Container Registry

The repository includes a GitHub Actions workflow that builds and publishes both Docker images to GHCR on pushes to `main`.

Expected image names:

```text
ghcr.io/fare-spec/birefnet-ui:main
ghcr.io/fare-spec/birefnet-ui:model-exporter-main
```

If the package is private, authenticate before pulling:

```bash
echo "$GHCR_TOKEN" | docker login ghcr.io -u Fare-spec --password-stdin
docker pull ghcr.io/fare-spec/birefnet-ui:main
docker pull ghcr.io/fare-spec/birefnet-ui:model-exporter-main
```

## Local Development

Docker is preferred. For local Rust development:

```bash
./run.sh
```

`run.sh` uses `LIBTORCH` if it is set. Otherwise it builds with `download-libtorch`.
For this project, `LIBTORCH_USE_PYTORCH=1` is the most reliable local mode when using BiRefNet models that depend on `torchvision` custom ops.

Manual equivalent:

```bash
LIBTORCH_USE_PYTORCH=1 \
LIBTORCH_BYPASS_VERSION_CHECK=1 \
BIREFNET_MODELS='birefnet-base|BiRefNet Base|models/birefnet-base.ts' \
cargo run --release
```

Run checks:

```bash
./check.sh
```

Limit CPU concurrency on small servers:

```bash
RAYON_NUM_THREADS=2 ./run.sh
```

## API

Endpoint:

```text
POST /birefnet/remove-background
```

Multipart fields:

- `images`, `files`, or `file`: one or more image files.
- `model`: model id exposed by `/models`, for example `birefnet-lite`.
- `bg_mode`: `transparent`, `white`, `black`, or `image`.
- `background_image`: required when `bg_mode=image`.

Example:

```bash
curl \
  -F 'images=@photo-1.jpg' \
  -F 'images=@photo-2.jpg' \
  -F 'model=birefnet-lite' \
  -F 'bg_mode=image' \
  -F 'background_image=@background.jpg' \
  http://127.0.0.1:3000/birefnet/remove-background \
  --output results.zip
```

Single-image requests return a PNG. Multi-image requests return a ZIP.

## Privacy And Memory

The application does not persist uploads, outputs, or background images. Data stays in memory during processing. Buffers controlled by the application are zeroized after processing; the final response buffer remains in memory only until the HTTP download is sent and released by the server.

Output PNG files are encoded from pixels with a new encoder. Source EXIF/GPS/device metadata, comments, and source creation dates are not copied. ZIP entries use the neutral timestamp `1980-01-01 00:00:00`.
