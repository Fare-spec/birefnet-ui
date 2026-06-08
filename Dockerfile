# syntax=docker/dockerfile:1

FROM rust:1-bookworm AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV LIBTORCH_BYPASS_VERSION_CHECK=1

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        pkg-config \
        unzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release --features download-libtorch \
    && LIBTORCH_DIR="$(find target/release/build -path '*/out/libtorch/libtorch' -type d | head -n 1)" \
    && test -n "$LIBTORCH_DIR" \
    && cp -a "$LIBTORCH_DIR" /opt/libtorch

FROM debian:bookworm-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --home-dir /app --shell /usr/sbin/nologin birefnet

WORKDIR /app

COPY --from=builder /app/target/release/birefnet /usr/local/bin/birefnet
COPY --from=builder /opt/libtorch /opt/libtorch
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /app/models \
    && chown -R birefnet:birefnet /app

ENV BIND_ADDR=0.0.0.0:3000
ENV BIREFNET_MODEL_DIR=/app/models
ENV LIBTORCH=/opt/libtorch
ENV LD_LIBRARY_PATH=/opt/libtorch/lib

EXPOSE 3000

USER birefnet

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["birefnet"]
