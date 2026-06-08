# syntax=docker/dockerfile:1

FROM python:3.13-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive
ENV LIBTORCH_USE_PYTORCH=1
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV PATH="/root/.cargo/bin:${PATH}"

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        build-essential \
        ca-certificates \
        curl \
        pkg-config \
    && rm -rf /var/lib/apt/lists/*

RUN pip install --no-cache-dir torch torchvision

RUN curl --proto '=https' --tlsv1.2 -sSf https://sh.rustup.rs \
    | sh -s -- -y --profile minimal

WORKDIR /app

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release

FROM python:3.13-slim AS runtime

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
        ca-certificates \
        curl \
        libgomp1 \
        libstdc++6 \
    && rm -rf /var/lib/apt/lists/* \
    && useradd --system --create-home --home-dir /app --shell /usr/sbin/nologin birefnet

RUN pip install --no-cache-dir torch torchvision

WORKDIR /app

COPY --from=builder /app/target/release/birefnet /usr/local/bin/birefnet
COPY docker-entrypoint.sh /usr/local/bin/docker-entrypoint.sh

RUN chmod +x /usr/local/bin/docker-entrypoint.sh \
    && mkdir -p /app/models \
    && chown -R birefnet:birefnet /app

ENV BIND_ADDR=0.0.0.0:3000
ENV BIREFNET_MODEL_DIR=/app/models
ENV LIBTORCH_USE_PYTORCH=1
ENV LIBTORCH_BYPASS_VERSION_CHECK=1
ENV LD_LIBRARY_PATH=/usr/local/lib/python3.13/site-packages/torch/lib:/usr/local/lib
ENV PYTHON_LIBRARY_PATH=/usr/local/lib/libpython3.13.so.1.0
ENV TORCHVISION_LIBRARY_PATH=/usr/local/lib/python3.13/site-packages/torchvision/_C.so

EXPOSE 3000

USER birefnet

ENTRYPOINT ["docker-entrypoint.sh"]
CMD ["birefnet"]
