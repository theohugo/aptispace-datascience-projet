FROM python:3.12-slim

ARG DEBIAN_FRONTEND=noninteractive
ARG QUARTO_VERSION=1.7.34
ARG TYPST_VERSION=0.13.1

WORKDIR /workspace

RUN apt-get update \
    && apt-get install -y --no-install-recommends \
    ca-certificates \
    curl \
    wget \
    git \
    unzip \
    xz-utils \
    libc6 \
    libstdc++6 \
    libgcc-s1 \
    && rm -rf /var/lib/apt/lists/*

# Install Quarto CLI (detect architecture)
RUN ARCH=$(dpkg --print-architecture); \
    if [ "$ARCH" = "arm64" ]; then \
      QUARTO_ARCH="arm64"; \
    else \
      QUARTO_ARCH="amd64"; \
    fi; \
    wget -q "https://github.com/quarto-dev/quarto-cli/releases/download/v${QUARTO_VERSION}/quarto-${QUARTO_VERSION}-linux-${QUARTO_ARCH}.deb" -O /tmp/quarto.deb \
    && apt-get update \
    && apt-get install -y --no-install-recommends /tmp/quarto.deb \
    && rm -f /tmp/quarto.deb \
    && rm -rf /var/lib/apt/lists/*

# Install Typst binary (detect architecture)
RUN ARCH=$(dpkg --print-architecture); \
    if [ "$ARCH" = "arm64" ]; then \
      TYPST_ARCH="aarch64-unknown-linux-musl"; \
    else \
      TYPST_ARCH="x86_64-unknown-linux-musl"; \
    fi; \
    wget -q "https://github.com/typst/typst/releases/download/v${TYPST_VERSION}/typst-${TYPST_ARCH}.tar.xz" -O /tmp/typst.tar.xz \
    && mkdir -p /tmp/typst \
    && tar -xJf /tmp/typst.tar.xz -C /tmp/typst --strip-components=1 \
    && install -m 0755 /tmp/typst/typst /usr/local/bin/typst \
    && rm -rf /tmp/typst /tmp/typst.tar.xz

# Install go-task
RUN sh -c "$(curl --location https://taskfile.dev/install.sh)" -- -d -b /usr/local/bin

# Python dependencies
COPY requirements.txt /tmp/requirements.txt
RUN pip install --upgrade pip \
    && pip install --no-cache-dir -r /tmp/requirements.txt

CMD ["bash"]
