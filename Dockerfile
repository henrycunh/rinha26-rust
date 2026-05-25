FROM rust:1.91-slim-bookworm AS builder
ARG TARGETARCH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates curl gzip \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN cargo build --release --bin build-index \
    && if [ "$TARGETARCH" = "amd64" ]; then \
        RUSTFLAGS="-C target-cpu=haswell" cargo build --release --bin rinha-fraud; \
    else \
        cargo build --release --bin rinha-fraud; \
    fi

RUN mkdir -p /out \
    && curl -fsSL --retry 3 \
        https://raw.githubusercontent.com/zanfranceschi/rinha-de-backend-2026/main/resources/references.json.gz \
        -o /tmp/references.json.gz \
    && gzip -cd /tmp/references.json.gz | BUILD_INDEX_IVF_DIMS=2,4 /build/target/release/build-index /out/references.ridx

FROM gcr.io/distroless/cc-debian12:nonroot

LABEL org.opencontainers.image.source="https://github.com/henrycunh/rinha26-rust"
LABEL org.opencontainers.image.description="Rust backend for Rinha de Backend 2026"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app
COPY --from=builder /build/target/release/rinha-fraud /app/rinha-fraud
COPY --from=builder /out/references.ridx /app/references.ridx

ENV PORT=8080
ENV INDEX_PATH=/app/references.ridx

EXPOSE 8080
ENTRYPOINT ["/app/rinha-fraud"]
