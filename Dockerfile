FROM rust:1.91-slim-bookworm AS builder
ARG TARGETARCH

RUN apt-get update \
    && apt-get install -y --no-install-recommends ca-certificates gcc \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

COPY Cargo.toml Cargo.lock ./
COPY src ./src

RUN if [ "$TARGETARCH" = "amd64" ]; then \
        cc -O3 -flto -DNDEBUG -march=haswell -mtune=haswell -fomit-frame-pointer -s -o /tmp/fd-lb src/fd_lb.c; \
    else \
        cc -O3 -flto -DNDEBUG -fomit-frame-pointer -s -o /tmp/fd-lb src/fd_lb.c; \
    fi

RUN if [ "$TARGETARCH" = "amd64" ]; then \
        RUSTFLAGS="-C target-cpu=haswell" cargo build --release --bin rinha-fraud; \
    else \
        cargo build --release --bin rinha-fraud; \
    fi

FROM gcr.io/distroless/cc-debian12:nonroot

LABEL org.opencontainers.image.source="https://github.com/henrycunh/rinha26-rust"
LABEL org.opencontainers.image.description="Rust backend for Rinha de Backend 2026"
LABEL org.opencontainers.image.licenses="MIT"

WORKDIR /app
COPY --from=builder /build/target/release/rinha-fraud /app/rinha-fraud
COPY --from=builder /tmp/fd-lb /app/fd-lb

ENV PORT=8080

EXPOSE 8080
ENTRYPOINT ["/app/rinha-fraud"]
