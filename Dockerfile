# syntax=docker/dockerfile:1.7

# --- Stage 1: build the Rust MLP server ---------------------------------
FROM rust:1.82-slim AS rust-build
WORKDIR /build
COPY src-tauri/Cargo.toml src-tauri/Cargo.toml
COPY src-tauri/src src-tauri/src
WORKDIR /build/src-tauri
RUN cargo build --release --bin mlp-server

# --- Stage 2: build the frontend static bundle --------------------------
FROM node:20-slim AS web-build
WORKDIR /web
COPY frontend/package.json frontend/package-lock.json ./
RUN npm ci --no-audit --no-fund
COPY frontend/ ./
RUN npm run build

# --- Stage 3: runtime ---------------------------------------------------
FROM python:3.11-slim AS runtime
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    MLP_URL=http://127.0.0.1:9000 \
    FRONTEND_DIST=/app/frontend/dist \
    PORT=8080

WORKDIR /app

COPY backend/pyproject.toml ./backend/pyproject.toml
COPY backend/app ./backend/app
RUN pip install --no-cache-dir -e ./backend

COPY --from=rust-build /build/src-tauri/target/release/mlp-server /usr/local/bin/mlp-server
COPY --from=web-build /web/dist ./frontend/dist

COPY docker/entrypoint.sh /usr/local/bin/entrypoint.sh
RUN chmod +x /usr/local/bin/entrypoint.sh

EXPOSE 8080
ENTRYPOINT ["/usr/local/bin/entrypoint.sh"]
