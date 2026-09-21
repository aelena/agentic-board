# --- web UI build (skipped gracefully when web/ has no package.json yet) -------------------------
FROM node:22-alpine AS web
WORKDIR /web
COPY web/ ./
RUN if [ -f package.json ]; then npm ci && npm run build; else mkdir -p dist; fi

# --- API -------------------------------------------------------------------------------------------
FROM python:3.12-slim AS api
ENV PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    REFINER_HOST=0.0.0.0 \
    REFINER_PORT=8000 \
    REFINER_RUNS_DIR=/data/runs \
    REFINER_BOARDS_DIR=/data/boards
WORKDIR /app
COPY pyproject.toml README.md ./
COPY src ./src
RUN pip install --upgrade pip uv && uv pip install --system .
COPY --from=web /web/dist ./web/dist
RUN mkdir -p /data/runs /data/boards
VOLUME ["/data"]
EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s CMD python -c "import httpx,sys; sys.exit(httpx.get('http://localhost:8000/api/health').status_code != 200)"
CMD ["refiner", "serve", "--workers", "2"]
