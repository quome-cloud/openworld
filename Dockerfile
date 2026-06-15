# OpenWorld inference server — a stateless FastAPI forward pass over world specs.
#
#   docker build -t openworld .
#   docker run --rm -p 8080:8080 openworld                 # serves the bundled specs/
#   docker run --rm -p 8080:8080 -v "$PWD/specs:/app/specs" openworld   # your own specs
#
# Then open http://localhost:8080/  (interactive /view per world, /docs for the API).
#
# python:3.14-slim tracks the latest 3.14.x patch, so rebuilds pick up CPython
# security fixes. The image installs only the core + serve/CLI layer (FastAPI,
# Uvicorn, Click, Rich); the numpy-backed analysis extras and the experiment
# pipeline are intentionally left out to keep the deployable image small.
FROM python:3.14-slim AS runtime

# Fail fast, no .pyc writes, unbuffered logs, no pip cache layer.
ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1

WORKDIR /app

# Install dependencies first (cached unless packaging metadata changes).
COPY pyproject.toml README.md ./
COPY openworld ./openworld
RUN pip install .

# Default world specs (override by mounting your own onto /app/specs).
COPY specs ./specs

# Run as an unprivileged user.
RUN useradd --create-home --uid 10001 openworld \
    && chown -R openworld:openworld /app
USER openworld

EXPOSE 8080

# Liveness: the registry index returns 200 once uvicorn is up.
HEALTHCHECK --interval=30s --timeout=3s --start-period=10s --retries=3 \
    CMD python -c "import urllib.request,sys; sys.exit(0 if urllib.request.urlopen('http://127.0.0.1:8080/', timeout=2).status==200 else 1)"

ENTRYPOINT ["openworld"]
CMD ["serve", "/app/specs", "--host", "0.0.0.0", "--port", "8080", "--allow-code", "--no-open"]
