# IOPaint as a CPU-only private service for the Focal site: LaMa erases what the
# user paints over, BRIA RMBG-1.4 removes backgrounds (non-commercial licence).
# Built and started by render.yaml; see services/iopaint/README.md.
FROM python:3.11-slim-bookworm

# opencv-python, an IOPaint dependency, links against these at import time.
RUN apt-get update \
    && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 \
    && rm -rf /var/lib/apt/lists/*

ENV PIP_NO_CACHE_DIR=1 \
    PIP_DISABLE_PIP_VERSION_CHECK=1 \
    PYTHONUNBUFFERED=1 \
    XDG_CACHE_HOME=/models \
    NUMBA_CACHE_DIR=/tmp/numba

WORKDIR /app
COPY requirements.txt constraints.txt ./
# CPU-only PyTorch: the default PyPI wheel pulls in several GB of CUDA libraries.
RUN pip install torch==2.15.0 torchvision==0.29.2 --index-url https://download.pytorch.org/whl/cpu \
    && pip install -r requirements.txt -c constraints.txt

RUN useradd --system --create-home iopaint \
    && mkdir /models \
    && chown iopaint /models
USER iopaint

# Bake the weights into the image; the running service never downloads.
COPY fetch_models.py ./
RUN python fetch_models.py

# server.py keeps one model in memory at a time so the service fits the 1c-2g
# plan; on 4 GB or more, `iopaint start` serves the same endpoints instead.
COPY server.py ./

# One CPU on the 1c-2g plan. The glibc settings hand freed buffers back to the
# OS between requests so peaks stay near 1.8 GB instead of creeping upwards.
ENV HF_HUB_OFFLINE=1 \
    OMP_NUM_THREADS=1 \
    MALLOC_MMAP_THRESHOLD_=131072 \
    MALLOC_ARENA_MAX=2

EXPOSE 8080
CMD ["sh", "-c", "exec uvicorn server:app --host 0.0.0.0 --port \"${PORT:-8080}\""]
