# 3d-print web app (browser uploader + POST /print) on a CUDA GPU, CPU fallback.
#
#   docker build -t 3d-print .
#   docker run --gpus all -p 8000:8000 -e PRINT_API_TOKEN=... -v 3d-print-models:/models 3d-print
#
# Weights (~2.7 GB) download into /models on first use; pass
# --build-arg BAKE_WEIGHTS=1 to bake them into the image instead.

# torchmcubes compiles against torch's CUDA, so build its wheel on the CUDA
# devel image (13.2 matches torch 2.15's cu13.2 wheels); the runtime stage only
# needs the CUDA libs that ship inside the torch wheels.
FROM nvidia/cuda:13.2.1-devel-ubuntu24.04 AS mcubes
COPY --from=ghcr.io/astral-sh/uv:0.9 /uv /usr/local/bin/uv
RUN apt-get update && apt-get install -y --no-install-recommends git build-essential \
    && rm -rf /var/lib/apt/lists/*
ENV UV_PYTHON_INSTALL_DIR=/opt/python CUDA_HOME=/usr/local/cuda TORCH_CUDA_ARCH_LIST="7.5;8.0;8.6;8.9;9.0"
RUN uv venv -p 3.11 /opt/venv
ENV VIRTUAL_ENV=/opt/venv PATH=/opt/venv/bin:$PATH
COPY research/triposr_requirements.txt /tmp/req.txt
RUN grep -E '^(torch|cmake|ninja|pybind11|scikit_build_core|numpy)==' /tmp/req.txt > /tmp/build.txt \
    && uv pip install -r /tmp/build.txt pip setuptools wheel \
    && pip wheel --no-build-isolation --no-deps -w /wheels git+https://github.com/tatsy/torchmcubes.git

FROM python:3.11-slim-bookworm
RUN apt-get update && apt-get install -y --no-install-recommends libgl1 libglib2.0-0 libgomp1 \
    && rm -rf /var/lib/apt/lists/*
WORKDIR /app
COPY research/triposr_requirements.txt research/
COPY server/requirements-web.txt server/
COPY --from=mcubes /wheels /wheels
RUN pip install --no-cache-dir -r research/triposr_requirements.txt -r server/requirements-web.txt /wheels/*.whl

ENV MODELS=/models
ENV HF_HOME=$MODELS/hf U2NET_HOME=$MODELS/rembg \
    TRIPOSR_PYTHON=/usr/local/bin/python TRIPOSR_JOBS_DIR=/tmp/jobs \
    TOKENIZERS_PARALLELISM=false PYTHONWARNINGS=ignore PYTHONUNBUFFERED=1
ARG BAKE_WEIGHTS=0
RUN if [ "$BAKE_WEIGHTS" = "1" ]; then python -c "import rembg; from huggingface_hub import hf_hub_download as d; [d('stabilityai/TripoSR', f) for f in ('config.yaml', 'model.ckpt')]; rembg.new_session()"; fi

COPY bin bin
COPY tools tools
COPY server server
COPY research/triposr research/triposr

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=60s \
    CMD python -c "import urllib.request; urllib.request.urlopen('http://localhost:8000/api/health')"
CMD ["uvicorn", "server.app:app", "--host", "0.0.0.0", "--port", "8000"]
