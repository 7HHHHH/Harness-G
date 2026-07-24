#!/usr/bin/env bash
set -euo pipefail

ENV_NAME=${ENV_NAME:-graphr1-sage}
PYTHON_VERSION=${PYTHON_VERSION:-3.11.11}

if ! command -v conda >/dev/null 2>&1; then
  echo "[ERROR] conda not found on PATH."
  exit 1
fi

if ! conda env list | awk '{print $1}' | grep -qx "$ENV_NAME"; then
  conda create -y -n "$ENV_NAME" "python==$PYTHON_VERSION"
fi

conda run -n "$ENV_NAME" python -m pip install --upgrade pip
conda run -n "$ENV_NAME" python -m pip install torch==2.4.0 --index-url https://download.pytorch.org/whl/cu124
conda run -n "$ENV_NAME" python -m pip install -e .
conda run -n "$ENV_NAME" python -m pip install -r requirements.txt
conda run -n "$ENV_NAME" python -m pip install pytest fastapi uvicorn requests
conda run -n "$ENV_NAME" python -m pip install spacy || echo "[WARN] spacy install failed; rule-based extractor remains available."
conda run -n "$ENV_NAME" python -m pip install flash-attn --no-build-isolation || echo "[WARN] flash-attn install failed; continue if training stack supports it."

echo "[Harness-G] conda environment ready: $ENV_NAME"
