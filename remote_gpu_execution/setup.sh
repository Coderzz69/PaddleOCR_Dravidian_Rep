#!/usr/bin/env bash
set -euo pipefail

# Run this file from the cluster home/work directory on the login node.
# Internet is required here. Compute-node training should use train.py.

BASE_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="${REPO_URL:-https://github.com/Coderzz69/PaddleOCR_Dravidian_Rep.git}"
REPO_DIR="${REPO_DIR:-${BASE_DIR}/PaddleOCR_Dravidian_Rep}"
CONDA_SH="${CONDA_SH:-${HOME}/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-paddle_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PADDLE_VERSION="${PADDLE_VERSION:-3.3.1}"
PADDLE_INDEX_URL="${PADDLE_INDEX_URL:-https://www.paddlepaddle.org.cn/packages/stable/cu126/}"
PRETRAIN_URL="${PRETRAIN_URL:-https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ta_PP-OCRv5_mobile_rec_pretrained.pdparams}"

echo "Base directory: ${BASE_DIR}"
echo "Repository: ${REPO_DIR}"

if [ ! -d "${REPO_DIR}/.git" ]; then
  git clone "${REPO_URL}" "${REPO_DIR}"
else
  git -C "${REPO_DIR}" pull --ff-only
fi

eval "$("${CONDA_SH}" shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${CONDA_ENV}"

cd "${REPO_DIR}"
python -m pip install --upgrade pip setuptools wheel
python -m pip uninstall -y paddlepaddle paddlepaddle-gpu >/dev/null 2>&1 || true
python -m pip install \
  --extra-index-url "${PADDLE_INDEX_URL}" \
  "paddlepaddle-gpu==${PADDLE_VERSION}" \
  -r requirements.txt \
  datasets \
  regex

mkdir -p pretrain_models
if [ ! -f "pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained.pdparams" ]; then
  curl -L --retry 5 --retry-delay 5 \
    -o pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained.pdparams \
    "${PRETRAIN_URL}"
else
  echo "Pretrained model already exists."
fi

cd "${BASE_DIR}"
python download_dataset.py
python sanity_check.py

echo "Setup complete. Submit/run train.py on the GPU compute node."
