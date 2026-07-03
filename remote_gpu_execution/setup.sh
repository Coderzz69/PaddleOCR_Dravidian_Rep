  #!/usr/bin/env bash
set -euo pipefail

# Run this script on the login node before submitting the SLURM job.
# The login node must have internet access. The compute node does not need it.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_URL="${REPO_URL:-https://github.com/Coderzz69/PaddleOCR_Dravidian.git}"
REPO_NAME="${REPO_NAME:-PaddleOCR_Dravidian}"
CONDA_SH="${CONDA_SH:-/home/user/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-paddle_env}"
PYTHON_VERSION="${PYTHON_VERSION:-3.10}"
PADDLE_VERSION="${PADDLE_VERSION:-3.3.1}"
PADDLE_INDEX_URL="${PADDLE_INDEX_URL:-https://www.paddlepaddle.org.cn/packages/stable/cu126/}"
PRETRAIN_URL="${PRETRAIN_URL:-https://paddle-model-ecology.bj.bcebos.com/paddlex/official_pretrained_model/ta_PP-OCRv5_mobile_rec_pretrained.pdparams}"

cd "${SCRIPT_DIR}"
echo "Working directory: ${PWD}"

if [ -f "${SCRIPT_DIR}/../tools/train.py" ]; then
  REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [ -d "${SCRIPT_DIR}/${REPO_NAME}/.git" ]; then
  REPO_DIR="${SCRIPT_DIR}/${REPO_NAME}"
else
  echo "1. Cloning repository..."
  git clone "${REPO_URL}" "${REPO_NAME}"
  REPO_DIR="${SCRIPT_DIR}/${REPO_NAME}"
fi

cd "${REPO_DIR}"
echo "Repository: ${REPO_DIR}"

echo "2. Setting up Conda and Paddle GPU environment..."
eval "$("${CONDA_SH}" shell.bash hook)"
if ! conda env list | awk '{print $1}' | grep -qx "${CONDA_ENV}"; then
  conda create -n "${CONDA_ENV}" "python=${PYTHON_VERSION}" -y
fi
conda activate "${CONDA_ENV}"

python -m pip install --upgrade pip setuptools wheel
python -m pip uninstall -y paddlepaddle paddlepaddle-gpu >/dev/null 2>&1 || true
python -m pip install \
  --extra-index-url "${PADDLE_INDEX_URL}" \
  "paddlepaddle-gpu==${PADDLE_VERSION}" \
  -r requirements.txt \
  regex

echo "3. Preparing dataset, if prep_dataset.py is available..."
if [ -f "${SCRIPT_DIR}/prep_dataset.py" ]; then
  python "${SCRIPT_DIR}/prep_dataset.py"
elif [ -f "${SCRIPT_DIR}/../prep_dataset.py" ]; then
  python "${SCRIPT_DIR}/../prep_dataset.py"
else
  echo "No prep_dataset.py found. Skipping dataset conversion."
  echo "Before training, ensure train/val labels exist under colab_dataset/ or train_data/."
fi

echo "4. Downloading Tamil PP-OCRv5 pretrained recognition weights..."
mkdir -p pretrain_models
if [ ! -f "pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained.pdparams" ]; then
  curl -L --retry 5 --retry-delay 5 \
    -o pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained.pdparams \
    "${PRETRAIN_URL}"
else
  echo "Pretrained weights already exist."
fi

echo "5. Checking Akshara dictionary and CUDA Paddle..."
test -f akshara/dict.txt
python - <<'PY'
import paddle
print("paddle:", paddle.__version__)
print("compiled_with_cuda:", paddle.device.is_compiled_with_cuda())
print("cuda_device_count:", paddle.device.cuda.device_count())
PY

echo "Setup complete. Submit train.sh from ${SCRIPT_DIR}."
