#!/usr/bin/env bash
#SBATCH --job-name=akshara_ocr
#SBATCH --output=akshara_ocr_%j.out
#SBATCH --error=akshara_ocr_%j.err
#SBATCH --partition=gpu
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --gres=gpu:1
#SBATCH --time=24:00:00
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G

set -euo pipefail

# Run this script on the GPU compute node. It performs no downloads.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_NAME="${REPO_NAME:-PaddleOCR_Dravidian}"
CONDA_SH="${CONDA_SH:-/home/user/miniconda3/bin/conda}"
CONDA_ENV="${CONDA_ENV:-paddle_env}"

if [ -f "${SCRIPT_DIR}/../tools/train.py" ]; then
  WORKDIR="$(cd "${SCRIPT_DIR}/.." && pwd)"
elif [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "${SLURM_SUBMIT_DIR}/${REPO_NAME}/tools/train.py" ]; then
  WORKDIR="${SLURM_SUBMIT_DIR}/${REPO_NAME}"
else
  WORKDIR="${SCRIPT_DIR}/${REPO_NAME}"
fi

cd "${WORKDIR}"
echo "Working directory: ${WORKDIR}"

eval "$("${CONDA_SH}" shell.bash hook)"
conda activate "${CONDA_ENV}"

CONFIG="${CONFIG:-configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml}"
PRETRAIN_MODEL="${PRETRAIN_MODEL:-./pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained}"
DICT_PATH="${DICT_PATH:-akshara/dict.txt}"
SAVE_MODEL_DIR="${SAVE_MODEL_DIR:-./output/akshara_gpu_finetune}"
BATCH_SIZE="${BATCH_SIZE:-128}"
EPOCH_NUM="${EPOCH_NUM:-75}"
RUN_VERIFICATION="${RUN_VERIFICATION:-1}"

if [ -d "./colab_dataset" ]; then
  TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-./colab_dataset}"
  TRAIN_LABELS="${TRAIN_LABELS:-./colab_dataset/train_list.txt}"
  VAL_DATA_DIR="${VAL_DATA_DIR:-./colab_dataset}"
  VAL_LABELS="${VAL_LABELS:-./colab_dataset/val_list.txt}"
else
  TRAIN_DATA_DIR="${TRAIN_DATA_DIR:-./train_data}"
  TRAIN_LABELS="${TRAIN_LABELS:-./train_data/train_list.txt}"
  VAL_DATA_DIR="${VAL_DATA_DIR:-./train_data}"
  VAL_LABELS="${VAL_LABELS:-./train_data/val_list.txt}"
fi

missing=0
for path in \
  "${CONFIG}" \
  "${DICT_PATH}" \
  "${PRETRAIN_MODEL}.pdparams" \
  "${TRAIN_DATA_DIR}" \
  "${TRAIN_LABELS}" \
  "${VAL_DATA_DIR}" \
  "${VAL_LABELS}"; do
  if [ ! -e "${path}" ]; then
    echo "ERROR: missing required path: ${path}" >&2
    missing=1
  fi
done
if [ "${missing}" -ne 0 ]; then
  exit 1
fi

nvidia-smi
python - <<'PY'
import paddle
print("paddle:", paddle.__version__)
print("compiled_with_cuda:", paddle.device.is_compiled_with_cuda())
count = paddle.device.cuda.device_count()
print("cuda_device_count:", count)
if not paddle.device.is_compiled_with_cuda() or count < 1:
    raise SystemExit("ERROR: Paddle cannot see a CUDA GPU on this node.")
PY

if [ "${RUN_VERIFICATION}" = "1" ]; then
  python verify_akshara.py
  python verify_model.py
fi

python tools/train.py -c "${CONFIG}" \
  -o Global.use_gpu=True \
     Global.distributed=False \
     Global.epoch_num="${EPOCH_NUM}" \
     Global.pretrained_model="${PRETRAIN_MODEL}" \
     Global.character_dict_path="${DICT_PATH}" \
     Global.save_model_dir="${SAVE_MODEL_DIR}" \
     Train.dataset.data_dir="${TRAIN_DATA_DIR}" \
     "Train.dataset.label_file_list=['${TRAIN_LABELS}']" \
     Eval.dataset.data_dir="${VAL_DATA_DIR}" \
     "Eval.dataset.label_file_list=['${VAL_LABELS}']" \
     Train.sampler.first_bs="${BATCH_SIZE}" \
     Train.loader.batch_size_per_card="${BATCH_SIZE}" \
     Eval.loader.batch_size_per_card="${BATCH_SIZE}"

python tools/export_model.py -c "${CONFIG}" \
  -o Global.pretrained_model="${SAVE_MODEL_DIR}/best_accuracy" \
     Global.character_dict_path="${DICT_PATH}" \
     Global.save_inference_dir=./inference/akshara_tamil_rec

echo "Job finished successfully."
