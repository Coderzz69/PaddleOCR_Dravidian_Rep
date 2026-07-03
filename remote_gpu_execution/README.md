# Remote GPU Training

These scripts are for the cluster workflow where:

- `setup.sh` runs on the login node with internet access.
- `train.sh` runs on the GPU compute node without internet access.
- The scripts are launched from a parent directory such as `IndicOcr_2/`.
- The repository is cloned as `PaddleOCR_Dravidian/`.
- Replace the `user` with you username

## Files

- `setup.sh`: clones/uses the repo, prepares the Conda environment, installs
  CUDA Paddle, optionally runs dataset preparation, and downloads pretrained
  Tamil PP-OCRv5 recognition weights.
- `train.sh`: SLURM-ready GPU training script. It performs no downloads.
- `gpu_setup.sh`: wrapper for `setup.sh`.
- `gpu_train.sh`: wrapper for `train.sh`.

## What Is Fixed Compared To The Old Scripts

The old scripts used:

- `PP-OCRv5_mobile_rec_pretrained.pdparams`
- `tamil_atomic_dict.txt`
- `ppocr/utils/dict/ta_dict.txt`

For this Akshara repository, that is not correct.

These scripts use:

- `pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained.pdparams`
- `akshara/dict.txt`
- `configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml`

Do not copy or regenerate any Tamil dictionary. The verified Akshara dictionary
is already in the repo at `akshara/dict.txt`.

## 1. Login Node Setup

Run on the login node:

```bash
cd /path/to/IndicOcr_2
bash remote_gpu_examples/setup.sh
```

If `remote_gpu_examples/` is inside the cloned repo, run:

```bash
cd /path/to/PaddleOCR_Dravidian
bash remote_gpu_examples/setup.sh
```

Defaults:

- Conda executable: `/home/user/miniconda3/bin/conda`
- Conda environment: `paddle_env`
- Python version for new env: `3.10`
- Paddle package: `paddlepaddle-gpu==3.3.1`
- Paddle CUDA index: `https://www.paddlepaddle.org.cn/packages/stable/cu126/`

Override if your cluster needs different values:

```bash
CONDA_SH=/home/user/miniconda3/bin/conda \
CONDA_ENV=paddle_env \
PADDLE_INDEX_URL=https://www.paddlepaddle.org.cn/packages/stable/cu118/ \
bash remote_gpu_examples/setup.sh
```

The setup script downloads:

```text
pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained.pdparams
```

This is the file used for fine-tuning.

## 2. Submit GPU Training

From the same parent directory:

```bash
sbatch remote_gpu_examples/train.sh
```

Or, if running from inside the repo:

```bash
sbatch remote_gpu_examples/train.sh
```

The training script auto-detects the repo location in either layout.

## 3. Dataset Paths

The script checks for `colab_dataset/` first:

```text
colab_dataset/train_list.txt
colab_dataset/val_list.txt
```

If `colab_dataset/` is absent, it falls back to:

```text
train_data/train_list.txt
train_data/val_list.txt
```

You can override paths:

```bash
TRAIN_DATA_DIR=./my_dataset \
TRAIN_LABELS=./my_dataset/train_list.txt \
VAL_DATA_DIR=./my_dataset \
VAL_LABELS=./my_dataset/val_list.txt \
sbatch remote_gpu_examples/train.sh
```

## 4. Common Training Options

```bash
BATCH_SIZE=64 \
EPOCH_NUM=75 \
SAVE_MODEL_DIR=./output/akshara_gpu_finetune \
sbatch remote_gpu_examples/train.sh
```

Useful variables:

- `CONDA_SH`: path to Conda executable.
- `CONDA_ENV`: Conda environment name.
- `CONFIG`: PaddleOCR recognition config.
- `PRETRAIN_MODEL`: pretrained model prefix without `.pdparams`.
- `DICT_PATH`: dictionary path.
- `BATCH_SIZE`: per-GPU batch size.
- `EPOCH_NUM`: number of epochs.
- `RUN_VERIFICATION`: set `0` to skip `verify_akshara.py` and `verify_model.py`.

## 5. Expected Output

Before training, `train.sh` should print:

```text
compiled_with_cuda: True
cuda_device_count: 1
```

It should also run:

```bash
python verify_akshara.py
python verify_model.py
```

The Paddle training log should show:

```text
character_dict_path : akshara/dict.txt
pretrained_model : ./pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained
```

Shape mismatch warnings for the final recognition heads are expected because the
Tamil pretrained model used the old character dictionary, while this repository
uses the 355-token Akshara dictionary.

## 6. Exported Model

After training, the script exports the best checkpoint to:

```text
inference/akshara_tamil_rec
```

Training checkpoints are saved under:

```text
output/akshara_gpu_finetune
```
