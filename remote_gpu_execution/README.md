# Remote GPU Execution

This folder contains the cluster helper files for training the [PaddleOCR Dravidian Akshara fork](../README.md) on a remote GPU machine.

> **See also**: [Akshara training guide](../README_AKSHARA.md) · [Architecture notes](../architecture.md)


The runnable files are:

```text
setup.sh
download_dataset.py
sanity_check.py
train.py
```

Only these four files need to be copied to the cluster home/work directory. This
README is documentation for the repository.

## What The Files Do

| File | Run Location | Purpose |
| --- | --- | --- |
| `setup.sh` | Login node with internet | Clones/pulls the repo, creates/uses Conda env, installs Paddle GPU, downloads pretrained weights, downloads dataset, runs sanity checks. |
| `download_dataset.py` | Login node with internet | Downloads `darknight054/indic-mozhi-ocr` Tamil split and converts it to PaddleOCR recognition format. |
| `sanity_check.py` | Login or compute node | Verifies repo paths, Akshara dictionary, pretrained weights, dataset labels, Paddle import, and first label. |
| `train.py` | GPU compute node without internet | Runs sanity check, optional verification scripts, training, and model export. |

## Defaults

Repository:

```text
https://github.com/Coderzz69/PaddleOCR_Dravidian_Rep.git
```

Dataset:

```text
darknight054/indic-mozhi-ocr
config: tamil
```

Converted dataset output:

```text
PaddleOCR_Dravidian_Rep/train_data/indic_mozhi_tamil/
```

Training config and assets:

```text
configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml
akshara/dict.txt
pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained.pdparams
```

## Login Node Setup

Place the four runnable files in a cluster work directory, then run:

```bash
bash setup.sh
```

`setup.sh` installs:

```text
paddlepaddle-gpu==3.3.1
datasets
regex
requirements.txt
```

The script assumes Conda is at:

```text
$HOME/miniconda3/bin/conda
```

Override if needed:

```bash
CONDA_SH=/home/stiwari/miniconda3/bin/conda \
CONDA_ENV=paddle_env \
bash setup.sh
```

For CUDA 11.8 clusters:

```bash
PADDLE_INDEX_URL=https://www.paddlepaddle.org.cn/packages/stable/cu118/ \
bash setup.sh
```

The default and tested Paddle index is CUDA 12.6:

```text
https://www.paddlepaddle.org.cn/packages/stable/cu126/
```

For a small dataset smoke download:

```bash
MAX_TRAIN_SAMPLES=1000 \
MAX_VAL_SAMPLES=100 \
MAX_TEST_SAMPLES=100 \
bash setup.sh
```

## GPU Compute Node Training

Submit the Python training script directly:

```bash
sbatch train.py
```

`train.py` has SLURM headers at the top. It performs no internet downloads.

Useful overrides:

```bash
BATCH_SIZE=64 \
EPOCH_NUM=75 \
SAVE_MODEL_DIR=./output/akshara_gpu_finetune \
sbatch train.py
```

The default training output is:

```text
PaddleOCR_Dravidian_Rep/output/akshara_gpu_finetune/
```

The default exported inference model is:

```text
PaddleOCR_Dravidian_Rep/inference/akshara_tamil_rec/
```

## Verified GPU Smoke Run

A fresh local GPU rehearsal was run on July 6, 2026 using the same remote-style
flow:

1. Deleted and recreated `.venv`.
2. Installed `paddlepaddle-gpu==3.3.1` from the CUDA 12.6 Paddle index.
3. Installed `requirements.txt`, `datasets`, and `regex`.
4. Converted a tiny Tamil subset from `darknight054/indic-mozhi-ocr`.
5. Ran `sanity_check.py`, `verify_akshara.py`, and `verify_model.py`.
6. Ran `train.py` on GPU for 2 epochs and exported the inference model.

Observed environment:

```text
GPU: NVIDIA GeForce RTX 3050 Laptop GPU
Driver: 595.80
CUDA reported by driver: 13.2
Paddle runtime: CUDA 12.6
PaddlePaddle GPU: 3.3.1
datasets: 5.0.0
regex: 2026.6.28
```

The successful training log showed:

```text
train with paddle 3.3.1 and device Place(gpu:0)
character_dict_path : akshara/dict.txt
pretrained_model : ./pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained
load pretrain successful from ./pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained
epoch: [1/2]
epoch: [2/2]
inference model is saved to ./inference/akshara_tamil_rec/inference
```

The 2-epoch smoke output was:

```text
output/remote_gpu_2epoch_smoke/latest.pdparams
output/remote_gpu_2epoch_smoke/latest.pdopt
output/remote_gpu_2epoch_smoke/latest.states
inference/akshara_tamil_rec/inference.pdiparams
inference/akshara_tamil_rec/inference.yml
inference/akshara_tamil_rec/inference.json
```

## CPU Smoke Test

For local testing only, CPU execution is supported:

```bash
REPO_DIR=/path/to/PaddleOCR_Dravidian_Rep \
TRAIN_DATA_DIR=./train_data \
TRAIN_LABELS=./train_data/train_list.txt \
VAL_DATA_DIR=./train_data \
VAL_LABELS=./train_data/val_list.txt \
USE_GPU=False \
RUN_VERIFICATION=0 \
EPOCH_NUM=1 \
BATCH_SIZE=1 \
SAVE_MODEL_DIR=./output/remote_cpu_smoke \
python train.py
```

This was verified locally with PaddlePaddle `3.3.1` on CPU. The smoke run loaded
the Tamil pretrained model, used `akshara/dict.txt`, completed training, and
exported an inference model.

## Expected Training Log Lines

The log should show:

```text
character_dict_path : akshara/dict.txt
pretrained_model : ./pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained
load pretrain successful from ./pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained
```

Shape mismatch warnings for the final recognition heads are expected because the
Tamil pretrained model used the old character dictionary, while this repo uses
the fixed 355-token Akshara dictionary.

## Troubleshooting

If `paddle.device.is_compiled_with_cuda()` is `False`, the CPU Paddle package is
installed instead of the GPU package. Re-run `setup.sh`, or reinstall with the
correct Paddle CUDA index.

If `nvidia-smi` fails on the compute node, the NVIDIA driver is not visible to
the job. Fix the cluster/GPU allocation before training.

If the final export cannot find `best_accuracy.pdparams`, `train.py` falls back
to `latest.pdparams`. This is expected for tiny smoke runs where no best
checkpoint is written.

If the Hugging Face dataset download is slow or rate limited, set `HF_TOKEN`
before running `setup.sh`.

## Guardrails

- Do not edit PaddleOCR architecture or model code from these scripts.
- Do not regenerate or rewrite `akshara/dict.txt`.
- Run `setup.sh` only where internet access is available.
- Run `train.py` on the compute node after setup has completed.
- Use `STRICT_AKSHARA=0` only if you intentionally want to skip labels containing tokens outside the dictionary.
