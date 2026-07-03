# Akshara PaddleOCR — Training and Usage Guide

This repository is a PaddleOCR checkout adapted for Dravidian/Tamil recognition with Akshara tokenization. For an overview of this fork, see the [main README](./README.md). For architecture and data-layer details, see [architecture.md](./architecture.md).

## Table of Contents

- [What Is Already Configured](#what-is-already-configured)
- [Prerequisites](#prerequisites)
- [Verify The Setup](#verify-the-setup)
- [Dataset Format](#dataset-format)
- [Important Length Setting](#important-length-setting)
- [Smoke Train](#smoke-train)
- [Full Training](#full-training)
- [Resume Training](#resume-training)
- [Evaluate A Trained Checkpoint](#evaluate-a-trained-checkpoint)
- [Run Recognition Inference](#run-recognition-inference)
- [Export For Deployment](#export-for-deployment)
- [Files To Know](#files-to-know)
- [Latest Local Smoke Verification](#latest-local-smoke-verification)
- [Troubleshooting](#troubleshooting)
- [Guardrails](#guardrails)

The OCR model architecture is unchanged. The recognition pipeline still uses PP-OCRv5, `MultiHead`, `CTCHead`, `CTCLoss`, and `CTCLabelDecode`. The only semantic change is that recognition labels are tokenized as Unicode grapheme clusters with:

```python
regex.findall(r"\X", text)
```


## What Is Already Configured

The PP-OCRv5 Akshara recognition configs point to the verified Akshara dictionary:

```yaml
Global:
  character_dict_path: akshara/dict.txt
  use_space_char: false
```

The active dictionary is:

```text
akshara/dict.txt
```

It is byte-identical to the verified backup dictionary:

```text
akshara_backup/dict.txt
```

Do not regenerate, sort, clean, or edit the dictionary unless you intentionally run a new audit.

The Akshara mobile config additionally fine-tunes from:

```text
pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained.pdparams
```

That pretrained model was trained with PaddleOCR's default Tamil character dictionary. Because the Akshara dictionary has a different vocabulary size, PaddleOCR loads all shape-compatible weights and skips the mismatched final classifier/token-embedding tensors. This is expected for Akshara fine-tuning.

## Prerequisites

The system Python in this workspace is Python 3.14, which does not have a compatible PaddlePaddle wheel. Use the repo-local virtual environment:

```bash
.venv/bin/python --version
.venv/bin/python -c "import paddle; print(paddle.__version__)"
```

Known working environment:

```text
Python 3.12.13
PaddlePaddle 3.3.1
regex 2026.6.28
```

If the environment needs to be recreated:

```bash
uv venv --python 3.12 .venv
uv pip install --python .venv/bin/python paddlepaddle -r requirements.txt
```

## Verify The Setup

Run these before training:

```bash
.venv/bin/python verify_akshara.py
.venv/bin/python verify_model.py
```

Expected highlights:

```text
dictionary tokens: 355
ctc vocabulary size: 356
nrtr vocabulary size: 359
Akshara encode -> decode round-trip: OK
classifier output dimension: 356
model forward + CTC loss: OK
```

## Dataset Format

PaddleOCR recognition training uses tab-separated label files.

Each line must be:

```text
relative/image/path.png<TAB>label text
```

Example:

```text
images/img_000001.png	தமிழ்
images/img_000002.png	தரிசனத்தின்
```

The image path is resolved relative to `Train.dataset.data_dir` or `Eval.dataset.data_dir`.

The default PP-OCRv5 configs expect:

```text
train_data/train_list.txt
train_data/val_list.txt
```

with images under:

```text
train_data/
```

For remote GPU training, the helper scripts download and convert:

```text
darknight054/indic-mozhi-ocr
config: tamil
```

The converted dataset is written to:

```text
train_data/indic_mozhi_tamil/train_list.txt
train_data/indic_mozhi_tamil/val_list.txt
train_data/indic_mozhi_tamil/test_list.txt
```

## Important Length Setting

The verified audit reports a maximum label length of 50 Akshara tokens.

The stock PP-OCRv5 config uses:

```yaml
max_text_length: 25
```

If you train on the full verified dataset and want to keep every encodable label, set:

```yaml
Global:
  max_text_length: &max_text_length 50
```

The `NRTRHead` already references this value through `*max_text_length`, so updating this anchor updates the auxiliary head too.

For quick smoke tests, `25` is fine if the subset labels are shorter.

## Smoke Train

A tiny smoke dataset is present only to prove the fine-tuning pipeline runs end-to-end. It is not useful for model quality.

Run:

```bash
.venv/bin/python tools/train.py \
  -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  -o Global.use_gpu=False \
     Global.epoch_num=1 \
     Global.print_batch_step=1 \
     Global.eval_batch_step='[0,1]' \
     Global.save_model_dir=./output/akshara_finetune_smoke \
     Global.save_epoch_step=1 \
     Global.distributed=False \
     Train.sampler.first_bs=2 \
     Train.sampler.fix_bs=True \
     Train.loader.batch_size_per_card=2 \
     Train.loader.drop_last=False \
     Train.loader.num_workers=0 \
     Eval.loader.batch_size_per_card=2 \
     Eval.loader.drop_last=False \
     Eval.loader.num_workers=0
```

Expected result:

```text
train dataloader has 3 iters
valid dataloader has 1 iters
load pretrain successful from ./pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained
save model in ./output/akshara_finetune_smoke/latest
```

Expected shape-mismatch warnings:

```text
head.ctc_head.fc.weight ... [120, 356] ... loaded ... [120, 515]
head.ctc_head.fc.bias ... [356] ... loaded ... [515]
head.gtc_head.embedding.embedding.weight ... [360, 384] ... loaded ... [519, 384]
head.gtc_head.tgt_word_prj.weight ... [384, 360] ... loaded ... [384, 519]
```

These warnings mean the old Tamil output layers do not match the Akshara vocabulary. PaddleOCR skips those tensors and still fine-tunes the compatible pretrained layers.

## Full Training

Prepare your real data:

```text
train_data/train_list.txt
train_data/val_list.txt
train_data/<your images>
```

For full audited labels, update `max_text_length` to `50`.

For the remote cluster workflow, use the four runnable files in
`remote_gpu_execution/`:

```text
setup.sh
download_dataset.py
sanity_check.py
train.py
```

Copy those files to the cluster home/work directory. Then run:

```bash
bash setup.sh
```

on the login node with internet access, followed by:

```bash
sbatch train.py
```

on the GPU compute node. See `remote_gpu_execution/README.md` for the full
cluster instructions.

Single-device fine-tuning:

```bash
.venv/bin/python tools/train.py \
  -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  -o Global.use_gpu=True \
     Global.distributed=False \
     Global.save_model_dir=./output/PP-OCRv5_mobile_rec_akshara
```

CPU training is possible but slow:

```bash
.venv/bin/python tools/train.py \
  -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  -o Global.use_gpu=False \
     Global.distributed=False \
     Global.save_model_dir=./output/PP-OCRv5_mobile_rec_akshara_cpu
```

Multi-GPU training:

```bash
.venv/bin/python -m paddle.distributed.launch \
  --log_dir=./log/ \
  --gpus '0,1,2,3' \
  tools/train.py \
  -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  -o Global.save_model_dir=./output/PP-OCRv5_mobile_rec_akshara
```

Use the server config instead if you want the larger PP-OCRv5 server recognizer:

```bash
.venv/bin/python tools/train.py \
  -c configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml \
  -o Global.save_model_dir=./output/PP-OCRv5_server_rec_akshara
```

Do not use `configs/rec/PP-OCRv5/multi_language/ta_PP-OCRv5_mobile_rec.yaml` for Akshara training as-is. That file is the stock Tamil character-dictionary config. For Akshara fine-tuning, use `configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml`, which points to `akshara/dict.txt`.

## Resume Training

Resume from a saved checkpoint:

```bash
.venv/bin/python tools/train.py \
  -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  -o Global.checkpoints=./output/PP-OCRv5_mobile_rec_akshara/latest \
     Global.save_model_dir=./output/PP-OCRv5_mobile_rec_akshara
```

Use the checkpoint prefix without `.pdparams`, `.pdopt`, or `.states`.

## Evaluate A Trained Checkpoint

Evaluate with the validation list in the config:

```bash
.venv/bin/python tools/eval.py \
  -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  -o Global.checkpoints=./output/PP-OCRv5_mobile_rec_akshara/best_accuracy \
     Global.use_gpu=True
```

For CPU:

```bash
.venv/bin/python tools/eval.py \
  -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  -o Global.checkpoints=./output/PP-OCRv5_mobile_rec_akshara/best_accuracy \
     Global.use_gpu=False
```

## Run Recognition Inference

Run inference on one image or an image directory:

```bash
.venv/bin/python tools/infer_rec.py \
  -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  -o Global.checkpoints=./output/PP-OCRv5_mobile_rec_akshara/best_accuracy \
     Global.infer_img=./path/to/image_or_dir \
     Global.use_gpu=True \
     Global.save_res_path=./output/rec/akshara_predictions.txt
```

CPU inference:

```bash
.venv/bin/python tools/infer_rec.py \
  -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  -o Global.checkpoints=./output/PP-OCRv5_mobile_rec_akshara/best_accuracy \
     Global.infer_img=./path/to/image_or_dir \
     Global.use_gpu=False \
     Global.save_res_path=./output/rec/akshara_predictions.txt
```

Results are written to:

```text
output/rec/akshara_predictions.txt
```

Each line contains the image path and decoded recognition result.

## Export For Deployment

Export a trained checkpoint:

```bash
.venv/bin/python tools/export_model.py \
  -c configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml \
  -o Global.pretrained_model=./output/PP-OCRv5_mobile_rec_akshara/best_accuracy \
     Global.save_inference_dir=./inference/PP-OCRv5_mobile_rec_akshara
```

The exported inference model will be saved under:

```text
inference/PP-OCRv5_mobile_rec_akshara/
```

Keep `akshara/dict.txt` with the exported model or deployment config. The decoder needs the same token order used during training.

## Files To Know

| File | Purpose |
| --- | --- |
| `akshara/dict.txt` | Active 355-token Akshara dictionary. |
| `configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml` | Mobile Akshara recognition training config. |
| `configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml` | Server Akshara recognition training config. |
| `ppocr/data/imaug/label_ops.py` | CTC and NRTR Akshara label encoding. |
| `ppocr/postprocess/rec_postprocess.py` | CTC decoding and word helper utilities. |
| `verify_akshara.py` | Tokenizer, encoder, decoder round-trip verification. |
| `verify_model.py` | Model shape and CTC-loss verification. |
| `remote_gpu_execution/setup.sh` | Login-node setup for remote GPU training. |
| `remote_gpu_execution/download_dataset.py` | Converts `darknight054/indic-mozhi-ocr` Tamil data to PaddleOCR format. |
| `remote_gpu_execution/sanity_check.py` | Checks repo, dictionary, weights, dataset, and Paddle runtime. |
| `remote_gpu_execution/train.py` | SLURM-ready training/export entry point. |
| `remote_gpu_execution/README.md` | Remote GPU execution instructions. |
| `architecture.md` | Architecture and data-layer notes. |

## Latest Local Smoke Verification

The remote execution files were smoke-tested locally on CPU with:

```text
PaddlePaddle 3.3.1
USE_GPU=False
EPOCH_NUM=1
BATCH_SIZE=1
```

Verified outcomes:

- `sanity_check.py` passed.
- `verify_akshara.py` passed.
- `verify_model.py` passed.
- `download_dataset.py` converted one sample each from the real Hugging Face train, validation, and test splits with `unknown=0`.
- `train.py` completed a CPU training smoke run using `akshara/dict.txt`.
- `train.py` exported an inference model to `inference/akshara_tamil_rec/`.

## Troubleshooting

### Shape mismatch warnings during training

Warnings like the following are **expected** and safe to ignore:

```text
head.ctc_head.fc.weight ... [120, 356] ... loaded ... [120, 515]
head.ctc_head.fc.bias ... [356] ... loaded ... [515]
head.gtc_head.embedding.embedding.weight ... [360, 384] ... loaded ... [519, 384]
head.gtc_head.tgt_word_prj.weight ... [384, 360] ... loaded ... [384, 519]
```

These occur because the Tamil pretrained model used a different dictionary size (515 tokens) than the Akshara dictionary (355 tokens). PaddleOCR skips these mismatched tensors and fine-tunes the compatible pretrained layers.

### Python version errors

The system Python may be Python 3.14, which does not have a compatible PaddlePaddle wheel. Always use the repo-local virtual environment:

```bash
.venv/bin/python --version   # Should show 3.12.x
```

See [Prerequisites](#prerequisites) for how to recreate the environment.

### Wrong config file

Do not use `configs/rec/PP-OCRv5/multi_language/ta_PP-OCRv5_mobile_rec.yaml` for Akshara training. That file is the stock Tamil character-dictionary config. For Akshara fine-tuning, always use `configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml`, which points to `akshara/dict.txt`.

## Guardrails

- Do not edit or reorder `akshara/dict.txt`.
- Keep `use_space_char: false` unless the dictionary is re-audited with a space token.
- Keep the decoder algorithm as `CTCLabelDecode`.
- Do not switch to NAT; this repository is still a CTC recognizer.
- Re-run both verification scripts after changing dictionary, config, or label encoding code.
