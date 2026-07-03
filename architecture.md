# PaddleOCR Dravidian Akshara Architecture

## Scope

This repository is a PaddleOCR checkout modified for Tamil/Dravidian text recognition with Akshara, or Unicode grapheme cluster, tokenization.

The implementation keeps the PP-OCRv5 recognition architecture unchanged:

- Backbone: `PPLCNetV3` for mobile, `PPHGNetV2_B4` for server
- Recognition head: `MultiHead`
- Primary head: `CTCHead`
- Auxiliary training head: `NRTRHead`
- Losses: `CTCLoss` and `NRTRLoss` through `MultiLoss`
- Inference decoder: `CTCLabelDecode`

Only recognition tokenization and token-length handling were changed from Unicode code points to Unicode grapheme clusters.

## Current Data Store

There is no SQL, SQLite, Postgres, MySQL, DuckDB, or LMDB database file currently present in this workspace.

The current project data layer is file-based:

| Path | Purpose |
| --- | --- |
| `akshara/dict.txt` | Active Akshara dictionary used by PaddleOCR configs. |
| `akshara_backup/dict.txt` | Verified source dictionary artifact. |
| `akshara_backup/valid_labels.csv` | Verified encodable label audit output. |
| `akshara_backup/report.md` | Dataset audit summary. |
| `pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained.pdparams` | Official Tamil PP-OCRv5 pretrained weights used for Akshara fine-tuning. |
| `train_data/train_list.txt` | Small smoke-training label list. |
| `train_data/val_list.txt` | Small smoke-evaluation label list. |
| `train_data/smoke/*.png` | Local smoke-training images copied from test fixtures. |
| `output/akshara_finetune_smoke/` | Fine-tune smoke-run checkpoints, states, config copy, and training log. |

Dictionary integrity:

- `akshara/dict.txt` line count: 355
- `akshara_backup/dict.txt` line count: 355
- SHA-256 for both: `ea8fa4896357e37c201749fae87cc818dfa0407919523f543097e15d9b1578d3`

Audit artifact summary:

- Total labels audited: 80022
- Valid labels: 79862
- Invalid labels: 160
- Retained labels: 99.80%
- Final grapheme vocabulary: 355 tokens

## Recognition Pipeline

### Training Flow

1. `tools/train.py` loads `configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml` or `configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml`.
2. `ppocr/utils/save_load.py` loads `Global.pretrained_model` when present.
3. For Akshara mobile fine-tuning, compatible tensors are loaded from `pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained.pdparams`; classifier/token tensors with dictionary-size mismatches are skipped.
4. `Train.dataset` reads tab-separated image paths and labels from `train_data/train_list.txt`.
5. Image transforms run:
   - `DecodeImage`
   - `RecConAug`
   - `RecAug`
   - `MultiLabelEncode`
   - `KeepKeys`
6. `MultiLabelEncode` produces:
   - `label_ctc` through `CTCLabelEncode`
   - `label_gtc` through `NRTRLabelEncode`
   - `length` from the CTC Akshara-token sequence length
7. `MultiHead` computes:
   - CTC logits from `CTCHead`
   - auxiliary NRTR logits from `NRTRHead`
8. `MultiLoss` computes:
   - `CTCLoss` over CTC logits and Akshara IDs
   - `NRTRLoss` over auxiliary NRTR labels

### Inference Flow

1. The recognizer emits CTC logits.
2. `CTCLabelDecode` selects the argmax token ID per timestep.
3. CTC duplicate removal and blank-token filtering remain ID-based and unchanged.
4. Remaining IDs map back to entries in `akshara/dict.txt`.
5. Token strings are joined into final Unicode text.

## Akshara Tokenization

The active recognition encoders use:

```python
regex.findall(r"\X", text)
```

This treats each Unicode grapheme cluster as one OCR token. That is the only intended semantic change. The model still sees integer token IDs, CTC blank ID `0`, and the same sequence modeling objective.

Active code paths:

| File | Responsibility |
| --- | --- |
| `ppocr/data/imaug/label_ops.py` | Adds grapheme tokenization and applies it to `CTCLabelEncode` and `NRTRLabelEncode`. |
| `ppocr/data/imaug/rec_img_aug.py` | Counts grapheme clusters in `RecConAug` length checks. |
| `ppocr/postprocess/rec_postprocess.py` | Uses grapheme iteration in `pred_reverse` and `get_word_info`; core CTC decode remains ID-based. |
| `configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml` | Points PP-OCRv5 mobile recognition to `akshara/dict.txt`. |
| `configs/rec/PP-OCRv5/PP-OCRv5_server_rec.yml` | Points PP-OCRv5 server recognition to `akshara/dict.txt`. |

## Configuration

Both PP-OCRv5 recognition configs use:

```yaml
Global:
  character_dict_path: akshara/dict.txt
  use_space_char: false
```

`use_space_char` is false so PaddleOCR does not append an extra space token to the verified 355-token dictionary.

For CTC decoding:

- Dictionary tokens: 355
- CTC blank: 1
- CTC classifier output dimension: 356

For NRTR auxiliary labels:

- Dictionary tokens: 355
- Special tokens: `blank`, `<unk>`, `<s>`, `</s>`
- NRTR vocabulary size: 359

## Verification Scripts

`verify_akshara.py` checks:

- dictionary loading
- grapheme tokenization
- CTC encoder
- NRTR encoder
- CTC decoder
- encode-to-decode round trips

`verify_model.py` checks:

- PP-OCRv5 mobile recognition model construction
- Akshara vocabulary size
- CTC classifier output dimension
- dummy forward pass
- CTC loss computation
- tensor shape compatibility

Verified local results:

- `verify_akshara.py`: passed
- `verify_model.py`: passed
- PP-OCRv5 mobile fine-tune smoke train: passed for 1 epoch on the small local subset

## Smoke Training Dataset

The current smoke dataset is intentionally minimal and exists only to verify fine-tuning pipeline compatibility:

- Train labels: 6
- Eval labels: 2
- Image source: copied local test fixture image
- Output directory: `output/akshara_finetune_smoke/`

The smoke run proves that Akshara tokenization, pretrained Tamil weight loading, model construction, forward pass, losses, dataloading, evaluation, and checkpoint saving are wired correctly. It is not a model-quality run.

## Runtime Environment

The system Python is 3.14 and does not have a compatible PaddlePaddle wheel. A repo-local virtual environment was created:

```text
.venv/
```

Known installed runtime:

- Python: 3.12.13
- PaddlePaddle: 3.3.1
- regex: 2026.6.28

Use `.venv/bin/python` for verification and training commands in this workspace.

## Non-Goals

This project does not introduce:

- NAT recognition
- architecture changes
- a new decoder algorithm
- a new loss
- dictionary regeneration
- dictionary reordering or cleanup
- a relational database

## Operational Notes

- Keep `akshara/dict.txt` byte-identical to the verified artifact unless a new audit is intentionally performed.
- Do not enable `use_space_char` with this dictionary unless the dictionary is explicitly re-audited with a space token.
- Full training should replace the smoke `train_data/*.txt` files with the real audited dataset label files.
- Smoke outputs under `output/akshara_finetune_smoke/` are disposable verification artifacts.
