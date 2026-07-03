# Tamil Akshara Dictionary Backup

This directory contains the verified source dictionary artifact for the [PaddleOCR Dravidian Akshara fork](../README.md).

> **See also**: [Akshara training guide](../README_AKSHARA.md) · [Architecture notes](../architecture.md)

## Dictionary Details

- **Dictionary size**: 355
- **Number of valid labels**: 79862
- **Number of review labels**: 0
- **Number of invalid labels**: 160
- **Date generated**: 2026-07-02 20:13:07
- **SHA256 checksum of dict.txt**: `ea8fa4896357e37c201749fae87cc818dfa0407919523f543097e15d9b1578d3`

## Dictionary Integrity

The active dictionary at `akshara/dict.txt` must be byte-identical to `akshara_backup/dict.txt`. Both files should have the same SHA-256 checksum listed above. Do not regenerate, reorder, clean, or edit the dictionary unless you intentionally run a new audit.

## Instructions for Reusing the Dictionary

To use this dictionary in a fresh PaddleOCR installation:

1. Copy `dict.txt` to your PaddleOCR `ppocr/utils/dict/` directory (or any preferred location).
2. Update your training/evaluation configuration YAML file (e.g., `configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml`).
3. Set the `character_dict_path` variable under `Global` in the config file to point to the new dictionary location.
4. Set `use_space_char: false` — the current dictionary does not include a space token. Do not enable `use_space_char` unless the dictionary is explicitly re-audited with a space token.
