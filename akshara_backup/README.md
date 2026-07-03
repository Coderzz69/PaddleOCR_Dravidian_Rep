# Tamil Akshara Dictionary Backup

- **Dictionary size**: 355
- **Number of valid labels**: 79862
- **Number of review labels**: 0
- **Number of invalid labels**: 160
- **Date generated**: 2026-07-02 20:13:07
- **SHA256 checksum of dict.txt**: ea8fa4896357e37c201749fae87cc818dfa0407919523f543097e15d9b1578d3

## Instructions for Reusing the Dictionary

To use this dictionary in a fresh PaddleOCR installation:
1. Copy `dict.txt` to your PaddleOCR `ppocr/utils/dict/` directory (or any preferred location).
2. Update your training/evaluation configuration YAML file (e.g., `configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml`).
3. Set the `character_dict_path` variable under `Global` in the config file to point to the new dictionary location.
4. Set `use_space_char: True` if your dictionary includes spaces (or appropriately based on your dictionary structure).
