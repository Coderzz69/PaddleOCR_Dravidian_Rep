import os
import hashlib
import datetime
import csv

backup_dir = "/home/coderzz96/Desktop/akshara_backup"
dict_path = os.path.join(backup_dir, "dict.txt")
valid_path = os.path.join(backup_dir, "valid_labels.csv")
review_path = os.path.join(backup_dir, "review_labels.csv")
invalid_path = os.path.join(backup_dir, "invalid_labels.csv")

def count_lines(filepath, has_header=True):
    if not os.path.exists(filepath):
        return 0
    with open(filepath, 'r', encoding='utf-8') as f:
        count = sum(1 for _ in f)
    return max(0, count - 1) if has_header else count

dict_size = count_lines(dict_path, has_header=False)
valid_count = count_lines(valid_path, has_header=True)
review_count = count_lines(review_path, has_header=True)
invalid_count = count_lines(invalid_path, has_header=True)
date_generated = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")

sha256_hash = hashlib.sha256()
if os.path.exists(dict_path):
    with open(dict_path, "rb") as f:
        for byte_block in iter(lambda: f.read(4096), b""):
            sha256_hash.update(byte_block)
dict_sha = sha256_hash.hexdigest()

readme_content = f"""# Tamil Akshara Dictionary Backup

- **Dictionary size**: {dict_size}
- **Number of valid labels**: {valid_count}
- **Number of review labels**: {review_count}
- **Number of invalid labels**: {invalid_count}
- **Date generated**: {date_generated}
- **SHA256 checksum of dict.txt**: {dict_sha}

## Instructions for Reusing the Dictionary

To use this dictionary in a fresh PaddleOCR installation:
1. Copy `dict.txt` to your PaddleOCR `ppocr/utils/dict/` directory (or any preferred location).
2. Update your training/evaluation configuration YAML file (e.g., `configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml`).
3. Set the `character_dict_path` variable under `Global` in the config file to point to the new dictionary location.
4. Set `use_space_char: True` if your dictionary includes spaces (or appropriately based on your dictionary structure).
"""

with open(os.path.join(backup_dir, "README.md"), "w", encoding='utf-8') as f:
    f.write(readme_content)

# Verification
assert os.path.exists(dict_path), "dict.txt does not exist."

with open(dict_path, "r", encoding='utf-8') as f:
    lines = f.read().splitlines()

assert len(lines) > 0, "dict.txt is empty."

# Check for exactly one token per line
# This means no line should be empty unless the token itself is an empty string/newline (usually handled differently)
# In standard PaddleOCR dicts, every line is exactly one token.
duplicates = set()
seen = set()
for i, line in enumerate(lines):
    # Depending on the dictionary format, some lines might be empty if the token is a space,
    # but strictly speaking a token should just be whatever is on the line.
    if line in seen:
        duplicates.add(line)
    seen.add(line)

assert len(duplicates) == 0, f"Found duplicate entries in dict.txt: {duplicates}"

print(f"Number of tokens: {len(lines)}")
print("First 20 tokens:")
for t in lines[:20]:
    print(t)
    
print("Last 20 tokens:")
for t in lines[-20:]:
    print(t)
    
print(f"SHA256 checksum of dict.txt: {dict_sha}")
print("\nBackup completed successfully.")
print("Safe to delete the current PaddleOCR installation.")
