#!/usr/bin/env python3
import csv
import os
import re
from pathlib import Path

import regex


BASE_DIR = Path(__file__).resolve().parent
default_hf_cache = Path.home() / ".cache" / "huggingface"
if not os.access(default_hf_cache, os.W_OK):
    hf_home = BASE_DIR / ".hf_home"
    hf_home.mkdir(parents=True, exist_ok=True)
    os.environ.setdefault("HF_HOME", str(hf_home))
    os.environ.setdefault("HF_DATASETS_CACHE", str(hf_home / "datasets"))

from datasets import load_dataset


REPO_DIR = Path(os.getenv("REPO_DIR", BASE_DIR / "PaddleOCR_Dravidian_Rep"))
DATASET_NAME = os.getenv("HF_DATASET_NAME", "darknight054/indic-mozhi-ocr")
DATASET_CONFIG = os.getenv("HF_DATASET_CONFIG", "tamil")
OUTPUT_DIR = Path(
    os.getenv("MOZHI_OUTPUT_DIR", REPO_DIR / "train_data" / "indic_mozhi_tamil")
)
DICT_PATH = Path(os.getenv("AKSHARA_DICT_PATH", REPO_DIR / "akshara" / "dict.txt"))
STRICT_AKSHARA = os.getenv("STRICT_AKSHARA", "1") == "1"
MAX_TRAIN_SAMPLES = int(os.getenv("MAX_TRAIN_SAMPLES", "0") or "0")
MAX_VAL_SAMPLES = int(os.getenv("MAX_VAL_SAMPLES", "0") or "0")
MAX_TEST_SAMPLES = int(os.getenv("MAX_TEST_SAMPLES", "0") or "0")

SPLITS = {
    "train": ("train", "train_list.txt", MAX_TRAIN_SAMPLES),
    "validation": ("val", "val_list.txt", MAX_VAL_SAMPLES),
    "test": ("test", "test_list.txt", MAX_TEST_SAMPLES),
}


def load_dictionary(path):
    if not path.exists():
        raise FileNotFoundError(f"Akshara dictionary not found: {path}")
    with path.open("r", encoding="utf-8") as f:
        return {line.rstrip("\n") for line in f if line.rstrip("\n")}


def safe_name(value, fallback):
    value = str(value or fallback)
    value = re.sub(r"[^A-Za-z0-9_.-]+", "_", value).strip("._")
    return value or str(fallback)


def clean_label(text):
    return str(text).replace("\t", " ").replace("\r", " ").replace("\n", " ").strip()


def unknown_tokens(text, vocab):
    return sorted({tok for tok in regex.findall(r"\X", text) if tok not in vocab})


def save_split(split_name, out_split, list_name, limit, vocab, unknown_writer):
    print(f"Loading {DATASET_NAME}/{DATASET_CONFIG} split={split_name}")
    dataset = load_dataset(DATASET_NAME, DATASET_CONFIG, split=split_name)
    if limit > 0:
        dataset = dataset.select(range(min(limit, len(dataset))))

    image_dir = OUTPUT_DIR / "images" / out_split
    image_dir.mkdir(parents=True, exist_ok=True)
    label_path = OUTPUT_DIR / list_name

    kept = 0
    skipped = 0
    unknown_count = 0
    with label_path.open("w", encoding="utf-8", newline="") as label_file:
        for idx, row in enumerate(dataset):
            label = clean_label(row.get("text", ""))
            if not label:
                skipped += 1
                continue

            bad_tokens = unknown_tokens(label, vocab)
            if bad_tokens:
                unknown_count += 1
                unknown_writer.writerow(
                    {
                        "split": split_name,
                        "id": row.get("id", idx),
                        "text": label,
                        "unknown_tokens": " ".join(bad_tokens),
                    }
                )
                if STRICT_AKSHARA:
                    continue

            image = row["image"]
            image_name = f"{safe_name(row.get('id'), idx)}.png"
            image_path = image_dir / image_name
            if image.mode not in ("RGB", "L"):
                image = image.convert("RGB")
            image.save(image_path)

            rel_path = image_path.relative_to(OUTPUT_DIR).as_posix()
            label_file.write(f"{rel_path}\t{label}\n")
            kept += 1

    print(
        f"{split_name}: wrote {kept} labels to {label_path} "
        f"(empty skipped={skipped}, unknown={unknown_count})"
    )
    return unknown_count


def main():
    vocab = load_dictionary(DICT_PATH)
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    unknown_path = OUTPUT_DIR / "unknown_akshara_tokens.csv"

    total_unknown = 0
    with unknown_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(
            f, fieldnames=["split", "id", "text", "unknown_tokens"]
        )
        writer.writeheader()
        for split_name, args in SPLITS.items():
            total_unknown += save_split(split_name, *args, vocab, writer)

    print(f"Unknown-token report: {unknown_path}")
    if STRICT_AKSHARA and total_unknown:
        raise SystemExit(
            "Found labels containing tokens not present in akshara/dict.txt. "
            "Review unknown_akshara_tokens.csv or rerun with STRICT_AKSHARA=0."
        )


if __name__ == "__main__":
    main()
