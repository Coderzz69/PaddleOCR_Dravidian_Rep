#!/usr/bin/env python3
import os
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
REPO_DIR = Path(os.getenv("REPO_DIR", BASE_DIR / "PaddleOCR_Dravidian_Rep"))
DATA_DIR = Path(os.getenv("TRAIN_DATA_DIR", REPO_DIR / "train_data" / "indic_mozhi_tamil"))

if not DATA_DIR.is_absolute():
    DATA_DIR = REPO_DIR / DATA_DIR


def require(path):
    if not Path(path).exists():
        raise SystemExit(f"Missing required path: {path}")
    print(f"OK: {path}")


def main():
    require(REPO_DIR / "tools" / "train.py")
    require(REPO_DIR / "configs" / "rec" / "PP-OCRv5" / "PP-OCRv5_mobile_rec.yml")
    require(REPO_DIR / "akshara" / "dict.txt")
    require(REPO_DIR / "pretrain_models" / "ta_PP-OCRv5_mobile_rec_pretrained.pdparams")
    require(DATA_DIR / "train_list.txt")
    require(DATA_DIR / "val_list.txt")

    import paddle

    print("paddle:", paddle.__version__)
    print("compiled_with_cuda:", paddle.device.is_compiled_with_cuda())
    try:
        print("cuda_device_count:", paddle.device.cuda.device_count())
    except Exception as exc:
        print("cuda_device_count_error:", repr(exc))

    dict_count = sum(
        1
        for line in (REPO_DIR / "akshara" / "dict.txt").read_text(encoding="utf-8").splitlines()
        if line
    )
    print("akshara_dict_tokens:", dict_count)

    with (DATA_DIR / "train_list.txt").open("r", encoding="utf-8") as f:
        first = f.readline().rstrip("\n")
    print("first_train_label:", first[:200])


if __name__ == "__main__":
    main()
