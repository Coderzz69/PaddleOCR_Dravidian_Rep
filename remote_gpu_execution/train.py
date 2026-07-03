#!/usr/bin/env python3
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

import os
import subprocess
import sys
from pathlib import Path


BASE_DIR = Path(__file__).resolve().parent
CONDA_SH = Path(os.getenv("CONDA_SH", str(Path.home() / "miniconda3" / "bin" / "conda")))
CONDA_ENV = os.getenv("CONDA_ENV", "paddle_env")
REPO_DIR = Path(os.getenv("REPO_DIR", BASE_DIR / "PaddleOCR_Dravidian_Rep"))
CONFIG = os.getenv("CONFIG", "configs/rec/PP-OCRv5/PP-OCRv5_mobile_rec.yml")
PRETRAIN_MODEL = os.getenv(
    "PRETRAIN_MODEL", "./pretrain_models/ta_PP-OCRv5_mobile_rec_pretrained"
)
DICT_PATH = os.getenv("DICT_PATH", "akshara/dict.txt")
SAVE_MODEL_DIR = os.getenv("SAVE_MODEL_DIR", "./output/akshara_gpu_finetune")
TRAIN_DATA_DIR = os.getenv("TRAIN_DATA_DIR", "./train_data/indic_mozhi_tamil")
TRAIN_LABELS = os.getenv(
    "TRAIN_LABELS", "./train_data/indic_mozhi_tamil/train_list.txt"
)
VAL_DATA_DIR = os.getenv("VAL_DATA_DIR", "./train_data/indic_mozhi_tamil")
VAL_LABELS = os.getenv("VAL_LABELS", "./train_data/indic_mozhi_tamil/val_list.txt")
BATCH_SIZE = os.getenv("BATCH_SIZE", "128")
EPOCH_NUM = os.getenv("EPOCH_NUM", "75")
RUN_VERIFICATION = os.getenv("RUN_VERIFICATION", "1") == "1"
USE_GPU = os.getenv("USE_GPU", "True")
EXPORT_MODEL = os.getenv("EXPORT_MODEL", "1") == "1"


def reexec_with_conda_python():
    conda_root = CONDA_SH.parent.parent
    conda_python = conda_root / "envs" / CONDA_ENV / "bin" / "python"
    if not conda_python.exists():
        return
    current = Path(sys.executable).resolve()
    target = conda_python.resolve()
    if current != target:
        os.execv(str(target), [str(target), *sys.argv])


def run(cmd, cwd=REPO_DIR):
    print("+", " ".join(cmd))
    env = os.environ.copy()
    home_cache = Path.home() / ".cache"
    if not os.access(home_cache, os.W_OK):
        local_home = BASE_DIR / ".runtime_home"
        (local_home / ".cache").mkdir(parents=True, exist_ok=True)
        env["HOME"] = str(local_home)
    subprocess.run(cmd, cwd=cwd, check=True, env=env)


def require(path):
    full = REPO_DIR / path
    if not full.exists():
        raise SystemExit(f"Missing required path: {full}")


def main():
    reexec_with_conda_python()

    require("tools/train.py")
    require(CONFIG)
    require(DICT_PATH)
    require(f"{PRETRAIN_MODEL}.pdparams")
    require(TRAIN_DATA_DIR)
    require(TRAIN_LABELS)
    require(VAL_DATA_DIR)
    require(VAL_LABELS)

    run([sys.executable, str(BASE_DIR / "sanity_check.py")], cwd=BASE_DIR)

    if RUN_VERIFICATION:
        run([sys.executable, "verify_akshara.py"])
        run([sys.executable, "verify_model.py"])

    train_cmd = [
        sys.executable,
        "tools/train.py",
        "-c",
        CONFIG,
        "-o",
        f"Global.use_gpu={USE_GPU}",
        "Global.distributed=False",
        f"Global.epoch_num={EPOCH_NUM}",
        f"Global.pretrained_model={PRETRAIN_MODEL}",
        f"Global.character_dict_path={DICT_PATH}",
        f"Global.save_model_dir={SAVE_MODEL_DIR}",
        f"Train.dataset.data_dir={TRAIN_DATA_DIR}",
        f"Train.dataset.label_file_list=['{TRAIN_LABELS}']",
        f"Eval.dataset.data_dir={VAL_DATA_DIR}",
        f"Eval.dataset.label_file_list=['{VAL_LABELS}']",
        f"Train.sampler.first_bs={BATCH_SIZE}",
        f"Train.loader.batch_size_per_card={BATCH_SIZE}",
        f"Eval.loader.batch_size_per_card={BATCH_SIZE}",
    ]
    run(train_cmd)

    if not EXPORT_MODEL:
        return

    best_prefix = Path(SAVE_MODEL_DIR) / "best_accuracy"
    latest_prefix = Path(SAVE_MODEL_DIR) / "latest"
    export_prefix = best_prefix if (REPO_DIR / f"{best_prefix}.pdparams").exists() else latest_prefix
    export_model_path = export_prefix.as_posix()
    if not export_prefix.is_absolute():
        export_model_path = f"./{export_model_path}"

    export_cmd = [
        sys.executable,
        "tools/export_model.py",
        "-c",
        CONFIG,
        "-o",
        f"Global.pretrained_model={export_model_path}",
        f"Global.character_dict_path={DICT_PATH}",
        "Global.save_inference_dir=./inference/akshara_tamil_rec",
    ]
    run(export_cmd)


if __name__ == "__main__":
    main()
