"""
Central settings for the handwriting OCR project.

Every other script imports its paths and numbers from here, so if you want to
change something (image size, batch size, number of epochs...) you only need
to change it in ONE place.

All paths are built relative to the project folder, so the project works no
matter where you put it on your computer.
"""

from pathlib import Path

# ---------------------------------------------------------------------------
# Folders
# ---------------------------------------------------------------------------
# This file lives in  handwriting_ocr/src/config.py  ->  parent.parent = handwriting_ocr/
PROJECT_DIR = Path(__file__).resolve().parent.parent

DATA_DIR = PROJECT_DIR / "data"
RAW_DIR = DATA_DIR / "raw"            # downloaded files exactly as they came from the internet
TRAIN_DIR = DATA_DIR / "train"        # images + labels.csv for training
VAL_DIR = DATA_DIR / "validation"     # images + labels.csv for validation
TEST_DIR = DATA_DIR / "test"          # images + labels.csv for the final test
SPLIT_DIRS = {"train": TRAIN_DIR, "validation": VAL_DIR, "test": TEST_DIR}

MODELS_DIR = PROJECT_DIR / "models"
RESULTS_DIR = PROJECT_DIR / "results"

MODEL_PATH = MODELS_DIR / "handwriting_ocr.keras"                 # best model from training
FINETUNED_MODEL_PATH = MODELS_DIR / "handwriting_ocr_finetuned.keras"
VOCAB_PATH = MODELS_DIR / "vocab.json"                            # character <-> number tables

# ---------------------------------------------------------------------------
# Dataset (IAM handwriting database, line level, hosted on Hugging Face)
# ---------------------------------------------------------------------------
DATASET_NAME = "Teklia/IAM-line"
DATASET_URL = "https://huggingface.co/datasets/Teklia/IAM-line/resolve/main/data/{split}.parquet"
DATASET_SPLITS = ["train", "validation", "test"]

# ---------------------------------------------------------------------------
# Image preprocessing
# ---------------------------------------------------------------------------
# Every line image is resized to this height while keeping its aspect ratio,
# so long lines become wide images and short lines narrow images (no squashing).
# Images in a batch are padded on the right to the widest image of that batch.
IMG_HEIGHT = 64
MAX_IMG_WIDTH = 2560      # safety limit; only absurdly long lines are shrunk

# The CNN shrinks the width by this factor, so a 1000-pixel-wide line gives
# 1000 / 4 = 250 "time steps" (columns). Each time step outputs one character
# or a CTC "blank".
DOWNSAMPLE = 4

# ---------------------------------------------------------------------------
# Training
# ---------------------------------------------------------------------------
SEED = 42                 # fixed random seed -> reproducible results
BATCH_SIZE = 16           # lower this (e.g. 8) if you run out of memory
EPOCHS = 60               # maximum; EarlyStopping usually stops earlier
LEARNING_RATE = 1e-3
FINETUNE_LEARNING_RATE = 1e-4   # 10x smaller: small careful steps when fine-tuning
AUGMENT = True            # small random changes to training images (helps generalisation)
