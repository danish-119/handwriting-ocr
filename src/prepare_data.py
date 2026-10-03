"""
Step 2: clean, check and preprocess the downloaded dataset.

Input : data/raw/{train,validation,test}.parquet     (from download_data.py)
Output: data/{train,validation,test}/images/*.png     (preprocessed line images)
        data/{train,validation,test}/labels.csv       (file name + transcription)
        results/data_report.json                      (what was checked / removed)

Splits: IAM already has OFFICIAL train / validation / test splits that are
"writer independent" (the people who wrote the test lines never wrote any
training line). We use those instead of a random split. That is the honest way
to measure how the model handles handwriting it has never seen.

Extra safety: if the exact same SENTENCE appears in both the training set and
the validation/test set (IAM has a few, because several writers copied the same
text), the training copy is removed, so every test sentence is truly unseen.

Run from the project folder:
    python src/prepare_data.py
"""

import hashlib
import io
import json
import shutil
import sys
from pathlib import Path

import pandas as pd
from PIL import Image, UnidentifiedImageError
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402
from preprocessing import preprocess_pil  # noqa: E402


def load_raw_split(split: str) -> pd.DataFrame:
    """Read one downloaded parquet file into a table with columns: image_bytes, text."""
    path = config.RAW_DIR / f"{split}.parquet"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run  python src/download_data.py  first.")
    df = pd.read_parquet(path)
    return pd.DataFrame({
        "image_bytes": [img["bytes"] if isinstance(img, dict) else None for img in df["image"]],
        "text": df["text"],
    })


def clean_split(df: pd.DataFrame, split: str, report: dict) -> pd.DataFrame:
    """Remove rows with missing labels, missing images, corrupted images or duplicates."""
    n_start = len(df)

    # Missing / empty transcriptions
    df["text"] = df["text"].astype("string").str.strip()
    missing_label = df["text"].isna() | (df["text"] == "")
    # Missing image data
    missing_image = df["image_bytes"].isna()

    # Corrupted images: try to fully decode every image
    corrupted = []
    for data in df["image_bytes"]:
        if data is None:
            corrupted.append(False)
            continue
        try:
            Image.open(io.BytesIO(data)).load()
            corrupted.append(False)
        except (UnidentifiedImageError, OSError):
            corrupted.append(True)
    corrupted = pd.Series(corrupted, index=df.index)

    df = df[~(missing_label | missing_image | corrupted)].copy()

    # Exact duplicate images (same bytes) inside this split
    df["md5"] = [hashlib.md5(b).hexdigest() for b in df["image_bytes"]]
    duplicates = df["md5"].duplicated()
    df = df[~duplicates].copy()

    report[split] = {
        "rows_downloaded": n_start,
        "missing_labels": int(missing_label.sum()),
        "missing_images": int(missing_image.sum()),
        "corrupted_images": int(corrupted.sum()),
        "duplicate_images": int(duplicates.sum()),
    }
    return df


def remove_overlap(splits: dict, report: dict) -> None:
    """Remove from TRAIN any image or sentence that also appears in validation or test."""
    held_out_md5 = set(splits["validation"]["md5"]) | set(splits["test"]["md5"])
    held_out_text = set(splits["validation"]["text"]) | set(splits["test"]["text"])
    train = splits["train"]
    leak = train["md5"].isin(held_out_md5) | train["text"].isin(held_out_text)
    splits["train"] = train[~leak].copy()
    report["train"]["removed_because_sentence_or_image_in_val_or_test"] = int(leak.sum())


def export_split(df: pd.DataFrame, split: str) -> pd.DataFrame:
    """Preprocess every image, save it as PNG and write labels.csv."""
    out_dir = config.SPLIT_DIRS[split]
    if out_dir.exists():
        shutil.rmtree(out_dir)  # start fresh so old files never mix with new ones
    image_dir = out_dir / "images"
    image_dir.mkdir(parents=True)

    rows = []
    for i, (data, text) in enumerate(tqdm(zip(df["image_bytes"], df["text"]), total=len(df),
                                          desc=f"export {split}")):
        image = preprocess_pil(Image.open(io.BytesIO(data)))
        name = f"{split}_{i:05d}.png"
        Image.fromarray(image).save(image_dir / name)
        rows.append({"file": f"images/{name}", "text": text,
                     "width": image.shape[1], "height": image.shape[0]})

    labels = pd.DataFrame(rows)
    labels.to_csv(out_dir / "labels.csv", index=False, encoding="utf-8")
    return labels


def prepare_dataset(force: bool = False) -> dict:
    """Run all cleaning steps. Skips the work if the exported data already exists."""
    if not force and all((d / "labels.csv").exists() for d in config.SPLIT_DIRS.values()):
        print("Prepared data already exists (use force=True to rebuild).")
        return json.loads((config.RESULTS_DIR / "data_report.json").read_text())

    report = {}
    splits = {s: clean_split(load_raw_split(s), s, report) for s in config.DATASET_SPLITS}
    remove_overlap(splits, report)

    for split, df in splits.items():
        labels = export_split(df, split)
        report[split]["rows_used"] = len(labels)

    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    (config.RESULTS_DIR / "data_report.json").write_text(json.dumps(report, indent=2))
    return report


def load_labels(split: str) -> pd.DataFrame:
    """Read data/<split>/labels.csv and add the full image path."""
    path = config.SPLIT_DIRS[split] / "labels.csv"
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run  python src/prepare_data.py  first.")
    # dtype=str + keep_default_na=False: a label like "1960" or "NA" stays text
    df = pd.read_csv(path, dtype={"text": str}, keep_default_na=False, encoding="utf-8")
    df["path"] = [str(config.SPLIT_DIRS[split] / f) for f in df["file"]]
    return df


if __name__ == "__main__":
    result = prepare_dataset(force="--force" in sys.argv)
    print(json.dumps(result, indent=2))
