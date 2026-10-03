"""
Step 1: download the IAM-line handwriting dataset.

Dataset : IAM Handwriting Database, line level ("Teklia/IAM-line" on Hugging Face)
Source  : https://huggingface.co/datasets/Teklia/IAM-line
          (original database: https://fki.tic.heia-fr.ch/databases/iam-handwriting-database)
License : listed as MIT on Hugging Face; the original IAM database is free for
          NON-COMMERCIAL RESEARCH use. Fine for learning, not for selling a product.
Content : 10,373 images of handwritten English text LINES (complete sentences or
          parts of sentences) written by ~650 different people, each with the exact
          text transcription. Official splits: 6,482 train / 976 validation / 2,915 test.

The files are saved to  data/raw/{train,validation,test}.parquet
A parquet file is a compressed table; here each row = (image bytes, text).

Run from the project folder:
    python src/download_data.py
Files that already exist are skipped, so running it twice is harmless.
"""

import sys
from pathlib import Path

import requests
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402


def download_file(url: str, destination: Path, retries: int = 3) -> None:
    """Download `url` to `destination` with a progress bar.

    The file is first written as  *.part  and only renamed at the end, so a
    broken/interrupted download never looks like a finished one.
    """
    destination.parent.mkdir(parents=True, exist_ok=True)
    temp_path = destination.with_suffix(destination.suffix + ".part")

    for attempt in range(1, retries + 1):
        try:
            with requests.get(url, stream=True, timeout=60) as response:
                response.raise_for_status()
                total = int(response.headers.get("content-length", 0))
                with open(temp_path, "wb") as f, tqdm(
                    total=total, unit="B", unit_scale=True, desc=destination.name
                ) as bar:
                    for chunk in response.iter_content(chunk_size=1024 * 1024):
                        f.write(chunk)
                        bar.update(len(chunk))
            temp_path.replace(destination)
            return
        except requests.RequestException as error:
            print(f"  Download failed (attempt {attempt}/{retries}): {error}")
            if attempt == retries:
                raise RuntimeError(
                    f"Could not download {url}. Check your internet connection and try again."
                ) from error


def download_dataset() -> dict:
    """Download all three splits (if missing) and return {split: path}."""
    paths = {}
    for split in config.DATASET_SPLITS:
        path = config.RAW_DIR / f"{split}.parquet"
        if path.exists() and path.stat().st_size > 0:
            print(f"[skip] {path.relative_to(config.PROJECT_DIR)} already downloaded")
        else:
            print(f"[download] {split} split ...")
            download_file(config.DATASET_URL.format(split=split), path)
        paths[split] = path
    print("Dataset is ready in", config.RAW_DIR.relative_to(config.PROJECT_DIR))
    return paths


if __name__ == "__main__":
    download_dataset()
