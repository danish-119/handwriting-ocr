"""
Step 6: fine-tune (continue training) the saved model on NEW handwriting data.

    Existing trained OCR model  ->  load  ->  add new handwriting  ->
    continue training with a small learning rate  ->  evaluate  ->  save separately

When is this useful?
  * The model reads YOUR handwriting badly (every writer is different).
  * Your images look different from IAM (phone photos, a pencil, lined paper ...).
  * You only have a little data: fine-tuning needs far fewer examples (even
    50-200 lines help) than training from scratch, because the model already
    knows what letters look like.

YOUR DATA must be a folder like this:
    data/my_handwriting/
        labels.csv            <- two columns:  file,text
        line001.jpg               line001.jpg,The weather was beautiful today.
        line002.jpg               line002.jpg,I went to the market yesterday
        ...
Each image should contain ONE line of handwriting.

Run from the project folder:
    python src/finetune.py                              # uses data/my_handwriting/
    python src/finetune.py --data path/to/folder --epochs 10
    python src/finetune.py --demo                       # build + use a demo dataset

Important details:
  * Learning rate is 10x smaller than in normal training: we only want to nudge
    the model, not destroy what it already learned.
  * "Replay": some original IAM training lines are mixed in, so the model does
    not forget how to read other people's handwriting (catastrophic forgetting).
  * The vocabulary (list of characters) is fixed. Characters that never appeared
    in IAM (e.g. 'é', '€', '@') cannot be learned without rebuilding the output
    layer; they are reported and treated as unknown.
  * The original model is never overwritten; the result is saved as
    models/handwriting_ocr_finetuned.keras
"""

import argparse
import shutil
import sys
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

import keras  # noqa: E402
from PIL import Image, ImageFilter  # noqa: E402

from dataset import make_dataset, encode_labels, sort_by_width  # noqa: E402
from metrics import corpus_cer, corpus_wer  # noqa: E402
from model import compile_model, load_trained_model  # noqa: E402
from prepare_data import load_labels  # noqa: E402
from preprocessing import load_image  # noqa: E402
from train import CERCallback, predict_dataset  # noqa: E402
from vocab import Vocabulary  # noqa: E402

MY_DATA_DIR = config.DATA_DIR / "my_handwriting"
DEMO_DIR = config.DATA_DIR / "finetune_demo"


# ---------------------------------------------------------------------------
# Reading the new data
# ---------------------------------------------------------------------------
def prepare_folder(folder: Path) -> pd.DataFrame:
    """Preprocess every image listed in folder/labels.csv (like prepare_data.py does).

    Returns a labels table with columns: path, text, width. Missing or corrupted
    images and empty labels are skipped with a warning.
    """
    folder = Path(folder)
    labels_file = folder / "labels.csv"
    if not labels_file.exists():
        raise FileNotFoundError(
            f"{labels_file} not found.\nCreate it with two columns 'file,text' "
            "(see the top of src/finetune.py), or run with --demo.")
    table = pd.read_csv(labels_file, dtype=str, keep_default_na=False, encoding="utf-8")
    if not {"file", "text"} <= set(table.columns):
        raise ValueError(f"{labels_file} must have the columns: file,text")

    out_dir = folder / "_processed"
    if out_dir.exists():
        shutil.rmtree(out_dir)
    out_dir.mkdir()

    rows = []
    for i, (file, text) in enumerate(zip(table["file"], table["text"])):
        text = text.strip()
        if not text:
            print(f"  [skip] {file}: empty label")
            continue
        try:
            image = load_image(folder / file)
        except (FileNotFoundError, ValueError) as error:
            print(f"  [skip] {error}")
            continue
        out = out_dir / f"{i:05d}.png"
        Image.fromarray(image).save(out)
        rows.append({"path": str(out), "text": text, "width": image.shape[1]})
    if not rows:
        raise ValueError(f"No usable images found in {folder}")
    print(f"  {len(rows)} usable lines in {folder}")
    return pd.DataFrame(rows)


def report_unknown_characters(texts, vocab: Vocabulary):
    unknown = sorted(set("".join(texts)) - set(vocab.characters))
    if unknown:
        print(f"  WARNING: these characters are not in the model's vocabulary and "
              f"cannot be learned: {unknown}")


# ---------------------------------------------------------------------------
# Demo data (only used when you don't have your own handwriting yet)
# ---------------------------------------------------------------------------
def _photo_style(image: Image.Image, rng: np.random.Generator) -> Image.Image:
    """Make a scanned IAM line look like a slanted, slightly blurry phone photo."""
    w, h = image.size
    margin = h // 2
    canvas = Image.new("L", (w + 2 * margin, h), 255)
    canvas.paste(image, (margin, 0))
    shear = 0.45  # strong forward slant, the same for every line ("one writer's style")
    canvas = canvas.transform(canvas.size, Image.Transform.AFFINE,
                              (1, shear, -shear * h / 2, 0, 1, 0), fillcolor=255)
    canvas = canvas.filter(ImageFilter.MinFilter(3))          # thicker pen (dark ink spreads)
    canvas = canvas.filter(ImageFilter.GaussianBlur(1.0))      # camera blur
    pixels = np.asarray(canvas, dtype=np.float32)
    pixels = 50 + pixels * (200 - 50) / 255.0                 # gray paper, dark-gray ink
    gradient = np.linspace(-25, 25, pixels.shape[1])[None, :]  # uneven lighting
    pixels = pixels + gradient + rng.normal(0, 5, pixels.shape)  # sensor noise
    return Image.fromarray(np.clip(pixels, 0, 255).astype(np.uint8))


def create_demo_dataset(n_train=300, n_val=100) -> Path:
    """Build data/finetune_demo/: IAM lines restyled to look like phone photos.

    This simulates "new handwriting data that looks different from what the model
    was trained on". Lines come from the TRAIN and VALIDATION splits only - the
    test set stays untouched. The demo's own validation part comes from the IAM
    validation split, so it was never trained on.
    """
    if (DEMO_DIR / "labels.csv").exists():
        print(f"Demo dataset already exists: {DEMO_DIR}")
        return DEMO_DIR
    rng = np.random.default_rng(config.SEED)
    DEMO_DIR.mkdir(parents=True, exist_ok=True)
    picks = [("train", load_labels("train").sample(n=n_train, random_state=1)),
             ("validation", load_labels("validation").sample(n=n_val, random_state=1))]
    rows = []
    for split, df in picks:
        for i, (path, text) in enumerate(zip(df["path"], df["text"])):
            name = f"{split}_{i:04d}.jpg"
            _photo_style(Image.open(path), rng).save(DEMO_DIR / name, quality=85)
            rows.append({"file": name, "text": text, "split": split})
    pd.DataFrame(rows).to_csv(DEMO_DIR / "labels.csv", index=False, encoding="utf-8")
    print(f"Demo dataset created: {DEMO_DIR} ({n_train} train + {n_val} validation lines)")
    return DEMO_DIR


# ---------------------------------------------------------------------------
# Fine-tuning
# ---------------------------------------------------------------------------
def score(model, df, vocab):
    df = sort_by_width(df)
    ds = make_dataset(df["path"].tolist(), encode_labels(df["text"].tolist(), vocab))
    predictions, _ = predict_dataset(model, ds, df, vocab)
    return corpus_cer(df["text"], predictions), corpus_wer(df["text"], predictions)


def finetune(data_dir=MY_DATA_DIR, base_model=config.MODEL_PATH, out_path=config.FINETUNED_MODEL_PATH,
             epochs=8, learning_rate=config.FINETUNE_LEARNING_RATE, replay=300, val_fraction=0.2):
    keras.utils.set_random_seed(config.SEED)
    vocab = Vocabulary.load(config.VOCAB_PATH)       # same vocabulary as the original model
    print(f"Loading base model {Path(base_model).name}")
    model = load_trained_model(base_model)

    print("Preparing new handwriting data ...")
    new_df = prepare_folder(data_dir)
    report_unknown_characters(new_df["text"], vocab)

    # Split the new data into train / validation (fixed seed = reproducible)
    raw = pd.read_csv(Path(data_dir) / "labels.csv", dtype=str, keep_default_na=False)
    if "split" in raw.columns and len(raw) == len(new_df):   # demo data has a predefined split
        is_val = (raw["split"] == "validation").to_numpy()
    else:
        rng = np.random.default_rng(config.SEED)
        is_val = rng.random(len(new_df)) < val_fraction
        if is_val.sum() == 0:
            is_val[rng.integers(len(new_df))] = True
    new_train, new_val = new_df[~is_val], sort_by_width(new_df[is_val])
    print(f"  new data: {len(new_train)} lines for fine-tuning, {len(new_val)} for checking")

    # A few original lines to check that the model does not forget IAM handwriting
    iam_val = sort_by_width(load_labels("validation").sample(n=200, random_state=config.SEED))

    before_new = score(model, new_val, vocab)
    before_iam = score(model, iam_val, vocab)
    print(f"BEFORE fine-tuning: new data CER {before_new[0]:.3f} WER {before_new[1]:.3f} | "
          f"IAM CER {before_iam[0]:.3f}")

    # Mix in some original training lines ("replay") to avoid forgetting
    replay_df = load_labels("train").sample(n=min(replay, len(new_train) * 2), random_state=config.SEED)
    mixed = pd.concat([new_train, replay_df[["path", "text", "width"]]], ignore_index=True)
    train_ds = make_dataset(mixed["path"].tolist(), encode_labels(mixed["text"].tolist(), vocab),
                            training=True)
    val_ds = make_dataset(new_val["path"].tolist(), encode_labels(new_val["text"].tolist(), vocab))

    compile_model(model, learning_rate)   # fresh optimizer with the SMALL learning rate
    callbacks = [
        CERCallback(val_ds, new_val, vocab, n_examples=min(2, len(new_val))),
        keras.callbacks.ModelCheckpoint(out_path, monitor="val_cer", mode="min", save_best_only=True),
        keras.callbacks.EarlyStopping(monitor="val_cer", mode="min", patience=3),
    ]
    model.fit(train_ds, validation_data=val_ds, epochs=epochs, callbacks=callbacks, verbose=2)

    best = load_trained_model(out_path)
    after_new = score(best, new_val, vocab)
    after_iam = score(best, iam_val, vocab)
    print(f"AFTER fine-tuning:  new data CER {after_new[0]:.3f} WER {after_new[1]:.3f} | "
          f"IAM CER {after_iam[0]:.3f}")
    print(f"Fine-tuned model saved to {out_path}")
    result = {"before_new_cer": before_new[0], "before_new_wer": before_new[1],
              "after_new_cer": after_new[0], "after_new_wer": after_new[1],
              "before_iam_cer": before_iam[0], "after_iam_cer": after_iam[0]}
    pd.Series(result).to_json(config.RESULTS_DIR / "finetune_results.json", indent=2)
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Fine-tune the OCR model on new handwriting.")
    parser.add_argument("--data", default=str(MY_DATA_DIR), help="folder with images + labels.csv")
    parser.add_argument("--demo", action="store_true", help="create and use the demo dataset")
    parser.add_argument("--epochs", type=int, default=8)
    parser.add_argument("--base-model", default=str(config.MODEL_PATH))
    args = parser.parse_args()
    folder = create_demo_dataset() if args.demo else Path(args.data)
    finetune(folder, base_model=Path(args.base_model), epochs=args.epochs)
