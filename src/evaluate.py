"""
Step 4: final evaluation on the TEST set (handwriting and sentences never used in training).

Run from the project folder:
    python src/evaluate.py
    python src/evaluate.py --model models/handwriting_ocr_finetuned.keras

Saves to results/:
    metrics.json                - test CER and WER
    test_predictions.csv        - every test line: actual, predicted, CER, confidence
    example_predictions.png     - good AND bad examples with their images
    error_analysis.txt          - which characters are confused most, CER by line length
    character_confusions.png    - bar chart of the most common character mistakes
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

from dataset import make_dataset_from_frame, sort_by_width  # noqa: E402
from metrics import align, cer, corpus_cer, corpus_wer, wer  # noqa: E402
from model import load_trained_model  # noqa: E402
from prepare_data import load_labels  # noqa: E402
from train import predict_dataset  # noqa: E402
from vocab import Vocabulary  # noqa: E402


def evaluate_split(model, vocab, split="test", limit=None):
    """Predict every line of a split and return a table with per-line results."""
    df = load_labels(split)
    if limit:
        df = df.sample(n=min(limit, len(df)), random_state=config.SEED)
    df = sort_by_width(df)
    ds = make_dataset_from_frame(df, vocab, training=False)
    predictions, confidences = predict_dataset(model, ds, df, vocab)
    df["predicted"] = predictions
    df["confidence"] = confidences
    df["cer"] = [cer(a, p) for a, p in zip(df["text"], predictions)]
    df["wer"] = [wer(a, p) for a, p in zip(df["text"], predictions)]
    return df


def summarize(df):
    return {
        "lines": int(len(df)),
        "CER": round(corpus_cer(df["text"], df["predicted"]), 4),
        "WER": round(corpus_wer(df["text"], df["predicted"]), 4),
        "perfect_lines": int((df["cer"] == 0).sum()),
    }


def show_examples(df, out_path, n_good=4, n_bad=4, seed=config.SEED):
    """Save a figure with good and bad predictions next to their images."""
    import matplotlib
    import matplotlib.pyplot as plt
    from PIL import Image

    rng = np.random.default_rng(seed)
    good = df[df["cer"] <= 0.05]
    bad = df[df["cer"] > 0.15]
    picks = [("GOOD", r) for _, r in good.sample(n=min(n_good, len(good)), random_state=seed).iterrows()]
    picks += [("NEEDS WORK", r) for _, r in bad.sample(n=min(n_bad, len(bad)), random_state=seed).iterrows()]
    if not picks:
        return None
    rng.shuffle(picks)

    fig, axes = plt.subplots(len(picks), 1, figsize=(13, 1.9 * len(picks)))
    axes = np.atleast_1d(axes)
    for ax, (tag, row) in zip(axes, picks):
        ax.imshow(Image.open(row["path"]), cmap="gray", vmin=0, vmax=255)
        ax.set_title(f"[{tag}  CER {row['cer']:.2f}]\nActual:    {row['text']}\nPredicted: {row['predicted']}",
                     loc="left", fontsize=9, family="monospace")
        ax.axis("off")
    fig.tight_layout()
    if out_path:
        fig.savefig(out_path, dpi=90)
    if matplotlib.get_backend().lower() == "agg":
        plt.close(fig)
    return fig


def error_analysis(df):
    """Count error types, most confused characters and CER by line length."""
    edits = Counter()
    confusions = Counter()
    for actual, predicted in zip(df["text"], df["predicted"]):
        for op, a, p in align(actual, predicted):
            edits[op] += 1
            show = lambda c: "␣" if c == " " else c  # noqa: E731 - make spaces visible
            if op == "substitution":
                confusions[f"'{show(a)}' read as '{show(p)}'"] += 1
            elif op == "deletion":
                confusions[f"'{show(a)}' missed"] += 1
            else:
                confusions[f"extra '{show(p)}'"] += 1

    lengths = pd.cut(df["text"].str.len(), bins=[0, 20, 40, 60, 100],
                     labels=["1-20 chars", "21-40 chars", "41-60 chars", "61+ chars"])
    by_length = df.groupby(lengths, observed=True).apply(
        lambda g: corpus_cer(g["text"], g["predicted"]), include_groups=False)
    return edits, confusions, by_length


def save_error_analysis(df, results_dir=config.RESULTS_DIR, suffix=""):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    edits, confusions, by_length = error_analysis(df)
    total = sum(edits.values())
    lines = ["ERROR ANALYSIS (test set)", "=" * 40, "",
             "Types of character errors:"]
    for op in ["substitution", "deletion", "insertion"]:
        lines.append(f"  {op:13s}: {edits[op]:6d}  ({edits[op] / max(total, 1):.0%})")
    lines += ["  substitution = wrong character, deletion = character missed,",
              "  insertion = extra character added", "",
              "Most common mistakes:"]
    for name, count in confusions.most_common(20):
        lines.append(f"  {count:5d}  {name}")
    lines += ["", "CER by line length:"]
    for name, value in by_length.items():
        lines.append(f"  {name:12s}: {value:.3f}")
    lines += ["", "Confidence vs. accuracy (is the confidence score useful?):"]
    df = df.assign(conf_bin=pd.qcut(df["confidence"], 4, labels=["lowest 25%", "low-mid", "mid-high", "highest 25%"],
                                    duplicates="drop"))
    for name, g in df.groupby("conf_bin", observed=True):
        lines.append(f"  confidence {name:12s}: CER {corpus_cer(g['text'], g['predicted']):.3f}")
    text = "\n".join(lines)
    (results_dir / f"error_analysis{suffix}.txt").write_text(text, encoding="utf-8")

    top = confusions.most_common(15)[::-1]
    if top:
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.barh([t[0] for t in top], [t[1] for t in top], color="tab:orange")
        ax.set_title("Most common character mistakes on the test set")
        ax.set_xlabel("count")
        fig.tight_layout()
        fig.savefig(results_dir / f"character_confusions{suffix}.png", dpi=100)
        plt.close(fig)
    return text


def main(model_path=config.MODEL_PATH, limit=None):
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)
    model = load_trained_model(model_path)
    vocab = Vocabulary.load(config.VOCAB_PATH)

    print(f"Evaluating {Path(model_path).name} on the test set ...")
    df = evaluate_split(model, vocab, "test", limit=limit)
    metrics = summarize(df)
    metrics["model"] = Path(model_path).name
    print(json.dumps(metrics, indent=2))

    suffix = "" if Path(model_path) == config.MODEL_PATH else "_" + Path(model_path).stem
    (config.RESULTS_DIR / f"metrics{suffix}.json").write_text(json.dumps(metrics, indent=2))
    df[["file", "text", "predicted", "cer", "wer", "confidence"]].to_csv(
        config.RESULTS_DIR / f"test_predictions{suffix}.csv", index=False, encoding="utf-8")

    import matplotlib
    matplotlib.use("Agg")
    show_examples(df, config.RESULTS_DIR / f"example_predictions{suffix}.png")
    print(save_error_analysis(df, suffix=suffix))

    print("\nSome test lines (all sentences were never seen during training):")
    for _, row in df.sample(n=min(8, len(df)), random_state=1).iterrows():
        print(f"  Actual:    {row['text']}\n  Predicted: {row['predicted']}   (CER {row['cer']:.2f})\n")
    return metrics, df


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate the OCR model on the test set.")
    parser.add_argument("--model", default=str(config.MODEL_PATH))
    parser.add_argument("--limit", type=int, default=None, help="evaluate only N test lines")
    args = parser.parse_args()
    main(Path(args.model), limit=args.limit)
