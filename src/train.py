"""
Step 3: train the CNN + BiLSTM + CTC handwriting model.

Run from the project folder:
    python src/train.py                 # train from scratch
    python src/train.py --epochs 30     # choose the maximum number of epochs
    python src/train.py --resume        # continue an interrupted training run

What it does:
1. Builds the character vocabulary from the TRAINING transcriptions only.
2. Builds the input pipelines (training with augmentation, validation without).
3. Trains the model. After every epoch it
     - prints training loss, validation loss and validation CER,
     - shows a few validation lines: Actual vs Predicted,
     - saves the model if the validation CER is the best so far
       -> models/handwriting_ocr.keras
4. Saves the loss curve and the training log in results/.

The TEST set is never touched here.
"""

import argparse
import sys
import time
import warnings
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

import keras  # noqa: E402

from dataset import make_dataset_from_frame, sort_by_width  # noqa: E402
from metrics import corpus_cer, corpus_wer  # noqa: E402
from model import build_model, compile_model, greedy_decode, load_trained_model  # noqa: E402
from prepare_data import load_labels  # noqa: E402
from vocab import Vocabulary  # noqa: E402

# Harmless Keras messages caused by our width-bucketed dataset (its length is not
# known in advance). Hidden so they don't look like errors.
warnings.filterwarnings("ignore", message=".*ran out of data.*")
warnings.filterwarnings("ignore", message=".*shuffle=True.*")

LAST_MODEL_PATH = config.MODELS_DIR / "handwriting_ocr_last.keras"   # for --resume
LOG_PATH = config.RESULTS_DIR / "training_log.csv"


def predict_dataset(model, ds, df, vocab):
    """Run the model on a (non-shuffled) dataset; return predicted texts and confidences."""
    texts, confidences = [], []
    widths = df["width"].tolist()
    for images, _ in ds:
        probs = np.asarray(model.predict_on_batch(images))
        for p in probs:
            text, conf = greedy_decode(p, vocab, width=widths[len(texts)])
            texts.append(text)
            confidences.append(conf)
    return texts, confidences


class CERCallback(keras.callbacks.Callback):
    """After each epoch: compute validation CER/WER and print Actual vs Predicted examples.

    Loss numbers are hard to interpret; CER tells you directly how many characters
    are wrong. The value is added to the logs as 'val_cer' so ModelCheckpoint and
    EarlyStopping can use it.
    """

    def __init__(self, val_ds, val_df, vocab, n_examples=3):
        super().__init__()
        self.val_ds, self.val_df, self.vocab = val_ds, val_df, vocab
        self.n_examples = n_examples
        self.epoch_start = time.time()

    def on_epoch_begin(self, epoch, logs=None):
        self.epoch_start = time.time()

    def on_epoch_end(self, epoch, logs=None):
        logs = logs if logs is not None else {}
        predictions, _ = predict_dataset(self.model, self.val_ds, self.val_df, self.vocab)
        actual = self.val_df["text"].tolist()
        logs["val_cer"] = corpus_cer(actual, predictions)
        logs["val_wer"] = corpus_wer(actual, predictions)
        minutes = (time.time() - self.epoch_start) / 60

        print(f"\nEpoch {epoch + 1}: training loss {logs.get('loss', float('nan')):.2f} | "
              f"validation loss {logs.get('val_loss', float('nan')):.2f} | "
              f"validation CER {logs['val_cer']:.3f} | validation WER {logs['val_wer']:.3f} | "
              f"{minutes:.1f} min")
        rng = np.random.default_rng(epoch)
        for i in rng.choice(len(actual), size=self.n_examples, replace=False):
            print(f"  Actual:    {actual[i]}")
            print(f"  Predicted: {predictions[i]}")
        sys.stdout.flush()


def save_loss_plot(log_path=LOG_PATH, out_path=config.RESULTS_DIR / "loss_curve.png"):
    """Draw training/validation loss and validation CER from the training log."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    log = pd.read_csv(log_path)
    epochs = log["epoch"] + 1
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    ax1.plot(epochs, log["loss"], "o-", label="training loss")
    ax1.plot(epochs, log["val_loss"], "o-", label="validation loss")
    ax1.set_xlabel("epoch"); ax1.set_ylabel("CTC loss"); ax1.set_title("Loss (lower is better)")
    ax1.legend(); ax1.grid(alpha=0.3)
    ax2.plot(epochs, log["val_cer"], "o-", color="tab:green", label="validation CER")
    if "val_wer" in log:
        ax2.plot(epochs, log["val_wer"], "o-", color="tab:red", label="validation WER")
    ax2.set_xlabel("epoch"); ax2.set_ylabel("error rate"); ax2.set_title("Error rates (lower is better)")
    ax2.legend(); ax2.grid(alpha=0.3)
    fig.tight_layout()
    out_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_path, dpi=110)
    plt.close(fig)
    return out_path


def train(epochs=config.EPOCHS, resume=False, verbose=2, limit=None):
    """Train the model. `limit` = use only that many training lines (quick test runs)."""
    keras.utils.set_random_seed(config.SEED)
    config.MODELS_DIR.mkdir(parents=True, exist_ok=True)
    config.RESULTS_DIR.mkdir(parents=True, exist_ok=True)

    train_df = load_labels("train")
    val_df = load_labels("validation")
    if limit:  # quick experiment on a small subset
        train_df = train_df.sample(n=min(limit, len(train_df)), random_state=config.SEED)
        val_df = val_df.sample(n=min(max(limit // 5, 32), len(val_df)), random_state=config.SEED)
    val_df = sort_by_width(val_df)

    # Vocabulary from TRAINING labels only (no information from validation/test)
    vocab = Vocabulary.from_texts(train_df["text"])
    vocab.save(config.VOCAB_PATH)
    print(vocab)

    train_ds = make_dataset_from_frame(train_df, vocab, training=True)
    val_ds = make_dataset_from_frame(val_df, vocab, training=False)

    initial_epoch = 0
    if resume and LAST_MODEL_PATH.exists() and LOG_PATH.exists():
        model = load_trained_model(LAST_MODEL_PATH)
        initial_epoch = len(pd.read_csv(LOG_PATH))
        print(f"Resuming from epoch {initial_epoch + 1}")
    else:
        model = compile_model(build_model(vocab.size), config.LEARNING_RATE)
    model.summary()

    best_cer = None
    if initial_epoch > 0:
        best_cer = float(pd.read_csv(LOG_PATH)["val_cer"].min())

    callbacks = [
        CERCallback(val_ds, val_df, vocab),     # must come first: it adds 'val_cer' to the logs
        keras.callbacks.ModelCheckpoint(config.MODEL_PATH, monitor="val_cer", mode="min",
                                        save_best_only=True, initial_value_threshold=best_cer,
                                        verbose=1),
        keras.callbacks.ModelCheckpoint(LAST_MODEL_PATH),   # every epoch, for --resume
        keras.callbacks.ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3,
                                          min_lr=1e-5, verbose=1),
        keras.callbacks.EarlyStopping(monitor="val_cer", mode="min", patience=10,
                                      restore_best_weights=False, verbose=1),
        keras.callbacks.CSVLogger(LOG_PATH, append=initial_epoch > 0),
    ]

    start = time.time()
    history = model.fit(train_ds, validation_data=val_ds, epochs=epochs,
                        initial_epoch=initial_epoch, callbacks=callbacks, verbose=verbose)
    hours = (time.time() - start) / 3600
    print(f"\nTraining finished in {hours:.2f} hours. Best model: {config.MODEL_PATH}")
    print("Loss plot:", save_loss_plot())
    return history


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train the handwriting OCR model.")
    parser.add_argument("--epochs", type=int, default=config.EPOCHS)
    parser.add_argument("--resume", action="store_true", help="continue an interrupted run")
    parser.add_argument("--limit", type=int, default=None,
                        help="use only N training lines (quick test)")
    args = parser.parse_args()
    train(epochs=args.epochs, resume=args.resume, limit=args.limit)
