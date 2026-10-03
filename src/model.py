"""
The handwriting recognition model: CNN + Bidirectional LSTM + CTC.

    image (64 x W)
       |  CNN  - looks at small patches of pixels and learns strokes, loops,
       |         letter parts... Pooling shrinks the image: height 64 -> 4,
       |         width W -> W/4.
       v
    feature map (4 x W/4 x 128)
       |  reshape: every one of the W/4 columns becomes ONE feature vector.
       |           The line is now a sequence read from left to right.
       v
    sequence (W/4 steps, each a 512-number vector)
       |  2 x Bidirectional LSTM - reads the sequence left->right AND right->left,
       |         so each position knows its neighbours (e.g. to tell "rn" from "m").
       v
    sequence (W/4 steps)
       |  Dense + softmax - for every step: probability of each character + "blank"
       v
    probabilities (W/4 x number_of_classes)
       |  CTC decoding - take the most likely class per step, merge repeats,
       |                 remove blanks:  "hh-e-ll-ll-oo" -> "hello"
       v
    text

CTC (Connectionist Temporal Classification) is the trick that makes this work
without telling the model WHERE each letter is in the image. During training,
the CTC loss adds up the probability of ALL possible ways the model could place
the characters of the correct text across the W/4 steps. The model only needs
the image and its text.
"""

import numpy as np
import keras
from keras import layers, ops

import config

CUSTOM_OBJECTS = {}  # filled below; passed to keras.models.load_model


@keras.saving.register_keras_serializable(package="handwriting_ocr")
def ctc_loss(y_true, y_pred):
    """CTC loss for a batch.

    y_true: int labels (batch, max_label_len), padded with 0 (= blank, never a real character)
    y_pred: softmax probabilities (batch, time_steps, num_classes)
    """
    y_true = ops.cast(y_true, "int32")
    label_length = ops.sum(ops.cast(ops.not_equal(y_true, 0), "int32"), axis=1)
    batch_size, time_steps = ops.shape(y_pred)[0], ops.shape(y_pred)[1]
    input_length = ops.full((batch_size,), time_steps, dtype="int32")
    # keras.ops.ctc_loss expects log-probabilities/logits; log(softmax) is exactly that.
    log_probs = ops.log(ops.clip(y_pred, 1e-8, 1.0))
    return ops.ctc_loss(y_true, log_probs, label_length, input_length, mask_index=0)


CUSTOM_OBJECTS["ctc_loss"] = ctc_loss


def conv_block(x, filters, pool):
    """Conv 3x3 -> BatchNorm -> ReLU (-> MaxPool)."""
    x = layers.Conv2D(filters, 3, padding="same", use_bias=False)(x)
    x = layers.BatchNormalization()(x)
    x = layers.Activation("relu")(x)
    if pool:
        x = layers.MaxPooling2D(pool_size=pool)(x)
    return x


def build_model(num_classes: int, lstm_units: int = 128) -> keras.Model:
    """Create the CNN + BiLSTM model. Width is `None` so any line length works."""
    inputs = keras.Input(shape=(config.IMG_HEIGHT, None, 1), name="image")

    # ---- CNN feature extractor ----------------------------------------------
    # Few filters where the image is still big (that is where the computing cost
    # is), more filters once the image has been shrunk.
    x = conv_block(inputs, 16, pool=(2, 2))     # 64 x W   -> 32 x W/2
    x = conv_block(x, 32, pool=(2, 2))          # 32 x W/2 -> 16 x W/4
    x = conv_block(x, 64, pool=None)
    x = conv_block(x, 64, pool=(2, 1))          # 16 -> 8 (height only; keep width detail)
    x = conv_block(x, 128, pool=(2, 1))         #  8 -> 4
    x = layers.SpatialDropout2D(0.1)(x)

    # ---- turn the 2-D feature map into a left-to-right sequence -------------
    # (batch, height=4, width=W/4, channels=128) -> (batch, W/4, 4*128)
    x = layers.Permute((2, 1, 3))(x)
    x = layers.Reshape((-1, (config.IMG_HEIGHT // 16) * 128))(x)
    x = layers.Dense(256, activation="relu")(x)
    x = layers.Dropout(0.2)(x)

    # ---- sequence model ------------------------------------------------------
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    x = layers.Dropout(0.25)(x)
    x = layers.Bidirectional(layers.LSTM(lstm_units, return_sequences=True))(x)
    x = layers.Dropout(0.25)(x)

    # ---- per-time-step character probabilities (class 0 = CTC blank) ----------
    outputs = layers.Dense(num_classes, activation="softmax", name="char_probs")(x)
    return keras.Model(inputs, outputs, name="handwriting_ocr")


def compile_model(model: keras.Model, learning_rate: float) -> keras.Model:
    model.compile(optimizer=keras.optimizers.Adam(learning_rate=learning_rate, clipnorm=5.0),
                  loss=ctc_loss)
    return model


def load_trained_model(path=config.MODEL_PATH) -> keras.Model:
    """Load a saved .keras model (including the custom CTC loss) without retraining."""
    from pathlib import Path
    if not Path(path).exists():
        raise FileNotFoundError(f"No trained model at {path}. Train first (src/train.py or the notebook).")
    return keras.models.load_model(path, custom_objects=CUSTOM_OBJECTS)


# ---------------------------------------------------------------------------
# Decoding: probabilities -> text
# ---------------------------------------------------------------------------
def greedy_decode(probs: np.ndarray, vocab, width: int = None):
    """Best-path CTC decoding for ONE image.

    probs : (time_steps, num_classes) softmax output
    width : real (unpadded) image width, so padding columns are ignored
    returns (text, confidence) where confidence is the average probability of the
    chosen characters (1.0 = model completely sure, low values = unsure).
    """
    if width is not None:
        probs = probs[: max(int(np.ceil(width / config.DOWNSAMPLE)), 1)]
    best = probs.argmax(axis=-1)
    best_p = probs.max(axis=-1)

    ids, confidences, previous = [], [], -1
    for t, c in enumerate(best):
        if c != previous and c != vocab.blank_id:   # new character (not a repeat, not blank)
            ids.append(c)
            confidences.append(best_p[t])
        previous = c
    confidence = float(np.mean(confidences)) if confidences else 0.0
    return vocab.decode(ids), confidence


def predict_texts(model, images, vocab, batch_size=config.BATCH_SIZE):
    """List of preprocessed float images (H, W, 1) -> list of (text, confidence)."""
    from dataset import batch_images
    results = []
    for start in range(0, len(images), batch_size):
        chunk = images[start: start + batch_size]
        probs = model.predict_on_batch(batch_images(chunk))
        probs = np.asarray(probs)
        for img, p in zip(chunk, probs):
            results.append(greedy_decode(p, vocab, width=img.shape[1]))
    return results
