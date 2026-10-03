"""
Builds the TensorFlow input pipelines (tf.data) that feed images + labels to the model.

What happens to each example:
    PNG file  ->  decode  ->  invert + scale to 0..1  ->  (training only) augmentation
              ->  group with lines of similar width  ->  pad to the widest line in the batch

Why group by width? Lines have very different lengths (a short line may be 300
pixels wide, a long one 2000). If we padded everything to 2000 pixels the
computer would waste most of its time on empty padding. Putting similar widths
together keeps the padding small and training fast.
"""

import math

import numpy as np
import tensorflow as tf
import keras

import config
from vocab import Vocabulary

AUTOTUNE = tf.data.AUTOTUNE
WIDTH_BUCKETS = [384, 640, 768, 896, 1024, 1152, 1280, 1536, 1792]


def encode_labels(texts, vocab: Vocabulary) -> np.ndarray:
    """List of strings -> 2-D int array, padded with 0 (= CTC blank, never a real character)."""
    encoded = [vocab.encode(t) for t in texts]
    max_len = max(len(e) for e in encoded)
    labels = np.zeros((len(encoded), max_len), dtype=np.int32)
    for i, e in enumerate(encoded):
        labels[i, : len(e)] = e
    return labels


def _load_png(path):
    return _to_float(_read_png(path))


def _read_png(path):
    return tf.io.decode_png(tf.io.read_file(path), channels=1)    # uint8, (H, W, 1)


def _to_float(image):
    return 1.0 - tf.cast(image, tf.float32) / 255.0                # ink = 1, paper = 0


def augment(image):
    """Small random changes so the model sees slightly different handwriting every epoch.

    Following published IAM experiments, with 50% probability we apply:
      * slant (shear)            - people write more upright or more slanted
      * horizontal stretch       - narrower or wider handwriting
      * stroke thickness change  - thin pen vs thick marker
      * ink intensity change     - faint vs dark ink
    The changes are small on purpose: the text must stay perfectly readable,
    otherwise the label would no longer describe the image.
    """
    if tf.random.uniform(()) > 0.5:
        return image
    h = tf.shape(image)[0]
    w = tf.shape(image)[1]

    # Horizontal stretch
    stretch = tf.random.uniform((), 0.85, 1.15)
    new_w = tf.maximum(tf.cast(tf.cast(w, tf.float32) * stretch, tf.int32), 8)
    image = tf.image.resize(image, (h, new_w))

    # Slant: shift each row sideways in proportion to its distance from the middle.
    # A margin is added first so slanted letters at the edges are not cut off.
    margin = 16
    image = tf.pad(image, [[0, 0], [margin, margin], [0, 0]])
    shear = tf.random.uniform((), -0.4, 0.4)
    cy = tf.cast(h, tf.float32) / 2.0
    # The transform maps each OUTPUT pixel (x, y) to the INPUT pixel (x + shear*(y - cy), y)
    transform = tf.stack([1.0, shear, -shear * cy, 0.0, 1.0, 0.0, 0.0, 0.0])
    image = keras.ops.image.affine_transform(
        image, transform, interpolation="bilinear", fill_mode="constant", fill_value=0.0
    )

    # Stroke thickness: dilate (thicker), erode (thinner) or keep, using a 2x2 window
    choice = tf.random.uniform((), 0, 3, dtype=tf.int32)
    batched = image[tf.newaxis]
    thicker = tf.nn.max_pool2d(batched, 2, 1, "SAME")[0]
    thinner = -tf.nn.max_pool2d(-batched, 2, 1, "SAME")[0]
    image = tf.switch_case(choice, [lambda: image, lambda: thicker, lambda: thinner])

    image = image * tf.random.uniform((), 0.7, 1.0)  # fainter / darker ink
    return tf.clip_by_value(image, 0.0, 1.0)


def make_dataset(paths, labels, batch_size=config.BATCH_SIZE, training=False):
    """tf.data pipeline yielding (images, labels) batches.

    images: float32 (batch, 64, width, 1), padded with 0 on the right
    labels: int32   (batch, max_label_len), padded with 0 (blank)
    """
    ds = tf.data.Dataset.from_tensor_slices((list(paths), labels))
    # Decode PNGs once and keep them in memory as bytes (uint8 = 4x smaller than floats)
    ds = ds.map(lambda p, y: (_read_png(p), y), num_parallel_calls=AUTOTUNE).cache()
    if training:  # shuffle the small uint8 images, before converting them to floats
        ds = ds.shuffle(len(paths), seed=config.SEED, reshuffle_each_iteration=True)
    ds = ds.map(lambda x, y: (_to_float(x), y), num_parallel_calls=AUTOTUNE)
    if training:
        if config.AUGMENT:
            ds = ds.map(lambda x, y: (augment(x), y), num_parallel_calls=AUTOTUNE)

    if not training:
        # Keep the original order (so predictions line up with the labels table).
        ds = ds.padded_batch(batch_size, padded_shapes=([config.IMG_HEIGHT, None, 1], [None]),
                             padding_values=(0.0, 0))
        return ds.map(_pad_width_to_multiple, num_parallel_calls=AUTOTUNE).prefetch(AUTOTUNE)

    return _bucket_and_pad(ds, batch_size)


def _bucket_and_pad(ds, batch_size):
    """Training batches: group lines of similar width into the same batch (less padding)."""
    ds = ds.bucket_by_sequence_length(
        element_length_func=lambda x, y: tf.shape(x)[1],
        bucket_boundaries=WIDTH_BUCKETS,
        bucket_batch_sizes=[batch_size] * (len(WIDTH_BUCKETS) + 1),
        padded_shapes=([config.IMG_HEIGHT, None, 1], [None]),
        padding_values=(0.0, 0),
        pad_to_bucket_boundary=False,
        drop_remainder=False,
    )
    # The CNN divides the width by DOWNSAMPLE, so pad widths to a multiple of it.
    ds = ds.map(_pad_width_to_multiple, num_parallel_calls=AUTOTUNE)
    return ds.prefetch(AUTOTUNE)


def _pad_width_to_multiple(images, labels):
    width = tf.shape(images)[2]
    target = (width + config.DOWNSAMPLE - 1) // config.DOWNSAMPLE * config.DOWNSAMPLE
    images = tf.pad(images, [[0, 0], [0, 0], [0, target - width], [0, 0]])
    return images, labels


def make_dataset_from_frame(df, vocab: Vocabulary, training=False, batch_size=config.BATCH_SIZE):
    """Convenience: labels table (from prepare_data.load_labels) -> tf.data pipeline.

    For validation/test, sort the table by width first (see sort_by_width) so each
    batch contains similar widths; the batches then follow the table's order.
    """
    return make_dataset(df["path"].tolist(), encode_labels(df["text"].tolist(), vocab),
                        batch_size=batch_size, training=training)


def batch_images(images):
    """List of float32 (H, W, 1) arrays -> one zero-padded batch (N, H, maxW, 1) for prediction."""
    max_w = max(img.shape[1] for img in images)
    max_w = int(math.ceil(max_w / config.DOWNSAMPLE) * config.DOWNSAMPLE)
    batch = np.zeros((len(images), config.IMG_HEIGHT, max_w, 1), dtype=np.float32)
    for i, img in enumerate(images):
        batch[i, :, : img.shape[1]] = img
    return batch


def sort_by_width(df):
    """Sort a labels table by image width (for fast, order-preserving evaluation)."""
    return df.sort_values("width", kind="stable").reset_index(drop=True)
