"""
Image preprocessing shared by training, evaluation and prediction.

It is VERY important that new images (e.g. a photo of your own handwriting) are
processed exactly like the training images, otherwise the model sees something
it has never seen before. That's why this lives in one shared file.

Steps for every image:
1. Grayscale        - colour carries no information about which letter it is.
                      We pick the grey version (brightness, or the red/green/blue
                      channel) where text and background differ most, and flip
                      light-on-dark text (dark mode, blackboards) to dark-on-light,
                      so the background colour does not matter.
2. Auto-contrast    - stretch the darkest pixel to black and the lightest to white,
                      so faint pencil and dark ink look similar.
   Clean background - gray / speckled paper (phone photos) becomes pure white.
   Ruled lines      - long straight lines of lined paper are erased (IAM has none).
3. Crop margins     - remove empty paper around the text (important for photos).
4. Resize height    - every line becomes IMG_HEIGHT (64) pixels tall. The width is
                      scaled by the SAME factor, so letters keep their shape
                      (no stretching or squashing).
The result still looks like normal handwriting (dark ink on white paper), so it
can be saved as a PNG and viewed. Just before the model sees it, to_model_input() then:
5. Inverts          - ink becomes bright and paper dark, so padding with 0 means
                      "nothing here".
6. Normalises       - pixel values 0..255 become 0.0..1.0 (neural networks train
                      much better on small numbers).
"""

from pathlib import Path

import numpy as np
from PIL import Image, ImageOps, UnidentifiedImageError

import config


def preprocess_pil(image: Image.Image, height: int = config.IMG_HEIGHT,
                   max_width: int = config.MAX_IMG_WIDTH) -> np.ndarray:
    """PIL image -> uint8 array of shape (height, width): dark ink on white paper."""
    image = ImageOps.exif_transpose(image)       # respect phone camera rotation
    gray = to_dark_on_light(image)               # 1. grayscale, dark text on light background
    gray = ImageOps.autocontrast(gray, cutoff=1)  # 2. auto-contrast
    gray = clean_background(gray)                 # 2b. paper -> pure white
    gray = remove_ruled_lines(gray)               # 2c. erase lines of lined paper

    # 3. crop to the bounding box of the ink (dark pixels), keeping a small margin
    pixels = np.asarray(gray)
    ink_rows = np.where((pixels < 128).any(axis=1))[0]
    ink_cols = np.where((pixels < 128).any(axis=0))[0]
    if len(ink_rows) > 0 and len(ink_cols) > 0:
        margin = 4
        top, bottom = max(ink_rows[0] - margin, 0), min(ink_rows[-1] + margin + 1, pixels.shape[0])
        left, right = max(ink_cols[0] - margin, 0), min(ink_cols[-1] + margin + 1, pixels.shape[1])
        gray = gray.crop((left, top, right, bottom))

    # 4. resize to a fixed height, keeping the aspect ratio
    w, h = gray.size
    new_w = max(int(round(w * height / h)), 8)
    if new_w > max_width:  # absurdly long line: squeeze horizontally (never happens with IAM)
        new_w = max_width
    gray = gray.resize((new_w, height), Image.Resampling.LANCZOS)

    return np.asarray(gray, dtype=np.uint8)


def to_dark_on_light(image: Image.Image) -> Image.Image:
    """Any image -> grayscale with DARK text on a LIGHT background.

    * Transparent images are placed on white.
    * Colour: red text on a green background can have the same brightness, so a
      plain grayscale conversion would make the text disappear. We try four grey
      versions (brightness, red, green, blue) and keep the one where the two
      groups of pixels (text / background) are most different (Otsu's measure).
    * Polarity: text normally covers less area than the background. If the dark
      pixels are the majority, the background is dark (dark mode, blackboard,
      white text on a coloured banner) and we invert the image.
    """
    if image.mode in ("RGBA", "LA") or (image.mode == "P" and "transparency" in image.info):
        rgba = image.convert("RGBA")
        white = Image.new("RGBA", rgba.size, (255, 255, 255, 255))
        image = Image.alpha_composite(white, rgba)
    if image.mode in ("L", "I;16", "I", "F", "1"):   # already grey: nothing to choose
        candidates = [np.asarray(image.convert("L"), dtype=np.float32)]
    else:
        rgb = np.asarray(image.convert("RGB"), dtype=np.float32)
        candidates = [np.asarray(image.convert("L"), dtype=np.float32),   # brightness (PIL's formula)
                      rgb[..., 0], rgb[..., 1], rgb[..., 2]]
    best, best_score, best_threshold = candidates[0], -1.0, 128
    for channel in candidates:
        threshold, score = _otsu(channel)
        if score > best_score * 1.15:   # prefer brightness unless a channel is clearly better
            best, best_score, best_threshold = channel, score, threshold
    if (best < best_threshold).mean() > 0.5:   # mostly dark -> dark background -> invert
        best = 255.0 - best
    return Image.fromarray(np.clip(np.rint(best), 0, 255).astype(np.uint8))


def clean_background(gray: Image.Image) -> Image.Image:
    """Make the paper pure white (phone photos have gray, speckled, unevenly lit paper).

    The IAM training images are clean scans with white paper. In a photo, the
    gray paper texture and small specks can look like faint strokes, and the
    model may "read" them as extra letters. We find the ink/paper threshold
    automatically (Otsu's method), then turn everything clearly brighter than
    the ink white and stretch the ink to the full dark range.
    """
    pixels = np.asarray(gray, dtype=np.float32)
    threshold = _otsu_threshold(pixels)
    paper = pixels[pixels > threshold]
    # Clean scans (like all IAM images: paper median >= 239, noise <= 20) are left
    # untouched - cleaning them would only make the ink slightly lighter.
    if paper.size == 0 or (np.median(paper) >= 232 and paper.std() <= 22):
        return gray
    paper_level = min(threshold + 0.5 * (255 - threshold), 250)   # halfway between threshold and white
    cleaned = np.clip(pixels * 255.0 / paper_level, 0, 255)       # paper -> 255, ink stays dark
    return Image.fromarray(cleaned.astype(np.uint8))


def remove_ruled_lines(gray: Image.Image, min_fraction: float = 0.3, min_length: int = 80) -> Image.Image:
    """Erase long straight horizontal lines (printed lines of lined paper, ruler-drawn lines).

    The IAM training images are written on unlined paper, so the model has never
    seen such a line. It runs through the letters and joins all words together
    (the model then also stops seeing the spaces between words).

    A pixel row counts as "line" where ink continues WITHOUT A BREAK for at least
    `min_fraction` of the image width (and at least `min_length` pixels).
    Handwriting never has such long, perfectly straight strokes, so letters are
    left alone. A line may move up/down by one pixel (slightly tilted photo).

    Limitation: wavy hand-drawn lines that touch the letters are not removed
    reliably - write on unlined paper for best results.
    """
    pixels = np.array(gray, dtype=np.uint8)
    height, width = pixels.shape
    run = max(int(width * min_fraction), min_length)
    if width < run or height < 3:
        return gray

    ink = pixels < _otsu_threshold(pixels)
    tolerant = ink.copy()                  # a row also "sees" the ink just above/below it
    tolerant[1:] |= ink[:-1]
    tolerant[:-1] |= ink[1:]

    # For every position: does an unbroken ink run of length `run` start here?
    counts = np.cumsum(np.pad(tolerant, ((0, 0), (1, 0))), axis=1, dtype=np.int32)
    starts = np.zeros((height, width), dtype=bool)
    starts[:, : width - run + 1] = (counts[:, run:] - counts[:, : width - run + 1]) == run
    if not starts.any():
        return gray  # no lines: image unchanged

    # Mark every pixel covered by such a run and paint its ink white (= paper)
    n_starts = np.cumsum(np.pad(starts, ((0, 0), (1, 0))), axis=1, dtype=np.int32)
    left = np.maximum(np.arange(width) - run + 1, 0)
    covered = (n_starts[:, 1:] - n_starts[:, left]) > 0
    pixels[covered & ink] = 255
    return Image.fromarray(pixels)


def to_model_input(image: np.ndarray) -> np.ndarray:
    """uint8 (H, W) dark-ink image -> float32 (H, W, 1) with ink = 1.0, paper = 0.0."""
    return (1.0 - image.astype(np.float32) / 255.0)[..., np.newaxis]


def load_image(path) -> np.ndarray:
    """Open an image file and preprocess it. Raises a friendly error if it is missing/corrupt."""
    path = Path(path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            return preprocess_pil(image)
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Could not read image (corrupted or not an image?): {path}") from error


def segment_lines(image: Image.Image, min_line_height: int = 15):
    """Split a picture of several handwritten lines into one image per line.

    The model reads ONE line at a time (like the IAM training data). For a photo
    of a whole paragraph we look for horizontal "empty paper" bands between lines:
    count the ink pixels in every row; rows with (almost) no ink separate lines.
    This simple method works for neatly written lines on plain paper; for
    tightly packed or slanted lines, crop the lines yourself instead.
    """
    gray = ImageOps.autocontrast(to_dark_on_light(ImageOps.exif_transpose(image)), cutoff=1)
    gray = remove_ruled_lines(clean_background(gray))   # ruled lines would look like rows full of ink
    pixels = np.asarray(gray, dtype=np.float32)
    ink = pixels < _otsu_threshold(pixels)
    row_ink = ink.sum(axis=1).astype(np.float32)
    row_ink = np.convolve(row_ink, np.ones(5) / 5, mode="same")   # smooth out noise
    is_text = row_ink > max(row_ink.max() * 0.05, 1)

    lines, start = [], None
    for y, text_row in enumerate(np.append(is_text, False)):
        if text_row and start is None:
            start = y
        elif not text_row and start is not None:
            if y - start >= min_line_height:
                pad = (y - start) // 4
                top, bottom = max(start - pad, 0), min(y + pad, pixels.shape[0])
                lines.append(gray.crop((0, top, gray.width, bottom)))
            start = None
    return lines if lines else [gray]


def _otsu_threshold(pixels: np.ndarray) -> float:
    """Automatic black/white threshold (Otsu's method)."""
    return _otsu(pixels)[0]


def _otsu(pixels: np.ndarray):
    """Otsu's method: (threshold, how well it separates the two groups of pixels)."""
    hist, _ = np.histogram(pixels, bins=256, range=(0, 256))
    total, sum_all = pixels.size, np.dot(np.arange(256), hist)
    weight_bg, sum_bg, best, threshold = 0, 0.0, -1.0, 128
    for t in range(256):
        weight_bg += hist[t]
        if weight_bg == 0 or weight_bg == total:
            continue
        sum_bg += t * hist[t]
        mean_bg = sum_bg / weight_bg
        mean_fg = (sum_all - sum_bg) / (total - weight_bg)
        between = weight_bg * (total - weight_bg) * (mean_bg - mean_fg) ** 2
        if between > best:
            best, threshold = between, t
    return threshold, best / (total * total)
