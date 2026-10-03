"""
Step 5: read handwriting from your own image(s).

Run from the project folder:
    python src/predict.py path/to/handwriting.png
    python src/predict.py page1.jpg page2.jpg
    python src/predict.py my_line.png --single-line      # image is exactly one line
    python src/predict.py note.jpg --model models/handwriting_ocr_finetuned.keras

Tips for best results (the model learned from scanned IAM forms):
  * dark pen on white, unlined paper, good even lighting, no shadows
  * one line of text per image works best; a few well-separated lines also work
    (they are split automatically)
  * crop away anything that isn't handwriting (table edge, fingers, etc.)
"""

import argparse
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # hide TensorFlow start-up messages

sys.path.insert(0, str(Path(__file__).resolve().parent))
import config  # noqa: E402

from PIL import Image, UnidentifiedImageError  # noqa: E402

from model import load_trained_model, predict_texts  # noqa: E402
from preprocessing import preprocess_pil, segment_lines, to_model_input  # noqa: E402
from vocab import Vocabulary  # noqa: E402


def confidence_label(confidence: float) -> str:
    if confidence >= 0.9:
        return "high"
    if confidence >= 0.75:
        return "medium"
    return "low - the text is probably partly wrong"


def recognize(image_path, model, vocab, single_line=False):
    """Return a list of (text, confidence), one per detected line."""
    path = Path(image_path)
    if not path.exists():
        raise FileNotFoundError(f"Image not found: {path}")
    try:
        with Image.open(path) as image:
            image.load()
            lines = [image] if single_line else segment_lines(image)
            inputs = [to_model_input(preprocess_pil(line)) for line in lines]
    except (UnidentifiedImageError, OSError) as error:
        raise ValueError(f"Could not read image (corrupted or not an image?): {path}") from error
    return predict_texts(model, inputs, vocab)


def main():
    parser = argparse.ArgumentParser(description="Recognize handwritten text in images.")
    parser.add_argument("images", nargs="+", help="image file(s) with handwriting")
    parser.add_argument("--model", default=None,
                        help="model file (default: fine-tuned model if it exists, else the base model)")
    parser.add_argument("--single-line", action="store_true",
                        help="do not try to split the image into several lines")
    args = parser.parse_args()

    model_path = Path(args.model) if args.model else (
        config.FINETUNED_MODEL_PATH if config.FINETUNED_MODEL_PATH.exists() else config.MODEL_PATH)
    try:
        model = load_trained_model(model_path)
        vocab = Vocabulary.load(config.VOCAB_PATH)
    except FileNotFoundError as error:
        sys.exit(str(error))
    print(f"(model: {model_path.name})")

    for image_path in args.images:
        try:
            results = recognize(image_path, model, vocab, single_line=args.single_line)
        except (FileNotFoundError, ValueError) as error:
            print(f"\n{image_path}: ERROR - {error}")
            continue
        print(f"\n{image_path}")
        print("Recognized text:")
        for text, _ in results:
            print(text)
        mean_conf = sum(c for _, c in results) / len(results)
        print(f"Confidence: {mean_conf:.0%} ({confidence_label(mean_conf)})"
              + (f"  [{len(results)} lines detected]" if len(results) > 1 else ""))


if __name__ == "__main__":
    main()
