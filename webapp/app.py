"""
A small local web page: upload a photo of handwriting -> see the recognized text.

Run from the project folder (with the virtual environment activated):
    python webapp/app.py
Then open  http://127.0.0.1:5000  in your browser.

Everything runs on YOUR computer: the image is never sent to the internet.
Stop the server with Ctrl + C in the terminal.
"""

import base64
import io
import os
import sys
from pathlib import Path

os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "3")  # hide TensorFlow start-up messages

PROJECT_DIR = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_DIR / "src"))

import config  # noqa: E402
from flask import Flask, jsonify, render_template, request  # noqa: E402
from PIL import Image, UnidentifiedImageError  # noqa: E402

from model import load_trained_model, predict_texts  # noqa: E402
from predict import confidence_label  # noqa: E402
from preprocessing import preprocess_pil, segment_lines, to_model_input  # noqa: E402
from vocab import Vocabulary  # noqa: E402

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = 15 * 1024 * 1024   # refuse uploads bigger than 15 MB

_models = {}  # loaded models are kept in memory, so only the first request is slow


def available_models():
    """Trained models in models/ (the 'last' checkpoint is only for resuming training)."""
    found = [p.name for p in sorted(config.MODELS_DIR.glob("*.keras")) if not p.stem.endswith("_last")]
    # put the fine-tuned model first if it exists
    return sorted(found, key=lambda name: "finetuned" not in name)


def get_model(name):
    """Load a model once; reload it automatically if the file was updated (e.g. by training)."""
    path = config.MODELS_DIR / name
    modified = path.stat().st_mtime
    if name not in _models or _models[name][0] != modified:
        _models[name] = (modified, load_trained_model(path))
    return _models[name][1]


def png_base64(array):
    """uint8 image array -> base64 PNG, so the browser can show what the model saw."""
    buffer = io.BytesIO()
    Image.fromarray(array).save(buffer, format="PNG")
    return base64.b64encode(buffer.getvalue()).decode("ascii")


@app.get("/")
def index():
    return render_template("index.html", models=available_models())


@app.post("/predict")
def predict():
    upload = request.files.get("image")
    if upload is None or upload.filename == "":
        return jsonify(error="Please choose an image first."), 400

    models = available_models()
    if not models:
        return jsonify(error="No trained model found in models/. Train the model first."), 500
    model_name = request.form.get("model") or models[0]
    if model_name not in models:
        return jsonify(error=f"Unknown model: {model_name}"), 400
    single_line = request.form.get("single_line") == "true"

    try:
        image = Image.open(upload.stream)
        image.load()
    except (UnidentifiedImageError, OSError):
        return jsonify(error="That file is not an image (or it is damaged). Try a JPG or PNG."), 400

    line_images = [image] if single_line else segment_lines(image)
    processed = [preprocess_pil(line) for line in line_images]
    results = predict_texts(get_model(model_name), [to_model_input(p) for p in processed],
                            Vocabulary.load(config.VOCAB_PATH))

    lines = [{"text": text, "confidence": round(conf, 3), "label": confidence_label(conf),
              "image": png_base64(p)}
             for (text, conf), p in zip(results, processed)]
    mean_conf = sum(line["confidence"] for line in lines) / len(lines)
    return jsonify(model=model_name, lines=lines,
                   text="\n".join(line["text"] for line in lines),
                   confidence=round(mean_conf, 3), label=confidence_label(mean_conf))


@app.errorhandler(413)
def too_large(_):
    return jsonify(error="Image is too large (max 15 MB)."), 413


if __name__ == "__main__":
    if not available_models():
        print("WARNING: no trained model in models/ yet - train first (notebook or src/train.py).")
    print("Open http://127.0.0.1:5000 in your browser.  (Ctrl + C to stop)")
    app.run(host="127.0.0.1", port=5000, debug=False)
