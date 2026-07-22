# app/app.py
"""
Minimal Flask demo for the Parkinson's FYP.

Upload a .wav recording, run it through the trained CNN-LSTM pipeline, and
display the HC/PD prediction with a confidence percentage.  Demo interface
for a project defense — not a production service.

Run locally
-----------
    python app/app.py

Then open http://127.0.0.1:5000 in a browser.
"""

from __future__ import annotations

import logging
import sys
from pathlib import Path

from flask import (
    Flask, flash, redirect, render_template, request, url_for,
    send_from_directory,
)
from werkzeug.utils import secure_filename

# ── Path bootstrap: make app/predict.py importable when run from anywhere ──
_APP_DIR = Path(__file__).resolve().parent
if str(_APP_DIR) not in sys.path:
    sys.path.insert(0, str(_APP_DIR))

from predict import predict_recording  # noqa: E402

logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(message)s",
                    datefmt="%H:%M:%S")

# Scratch folder for temporary uploads (created if absent).
UPLOAD_DIR = _APP_DIR / "uploads"
UPLOAD_DIR.mkdir(parents=True, exist_ok=True)

ALLOWED_EXTENSIONS = {".wav"}
MAX_CONTENT_MB = 50   # reject absurdly large uploads

app = Flask(__name__)
app.config["SECRET_KEY"] = "fyp-demo-not-secret"   # only for flash messages
app.config["MAX_CONTENT_LENGTH"] = MAX_CONTENT_MB * 1024 * 1024


def _is_wav(filename: str) -> bool:
    """Return True if the filename has a .wav extension (case-insensitive)."""
    return Path(filename).suffix.lower() in ALLOWED_EXTENSIONS


@app.route("/", methods=["GET"])
def upload_page():
    """Render the upload form."""
    return render_template("upload.html")


@app.route("/predict", methods=["POST"])
def predict():
    """Handle an upload, run inference, and render the result.

    The uploaded file is intentionally kept (not deleted) so it can be played
    back on the result page.  See cleanup_uploads() for how old files are
    pruned — for a live demo the uploads/ folder holds only a few files.
    """
    if "audio" not in request.files:
        flash("No file part in the request.")
        return redirect(url_for("upload_page"))

    file = request.files["audio"]
    if file.filename == "":
        flash("No file selected.")
        return redirect(url_for("upload_page"))

    if not _is_wav(file.filename):
        flash("Only .wav files are accepted.")
        return redirect(url_for("upload_page"))

    filename = secure_filename(file.filename)
    saved_path = UPLOAD_DIR / filename
    file.save(str(saved_path))

    try:
        result = predict_recording(saved_path)
    except Exception as exc:
        app.logger.exception("Inference failed")
        # Inference failed, so the file is useless — remove it now.
        saved_path.unlink(missing_ok=True)
        flash(f"Could not process the recording: {exc}")
        return redirect(url_for("upload_page"))

    # Note: the file is deliberately NOT deleted here — the result page plays
    # it back via the /uploads/<filename> route below.
    return render_template(
        "result.html",
        filename=filename,
        label=result["label"],
        confidence=result["confidence"],
        probability_pd=result["probability_pd"],
        probability_pd_pct=round(result["probability_pd"] * 100, 1),
        num_windows=result["num_windows"],
    )

@app.route("/uploads/<path:filename>")
def serve_upload(filename: str):
    """Serve a saved upload so the result page can play it back.

    secure_filename is re-applied as a defensive measure against path
    traversal, since this filename comes from the URL.
    """
    safe = secure_filename(filename)
    return send_from_directory(UPLOAD_DIR, safe)

def cleanup_uploads() -> None:
    """Delete leftover uploads from previous runs (called at startup)."""
    for f in UPLOAD_DIR.glob("*.wav"):
        try:
            f.unlink()
        except OSError:
            pass

if __name__ == "__main__":
    # debug=True is convenient for a local demo; turn off if presenting on a
    # shared network.
    cleanup_uploads()
    app.run(host="127.0.0.1", port=5000, debug=True)