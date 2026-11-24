import os
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
from flask import Flask, render_template, request
from tensorflow.keras.models import load_model
from werkzeug.utils import secure_filename

BASE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = BASE_DIR.parent
MODEL_PATH = PROJECT_DIR / "models" / "ecg_best_ann.keras"
SCALER_PATH = PROJECT_DIR / "models" / "ecg_scaler.pkl"

CLASS_NAMES = [
    "Normal (N)",
    "Supraventricular (S)",
    "Ventricular (V)",
    "Fusion (F)",
    "Unknown (Q)",
]
CLASS_SHORT = ["N", "S", "V", "F", "Q"]
MAX_UPLOAD_BYTES = 500 * 1024 * 1024
CHUNK_SIZE = 20_000
PREVIEW_LIMIT = 15

app = Flask(__name__)
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_BYTES

if not MODEL_PATH.is_file():
    raise FileNotFoundError(
        f"Trained model not found: {MODEL_PATH}. "
        "Save your trained model as models/ecg_best_ann.keras."
    )

model = load_model(MODEL_PATH, compile=False)
scaler = joblib.load(SCALER_PATH) if SCALER_PATH.is_file() else None


def prepare_features(frame):
    """Validate a CSV chunk and return its 187 ECG signal columns."""
    if frame.shape[1] == 188:
        frame = frame.iloc[:, :187]
    elif frame.shape[1] != 187:
        raise ValueError(
            f"Expected 187 signal columns (or 188 columns with a label), "
            f"but found {frame.shape[1]}."
        )

    features = frame.apply(pd.to_numeric, errors="coerce")
    if features.isna().any().any():
        raise ValueError(
            "The CSV contains non-numeric or missing values in the ECG signal columns."
        )

    values = features.to_numpy(dtype=np.float32)
    if not np.isfinite(values).all():
        raise ValueError("The CSV contains infinite or invalid numeric values.")

    return values


def predict_chunk(values):
    """Apply the training-time scaler, when available, and predict classes."""
    model_values = scaler.transform(values) if scaler is not None else values
    probabilities = model.predict(model_values, batch_size=1024, verbose=0)
    predictions = np.argmax(probabilities, axis=1)
    return predictions, probabilities


def analyze_csv(file_stream):
    """Process large CSV files in chunks without loading the whole file."""
    class_counts = np.zeros(5, dtype=np.int64)
    previews = []
    total_samples = 0

    try:
        reader = pd.read_csv(
            file_stream,
            header=None,
            chunksize=CHUNK_SIZE,
            low_memory=False,
        )

        for chunk in reader:
            values = prepare_features(chunk)
            predictions, probabilities = predict_chunk(values)

            class_counts += np.bincount(predictions, minlength=5)
            total_samples += len(values)

            remaining = PREVIEW_LIMIT - len(previews)
            if remaining > 0:
                for index in range(min(remaining, len(values))):
                    predicted_class = int(predictions[index])
                    previews.append({
                        "index": len(previews) + 1,
                        "waveform": values[index].astype(float).tolist(),
                        "prediction": predicted_class,
                        "prediction_label": CLASS_NAMES[predicted_class],
                        "confidence": float(probabilities[index].max() * 100),
                        "probabilities": (
                            probabilities[index].astype(float) * 100
                        ).tolist(),
                    })

    except pd.errors.EmptyDataError as exc:
        raise ValueError("The uploaded CSV file is empty.") from exc
    except pd.errors.ParserError as exc:
        raise ValueError("The uploaded file is not a valid CSV.") from exc

    if total_samples == 0:
        raise ValueError("The uploaded CSV does not contain any ECG samples.")

    first_sample = previews[0]
    normal_percentage = float(class_counts[0] / total_samples * 100)
    arrhythmia_count = int(total_samples - class_counts[0])

    return {
        "filename": "",
        "sample_count": int(total_samples),
        "class_counts": class_counts.astype(int).tolist(),
        "normal_percentage": normal_percentage,
        "arrhythmia_count": arrhythmia_count,
        "samples": previews,
        "selected_sample": first_sample,
        "prediction": first_sample["prediction"],
        "prediction_label": first_sample["prediction_label"],
        "confidence": first_sample["confidence"],
        "probabilities": first_sample["probabilities"],
        "waveform": first_sample["waveform"],
    }


@app.route("/", methods=["GET", "POST"])
def index():
    result = None
    error = None

    if request.method == "POST":
        uploaded_file = request.files.get("ecg_file")

        if uploaded_file is None or not uploaded_file.filename:
            error = "Please choose a CSV file before starting the analysis."
        elif not uploaded_file.filename.lower().endswith(".csv"):
            error = "Only CSV files are supported."
        else:
            try:
                result = analyze_csv(uploaded_file.stream)
                result["filename"] = secure_filename(uploaded_file.filename)
            except ValueError as exc:
                error = str(exc)
            except Exception as exc:
                app.logger.exception("ECG analysis failed")
                error = f"Analysis could not be completed: {exc}"

    return render_template(
        "index.html",
        result=result,
        error=error,
        class_names=CLASS_NAMES,
        class_short=CLASS_SHORT,
    )


@app.errorhandler(413)
def file_too_large(_error):
    return render_template(
        "index.html",
        result=None,
        error="File is larger than the 500 MB upload limit.",
        class_names=CLASS_NAMES,
        class_short=CLASS_SHORT,
    ), 413


if __name__ == "__main__":
    app.run(host="127.0.0.1", port=5000, debug=False)
