
"""
Unknown Attack Behaviour Detector — Flask Application

Educational anomaly-detection dashboard using Isolation Forest.
Flagged records are statistical anomalies, not confirmed attacks.
"""

from __future__ import annotations

import io
import json
import os
import uuid

import joblib
import numpy as np
import pandas as pd
from flask import Flask, jsonify, render_template, request, send_file
from werkzeug.utils import secure_filename

# ---------------------------------------------------------------------
# Configuration
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
UPLOAD_FOLDER = os.path.join(BASE_DIR, "uploads")

DEMO_CSV = os.path.join(DATA_DIR, "demo_dataset.csv")
MODEL_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

MAX_UPLOAD_MB = 10
ALLOWED_EXTENSIONS = {"csv"}

CONTAMINATION = 0.1

# Heuristic score bands, not confirmed attack severity.
SEVERITY_LOW_THRESHOLD = -0.1
SEVERITY_MEDIUM_THRESHOLD = -0.25

app = Flask(__name__)
app.config["UPLOAD_FOLDER"] = UPLOAD_FOLDER
app.config["MAX_CONTENT_LENGTH"] = MAX_UPLOAD_MB * 1024 * 1024

# Require a secret key supplied through the environment.
# Never silently use a predictable default key.
app.secret_key = os.environ.get("FLASK_SECRET")

if not app.secret_key:
    raise RuntimeError(
        "FLASK_SECRET is missing. Set the environment variable "
        "before starting the application."
    )

os.makedirs(UPLOAD_FOLDER, exist_ok=True)
os.makedirs(MODELS_DIR, exist_ok=True)

_last_prediction_df: pd.DataFrame | None = None


# ---------------------------------------------------------------------
# Model and data helpers
# ---------------------------------------------------------------------

def allowed_file(filename: str) -> bool:
    return (
        "." in filename
        and filename.rsplit(".", 1)[1].lower() in ALLOWED_EXTENSIONS
    )


def load_model():
    """Load the saved model, scaler, and ordered feature names."""
    required_files = [
        MODEL_PATH,
        SCALER_PATH,
        FEATURES_PATH,
    ]

    missing = [path for path in required_files if not os.path.exists(path)]

    if missing:
        raise FileNotFoundError(
            "Model files are missing. Check the models folder and run "
            "'python train_model.py' if training is required."
        )

    # Only load model files from a trusted source.
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)

    with open(FEATURES_PATH, "r", encoding="utf-8") as file:
        features = json.load(file)

    if not isinstance(features, list) or not features:
        raise ValueError("models/features.json must contain a feature list.")

    return model, scaler, features


def load_metrics() -> dict:
    """Load saved model metrics when available."""
    if not os.path.exists(METRICS_PATH):
        return {}

    with open(METRICS_PATH, "r", encoding="utf-8") as file:
        return json.load(file)


def clean_df(df: pd.DataFrame) -> pd.DataFrame:
    """Clean column names and replace infinite values."""
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]
    return df.replace([np.inf, -np.inf], np.nan)


def normalise_name(name: str) -> str:
    """Normalise a feature name for matching CSV column aliases."""
    return "".join(
        character.lower()
        for character in str(name)
        if character.isalnum()
    )


# Common aliases used by the sample CSV and CIC-IDS-style datasets.
FEATURE_ALIASES = {
    "flowduration": [
        "flow_duration",
        "Flow Duration",
    ],
    "totalfwdpackets": [
        "total_fwd_packets",
        "Total Fwd Packets",
    ],
    "totalbackwardpackets": [
        "total_bwd_packets",
        "total_backward_packets",
        "Total Backward Packets",
    ],
    "totallengthoffwdpackets": [
        "fwd_bytes",
        "total_length_of_fwd_packets",
        "Total Length of Fwd Packets",
    ],
    "totallengthofbwdpackets": [
        "bwd_bytes",
        "total_length_of_bwd_packets",
        "Total Length of Bwd Packets",
    ],
    "flowbytess": [
        "flow_bytes_per_sec",
        "Flow Bytes/s",
    ],
    "flowpacketss": [
        "flow_pkts_per_sec",
        "Flow Packets/s",
    ],
}


def resolve_source_column(df: pd.DataFrame, feature: str):
    """Find a matching CSV column for a trained model feature."""
    if feature in df.columns:
        return feature

    target = normalise_name(feature)

    # First try a punctuation/case-insensitive match.
    for column in df.columns:
        if normalise_name(column) == target:
            return column

    # Then try known aliases.
    aliases = FEATURE_ALIASES.get(target, [])

    for alias in aliases:
        if alias in df.columns:
            return alias

        alias_normalised = normalise_name(alias)

        for column in df.columns:
            if normalise_name(column) == alias_normalised:
                return column

    return None


def impute_and_scale(
    df: pd.DataFrame,
    feature_cols: list,
    scaler,
) -> np.ndarray:
    """
    Build features in the exact order expected by the trained scaler.

    Recognised aliases are mapped to model feature names.
    Missing or non-numeric values are imputed with zero when no
    valid value exists in that feature column.
    """
    X = pd.DataFrame(index=df.index)

    for feature in feature_cols:
        source_column = resolve_source_column(df, feature)

        if source_column is None:
            X[feature] = 0.0
            continue

        series = pd.to_numeric(df[source_column], errors="coerce")
        median_value = series.median()

        if pd.isna(median_value):
            median_value = 0.0

        X[feature] = series.fillna(median_value)

    X = X.replace([np.inf, -np.inf], np.nan).fillna(0.0)

    try:
        scaled = scaler.transform(X[feature_cols])
    except Exception as exc:
        raise ValueError(
            "Could not scale the input features. The saved scaler may "
            "not match models/features.json. Check train_model.py."
        ) from exc

    return scaled


def assign_severity(score: float) -> str:
    """Assign a heuristic score band, not an attack classification."""
    if score >= SEVERITY_LOW_THRESHOLD:
        return "Low"
    if score >= SEVERITY_MEDIUM_THRESHOLD:
        return "Medium"
    return "High"


def explain_record(
    feature_cols: list,
    scaled_row: np.ndarray,
) -> str:
    """
    Provide a simple explanation based on the largest absolute
    scaled feature values. This is not a causal explanation.
    """
    values = np.asarray(scaled_row).flatten()
    deviations = []

    for index, feature in enumerate(feature_cols):
        value = float(values[index])

        if abs(value) > 1.5:
            direction = "high" if value > 0 else "low"
            deviations.append((abs(value), feature, direction))

    deviations.sort(reverse=True)

    if not deviations:
        return (
            "The model flagged this record as anomalous. "
            "No single feature passed the explanation heuristic."
        )

    details = [
        f"{feature} is unusually {direction}"
        for _, feature, direction in deviations[:3]
    ]

    return (
        "Flagged because: "
        + "; ".join(details)
        + ". Statistical indication only, not proof of an attack."
    )


# ---------------------------------------------------------------------
# Core analysis
# ---------------------------------------------------------------------

def run_analysis(
    df: pd.DataFrame,
    source_label: str = "Uploaded CSV",
) -> dict:
    """Analyse network-flow rows and return dashboard data."""
    global _last_prediction_df

    if df.empty:
        raise ValueError("The dataset is empty.")

    if len(df) < 2:
        raise ValueError("The dataset must contain at least 2 records.")

    df = clean_df(df)

    model, scaler, feature_cols = load_model()

    if not hasattr(model, "predict") or not hasattr(model, "score_samples"):
        raise ValueError(
            "The saved model does not support Isolation Forest analysis."
        )

    X_scaled = impute_and_scale(df, feature_cols, scaler)

    predictions = model.predict(X_scaled)
    scores = model.score_samples(X_scaled)

    result_df = df.copy()
    result_df["_prediction"] = predictions
    result_df["_anomaly_score"] = scores
    result_df["_is_anomaly"] = predictions == -1

    result_df["_severity"] = [
        assign_severity(float(score)) if prediction == -1 else "Normal"
        for prediction, score in zip(predictions, scores)
    ]

    result_df["_explanation"] = [
        explain_record(feature_cols, X_scaled[index:index + 1])
        if predictions[index] == -1
        else "Traffic pattern was not flagged by this model."
        for index in range(len(result_df))
    ]

    _last_prediction_df = result_df.copy()

    total = len(result_df)
    n_anomaly = int(np.sum(predictions == -1))
    n_normal = total - n_anomaly
    anomaly_pct = round(100.0 * n_anomaly / total, 2)

    # Use the model's actual score cutoff when available.
    model_threshold = getattr(model, "offset_", None)

    if model_threshold is None:
        model_threshold = float(
            np.percentile(scores, CONTAMINATION * 100)
        )
    else:
        model_threshold = float(model_threshold)

    hist_counts, hist_edges = np.histogram(scores, bins=30)

    score_chart = {
        "labels": [round(float(value), 4) for value in hist_edges[:-1]],
        "values": [int(value) for value in hist_counts],
        "anomaly_threshold": model_threshold,
    }

    anomaly_df = result_df[result_df["_is_anomaly"]].copy()
    anomaly_df = anomaly_df.sort_values("_anomaly_score").head(50)

    display_cols = [
        feature for feature in feature_cols
        if resolve_source_column(result_df, feature) is not None
    ][:6]

    suspicious_records = []

    for index, row in anomaly_df.iterrows():
        record = {
            "id": (
                int(index)
                if isinstance(index, (int, np.integer))
                else str(index)
            ),
            "anomaly_score": round(float(row["_anomaly_score"]), 4),
            "severity": str(row["_severity"]),
            "explanation": str(row["_explanation"]),
        }

        for feature in display_cols:
            source_column = resolve_source_column(result_df, feature)
            value = row[source_column] if source_column else None

            if pd.isna(value):
                value = None
            elif isinstance(value, np.integer):
                value = int(value)
            elif isinstance(value, (np.floating, float)):
                value = round(float(value), 4)

            record[feature] = value

        suspicious_records.append(record)

    sample_scores = [
        round(float(value), 4) for value in scores[:200]
    ]

    return {
        "source": source_label,
        "total": int(total),
        "n_normal": n_normal,
        "n_anomaly": n_anomaly,
        "anomaly_pct": anomaly_pct,
        "score_chart": score_chart,
        "sample_scores": sample_scores,
        "suspicious_records": suspicious_records,
        "display_cols": display_cols,
        "disclaimer": (
            "Anomaly detection identifies statistical outliers in network "
            "flow data. Flagged records are NOT confirmed attacks. "
            "Results require expert review."
        ),
    }


# ---------------------------------------------------------------------
# Dashboard route
# ---------------------------------------------------------------------

@app.route("/")
def index():
    return render_template("index.html")


# ---------------------------------------------------------------------
# Demo analysis
# ---------------------------------------------------------------------

@app.route("/api/demo", methods=["GET"])
def api_demo():
    try:
        if not os.path.exists(DEMO_CSV):
            return jsonify({"error": "Demo dataset not found."}), 404

        df = pd.read_csv(DEMO_CSV, low_memory=False)

        result = run_analysis(
            df,
            source_label="Synthetic Demo Data (NOT real traffic)",
        )
        return jsonify(result)

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except Exception:
        app.logger.exception("Demo analysis failed")
        return jsonify({
            "error": "Demo analysis failed. Check the Flask terminal for details."
        }), 500


# ---------------------------------------------------------------------
# CSV upload and analysis
# ---------------------------------------------------------------------

@app.route("/api/upload", methods=["POST"])
def api_upload():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    safe_name = secure_filename(file.filename)

    if not safe_name or not allowed_file(safe_name):
        return jsonify({"error": "Only .csv files are accepted."}), 400

    save_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        f"{uuid.uuid4().hex}_{safe_name}",
    )

    try:
        file.save(save_path)
        df = pd.read_csv(save_path, low_memory=False)

        if df.empty:
            return jsonify({"error": "Uploaded CSV is empty."}), 400

        if len(df.columns) < 2:
            return jsonify({
                "error": "CSV must have at least 2 columns."
            }), 400

        result = run_analysis(df, source_label=safe_name)
        return jsonify(result)

    except FileNotFoundError as exc:
        return jsonify({"error": str(exc)}), 503
    except ValueError as exc:
        return jsonify({"error": str(exc)}), 400
    except pd.errors.ParserError:
        return jsonify({"error": "Could not parse the CSV file."}), 400
    except Exception:
        app.logger.exception("CSV analysis failed")
        return jsonify({
            "error": "CSV analysis failed. Check the Flask terminal for details."
        }), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


# ---------------------------------------------------------------------
# Model retraining
# ---------------------------------------------------------------------

@app.route("/api/train", methods=["POST"])
def api_train():
    if "file" not in request.files:
        return jsonify({"error": "No file part in request."}), 400

    file = request.files["file"]

    if not file.filename:
        return jsonify({"error": "No file selected."}), 400

    safe_name = secure_filename(file.filename)

    if not safe_name or not allowed_file(safe_name):
        return jsonify({"error": "Only .csv files are accepted."}), 400

    save_path = os.path.join(
        app.config["UPLOAD_FOLDER"],
        f"{uuid.uuid4().hex}_{safe_name}",
    )

    try:
        file.save(save_path)

        # WARNING: train_model.train() may overwrite the active model files.
        # Keep this application local unless retraining is protected by
        # appropriate authentication and authorization.
        from train_model import train

        metrics = train(save_path)

        return jsonify({
            "message": "Model retraining completed.",
            "metrics": metrics,
        })

    except Exception:
        app.logger.exception("Model retraining failed")
        return jsonify({
            "error": "Model retraining failed. Check the Flask terminal for details."
        }), 500
    finally:
        if os.path.exists(save_path):
            os.remove(save_path)


# ---------------------------------------------------------------------
# Model information
# ---------------------------------------------------------------------

@app.route("/api/model-info", methods=["GET"])
def api_model_info():
    try:
        metrics = load_metrics()

        if not metrics:
            return jsonify({"error": "Model metrics are not available."}), 404

        safe_metrics = {
            key: value
            for key, value in metrics.items()
            if key not in {"features", "model_path", "scaler_path"}
        }

        safe_metrics["n_features"] = metrics.get("n_features", 0)
        safe_metrics["feature_names"] = metrics.get("features", [])
        safe_metrics["model_type"] = "Isolation Forest"
        safe_metrics["disclaimer"] = (
            "Metrics shown are from training-data evaluation (optimistic). "
            "For reliable metrics, evaluate on a separate held-out test set."
        )

        return jsonify(safe_metrics)

    except Exception:
        app.logger.exception("Model information request failed")
        return jsonify({
            "error": "Could not load model information."
        }), 500


# ---------------------------------------------------------------------
# Download last prediction results
# ---------------------------------------------------------------------

@app.route("/api/download", methods=["GET"])
def api_download():
    global _last_prediction_df

    if _last_prediction_df is None:
        return jsonify({
            "error": "Run an analysis before downloading results."
        }), 404

    output_df = _last_prediction_df.copy()

    rename_columns = {
        "_prediction": "prediction_raw",
        "_anomaly_score": "anomaly_score",
        "_is_anomaly": "is_anomaly",
        "_severity": "severity",
        "_explanation": "explanation",
    }

    output_df = output_df.rename(columns=rename_columns)

    internal_columns = [
        column for column in output_df.columns
        if column.startswith("_")
    ]
    output_df = output_df.drop(
        columns=internal_columns,
        errors="ignore",
    )

    output = io.StringIO()
    output_df.to_csv(output, index=False)

    return send_file(
        io.BytesIO(output.getvalue().encode("utf-8")),
        mimetype="text/csv",
        as_attachment=True,
        download_name="attack_detector_predictions.csv",
    )


# ---------------------------------------------------------------------
# Sample CSV template
# ---------------------------------------------------------------------

@app.route("/api/sample-csv", methods=["GET"])
def api_sample_csv():
    """
    Generate a sample CSV using the actual saved model feature names.
    This avoids a mismatch between the template and the trained model.
    """
    try:
        _, _, feature_cols = load_model()

        example_values = {
            "flowduration": [1000000, 50000, 9999999],
            "totalfwdpackets": [10, 5, 1],
            "totalbackwardpackets": [8, 3, 0],
            "totallengthoffwdpackets": [5000, 200, 50000],
            "totallengthofbwdpackets": [4000, 150, 0],
            "flowbytess": [5000.0, 4000.0, 500000.0],
            "flowpacketss": [10.0, 100.0, 0.1],
        }

        sample = {}

        for feature in feature_cols:
            key = normalise_name(feature)
            values = example_values.get(key)

            if values is None:
                values = [0.0, 0.0, 0.0]

            sample[feature] = values

        sample["Label"] = ["BENIGN", "BENIGN", "Attack"]

        output = io.StringIO()
        pd.DataFrame(sample).to_csv(output, index=False)

        return send_file(
            io.BytesIO(output.getvalue().encode("utf-8")),
            mimetype="text/csv",
            as_attachment=True,
            download_name="sample_network_flow_template.csv",
        )

    except Exception:
        app.logger.exception("Sample CSV generation failed")
        return jsonify({
            "error": "Could not generate the sample CSV. Check model files."
        }), 500


# ---------------------------------------------------------------------
# Upload-size error handler
# ---------------------------------------------------------------------

@app.errorhandler(413)
def file_too_large(error):
    return jsonify({
        "error": f"File too large. Maximum size is {MAX_UPLOAD_MB} MB."
    }), 413


# ---------------------------------------------------------------------
# Application entry point
# ---------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("  Unknown Attack Behaviour Detector — Flask Server")
    print("=" * 60)
    print("  Dashboard : http://127.0.0.1:5000")
    print("=" * 60)

    # Debug mode stays disabled.
    app.run(debug=False, host="127.0.0.1", port=5000)
