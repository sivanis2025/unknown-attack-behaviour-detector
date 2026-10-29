
"""
Unknown Attack Behaviour Detector
Model training script using Isolation Forest.

Usage:
    python train_model.py
    python train_model.py --csv data/my_data.csv

Note:
    This model identifies statistical anomalies. It does not confirm
    that an observation is a real cyberattack.
"""

import argparse
import json
import os
import sys
import warnings

import joblib
import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    precision_recall_fscore_support,
)
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

# ---------------------------------------------------------------------
# Paths and configuration
# ---------------------------------------------------------------------

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
DATA_DIR = os.path.join(BASE_DIR, "data")
DEMO_CSV = os.path.join(DATA_DIR, "demo_dataset.csv")

MODEL_PATH = os.path.join(MODELS_DIR, "isolation_forest.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "scaler.pkl")
FEATURES_PATH = os.path.join(MODELS_DIR, "features.json")
METRICS_PATH = os.path.join(MODELS_DIR, "metrics.json")

CONTAMINATION = 0.10
N_ESTIMATORS = 100
RANDOM_STATE = 42
MIN_VARIANCE_THRESHOLD = 1e-6

LABEL_COLUMNS = {
    "label",
    "class",
    "attack_type",
    "category",
}

PREFERRED_FEATURES = [
    "Flow Duration",
    "Total Fwd Packets",
    "Total Backward Packets",
    "Total Length of Fwd Packets",
    "Total Length of Bwd Packets",
    "Flow Bytes/s",
    "Flow Packets/s",
    "Fwd Packet Length Max",
    "Fwd Packet Length Min",
    "Fwd Packet Length Mean",
    "Fwd Packet Length Std",
    "Bwd Packet Length Max",
    "Bwd Packet Length Min",
    "Bwd Packet Length Mean",
    "Bwd Packet Length Std",
    "Flow IAT Mean",
    "Flow IAT Std",
    "Flow IAT Max",
    "Flow IAT Min",
    "Fwd IAT Total",
    "Fwd IAT Mean",
    "Fwd IAT Std",
    "Fwd IAT Max",
    "Fwd IAT Min",
    "Bwd IAT Total",
    "Bwd IAT Mean",
    "Bwd IAT Std",
    "Bwd IAT Max",
    "Bwd IAT Min",
    "Fwd PSH Flags",
    "Bwd PSH Flags",
    "Fwd URG Flags",
    "Bwd URG Flags",
    "Fwd Header Length",
    "Bwd Header Length",
    "Fwd Packets/s",
    "Bwd Packets/s",
    "Min Packet Length",
    "Max Packet Length",
    "Packet Length Mean",
    "Packet Length Std",
    "Packet Length Variance",
    "FIN Flag Count",
    "SYN Flag Count",
    "RST Flag Count",
    "PSH Flag Count",
    "ACK Flag Count",
    "URG Flag Count",
    "CWE Flag Count",
    "ECE Flag Count",
    "Down/Up Ratio",
    "Average Packet Size",
    "Avg Fwd Segment Size",
    "Avg Bwd Segment Size",
    "Subflow Fwd Packets",
    "Subflow Fwd Bytes",
    "Subflow Bwd Packets",
    "Subflow Bwd Bytes",
    "Init_Win_bytes_forward",
    "Init_Win_bytes_backward",
    "act_data_pkt_fwd",
    "min_seg_size_forward",
    "Active Mean",
    "Active Std",
    "Active Max",
    "Active Min",
    "Idle Mean",
    "Idle Std",
    "Idle Max",
    "Idle Min",

    # Short feature names used by some demo CSV files.
    "flow_duration",
    "total_fwd_packets",
    "total_bwd_packets",
    "fwd_bytes",
    "bwd_bytes",
    "fwd_pkt_len_mean",
    "bwd_pkt_len_mean",
    "flow_bytes_per_sec",
    "flow_pkts_per_sec",
    "fwd_iat_mean",
    "bwd_iat_mean",
    "syn_flag",
    "ack_flag",
    "psh_flag",
    "rst_flag",
    "fin_flag",
]


# ---------------------------------------------------------------------
# Data loading and cleaning
# ---------------------------------------------------------------------

def normalise_name(name):
    """Convert a column name to lowercase letters and digits."""
    return "".join(
        character.lower()
        for character in str(name)
        if character.isalnum()
    )


def load_csv(path):
    """Read a CSV and clean whitespace around column names."""
    print(f"[INFO] Loading dataset: {path}")

    df = pd.read_csv(path, low_memory=False)
    df.columns = [str(column).strip() for column in df.columns]

    if df.empty:
        raise ValueError("The supplied CSV contains no records.")

    if len(df.columns) < 2:
        raise ValueError(
            "The CSV must contain at least two columns."
        )

    print(
        f"[INFO] Loaded {len(df):,} records "
        f"and {len(df.columns)} columns."
    )

    return df


def clean_dataframe(df):
    """Convert infinite numeric values to missing values."""
    df = df.copy()
    df.columns = [str(column).strip() for column in df.columns]

    return df.replace([np.inf, -np.inf], np.nan)


# ---------------------------------------------------------------------
# Label extraction
# ---------------------------------------------------------------------

def extract_labels(df):
    """
    Convert an available label column into binary labels:
        0 = Normal
        1 = Anomaly

    Returns None if there is no recognised label column.
    """
    label_lookup = {
        normalise_name(column): column
        for column in df.columns
    }

    for possible_name in LABEL_COLUMNS:
        actual_column = label_lookup.get(
            normalise_name(possible_name)
        )

        if actual_column is None:
            continue

        raw = df[actual_column].astype(str).str.strip().str.upper()

        normal_values = {
            "BENIGN",
            "NORMAL",
            "0",
            "FALSE",
            "NON-ATTACK",
            "NON ATTACK",
        }

        y_true = raw.apply(
            lambda value: 0 if value in normal_values else 1
        )

        normal_count = int((y_true == 0).sum())
        anomaly_count = int((y_true == 1).sum())

        print(
            f"[INFO] Labels in '{actual_column}': "
            f"{normal_count} normal, {anomaly_count} anomaly."
        )

        return y_true

    print("[INFO] No label column found.")
    return None


# ---------------------------------------------------------------------
# Feature selection
# ---------------------------------------------------------------------

def select_features(df):
    """
    Select useful numeric columns.

    Preferred network-flow features are selected first.
    Labels are never used as model input features.
    Constant and near-constant features are removed.
    """
    label_names = {
        normalise_name(name)
        for name in LABEL_COLUMNS
    }

    numeric_columns = df.select_dtypes(
        include=[np.number]
    ).columns.tolist()

    numeric_columns = [
        column
        for column in numeric_columns
        if normalise_name(column) not in label_names
    ]

    if not numeric_columns:
        raise ValueError(
            "No numeric feature columns were found. "
            "Check the CSV column types."
        )

    preferred_lookup = {
        normalise_name(name): name
        for name in PREFERRED_FEATURES
    }

    preferred = [
        column
        for column in numeric_columns
        if normalise_name(column) in preferred_lookup
    ]

    preferred_order = {
        normalise_name(name): index
        for index, name in enumerate(PREFERRED_FEATURES)
    }

    preferred.sort(
        key=lambda column: preferred_order.get(
            normalise_name(column), len(PREFERRED_FEATURES)
        )
    )

    remaining = [
        column
        for column in numeric_columns
        if column not in preferred
    ]

    candidates = preferred + remaining

    # Remove columns containing only missing values.
    candidates = [
        column
        for column in candidates
        if df[column].notna().any()
    ]

    if not candidates:
        raise ValueError(
            "No usable numeric features remain after cleaning."
        )

    # Convert values to numeric and fill missing values with medians.
    temporary = pd.DataFrame(index=df.index)

    for column in candidates:
        numeric_series = pd.to_numeric(
            df[column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

        median_value = numeric_series.median()

        if pd.isna(median_value):
            median_value = 0.0

        temporary[column] = numeric_series.fillna(median_value)

    variances = temporary.var()

    useful_features = [
        column
        for column in candidates
        if np.isfinite(variances[column])
        and variances[column] > MIN_VARIANCE_THRESHOLD
    ]

    if not useful_features:
        raise ValueError(
            "All numeric features are constant or nearly constant."
        )

    dropped = [
        column
        for column in candidates
        if column not in useful_features
    ]

    if dropped:
        print(
            "[INFO] Dropped constant/low-variance features:",
            dropped,
        )

    print(f"[INFO] Selected {len(useful_features)} features:")
    for feature in useful_features:
        print(f"       - {feature}")

    return useful_features


# ---------------------------------------------------------------------
# Preprocessing
# ---------------------------------------------------------------------

def preprocess(df, feature_columns, scaler=None):
    """
    Prepare numeric features for the model.

    Important: do not clip the training values here. The dashboard
    applies the same median-imputation and scaling approach during
    inference. This avoids training/inference preprocessing mismatch.
    """
    X = pd.DataFrame(index=df.index)

    for column in feature_columns:
        if column not in df.columns:
            raise ValueError(
                f"Required training feature is missing: {column}"
            )

        values = pd.to_numeric(
            df[column],
            errors="coerce",
        ).replace([np.inf, -np.inf], np.nan)

        median_value = values.median()

        if pd.isna(median_value):
            median_value = 0.0

        X[column] = values.fillna(median_value)

    if scaler is None:
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)
    else:
        X_scaled = scaler.transform(X)

    return X_scaled, scaler


# ---------------------------------------------------------------------
# Evaluation
# ---------------------------------------------------------------------

def evaluate(y_true, predictions):
    """
    Evaluate predictions where:
        Isolation Forest -1 = anomaly
        Isolation Forest  1 = normal
    """
    actual = np.asarray(y_true, dtype=int)
    predicted = np.where(np.asarray(predictions) == -1, 1, 0)

    precision, recall, f1, _ = (
        precision_recall_fscore_support(
            actual,
            predicted,
            labels=[0, 1],
            average="binary",
            pos_label=1,
            zero_division=0,
        )
    )

    matrix = confusion_matrix(
        actual,
        predicted,
        labels=[0, 1],
    ).tolist()

    report = classification_report(
        actual,
        predicted,
        labels=[0, 1],
        target_names=["Normal", "Anomaly"],
        zero_division=0,
    )

    return {
        "precision": round(float(precision), 4),
        "recall": round(float(recall), 4),
        "f1_score": round(float(f1), 4),
        "confusion_matrix": matrix,
        "classification_report": report,
    }


# ---------------------------------------------------------------------
# Training pipeline
# ---------------------------------------------------------------------

def train(csv_path):
    """Train the Isolation Forest and save model artefacts."""
    os.makedirs(MODELS_DIR, exist_ok=True)

    # 1. Load and clean the data.
    df = load_csv(csv_path)
    df = clean_dataframe(df)

    # 2. Extract optional labels before selecting model features.
    y_true = extract_labels(df)

    # 3. Select numeric network-flow features.
    feature_columns = select_features(df)

    # 4. Fit the scaler and transform the data.
    X_scaled, scaler = preprocess(df, feature_columns)

    # 5. Train the anomaly-detection model.
    print(
        "[INFO] Training Isolation Forest: "
        f"{N_ESTIMATORS} estimators, "
        f"contamination={CONTAMINATION}"
    )

    model = IsolationForest(
        n_estimators=N_ESTIMATORS,
        contamination=CONTAMINATION,
        random_state=RANDOM_STATE,
        n_jobs=-1,
    )

    model.fit(X_scaled)

    print("[INFO] Training completed.")

    # 6. Generate predictions on the training records.
    predictions = model.predict(X_scaled)
    anomaly_count = int(np.sum(predictions == -1))

    print(
        f"[INFO] Flagged {anomaly_count} of "
        f"{len(predictions)} training records as anomalies."
    )

    # 7. Store metadata expected by the dashboard.
    metrics = {
        "n_records": int(len(df)),
        "n_features": int(len(feature_columns)),
        "n_estimators": N_ESTIMATORS,
        "contamination": CONTAMINATION,
        "features": feature_columns,
        "has_evaluation": False,
        "model_type": "Isolation Forest",
        "evaluation_note": (
            "Training-data evaluation is optimistic. "
            "Use a separate held-out test set for reliable metrics."
        ),
    }

    if y_true is not None:
        print(
            "\n[INFO] Evaluating predictions on training data only."
        )
        print(
            "[WARNING] These metrics do not measure performance "
            "on unseen network traffic."
        )

        evaluation = evaluate(y_true, predictions)
        metrics.update(evaluation)
        metrics["has_evaluation"] = True

        print("\nClassification Report:")
        print(evaluation["classification_report"])

        tn, fp, fn, tp = evaluation["confusion_matrix"][0][0], (
            evaluation["confusion_matrix"][0][1]
        ), evaluation["confusion_matrix"][1][0], (
            evaluation["confusion_matrix"][1][1]
        )

        print("Confusion Matrix:")
        print(f"  True Negatives : {tn}")
        print(f"  False Positives: {fp}")
        print(f"  False Negatives: {fn}")
        print(f"  True Positives : {tp}")

    # 8. Save the trained model, scaler, ordered features and metrics.
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    with open(FEATURES_PATH, "w", encoding="utf-8") as file:
        json.dump(feature_columns, file, indent=2)

    with open(METRICS_PATH, "w", encoding="utf-8") as file:
        json.dump(metrics, file, indent=2)

    print("\n[INFO] Saved model artefacts:")
    print(f"       Model   : {MODEL_PATH}")
    print(f"       Scaler  : {SCALER_PATH}")
    print(f"       Features: {FEATURES_PATH}")
    print(f"       Metrics : {METRICS_PATH}")

    return metrics


# ---------------------------------------------------------------------
# Command-line entry point
# ---------------------------------------------------------------------

def main():
    parser = argparse.ArgumentParser(
        description=(
            "Train the Unknown Attack Behaviour Detector "
            "using Isolation Forest."
        )
    )

    parser.add_argument(
        "--csv",
        default=DEMO_CSV,
        help="Path to the CSV dataset. Defaults to the demo dataset.",
    )

    args = parser.parse_args()

    if not os.path.isfile(args.csv):
        print(f"[ERROR] CSV file not found: {args.csv}")
        sys.exit(1)

    print("=" * 60)
    print(" Unknown Attack Behaviour Detector — Training")
    print("=" * 60)

    try:
        metrics = train(args.csv)

        print("\n[DONE] Training completed successfully.")

        if metrics.get("has_evaluation"):
            print(f"Precision: {metrics['precision']:.4f}")
            print(f"Recall   : {metrics['recall']:.4f}")
            print(f"F1-score : {metrics['f1_score']:.4f}")

        print(
            "\nReminder: training-data metrics are not a reliable "
            "estimate of performance on unseen traffic."
        )

    except (ValueError, OSError, pd.errors.ParserError) as exc:
        print(f"\n[ERROR] Training failed: {exc}")
        sys.exit(1)
    except Exception as exc:
        print(f"\n[ERROR] Unexpected training error: {exc}")
        raise


if __name__ == "__main__":
    main()
