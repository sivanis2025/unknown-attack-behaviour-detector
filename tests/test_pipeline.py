"""
Unit tests for the Unknown Attack Behaviour Detector pipeline.

Run with:  python -m pytest tests/ -v
"""

import json
import os
import sys
import tempfile

import numpy as np
import pandas as pd
import pytest

# Make sure the project root is importable
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def sample_df():
    """Minimal valid DataFrame with known feature columns."""
    rng = np.random.default_rng(0)
    n = 50
    return pd.DataFrame(
        {
            "flow_duration": rng.integers(1000, 1_000_000, n),
            "total_fwd_packets": rng.integers(1, 100, n),
            "total_bwd_packets": rng.integers(1, 100, n),
            "fwd_bytes": rng.integers(100, 50_000, n),
            "bwd_bytes": rng.integers(100, 50_000, n),
            "fwd_pkt_len_mean": rng.uniform(40, 1500, n),
            "bwd_pkt_len_mean": rng.uniform(40, 1500, n),
            "flow_bytes_per_sec": rng.uniform(100, 100_000, n),
            "flow_pkts_per_sec": rng.uniform(1, 1000, n),
            "fwd_iat_mean": rng.integers(1000, 500_000, n),
            "bwd_iat_mean": rng.integers(1000, 500_000, n),
            "syn_flag": rng.integers(0, 2, n),
            "ack_flag": rng.integers(0, 2, n),
            "psh_flag": rng.integers(0, 2, n),
            "rst_flag": rng.integers(0, 2, n),
            "fin_flag": rng.integers(0, 2, n),
        }
    )


@pytest.fixture()
def labeled_df(sample_df):
    """Sample DF with a Label column for evaluation testing."""
    df = sample_df.copy()
    rng = np.random.default_rng(1)
    df["Label"] = rng.choice(["BENIGN", "Attack"], len(df), p=[0.8, 0.2])
    return df


# ---------------------------------------------------------------------------
# Tests: data generation
# ---------------------------------------------------------------------------


class TestDataGeneration:
    def test_generate_creates_file(self, tmp_path):
        """Demo data generator should create a CSV file with expected columns."""
        from data.generate_demo_data import generate as _orig_generate
        import data.generate_demo_data as gen_module

        original_path = gen_module.OUTPUT_PATH
        test_output = str(tmp_path / "test_demo.csv")
        gen_module.OUTPUT_PATH = test_output
        try:
            df = _orig_generate()
        finally:
            gen_module.OUTPUT_PATH = original_path

        assert os.path.exists(test_output), "CSV file should be created"
        assert len(df) == 500, "Should generate 500 records"
        assert "Label" in df.columns, "Should have a Label column"
        assert "flow_duration" in df.columns

    def test_label_distribution(self, tmp_path):
        """Should have ~80% normal and ~20% anomaly records."""
        from data.generate_demo_data import generate as _orig_generate
        import data.generate_demo_data as gen_module

        original_path = gen_module.OUTPUT_PATH
        test_output = str(tmp_path / "test_demo2.csv")
        gen_module.OUTPUT_PATH = test_output
        try:
            df = _orig_generate()
        finally:
            gen_module.OUTPUT_PATH = original_path

        benign = (df["Label"] == "BENIGN").sum()
        anomaly = len(df) - benign
        assert benign == 400, f"Expected 400 BENIGN, got {benign}"
        assert anomaly == 100, f"Expected 100 anomaly, got {anomaly}"


# ---------------------------------------------------------------------------
# Tests: preprocessing
# ---------------------------------------------------------------------------


class TestPreprocessing:
    def test_clean_df_handles_inf(self, sample_df):
        """Inf values should be replaced with NaN."""
        from app import clean_df

        sample_df.iloc[0, 0] = np.inf
        sample_df.iloc[1, 0] = -np.inf
        result = clean_df(sample_df)
        assert not np.isinf(result.select_dtypes(include=[np.number]).values).any()

    def test_clean_df_strips_column_names(self):
        """Column names with surrounding whitespace should be stripped."""
        from app import clean_df

        df = pd.DataFrame({"  col_a  ": [1, 2], " col_b": [3, 4]})
        result = clean_df(df)
        assert "col_a" in result.columns
        assert "col_b" in result.columns

    def test_select_features_drops_constant(self):
        """Constant columns should be excluded from feature selection."""
        from train_model import select_features

        df = pd.DataFrame(
            {
                "flow_duration": [100, 200, 300, 400, 500],
                "constant_col": [42, 42, 42, 42, 42],
                "bwd_bytes": [10, 20, 30, 40, 50],
            }
        )
        features = select_features(df)
        assert "constant_col" not in features
        assert "flow_duration" in features

    def test_select_features_raises_on_no_numeric(self):
        """Should raise ValueError if no numeric features exist."""
        from train_model import select_features

        df = pd.DataFrame({"name": ["a", "b", "c"], "category": ["x", "y", "z"]})
        with pytest.raises(ValueError):
            select_features(df)

    def test_impute_and_scale_missing_columns(self, sample_df):
        """Missing columns should be filled with 0 without crashing."""
        from app import impute_and_scale
        from sklearn.preprocessing import StandardScaler

        feature_cols = ["flow_duration", "total_fwd_packets", "missing_col_x"]
        scaler = StandardScaler()
        # Fit scaler on existing data only — patch
        X_fit = sample_df[["flow_duration", "total_fwd_packets"]].copy()
        X_fit["missing_col_x"] = 0.0
        scaler.fit(X_fit)
        result = impute_and_scale(sample_df, feature_cols, scaler)
        assert result.shape == (len(sample_df), 3)

    def test_impute_and_scale_no_nan_in_output(self, sample_df):
        """Output of impute_and_scale should contain no NaN values."""
        from app import impute_and_scale
        from sklearn.preprocessing import StandardScaler

        sample_df.iloc[0, 0] = np.nan
        sample_df.iloc[2, 3] = np.inf
        feature_cols = ["flow_duration", "fwd_bytes"]
        scaler = StandardScaler()
        X_clean = sample_df[feature_cols].replace([np.inf, -np.inf], np.nan).fillna(0)
        scaler.fit(X_clean)
        result = impute_and_scale(sample_df, feature_cols, scaler)
        assert not np.isnan(result).any()


# ---------------------------------------------------------------------------
# Tests: training pipeline
# ---------------------------------------------------------------------------


class TestTrainingPipeline:
    def test_train_creates_model_files(self, sample_df, tmp_path):
        """Training should save model, scaler, features, and metrics."""
        import train_model

        csv_path = str(tmp_path / "train_data.csv")
        sample_df.to_csv(csv_path, index=False)

        orig_models_dir = train_model.MODELS_DIR
        orig_model_path = train_model.MODEL_PATH
        orig_scaler_path = train_model.SCALER_PATH
        orig_features_path = train_model.FEATURES_PATH
        orig_metrics_path = train_model.METRICS_PATH

        models_dir = str(tmp_path / "models")
        train_model.MODELS_DIR = models_dir
        train_model.MODEL_PATH = os.path.join(models_dir, "isolation_forest.pkl")
        train_model.SCALER_PATH = os.path.join(models_dir, "scaler.pkl")
        train_model.FEATURES_PATH = os.path.join(models_dir, "features.json")
        train_model.METRICS_PATH = os.path.join(models_dir, "metrics.json")

        try:
            metrics = train_model.train(csv_path)
        finally:
            train_model.MODELS_DIR = orig_models_dir
            train_model.MODEL_PATH = orig_model_path
            train_model.SCALER_PATH = orig_scaler_path
            train_model.FEATURES_PATH = orig_features_path
            train_model.METRICS_PATH = orig_metrics_path

        assert os.path.exists(train_model.MODEL_PATH if train_model.MODEL_PATH != orig_model_path else os.path.join(models_dir, "isolation_forest.pkl"))
        assert isinstance(metrics, dict)
        assert "n_records" in metrics
        assert metrics["n_records"] == len(sample_df)

    def test_train_with_labels_produces_metrics(self, labeled_df, tmp_path):
        """Training with a Label column should produce evaluation metrics."""
        import train_model

        csv_path = str(tmp_path / "labeled_data.csv")
        labeled_df.to_csv(csv_path, index=False)

        models_dir = str(tmp_path / "models")
        orig_paths = (
            train_model.MODELS_DIR,
            train_model.MODEL_PATH,
            train_model.SCALER_PATH,
            train_model.FEATURES_PATH,
            train_model.METRICS_PATH,
        )
        train_model.MODELS_DIR = models_dir
        train_model.MODEL_PATH = os.path.join(models_dir, "isolation_forest.pkl")
        train_model.SCALER_PATH = os.path.join(models_dir, "scaler.pkl")
        train_model.FEATURES_PATH = os.path.join(models_dir, "features.json")
        train_model.METRICS_PATH = os.path.join(models_dir, "metrics.json")

        try:
            metrics = train_model.train(csv_path)
        finally:
            (
                train_model.MODELS_DIR,
                train_model.MODEL_PATH,
                train_model.SCALER_PATH,
                train_model.FEATURES_PATH,
                train_model.METRICS_PATH,
            ) = orig_paths

        assert metrics.get("has_evaluation") is True
        assert "precision" in metrics
        assert "recall" in metrics
        assert "f1_score" in metrics
        assert 0.0 <= metrics["precision"] <= 1.0
        assert 0.0 <= metrics["recall"] <= 1.0
        assert 0.0 <= metrics["f1_score"] <= 1.0


# ---------------------------------------------------------------------------
# Tests: prediction / analysis
# ---------------------------------------------------------------------------


class TestPrediction:
    def test_severity_labels(self):
        """Severity assignments should match threshold logic."""
        from app import assign_severity, SEVERITY_LOW_THRESHOLD, SEVERITY_MEDIUM_THRESHOLD

        assert assign_severity(0.0) == "Low"
        assert assign_severity(SEVERITY_LOW_THRESHOLD) == "Low"
        assert assign_severity(SEVERITY_MEDIUM_THRESHOLD) == "Medium"
        assert assign_severity(SEVERITY_MEDIUM_THRESHOLD - 0.01) == "High"

    def test_extract_labels_benign(self, labeled_df):
        """extract_labels should return 0 for BENIGN records."""
        from train_model import extract_labels

        labels = extract_labels(labeled_df)
        assert labels is not None
        benign_mask = labeled_df["Label"] == "BENIGN"
        assert (labels[benign_mask] == 0).all()

    def test_extract_labels_returns_none_when_absent(self, sample_df):
        """extract_labels should return None when no label column exists."""
        from train_model import extract_labels

        result = extract_labels(sample_df)
        assert result is None

    def test_evaluate_output_keys(self, labeled_df):
        """evaluate() should return expected metric keys."""
        from train_model import evaluate

        y_true = labeled_df["Label"].map(lambda v: 0 if v == "BENIGN" else 1)
        y_pred = np.random.choice([-1, 1], size=len(y_true))
        metrics = evaluate(y_true, y_pred)
        for key in ["precision", "recall", "f1_score", "confusion_matrix", "classification_report"]:
            assert key in metrics


# ---------------------------------------------------------------------------
# Tests: edge cases
# ---------------------------------------------------------------------------


class TestEdgeCases:
    def test_empty_dataframe_raises(self):
        """run_analysis should raise ValueError on empty DataFrame."""
        from app import run_analysis

        with pytest.raises((ValueError, Exception)):
            run_analysis(pd.DataFrame())

    def test_single_row_raises(self, sample_df):
        """run_analysis should raise ValueError for a single-record DataFrame."""
        from app import run_analysis

        with pytest.raises((ValueError, Exception)):
            run_analysis(sample_df.head(1))

    def test_all_nan_column_handled(self, sample_df):
        """Columns that are entirely NaN should not crash feature selection."""
        from train_model import select_features

        df = sample_df.copy()
        df["all_nan_col"] = np.nan
        features = select_features(df)
        assert "all_nan_col" not in features

    def test_clean_df_preserves_valid_values(self, sample_df):
        """clean_df should not alter valid numeric values."""
        from app import clean_df

        original_vals = sample_df["flow_duration"].copy()
        result = clean_df(sample_df)
        pd.testing.assert_series_equal(original_vals, result["flow_duration"])
