"""
ml_plugins/anomaly_detection/plugin.py
=========================================

Unsupervised anomaly detection via Isolation Forest. Like clustering,
no target column and no train/test split — evaluate() scores the fit
against the same data used to fit, via the fraction of points flagged
as anomalous, which is the standard sanity check for this algorithm
family (a contamination rate wildly different from the configured
`contamination` param indicates something's off with the feature set).

PARAMS:
    feature_columns: list[str], optional — defaults to all numeric columns.
    contamination: float, default 0.05 — expected fraction of anomalies
        (Isolation Forest needs this as a prior; 0.05 means "assume
        about 5% of rows are anomalous" unless told otherwise).
    random_state: int, default 42.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

from exceptions.domain_exceptions import PluginValidationError
from logging_setup.logger import get_logger
from ml_plugins.base_plugin import BaseMLPlugin, PluginResult, PluginValidationResult

logger = get_logger(__name__)

_MIN_ROWS_REQUIRED = 20   # Isolation Forest needs a reasonable sample to build trees


class AnomalyDetectionPlugin(BaseMLPlugin):
    """Isolation Forest anomaly detection on numeric feature columns."""

    @property
    def capability_name(self) -> str:
        return "anomaly_detection"

    @property
    def description(self) -> str:
        return "Flags unusual rows (outliers) in numeric data using Isolation Forest."

    def validate_requirements(self, available_columns: dict[str, str]) -> PluginValidationResult:
        numeric_cols = [
            col for col, dtype in available_columns.items()
            if any(t in dtype.upper() for t in ("INT", "REAL", "FLOAT", "NUMERIC", "DOUBLE"))
        ]
        if not numeric_cols:
            return PluginValidationResult(
                is_valid=False,
                missing_requirements=["at least one numeric column"],
                suggested_columns={},
            )
        return PluginValidationResult(is_valid=True, missing_requirements=[], suggested_columns={})

    def preprocess_data(
        self,
        df: pd.DataFrame,
        feature_columns: list[str] | None = None,
        **_: Any,
    ) -> dict[str, Any]:
        """
        Selects numeric columns and standardizes them — same
        distance-sensitivity rationale as ClusteringPlugin: without
        scaling, a large-magnitude column would dominate which points
        get flagged as isolated/anomalous.
        """
        candidate_features = feature_columns or list(df.select_dtypes(include=[np.number]).columns)
        if not candidate_features:
            raise PluginValidationError(
                message="No numeric columns available for anomaly detection.",
                user_message="Need at least one numeric column to check for anomalies.",
            )
        if len(df) < _MIN_ROWS_REQUIRED:
            raise PluginValidationError(
                message=f"Only {len(df)} rows; need at least {_MIN_ROWS_REQUIRED}.",
                user_message=f"Not enough data for reliable anomaly detection (need at least {_MIN_ROWS_REQUIRED} rows).",
            )

        X = df[candidate_features].fillna(df[candidate_features].mean(numeric_only=True))
        scaler = StandardScaler()
        X_scaled = scaler.fit_transform(X)

        return {
            "X_scaled": X_scaled,
            "X_original": X.reset_index(drop=True),
            "feature_columns": candidate_features,
        }

    def train(
        self,
        prepared_data: dict[str, Any],
        contamination: float = 0.05,
        random_state: int = 42,
        **_: Any,
    ) -> Any:
        model = IsolationForest(contamination=contamination, random_state=random_state)
        model.fit(prepared_data["X_scaled"])
        logger.info("anomaly_model_trained", contamination=contamination, n_rows=len(prepared_data["X_scaled"]))
        return model

    def evaluate(self, model: Any, prepared_data: dict[str, Any], **_: Any) -> dict[str, float]:
        predictions = model.predict(prepared_data["X_scaled"])   # -1 = anomaly, 1 = normal
        n_anomalies = int((predictions == -1).sum())
        n_total = len(predictions)

        return {
            "anomalies_found": n_anomalies,
            "total_rows": n_total,
            "anomaly_rate_pct": round(n_anomalies / n_total * 100, 2) if n_total else 0.0,
        }

    def predict(self, model: Any, data: dict[str, Any], **_: Any) -> pd.DataFrame:
        predictions = model.predict(data["X_scaled"])
        scores = model.decision_function(data["X_scaled"])   # lower = more anomalous

        result = data["X_original"].copy()
        result["is_anomaly"] = predictions == -1
        result["anomaly_score"] = np.round(scores, 4)
        return result.sort_values("anomaly_score").reset_index(drop=True)

    def explain(self, result: PluginResult) -> str:
        stats = result.summary_stats
        n_anomalies = stats.get("anomalies_found", 0)
        rate = stats.get("anomaly_rate_pct", 0)
        total = stats.get("total_rows", 0)

        return (
            f"Flagged {n_anomalies} of {total} rows ({rate:.1f}%) as unusual — these rows differ "
            f"substantially from the typical pattern in the data and may warrant closer review."
        )
