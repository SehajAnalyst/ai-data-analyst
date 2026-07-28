"""
ml_plugins/clustering/plugin.py
==================================

Unsupervised clustering via KMeans. No target column — groups rows
by similarity across numeric feature columns.

Genuinely different data shape from the supervised plugins: no
train/test split (there's no ground truth to hold out against), and
evaluate() scores the fit against the SAME data used to fit, via
silhouette score — this is standard, correct practice for clustering,
not a shortcut.

PARAMS:
    feature_columns: list[str], optional — defaults to all numeric columns.
    n_clusters: int, default 3.
    random_state: int, default 42.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.preprocessing import StandardScaler

from exceptions.domain_exceptions import PluginValidationError
from logging_setup.logger import get_logger
from ml_plugins.base_plugin import BaseMLPlugin, PluginResult, PluginValidationResult

logger = get_logger(__name__)

_MIN_ROWS_REQUIRED = 10


class ClusteringPlugin(BaseMLPlugin):
    """KMeans clustering on numeric feature columns."""

    @property
    def capability_name(self) -> str:
        return "clustering"

    @property
    def description(self) -> str:
        return "Groups rows into clusters of similar records based on numeric columns, using KMeans."

    def validate_requirements(self, available_columns: dict[str, str]) -> PluginValidationResult:
        numeric_cols = [
            col for col, dtype in available_columns.items()
            if any(t in dtype.upper() for t in ("INT", "REAL", "FLOAT", "NUMERIC", "DOUBLE"))
        ]
        if len(numeric_cols) < 2:
            return PluginValidationResult(
                is_valid=False,
                missing_requirements=["at least two numeric columns to cluster on"],
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
        Selects numeric feature columns and standardizes them (zero
        mean, unit variance). Standardization matters for KMeans
        specifically: it's a distance-based algorithm, so a column
        like "salary" (scale: tens of thousands) would completely
        dominate a column like "years_experience" (scale: single
        digits) without scaling — clusters would just reflect salary,
        not the intended multi-feature similarity.
        """
        candidate_features = feature_columns or list(df.select_dtypes(include=[np.number]).columns)
        if len(candidate_features) < 2:
            raise PluginValidationError(
                message=f"Only {len(candidate_features)} numeric column(s) available; need at least 2 to cluster.",
                user_message="Need at least two numeric columns to find meaningful groups.",
            )
        if len(df) < _MIN_ROWS_REQUIRED:
            raise PluginValidationError(
                message=f"Only {len(df)} rows; need at least {_MIN_ROWS_REQUIRED}.",
                user_message=f"Not enough data to cluster reliably (need at least {_MIN_ROWS_REQUIRED} rows).",
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
        n_clusters: int = 3,
        random_state: int = 42,
        **_: Any,
    ) -> Any:
        n_rows = len(prepared_data["X_scaled"])
        effective_k = min(n_clusters, max(2, n_rows // 2))
        if effective_k != n_clusters:
            logger.warning("clustering_k_reduced", requested=n_clusters, effective=effective_k, n_rows=n_rows)

        model = KMeans(n_clusters=effective_k, random_state=random_state, n_init=10)
        model.fit(prepared_data["X_scaled"])
        logger.info("clustering_model_trained", n_clusters=effective_k)
        return model

    def evaluate(self, model: Any, prepared_data: dict[str, Any], **_: Any) -> dict[str, float]:
        labels = model.labels_
        X_scaled = prepared_data["X_scaled"]

        # silhouette_score requires at least 2 distinct labels and
        # fewer labels than samples — both guaranteed by train()'s
        # effective_k clamp, but guarded here defensively anyway.
        if len(set(labels)) < 2:
            silhouette = 0.0
        else:
            silhouette = float(silhouette_score(X_scaled, labels))

        return {
            "n_clusters": int(model.n_clusters),
            "silhouette_score": round(silhouette, 4),
            "inertia": round(float(model.inertia_), 2),
        }

    def predict(self, model: Any, data: dict[str, Any], **_: Any) -> pd.DataFrame:
        result = data["X_original"].copy()
        result["cluster"] = model.labels_
        return result

    def explain(self, result: PluginResult) -> str:
        stats = result.summary_stats
        silhouette = stats.get("silhouette_score", 0)
        n_clusters = stats.get("n_clusters", 0)

        if silhouette >= 0.5:
            quality = "well-separated"
        elif silhouette >= 0.25:
            quality = "somewhat overlapping"
        else:
            quality = "largely overlapping — the groups may not be meaningfully distinct"

        return (
            f"The data was grouped into {n_clusters} clusters. The groups are {quality} "
            f"(silhouette score {silhouette:.2f}, where 1.0 is perfectly separated and 0 is no structure)."
        )
