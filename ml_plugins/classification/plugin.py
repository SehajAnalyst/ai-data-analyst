"""
ml_plugins/classification/plugin.py
======================================

Supervised classification: predicts a categorical target column.
Supports "logistic" (LogisticRegression) or "random_forest"
(RandomForestClassifier) via model_type param, default "logistic".

PARAMS:
    target_column: str, REQUIRED — column to classify (must have
        between 2 and 20 distinct values — see validate_requirements).
    feature_columns: list[str], optional — defaults to all other
        numeric columns.
    model_type: "logistic" | "random_forest", default "logistic".
    test_size: float, default 0.2.
    random_state: int, default 42.
    n_estimators: int, default 100 — only for "random_forest".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score
from sklearn.model_selection import train_test_split

from exceptions.domain_exceptions import PluginValidationError
from logging_setup.logger import get_logger
from ml_plugins.base_plugin import BaseMLPlugin, PluginResult, PluginValidationResult

logger = get_logger(__name__)

_MIN_ROWS_REQUIRED = 10
_MAX_CLASSES = 20   # above this, "classification" is not the right tool


class ClassificationPlugin(BaseMLPlugin):
    """Logistic Regression or Random Forest Classifier on a categorical target."""

    @property
    def capability_name(self) -> str:
        return "classification"

    @property
    def description(self) -> str:
        return (
            "Predicts a category or label (e.g. churned yes/no, risk tier) from "
            "other columns using Logistic Regression or Random Forest Classification."
        )

    def validate_requirements(self, available_columns: dict[str, str]) -> PluginValidationResult:
        numeric_cols = [
            col for col, dtype in available_columns.items()
            if any(t in dtype.upper() for t in ("INT", "REAL", "FLOAT", "NUMERIC", "DOUBLE"))
        ]
        if len(numeric_cols) < 1:
            return PluginValidationResult(
                is_valid=False,
                missing_requirements=["at least one numeric feature column"],
                suggested_columns={},
            )
        return PluginValidationResult(is_valid=True, missing_requirements=[], suggested_columns={})

    def preprocess_data(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        feature_columns: list[str] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        **_: Any,
    ) -> dict[str, Any]:
        if not target_column:
            raise PluginValidationError(
                message="ClassificationPlugin requires a target_column parameter.",
                user_message="Please specify which column you want to classify.",
            )
        if target_column not in df.columns:
            raise PluginValidationError(
                message=f"target_column '{target_column}' not found. Available: {list(df.columns)}",
                user_message=f"Column '{target_column}' doesn't exist in this data.",
            )

        n_classes = df[target_column].nunique()
        if n_classes < 2:
            raise PluginValidationError(
                message=f"target_column '{target_column}' has only {n_classes} distinct value(s).",
                user_message=f"'{target_column}' needs at least 2 distinct categories to classify.",
            )
        if n_classes > _MAX_CLASSES:
            raise PluginValidationError(
                message=f"target_column '{target_column}' has {n_classes} distinct values — too many for classification.",
                user_message=f"'{target_column}' has too many distinct values to treat as categories.",
            )
        if len(df) < _MIN_ROWS_REQUIRED:
            raise PluginValidationError(
                message=f"Only {len(df)} rows; need at least {_MIN_ROWS_REQUIRED}.",
                user_message=f"Not enough data for reliable classification (need at least {_MIN_ROWS_REQUIRED} rows).",
            )

        candidate_features = feature_columns or [
            c for c in df.select_dtypes(include=[np.number]).columns if c != target_column
        ]
        if not candidate_features:
            raise PluginValidationError(
                message="No numeric feature columns available besides the target.",
                user_message="There aren't enough numeric columns to build a classifier.",
            )

        X = df[candidate_features].fillna(df[candidate_features].mean(numeric_only=True))
        y = df[target_column]

        # stratify keeps class proportions consistent across the split —
        # important for imbalanced categories (e.g. 90/10 churn split),
        # falls back to unstratified if any class has too few members
        # for stratification to succeed.
        try:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state, stratify=y
            )
        except ValueError:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=test_size, random_state=random_state
            )

        return {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "feature_columns": candidate_features,
            "target_column": target_column,
            "classes": sorted(y.unique().tolist(), key=str),
        }

    def train(
        self,
        prepared_data: dict[str, Any],
        model_type: str = "logistic",
        n_estimators: int = 100,
        random_state: int = 42,
        **_: Any,
    ) -> Any:
        if model_type == "random_forest":
            model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
        elif model_type == "logistic":
            model = LogisticRegression(max_iter=1000, random_state=random_state)
        else:
            raise PluginValidationError(
                message=f"Unknown model_type '{model_type}'. Use 'logistic' or 'random_forest'.",
                user_message=f"'{model_type}' isn't a supported classification model.",
            )

        model.fit(prepared_data["X_train"], prepared_data["y_train"])
        logger.info("classification_model_trained", model_type=model_type, n_classes=len(prepared_data["classes"]))
        return model

    def evaluate(self, model: Any, prepared_data: dict[str, Any], **_: Any) -> dict[str, float]:
        y_pred = model.predict(prepared_data["X_test"])
        y_test = prepared_data["y_test"]
        average = "binary" if len(prepared_data["classes"]) == 2 else "weighted"

        # pos_label required for binary average when classes aren't 0/1 —
        # sklearn defaults to pos_label=1, which breaks for string labels
        # like "yes"/"no". Falling back to weighted average sidesteps
        # needing to guess the "positive" class for arbitrary label sets.
        try:
            precision = precision_score(y_test, y_pred, average=average, zero_division=0)
            recall = recall_score(y_test, y_pred, average=average, zero_division=0)
            f1 = f1_score(y_test, y_pred, average=average, zero_division=0)
        except ValueError:
            precision = precision_score(y_test, y_pred, average="weighted", zero_division=0)
            recall = recall_score(y_test, y_pred, average="weighted", zero_division=0)
            f1 = f1_score(y_test, y_pred, average="weighted", zero_division=0)

        return {
            "accuracy": round(float(accuracy_score(y_test, y_pred)), 4),
            "precision": round(float(precision), 4),
            "recall": round(float(recall), 4),
            "f1_score": round(float(f1), 4),
            "test_rows": len(y_test),
        }

    def predict(self, model: Any, data: dict[str, Any], **_: Any) -> pd.DataFrame:
        X_test = data["X_test"]
        y_test = data["y_test"]
        predictions = model.predict(X_test)

        result = X_test.copy()
        result["actual"] = y_test.values
        result["predicted"] = predictions
        result["correct"] = result["actual"] == result["predicted"]
        return result.reset_index(drop=True)

    def explain(self, result: PluginResult) -> str:
        stats = result.summary_stats
        accuracy = stats.get("accuracy", 0)
        f1 = stats.get("f1_score", 0)

        if accuracy >= 0.85:
            quality = "performs well"
        elif accuracy >= 0.65:
            quality = "performs moderately"
        else:
            quality = "performs poorly — treat predictions with caution"

        return (
            f"The classification model {quality}, correctly predicting {accuracy:.0%} "
            f"of cases (F1 score {f1:.2f}), evaluated on {stats.get('test_rows', 0)} held-out rows."
        )
