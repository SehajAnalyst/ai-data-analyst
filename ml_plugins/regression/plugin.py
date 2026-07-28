"""
ml_plugins/regression/plugin.py
==================================

Supervised regression: predicts a numeric target column from other
numeric/categorical columns in the DataFrame.

Supports two algorithms via the `model_type` param ("linear" or
"random_forest", default "linear"). One plugin class, not two,
because both share the exact same preprocess/evaluate/predict shape —
only the estimator differs. Splitting into two plugin classes would
duplicate everything except one line.

PARAMS (passed through run()'s params dict):
    target_column: str, REQUIRED — the column to predict.
    feature_columns: list[str], optional — defaults to all other
        numeric columns if not given.
    model_type: "linear" | "random_forest", default "linear".
    test_size: float, default 0.2 — fraction held out for evaluation.
    random_state: int, default 42 — for reproducible splits/fitting.
    n_estimators: int, default 100 — only used when model_type is
        "random_forest".
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

from exceptions.domain_exceptions import PluginValidationError
from logging_setup.logger import get_logger
from ml_plugins.base_plugin import BaseMLPlugin, PluginResult, PluginValidationResult

logger = get_logger(__name__)

_MIN_ROWS_REQUIRED = 10   # below this, a train/test split is not meaningful


class RegressionPlugin(BaseMLPlugin):
    """Linear Regression or Random Forest Regressor on a numeric target."""

    @property
    def capability_name(self) -> str:
        return "regression"

    @property
    def description(self) -> str:
        return (
            "Predicts a numeric value (e.g. salary, price, revenue) from other "
            "columns using Linear Regression or Random Forest Regression."
        )

    def validate_requirements(self, available_columns: dict[str, str]) -> PluginValidationResult:
        numeric_cols = [
            col for col, dtype in available_columns.items()
            if any(t in dtype.upper() for t in ("INT", "REAL", "FLOAT", "NUMERIC", "DOUBLE"))
        ]
        if len(numeric_cols) < 2:
            return PluginValidationResult(
                is_valid=False,
                missing_requirements=["at least two numeric columns (one target, one feature)"],
                suggested_columns={},
            )
        return PluginValidationResult(
            is_valid=True,
            missing_requirements=[],
            suggested_columns={"target_column": numeric_cols[-1]},
        )

    def preprocess_data(
        self,
        df: pd.DataFrame,
        target_column: str | None = None,
        feature_columns: list[str] | None = None,
        test_size: float = 0.2,
        random_state: int = 42,
        **_: Any,
    ) -> dict[str, Any]:
        """
        Splits df into train/test sets.

        Returns a dict (not a bare tuple) so the keys are self-
        documenting at the train()/evaluate()/predict() call sites —
        {"X_train", "X_test", "y_train", "y_test", "feature_columns"}.

        Raises:
            PluginValidationError: target_column missing/not provided,
                not numeric, or too few rows for a meaningful split.
        """
        if not target_column:
            raise PluginValidationError(
                message="RegressionPlugin requires a target_column parameter.",
                user_message="Please specify which column you want to predict.",
            )
        if target_column not in df.columns:
            raise PluginValidationError(
                message=f"target_column '{target_column}' not found in data. Available: {list(df.columns)}",
                user_message=f"Column '{target_column}' doesn't exist in this data.",
            )
        if len(df) < _MIN_ROWS_REQUIRED:
            raise PluginValidationError(
                message=f"Only {len(df)} rows available; need at least {_MIN_ROWS_REQUIRED} for regression.",
                user_message=f"Not enough data for a reliable prediction (need at least {_MIN_ROWS_REQUIRED} rows).",
            )

        candidate_features = feature_columns or [
            c for c in df.select_dtypes(include=[np.number]).columns if c != target_column
        ]
        if not candidate_features:
            raise PluginValidationError(
                message="No numeric feature columns available besides the target.",
                user_message="There aren't enough numeric columns to build a prediction model.",
            )

        X = df[candidate_features].fillna(df[candidate_features].mean(numeric_only=True))
        y = df[target_column]

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=random_state
        )

        return {
            "X_train": X_train, "X_test": X_test,
            "y_train": y_train, "y_test": y_test,
            "feature_columns": candidate_features,
            "target_column": target_column,
        }

    def train(
        self,
        prepared_data: dict[str, Any],
        model_type: str = "linear",
        n_estimators: int = 100,
        random_state: int = 42,
        **_: Any,
    ) -> Any:
        if model_type == "random_forest":
            model = RandomForestRegressor(n_estimators=n_estimators, random_state=random_state)
        elif model_type == "linear":
            model = LinearRegression()
        else:
            raise PluginValidationError(
                message=f"Unknown model_type '{model_type}' for regression. Use 'linear' or 'random_forest'.",
                user_message=f"'{model_type}' isn't a supported regression model.",
            )

        model.fit(prepared_data["X_train"], prepared_data["y_train"])
        logger.info("regression_model_trained", model_type=model_type, n_features=len(prepared_data["feature_columns"]))
        return model

    def evaluate(self, model: Any, prepared_data: dict[str, Any], **_: Any) -> dict[str, float]:
        y_pred = model.predict(prepared_data["X_test"])
        y_test = prepared_data["y_test"]

        return {
            "r2_score": round(float(r2_score(y_test, y_pred)), 4),
            "mae": round(float(mean_absolute_error(y_test, y_pred)), 4),
            "rmse": round(float(np.sqrt(mean_squared_error(y_test, y_pred))), 4),
            "test_rows": len(y_test),
        }

    def predict(self, model: Any, data: dict[str, Any], **_: Any) -> pd.DataFrame:
        """
        Predicts on the held-out test set (not new unseen data — this
        version doesn't accept fresh prediction inputs beyond the
        train/test split created in preprocess_data; see TODOs).
        Returns a DataFrame with actual vs predicted for inspection.
        """
        X_test = data["X_test"]
        y_test = data["y_test"]
        predictions = model.predict(X_test)

        result = X_test.copy()
        result["actual"] = y_test.values
        result["predicted"] = np.round(predictions, 2)
        result["residual"] = np.round(result["actual"] - result["predicted"], 2)
        return result.reset_index(drop=True)

    def explain(self, result: PluginResult) -> str:
        stats = result.summary_stats
        r2 = stats.get("r2_score", 0)
        rmse = stats.get("rmse", 0)

        if r2 >= 0.8:
            fit_quality = "a strong fit"
        elif r2 >= 0.5:
            fit_quality = "a moderate fit"
        else:
            fit_quality = "a weak fit — predictions may not be reliable"

        return (
            f"The regression model explains {r2:.0%} of the variation in the target "
            f"({fit_quality}), with a typical prediction error of ±{rmse:.2f}, "
            f"evaluated on {stats.get('test_rows', 0)} held-out rows."
        )
