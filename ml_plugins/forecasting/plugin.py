"""
ml_plugins/forecasting/plugin.py
===================================

Time series forecasting via ARIMA (statsmodels). Chosen over Prophet —
see the note below, backed by direct testing, not a guess.

WHY ARIMA, NOT PROPHET
------------------------
Verified directly before choosing: Prophet's cmdstanpy backend raised
`ValueError: No CmdStan installation found` when its path was queried
explicitly, yet a subsequent `.fit()` call succeeded anyway — meaning
it silently triggered some first-run fallback/build behavior whose
exact trigger (network access? a writable cache dir? a bundled
binary?) wasn't obvious from the outside. That's an environment-
dependent behavior that a production system should not depend on
implicitly. ARIMA has no such risk: it's pure numerical computation
via statsmodels, with no external binary, network call, or compiler
dependency at fit time.

WHY evaluate() DOESN'T JUST SCORE THE MODEL FROM train()
-------------------------------------------------------------------
This is the one plugin that deviates from "evaluate() scores exactly
the model object train() produced," and that deviation is deliberate,
not an inconsistency to hide:

  train() fits ARIMA on the FULL available series. This is the model
  used by predict() to produce the actual forward-looking forecast —
  and forecasting forward should always use every known data point,
  never withhold real, recent history from the production model.

  evaluate() cannot honestly assess forecast quality using that same
  full-data model, because forecasting accuracy must be measured
  out-of-sample (on data the model didn't see). So evaluate() performs
  its own internal backtest: it fits a SEPARATE, temporary ARIMA model
  on all but the last `test_periods` points, forecasts that held-out
  tail, and compares against the real values. This is standard time-
  series evaluation practice (backtesting), not a shortcut — the
  alternative (scoring the full-data model against data it already
  saw) would silently overstate accuracy.

PARAMS:
    date_column: str, REQUIRED.
    value_column: str, REQUIRED.
    order: tuple[int,int,int], default (1,1,1) — the ARIMA (p,d,q) order.
    test_periods: int, default 6 — how many trailing periods to hold
        out for the evaluate() backtest.
    horizon: int, default 12 — how many periods ahead predict() forecasts.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from statsmodels.tsa.arima.model import ARIMA

from exceptions.domain_exceptions import PluginValidationError
from logging_setup.logger import get_logger
from ml_plugins.base_plugin import BaseMLPlugin, PluginResult, PluginValidationResult

logger = get_logger(__name__)


class ForecastingPlugin(BaseMLPlugin):
    """ARIMA time-series forecasting."""

    @property
    def capability_name(self) -> str:
        return "forecasting"

    @property
    def description(self) -> str:
        return "Forecasts future values of a numeric metric over time (e.g. monthly sales) using ARIMA."

    def validate_requirements(self, available_columns: dict[str, str]) -> PluginValidationResult:
        date_hint_cols = [
            c for c in available_columns
            if any(h in c.lower() for h in ("date", "time", "month", "year", "period"))
        ]
        numeric_cols = [
            col for col, dtype in available_columns.items()
            if any(t in dtype.upper() for t in ("INT", "REAL", "FLOAT", "NUMERIC", "DOUBLE"))
        ]
        missing = []
        if not date_hint_cols:
            missing.append("a date/time column")
        if not numeric_cols:
            missing.append("a numeric value column")

        return PluginValidationResult(
            is_valid=not missing,
            missing_requirements=missing,
            suggested_columns={
                **({"date_column": date_hint_cols[0]} if date_hint_cols else {}),
                **({"value_column": numeric_cols[0]} if numeric_cols else {}),
            },
        )

    def preprocess_data(
        self,
        df: pd.DataFrame,
        date_column: str | None = None,
        value_column: str | None = None,
        test_periods: int = 6,
        **_: Any,
    ) -> dict[str, Any]:
        """
        Parses the date column, sorts chronologically, and builds a
        regularly-indexed pd.Series. ARIMA forecasting needs a known
        step size to project forward dates — if the dates can't be
        parsed or a regular frequency can't be inferred, this raises
        rather than silently guessing a frequency that would produce
        wrong forecast dates.
        """
        if not date_column or not value_column:
            raise PluginValidationError(
                message="ForecastingPlugin requires date_column and value_column parameters.",
                user_message="Please specify which column has the dates and which has the values to forecast.",
            )
        if date_column not in df.columns or value_column not in df.columns:
            raise PluginValidationError(
                message=f"date_column/value_column not found. Available: {list(df.columns)}",
                user_message="One of the specified columns doesn't exist in this data.",
            )

        min_required = test_periods + 10
        if len(df) < min_required:
            raise PluginValidationError(
                message=f"Only {len(df)} rows; need at least {min_required} for a {test_periods}-period backtest.",
                user_message=f"Not enough historical data for a reliable forecast (need at least {min_required} periods).",
            )

        work = df[[date_column, value_column]].copy()
        work[date_column] = pd.to_datetime(work[date_column], errors="coerce")
        if work[date_column].isna().any():
            raise PluginValidationError(
                message=f"Some values in '{date_column}' could not be parsed as dates.",
                user_message=f"Some values in '{date_column}' aren't valid dates.",
            )

        work = work.sort_values(date_column).drop_duplicates(subset=[date_column])
        series = work.set_index(date_column)[value_column]

        inferred_freq = pd.infer_freq(series.index)
        if inferred_freq is None:
            raise PluginValidationError(
                message=f"Could not infer a regular time frequency from '{date_column}'.",
                user_message=(
                    f"The dates in '{date_column}' aren't evenly spaced, so a reliable "
                    "forecast can't be produced. Try aggregating to a regular interval first "
                    "(e.g. group by month)."
                ),
            )
        series = series.asfreq(inferred_freq)
        if series.isna().any():
            series = series.interpolate()

        return {
            "series": series,
            "test_periods": test_periods,
            "date_column": date_column,
            "value_column": value_column,
        }

    def train(self, prepared_data: dict[str, Any], order: tuple[int, int, int] = (1, 1, 1), **_: Any) -> Any:
        """
        Fits ARIMA on the FULL series — this is the model predict()
        uses for the actual forward forecast. See module docstring
        for why evaluate() does NOT reuse this same fitted model.
        """
        series = prepared_data["series"]
        try:
            model = ARIMA(series, order=order).fit()
        except Exception as exc:
            raise PluginValidationError(
                message=f"ARIMA fit failed with order {order}: {exc}",
                user_message="Could not fit a forecast model to this data. Try a different date range.",
            ) from exc

        logger.info("forecasting_model_trained", order=order, n_periods=len(series))
        return model

    def evaluate(
        self,
        model: Any,
        prepared_data: dict[str, Any],
        order: tuple[int, int, int] = (1, 1, 1),
        **_: Any,
    ) -> dict[str, float]:
        """
        Backtests via a SEPARATE model fit on all-but-the-last
        test_periods points — see module docstring for why this
        doesn't reuse `model` (which was fit on the full series).
        """
        series = prepared_data["series"]
        test_periods = prepared_data["test_periods"]

        train_series = series[:-test_periods]
        test_series = series[-test_periods:]

        backtest_model = ARIMA(train_series, order=order).fit()
        forecast = backtest_model.forecast(steps=test_periods)

        actual = test_series.values
        predicted = forecast.values
        mae = float(np.mean(np.abs(actual - predicted)))
        rmse = float(np.sqrt(np.mean((actual - predicted) ** 2)))
        nonzero_mask = actual != 0
        mape = (
            float(np.mean(np.abs((actual[nonzero_mask] - predicted[nonzero_mask]) / actual[nonzero_mask])) * 100)
            if nonzero_mask.any() else float("nan")
        )

        return {
            "backtest_mae": round(mae, 2),
            "backtest_rmse": round(rmse, 2),
            "backtest_mape_pct": round(mape, 2) if not np.isnan(mape) else -1.0,
            "aic": round(float(model.aic), 2),
            "test_periods": test_periods,
        }

    def predict(self, model: Any, data: dict[str, Any], horizon: int = 12, **_: Any) -> pd.DataFrame:
        """
        Uses the full-data model (from train()) to forecast `horizon`
        periods forward with 95% confidence intervals.
        """
        forecast_result = model.get_forecast(steps=horizon)
        ci = forecast_result.conf_int(alpha=0.05)

        return pd.DataFrame({
            "date": forecast_result.predicted_mean.index,
            "forecast": np.round(forecast_result.predicted_mean.values, 2),
            "lower_ci": np.round(ci.iloc[:, 0].values, 2),
            "upper_ci": np.round(ci.iloc[:, 1].values, 2),
        })

    def explain(self, result: PluginResult) -> str:
        stats = result.summary_stats
        mape = stats.get("backtest_mape_pct", -1)
        forecast_df = result.result_data

        if mape < 0:
            accuracy_note = "backtest accuracy could not be computed (some actual values were zero)"
        elif mape <= 10:
            accuracy_note = f"historically accurate to within {mape:.1f}% on held-out data"
        elif mape <= 25:
            accuracy_note = f"moderately accurate (±{mape:.1f}% on held-out data) — use with some caution"
        else:
            accuracy_note = f"a wide historical error margin (±{mape:.1f}%) — treat this forecast as indicative only"

        n_periods = len(forecast_df) if forecast_df is not None else 0
        return (
            f"Forecasted the next {n_periods} periods. This model was {accuracy_note}. "
            f"Confidence intervals widen further into the future, which is expected."
        )
