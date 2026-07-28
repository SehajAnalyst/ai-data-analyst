"""
tests/unit/ml_plugins/test_forecasting.py
=============================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exceptions.domain_exceptions import PluginValidationError
from ml_plugins.forecasting.plugin import ForecastingPlugin


@pytest.fixture
def monthly_trend_data() -> pd.DataFrame:
    rng = np.random.RandomState(4)
    dates = pd.date_range("2023-01-01", periods=30, freq="MS")
    values = 10000 + np.arange(30) * 300 + rng.normal(0, 200, 30)
    return pd.DataFrame({"sale_month": dates.strftime("%Y-%m-%d"), "revenue": values})


class TestForecastingPlugin:
    def test_capability_name(self):
        assert ForecastingPlugin().capability_name == "forecasting"

    def test_forecast_produces_requested_horizon(self, monthly_trend_data):
        plugin = ForecastingPlugin()
        result = plugin.run(monthly_trend_data, params={
            "date_column": "sale_month", "value_column": "revenue", "horizon": 6,
        })
        assert len(result.result_data) == 6

    def test_forecast_columns_present(self, monthly_trend_data):
        plugin = ForecastingPlugin()
        result = plugin.run(monthly_trend_data, params={
            "date_column": "sale_month", "value_column": "revenue",
        })
        for col in ("date", "forecast", "lower_ci", "upper_ci"):
            assert col in result.result_data.columns

    def test_confidence_interval_widens_with_horizon(self, monthly_trend_data):
        """Real property: forecast uncertainty should increase further
        into the future, not stay flat or shrink."""
        plugin = ForecastingPlugin()
        result = plugin.run(monthly_trend_data, params={
            "date_column": "sale_month", "value_column": "revenue", "horizon": 6,
        })
        df = result.result_data
        first_width = df.iloc[0]["upper_ci"] - df.iloc[0]["lower_ci"]
        last_width = df.iloc[-1]["upper_ci"] - df.iloc[-1]["lower_ci"]
        assert last_width > first_width

    def test_backtest_accuracy_reasonable_on_clean_trend(self, monthly_trend_data):
        plugin = ForecastingPlugin()
        result = plugin.run(monthly_trend_data, params={
            "date_column": "sale_month", "value_column": "revenue",
        })
        # clean linear trend + modest noise should backtest well
        assert result.summary_stats["backtest_mape_pct"] < 15

    def test_missing_params_raises(self, monthly_trend_data):
        plugin = ForecastingPlugin()
        with pytest.raises(PluginValidationError):
            plugin.run(monthly_trend_data, params={})

    def test_unparseable_dates_raise(self):
        plugin = ForecastingPlugin()
        df = pd.DataFrame({
            "date_col": ["not a date"] * 20,
            "value": range(20),
        })
        with pytest.raises(PluginValidationError):
            plugin.run(df, params={"date_column": "date_col", "value_column": "value"})

    def test_irregular_dates_raise(self):
        """Dates that can't be resolved to a regular frequency must
        raise, not silently produce wrong forecast dates."""
        plugin = ForecastingPlugin()
        irregular_dates = ["2023-01-01", "2023-01-03", "2023-01-04", "2023-01-09"] * 6
        df = pd.DataFrame({"date_col": irregular_dates[:24], "value": range(24)})
        with pytest.raises(PluginValidationError):
            plugin.run(df, params={"date_column": "date_col", "value_column": "value"})

    def test_too_few_rows_raises(self):
        plugin = ForecastingPlugin()
        dates = pd.date_range("2023-01-01", periods=5, freq="MS")
        df = pd.DataFrame({"date_col": dates, "value": range(5)})
        with pytest.raises(PluginValidationError):
            plugin.run(df, params={"date_column": "date_col", "value_column": "value"})
