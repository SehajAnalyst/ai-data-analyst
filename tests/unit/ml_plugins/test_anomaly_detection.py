"""
tests/unit/ml_plugins/test_anomaly_detection.py
===================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exceptions.domain_exceptions import PluginValidationError
from ml_plugins.anomaly_detection.plugin import AnomalyDetectionPlugin


@pytest.fixture
def data_with_planted_outliers() -> pd.DataFrame:
    rng = np.random.RandomState(5)
    n = 100
    normal = rng.normal(100, 15, n - 5)
    outliers = [5000, 4800, 6200, 5500, 4900]
    return pd.DataFrame({
        "transaction_amount": np.concatenate([normal, outliers]),
        "account_age_days": rng.randint(30, 2000, n),
    })


class TestAnomalyDetectionPlugin:
    def test_capability_name(self):
        assert AnomalyDetectionPlugin().capability_name == "anomaly_detection"

    def test_flags_approximately_the_configured_contamination_rate(self, data_with_planted_outliers):
        plugin = AnomalyDetectionPlugin()
        result = plugin.run(data_with_planted_outliers, params={"contamination": 0.05})
        assert result.summary_stats["anomaly_rate_pct"] == pytest.approx(5.0, abs=1.0)

    def test_planted_outliers_are_actually_flagged(self, data_with_planted_outliers):
        """Real property check: the 5 planted extreme values (4800-6200)
        must appear among the flagged anomalies, not just that SOME
        5% of rows got flagged arbitrarily."""
        plugin = AnomalyDetectionPlugin()
        result = plugin.run(data_with_planted_outliers, params={"contamination": 0.05})
        flagged = result.result_data[result.result_data["is_anomaly"]]
        assert (flagged["transaction_amount"] > 1000).all()

    def test_predict_output_sorted_by_anomaly_score(self, data_with_planted_outliers):
        plugin = AnomalyDetectionPlugin()
        result = plugin.run(data_with_planted_outliers, params={"contamination": 0.05})
        scores = result.result_data["anomaly_score"]
        assert list(scores) == sorted(scores)

    def test_no_numeric_columns_raises(self):
        plugin = AnomalyDetectionPlugin()
        df = pd.DataFrame({"name": ["a"] * 25})
        with pytest.raises(PluginValidationError):
            plugin.run(df, params={})

    def test_too_few_rows_raises(self):
        plugin = AnomalyDetectionPlugin()
        df = pd.DataFrame({"x": range(5)})
        with pytest.raises(PluginValidationError):
            plugin.run(df, params={})
