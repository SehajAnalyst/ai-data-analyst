"""
tests/unit/ml_plugins/test_classification.py
================================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exceptions.domain_exceptions import PluginValidationError
from ml_plugins.classification.plugin import ClassificationPlugin


@pytest.fixture
def binary_data() -> pd.DataFrame:
    rng = np.random.RandomState(1)
    n = 150
    df = pd.DataFrame({
        "income": rng.randint(30000, 150000, n),
        "age": rng.randint(22, 65, n),
    })
    df["approved"] = ((df["income"] > 70000) & (df["age"] > 30)).astype(int)
    return df


class TestClassificationPlugin:
    def test_capability_name(self):
        assert ClassificationPlugin().capability_name == "classification"

    def test_run_produces_reasonable_accuracy(self, binary_data):
        plugin = ClassificationPlugin()
        result = plugin.run(binary_data, params={"target_column": "approved", "model_type": "logistic"})
        assert result.summary_stats["accuracy"] > 0.5   # better than random guessing

    def test_random_forest_captures_nonlinear_boundary_better_than_logistic(self, binary_data):
        """The synthetic rule (income>70000 AND age>30) is a non-linear
        boundary. Random Forest should fit it at least as well as
        Logistic Regression's linear boundary — this is a real
        algorithmic property, not an arbitrary assertion."""
        plugin = ClassificationPlugin()
        logistic_result = plugin.run(binary_data, params={"target_column": "approved", "model_type": "logistic"})
        forest_result = plugin.run(binary_data, params={"target_column": "approved", "model_type": "random_forest"})
        assert forest_result.summary_stats["accuracy"] >= logistic_result.summary_stats["accuracy"]

    def test_single_class_target_raises(self):
        plugin = ClassificationPlugin()
        df = pd.DataFrame({"x": range(20), "y": [1] * 20})
        with pytest.raises(PluginValidationError):
            plugin.run(df, params={"target_column": "y"})

    def test_too_many_classes_raises(self):
        plugin = ClassificationPlugin()
        df = pd.DataFrame({"x": range(50), "y": range(50)})   # 50 distinct classes
        with pytest.raises(PluginValidationError):
            plugin.run(df, params={"target_column": "y"})

    def test_missing_target_column_raises(self, binary_data):
        plugin = ClassificationPlugin()
        with pytest.raises(PluginValidationError):
            plugin.run(binary_data, params={})

    def test_predict_output_has_correct_column(self, binary_data):
        plugin = ClassificationPlugin()
        result = plugin.run(binary_data, params={"target_column": "approved"})
        assert "correct" in result.result_data.columns
        assert result.result_data["correct"].dtype == bool

    def test_metrics_are_bounded_between_0_and_1(self, binary_data):
        plugin = ClassificationPlugin()
        result = plugin.run(binary_data, params={"target_column": "approved"})
        for metric in ("accuracy", "precision", "recall", "f1_score"):
            assert 0.0 <= result.summary_stats[metric] <= 1.0
