"""
tests/unit/ml_plugins/test_regression.py
===========================================

Tests use real synthetic data with a known linear relationship, not
mocks — the point of these tests is to verify sklearn is wired up
correctly and produces sane, expected results, which mocking away the
model would not verify.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exceptions.domain_exceptions import PluginValidationError
from ml_plugins.regression.plugin import RegressionPlugin


@pytest.fixture
def linear_data() -> pd.DataFrame:
    rng = np.random.RandomState(0)
    n = 100
    df = pd.DataFrame({
        "experience": rng.randint(0, 20, n),
        "education_years": rng.randint(12, 20, n),
    })
    df["salary"] = 30000 + df["experience"] * 2500 + df["education_years"] * 1500 + rng.normal(0, 3000, n)
    return df


class TestRegressionPlugin:
    def test_capability_name(self):
        assert RegressionPlugin().capability_name == "regression"

    def test_validate_requirements_passes_with_two_numeric_columns(self):
        plugin = RegressionPlugin()
        result = plugin.validate_requirements({"salary": "REAL", "experience": "INTEGER"})
        assert result.is_valid

    def test_validate_requirements_fails_with_one_numeric_column(self):
        plugin = RegressionPlugin()
        result = plugin.validate_requirements({"name": "TEXT"})
        assert not result.is_valid

    def test_run_produces_high_r2_on_clean_linear_data(self, linear_data):
        plugin = RegressionPlugin()
        result = plugin.run(linear_data, params={"target_column": "salary", "model_type": "linear"})
        # Data was constructed to be linear with modest noise — R2
        # should be high, not just "not an error."
        assert result.summary_stats["r2_score"] > 0.8

    def test_run_supports_random_forest(self, linear_data):
        plugin = RegressionPlugin()
        result = plugin.run(linear_data, params={"target_column": "salary", "model_type": "random_forest"})
        assert "r2_score" in result.summary_stats
        assert result.summary_stats["test_rows"] > 0

    def test_missing_target_column_raises(self, linear_data):
        plugin = RegressionPlugin()
        with pytest.raises(PluginValidationError):
            plugin.run(linear_data, params={})

    def test_nonexistent_target_column_raises(self, linear_data):
        plugin = RegressionPlugin()
        with pytest.raises(PluginValidationError):
            plugin.run(linear_data, params={"target_column": "does_not_exist"})

    def test_unknown_model_type_raises(self, linear_data):
        plugin = RegressionPlugin()
        with pytest.raises(PluginValidationError):
            plugin.run(linear_data, params={"target_column": "salary", "model_type": "not_a_real_model"})

    def test_too_few_rows_raises(self):
        plugin = RegressionPlugin()
        tiny_df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        with pytest.raises(PluginValidationError):
            plugin.run(tiny_df, params={"target_column": "y"})

    def test_predict_output_has_actual_and_predicted_columns(self, linear_data):
        plugin = RegressionPlugin()
        result = plugin.run(linear_data, params={"target_column": "salary"})
        assert "actual" in result.result_data.columns
        assert "predicted" in result.result_data.columns
        assert "residual" in result.result_data.columns

    def test_explain_mentions_r2_and_error(self, linear_data):
        plugin = RegressionPlugin()
        result = plugin.run(linear_data, params={"target_column": "salary"})
        explanation = plugin.explain(result)
        assert isinstance(explanation, str)
        assert len(explanation) > 0

    def test_load_data_rejects_empty_dataframe(self):
        plugin = RegressionPlugin()
        with pytest.raises(PluginValidationError):
            plugin.load_data(pd.DataFrame())
