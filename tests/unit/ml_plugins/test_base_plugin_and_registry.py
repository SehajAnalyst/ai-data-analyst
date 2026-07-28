"""
tests/unit/ml_plugins/test_base_plugin_and_registry.py
==========================================================

Tests the run() template method itself (using a minimal fake plugin,
not a real ML algorithm — this test is about the ORCHESTRATION shape,
not about any specific model), plus plugin_registry.py and
bootstrap.py.
"""

from __future__ import annotations

import pandas as pd
import pytest

from exceptions.domain_exceptions import PluginNotFoundError
from ml_plugins.base_plugin import BaseMLPlugin, PluginResult, PluginValidationResult
from ml_plugins.bootstrap import register_all_plugins
from ml_plugins.plugin_registry import (
    get_plugin,
    list_available_capabilities,
    register_plugin,
)


class _FakePlugin(BaseMLPlugin):
    """Minimal concrete plugin for testing run()'s template-method
    composition, independent of any real ML library."""

    def __init__(self) -> None:
        self.calls: list[str] = []

    @property
    def capability_name(self) -> str:
        return "fake"

    @property
    def description(self) -> str:
        return "A fake plugin for testing."

    def validate_requirements(self, available_columns):
        return PluginValidationResult(is_valid=True, missing_requirements=[], suggested_columns={})

    def preprocess_data(self, df, **kwargs):
        self.calls.append("preprocess_data")
        return {"df": df}

    def train(self, prepared_data, **kwargs):
        self.calls.append("train")
        return "fake_model"

    def evaluate(self, model, prepared_data, **kwargs):
        self.calls.append("evaluate")
        return {"fake_metric": 1.0}

    def predict(self, model, data, **kwargs):
        self.calls.append("predict")
        return "fake_predictions"

    def explain(self, result):
        return "fake explanation"


class TestBaseMLPluginTemplateMethod:
    def test_load_data_rejects_none(self):
        plugin = _FakePlugin()
        with pytest.raises(Exception):
            plugin.load_data(None)

    def test_load_data_rejects_empty_dataframe(self):
        plugin = _FakePlugin()
        with pytest.raises(Exception):
            plugin.load_data(pd.DataFrame())

    def test_load_data_returns_a_copy_not_the_same_object(self):
        plugin = _FakePlugin()
        df = pd.DataFrame({"x": [1, 2, 3]})
        loaded = plugin.load_data(df)
        assert loaded is not df
        assert loaded.equals(df)

    def test_run_calls_lifecycle_steps_in_order(self):
        plugin = _FakePlugin()
        df = pd.DataFrame({"x": [1, 2, 3]})
        plugin.run(df, params={"some_param": 1})
        assert plugin.calls == ["preprocess_data", "train", "evaluate", "predict"]

    def test_run_returns_plugin_result_with_correct_shape(self):
        plugin = _FakePlugin()
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = plugin.run(df, params={})
        assert isinstance(result, PluginResult)
        assert result.result_data == "fake_predictions"
        assert result.summary_stats == {"fake_metric": 1.0}

    def test_run_with_no_params_uses_empty_dict(self):
        plugin = _FakePlugin()
        df = pd.DataFrame({"x": [1, 2, 3]})
        result = plugin.run(df)   # params omitted entirely
        assert isinstance(result, PluginResult)


class TestPluginRegistry:
    def test_get_unregistered_plugin_raises(self):
        with pytest.raises(PluginNotFoundError):
            get_plugin("definitely_not_registered_xyz")

    def test_register_and_get_plugin(self):
        fake = _FakePlugin()
        register_plugin(fake)
        retrieved = get_plugin("fake")
        assert retrieved is fake

    def test_list_available_capabilities_includes_registered(self):
        register_plugin(_FakePlugin())
        assert "fake" in list_available_capabilities()


class TestBootstrap:
    def test_register_all_plugins_registers_all_five(self):
        register_all_plugins()
        capabilities = list_available_capabilities()
        for expected in ("regression", "classification", "clustering", "forecasting", "anomaly_detection"):
            assert expected in capabilities

    def test_bootstrap_is_idempotent(self):
        register_all_plugins()
        first = set(list_available_capabilities())
        register_all_plugins()
        second = set(list_available_capabilities())
        assert first == second

    def test_all_registered_plugins_are_retrievable(self):
        register_all_plugins()
        for name in ("regression", "classification", "clustering", "forecasting", "anomaly_detection"):
            plugin = get_plugin(name)
            assert plugin.capability_name == name
            assert isinstance(plugin.description, str)
            assert len(plugin.description) > 0
