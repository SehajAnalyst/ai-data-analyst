"""
ml_plugins/base_plugin.py
============================

Abstract interface every ML capability (Regression, Classification,
Clustering, Forecasting, Anomaly Detection) must implement.

INTERFACE HISTORY — WHY run() IS NOW CONCRETE, NOT ABSTRACT
-------------------------------------------------------------------
This interface originally specified only two behavioral methods:
`run(data, params) -> PluginResult` and `explain(result) -> str`. That
was correct for the skeleton stage, before any plugin existed to
reveal what "run" actually needed to do internally.

Implementing five real plugins surfaced a genuine requirement: each
one needs a standardized, granular lifecycle —
load_data -> preprocess_data -> train -> evaluate -> predict — both
so each stage is independently testable (you can unit-test
preprocess_data without fitting a model) and so future features
(conversational param elicitation, retraining, incremental evaluation)
have a stable seam to hook into instead of reaching into a monolithic
run() implementation per plugin.

Resolution: `run()` is now a CONCRETE template method defined here,
composing the five abstract lifecycle steps below. Every concrete
plugin implements the five steps; none of them override `run()`
directly. This preserves the original contract exactly — anything
that already calls `plugin.run(data, params)` (plugin_registry.py,
any future orchestrator) is unaffected — while giving each plugin a
consistent internal shape.

WHY THE LIFECYCLE METHODS USE LOOSE (Any) TYPES, NOT STRICT ONES
-------------------------------------------------------------------------
Regression/Classification are supervised: preprocess_data produces an
(X_train, X_test, y_train, y_test) split, train(X_train, y_train)
fits a model, evaluate(model, X_test, y_test) scores it on held-out
data. Clustering and Anomaly Detection are unsupervised: there is no
y and no train/test split — evaluate() scores the fit against the
same data (e.g. silhouette score), which is standard practice for
those algorithm families. Forecasting fits a time-ordered series and
evaluates against a held-out tail of periods, then a final predict()
refits on the full series to produce the actual forward forecast.

These four data shapes are genuinely different, not superficially so.
Forcing one strict generic signature across all of them would either
be dishonest (claiming a shared type that isn't real) or would block
real ML practice (e.g. denying clustering the ability to evaluate
without a y it structurally doesn't have). The ABC therefore
guarantees METHOD NAMES and the overall PIPELINE SHAPE — not the
exact type flowing through `prepared_data`/`model` — matching how
this loose-typing tradeoff is handled honestly rather than hidden.

REALISTIC SCOPE WARNING (carried over from the architecture phase):
ML plugins are not just "call sklearn and return a result." Each
plugin needs to: validate that the connected schema even has suitable
data for the task (e.g. Forecasting needs a date/time column and a
numeric target), potentially elicit missing parameters
conversationally (which time column? which target?), and explain
results in plain English same as the SQL pipeline does. This
interface reflects that — see `required_columns()` and `explain()`
below — rather than pretending it's a thin wrapper.
"""

from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any

import pandas as pd

from exceptions.domain_exceptions import PluginValidationError
from logging_setup.logger import get_logger

logger = get_logger(__name__)


@dataclass
class PluginValidationResult:
    is_valid: bool
    missing_requirements: list[str]   # human-readable, e.g. "a datetime column"
    suggested_columns: dict[str, str]  # e.g. {"time_column": "order_date"} — best-guess
                                          # suggestions the orchestrator can confirm
                                          # with the user rather than asking from scratch


@dataclass
class PluginResult:
    result_data: Any            # plugin-specific result shape (e.g. forecast dataframe)
    chart_selection: object | None   # reuses core/visualization types where applicable
    summary_stats: dict[str, Any]


class BaseMLPlugin(ABC):
    """
    Every concrete plugin (ml_plugins/forecasting/, etc.) implements
    this. Mirrors the Strategy-pattern shape of BaseLLMProvider and
    BaseDBConnector deliberately — same architectural pattern applied
    a third time, for consistency across the codebase.
    """

    @property
    @abstractmethod
    def capability_name(self) -> str:
        """Short identifier, e.g. 'forecasting'. Used by
        plugin_registry.py for routing and by
        intent_classifier.IntentClassification.suggested_ml_capability
        to indicate which plugin a request should route to."""
        raise NotImplementedError

    @property
    @abstractmethod
    def description(self) -> str:
        """Human-readable description, ALSO used by
        intent_classifier.py (or a future routing prompt) to help the
        LLM decide whether a user message maps to this capability."""
        raise NotImplementedError

    @abstractmethod
    def validate_requirements(self, available_columns: dict[str, str]) -> PluginValidationResult:
        """
        Checks whether the connected schema has what this plugin
        needs (e.g. Forecasting requires a datetime + numeric column).

        Args:
            available_columns: column name -> data type, for the
                table(s) under discussion.

        Returns:
            PluginValidationResult. If is_valid is False, the
            orchestrator should surface missing_requirements to the
            user conversationally rather than failing silently or
            attempting to run the plugin anyway.
        """
        raise NotImplementedError

    # ── Granular lifecycle (each plugin implements these) ──────────────────

    def load_data(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Defensive entry point for the DataFrame the orchestrator
        already fetched via the normal SQL pipeline. This is NOT a
        raw-database-connection fetch — plugins never open their own
        DB connection, consistent with the "LLM/AI layer proposes,
        deterministic code executes" boundary used throughout core/
        (see module docstring). Fetching is the caller's job; this
        method is a validation/defensive-copy step every plugin needs.

        Concrete (not abstract) because the same three checks apply
        to every plugin — subclasses may override if a capability
        genuinely needs different loading behavior, but none currently
        do.

        Raises:
            PluginValidationError: df is empty or None.
        """
        if df is None or df.empty:
            raise PluginValidationError(
                message=f"{self.capability_name}: input DataFrame is empty or None.",
                user_message="There's no data to run this analysis on.",
            )
        return df.copy()

    @abstractmethod
    def preprocess_data(self, df: pd.DataFrame, **kwargs: Any) -> Any:
        """
        Transforms the loaded DataFrame into whatever shape this
        plugin's train/evaluate/predict steps need — e.g. an
        (X_train, X_test, y_train, y_test) tuple for supervised
        plugins, a feature matrix for unsupervised ones, or an
        indexed time series for forecasting. See module docstring for
        why this return type is intentionally plugin-specific.

        Raises:
            PluginValidationError: data doesn't meet this plugin's
                requirements (e.g. target column missing, insufficient
                rows for a meaningful train/test split).
        """
        raise NotImplementedError

    @abstractmethod
    def train(self, prepared_data: Any, **kwargs: Any) -> Any:
        """Fits and returns a model object using the output of
        preprocess_data(). The returned object's type is
        plugin-specific (a fitted sklearn estimator, a fitted
        statsmodels ARIMA results object, etc.)."""
        raise NotImplementedError

    @abstractmethod
    def evaluate(self, model: Any, prepared_data: Any, **kwargs: Any) -> dict[str, float]:
        """
        Returns quality metrics for the fitted model — held-out
        metrics for supervised/forecasting plugins (RMSE, accuracy,
        etc.), fit-quality metrics for unsupervised ones (silhouette
        score, inertia, anomaly fraction). Always a flat dict of
        metric name -> float, regardless of plugin type — this is the
        one piece of the lifecycle kept strictly uniform, since
        "return some numeric quality metrics" is genuinely universal
        across all five algorithm families.
        """
        raise NotImplementedError

    @abstractmethod
    def predict(self, model: Any, data: Any, **kwargs: Any) -> Any:
        """
        Produces the actual output the user asked for: predicted
        values, cluster assignments, forecasted periods, or anomaly
        flags. Return type is plugin-specific — see module docstring.
        """
        raise NotImplementedError

    @abstractmethod
    def explain(self, result: PluginResult) -> str:
        """Plain-English explanation of the result, same role as
        core/insights/insight_generator.py plays for SQL results —
        kept as a plugin responsibility rather than a generic shared
        function, since what's worth explaining is plugin-specific
        (e.g. forecast confidence intervals vs. cluster
        characteristics are explained completely differently)."""
        raise NotImplementedError

    # ── Orchestration (concrete — do not override in subclasses) ───────────

    def run(self, data: pd.DataFrame, params: dict[str, Any] | None = None) -> PluginResult:
        """
        Template method composing the five lifecycle steps above into
        the PluginResult contract that plugin_registry.py and any
        future orchestrator already depend on. See module docstring
        for why this is concrete rather than each plugin implementing
        its own run().

        Args:
            data: DataFrame already fetched by the caller via the
                normal SQL pipeline (not fetched by this method — see
                load_data()'s docstring).
            params: plugin-specific parameters (e.g. target_column,
                n_clusters, horizon). Passed through to every lifecycle
                step as **kwargs, so each plugin reads only the keys
                it needs.

        Returns:
            PluginResult with result_data = predict()'s output and
            summary_stats = evaluate()'s metrics dict.

        Raises:
            PluginValidationError: raised by load_data() or
                preprocess_data() if the data doesn't meet this
                plugin's requirements.
        """
        params = params or {}
        logger.info("ml_plugin_run_started", capability=self.capability_name, params=list(params.keys()))

        loaded = self.load_data(data)
        prepared = self.preprocess_data(loaded, **params)
        model = self.train(prepared, **params)
        metrics = self.evaluate(model, prepared, **params)
        predictions = self.predict(model, prepared, **params)

        logger.info("ml_plugin_run_succeeded", capability=self.capability_name, metrics=metrics)

        return PluginResult(
            result_data=predictions,
            chart_selection=None,
            summary_stats=metrics,
        )
