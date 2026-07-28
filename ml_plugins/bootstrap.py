"""
ml_plugins/bootstrap.py
==========================

Registers all concrete ML plugins into plugin_registry at startup.
This is the "explicit startup routine called from app/main.py"
plugin_registry.py's own docstring anticipated — it didn't exist
until plugins existed to register.

WHY A SEPARATE FILE RATHER THAN REGISTERING INLINE IN EACH plugin.py
-------------------------------------------------------------------------
Registering at plugin.py's module import time (e.g. a bare
`register_plugin(RegressionPlugin())` at the bottom of
regression/plugin.py) would mean IMPORTING a plugin module has the
side effect of mutating global registry state — surprising behavior
for anything that imports RegressionPlugin for testing without
wanting it auto-registered. Keeping registration as one explicit,
idempotent function called once at app startup makes the side effect
visible and intentional, not an import-time surprise.
"""

from __future__ import annotations

from logging_setup.logger import get_logger
from ml_plugins.plugin_registry import list_available_capabilities, register_plugin

logger = get_logger(__name__)

_bootstrapped = False


def register_all_plugins() -> None:
    """
    Registers every implemented ML plugin. Idempotent — safe to call
    more than once (e.g. across Streamlit reruns); only the first call
    has an effect, matching the pattern used by
    logging_setup.logger.configure_logging().
    """
    global _bootstrapped
    if _bootstrapped:
        return

    from ml_plugins.anomaly_detection.plugin import AnomalyDetectionPlugin
    from ml_plugins.classification.plugin import ClassificationPlugin
    from ml_plugins.clustering.plugin import ClusteringPlugin
    from ml_plugins.forecasting.plugin import ForecastingPlugin
    from ml_plugins.regression.plugin import RegressionPlugin

    register_plugin(RegressionPlugin())
    register_plugin(ClassificationPlugin())
    register_plugin(ClusteringPlugin())
    register_plugin(ForecastingPlugin())
    register_plugin(AnomalyDetectionPlugin())

    _bootstrapped = True
    logger.info("ml_plugins_registered", capabilities=list_available_capabilities())
