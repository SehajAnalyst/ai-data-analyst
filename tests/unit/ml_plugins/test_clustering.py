"""
tests/unit/ml_plugins/test_clustering.py
============================================
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from exceptions.domain_exceptions import PluginValidationError
from ml_plugins.clustering.plugin import ClusteringPlugin


@pytest.fixture
def three_blob_data() -> pd.DataFrame:
    rng = np.random.RandomState(2)
    n = 30
    group1 = pd.DataFrame({"income": rng.normal(30000, 2000, n), "spend": rng.normal(500, 50, n)})
    group2 = pd.DataFrame({"income": rng.normal(80000, 2000, n), "spend": rng.normal(2000, 100, n)})
    group3 = pd.DataFrame({"income": rng.normal(150000, 3000, n), "spend": rng.normal(5000, 200, n)})
    return pd.concat([group1, group2, group3], ignore_index=True)


class TestClusteringPlugin:
    def test_capability_name(self):
        assert ClusteringPlugin().capability_name == "clustering"

    def test_recovers_correct_number_of_clusters(self, three_blob_data):
        plugin = ClusteringPlugin()
        result = plugin.run(three_blob_data, params={"n_clusters": 3})
        assert result.summary_stats["n_clusters"] == 3

    def test_well_separated_blobs_give_high_silhouette(self, three_blob_data):
        """Real property check: three genuinely separated groups
        should score highly on silhouette (closer to 1.0), not just
        return SOME number."""
        plugin = ClusteringPlugin()
        result = plugin.run(three_blob_data, params={"n_clusters": 3})
        assert result.summary_stats["silhouette_score"] > 0.7

    def test_cluster_sizes_roughly_match_input_groups(self, three_blob_data):
        plugin = ClusteringPlugin()
        result = plugin.run(three_blob_data, params={"n_clusters": 3})
        cluster_counts = result.result_data["cluster"].value_counts()
        assert len(cluster_counts) == 3
        # each of the 3 planted groups had 30 rows
        for count in cluster_counts:
            assert 20 <= count <= 40   # allow some slack for boundary points

    def test_single_numeric_column_raises(self):
        plugin = ClusteringPlugin()
        df = pd.DataFrame({"x": range(20)})
        with pytest.raises(PluginValidationError):
            plugin.run(df, params={})

    def test_too_few_rows_raises(self):
        plugin = ClusteringPlugin()
        df = pd.DataFrame({"x": [1, 2, 3], "y": [1, 2, 3]})
        with pytest.raises(PluginValidationError):
            plugin.run(df, params={})

    def test_k_clamped_when_larger_than_reasonable_for_row_count(self):
        """Requesting 10 clusters on 12 rows should be clamped down,
        not crash or silently produce a nonsensical result."""
        plugin = ClusteringPlugin()
        df = pd.DataFrame({"x": range(12), "y": range(12)})
        result = plugin.run(df, params={"n_clusters": 10})
        assert result.summary_stats["n_clusters"] < 10

    def test_predict_output_has_cluster_column(self, three_blob_data):
        plugin = ClusteringPlugin()
        result = plugin.run(three_blob_data, params={"n_clusters": 3})
        assert "cluster" in result.result_data.columns
