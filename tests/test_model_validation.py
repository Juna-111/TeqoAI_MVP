import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np
import pandas as pd
try:
    import torch
except ImportError:
    torch = None

from src import model

from src.model import (
    action_categories,
    align_features,
    build_features,
    chronological_validation_splits,
    load_autoencoder_artifacts,
    save_autoencoder_artifacts,
    score_autoencoder,
    temporal_split,
    train_autoencoder,
)
from src.digital_twin import build_graph, format_feature_name, get_attack_paths
from src.risk_engine import compute_risk


class ModelValidationTests(unittest.TestCase):
    def test_autoencoder_is_optional_when_pytorch_is_unavailable(self):
        events = pd.DataFrame({"feature": [0.0, 1.0]})
        original_import = __import__

        def import_without_torch(name, *args, **kwargs):
            if name == "torch":
                raise ModuleNotFoundError("PyTorch is not installed")
            return original_import(name, *args, **kwargs)

        with patch("builtins.__import__", side_effect=import_without_torch):
            autoencoder, scores, scaler = train_autoencoder(events, events)

        self.assertIsNone(autoencoder)
        self.assertIsNone(scores)
        self.assertIsNone(scaler)

    def test_future_action_categories_are_not_learned_from_training_rows(self):
        events = pd.DataFrame(
            [
                ["windows_auth", 1, "u1", "c1", "s1", "Kerberos/Network", "Success", 0],
                ["windows_auth", 2, "u1", "c1", "s2", "NovelAuth/NovelLogon", "Success", 1],
            ],
            columns=[
                "source", "time", "user", "device", "target_device",
                "action", "result", "is_malicious",
            ],
        )
        categories = action_categories(events.iloc[:1])

        features, _ = build_features(events, categories)

        self.assertNotIn("auth_NovelAuth", features.columns)
        self.assertNotIn("logon_NovelLogon", features.columns)
        self.assertEqual(features.iloc[1]["auth_Kerberos"], 0)
        self.assertEqual(features.iloc[1]["logon_Network"], 0)

    def test_time_features_are_causal_and_independent_of_simultaneous_row_order(self):
        events = pd.DataFrame(
            [
                ["windows_auth", 20, "u1", "c3", "s2", "Kerberos/Network", "Success", 0],
                ["windows_auth", 10, "u1", "c2", "s1", "Kerberos/Network", "Success", 0],
                ["windows_auth", 10, "u1", "c1", "s1", "Kerberos/Network", "Success", 0],
            ],
            columns=[
                "source", "time", "user", "device", "target_device",
                "action", "result", "is_malicious",
            ],
        )

        features, prepared = build_features(events)

        same_time = prepared["time"].eq(10)
        self.assertEqual(features.loc[same_time, "rolling_count_1h"].tolist(), [1, 1])
        self.assertEqual(features.loc[same_time, "is_new_target_device"].tolist(), [1, 1])
        self.assertEqual(features.loc[prepared["time"].eq(20), "rolling_count_1h"].item(), 3)
        self.assertEqual(features.loc[prepared["time"].eq(20), "is_new_target_device"].item(), 1)

        shuffled = events.iloc[[2, 1, 0]].reset_index(drop=True)
        shuffled_features, shuffled_prepared = build_features(shuffled)
        pd.testing.assert_frame_equal(prepared, shuffled_prepared)
        pd.testing.assert_frame_equal(features, shuffled_features)

    def test_align_features_uses_training_schema_and_zero_fills_missing_columns(self):
        features = pd.DataFrame({"extra": [9], "kept": [2]})

        aligned = align_features(features, ["kept", "unseen"])

        self.assertEqual(list(aligned.columns), ["kept", "unseen"])
        self.assertEqual(aligned.iloc[0].tolist(), [2, 0])

    @unittest.skipIf(torch is None, "optional PyTorch dependency is not installed")
    def test_autoencoder_training_is_reproducible_and_artifacts_support_batched_scoring(self):
        events = pd.DataFrame(
            {
                "feature_a": [0.0, 0.1, 0.2, 1.0, 1.1],
                "feature_b": [1.0, 1.2, 1.1, 0.0, 0.2],
            }
        )

        torch_rng_before = torch.random.get_rng_state().clone()
        model, scores, scaler = train_autoencoder(
            events, events, epochs=2, random_state=17, batch_size=2
        )
        self.assertTrue(torch.equal(torch.random.get_rng_state(), torch_rng_before))
        repeated_model, repeated_scores, _ = train_autoencoder(
            events, events, epochs=2, random_state=17, batch_size=2
        )
        self.assertTrue(np.allclose(scores, repeated_scores))

        with tempfile.TemporaryDirectory() as directory:
            model_path = Path(directory) / "autoencoder.pt"
            scaler_path = Path(directory) / "autoencoder_scaler.pkl"
            save_autoencoder_artifacts(model, scaler, model_path, scaler_path)
            loaded_model, loaded_scaler = load_autoencoder_artifacts(
                model_path, scaler_path
            )
            loaded_scores = score_autoencoder(
                loaded_model, loaded_scaler, events, batch_size=2
            )

        self.assertTrue(np.allclose(scores, loaded_scores))
        self.assertEqual(len(repeated_scores), len(events))
        self.assertEqual(repeated_model.training, model.training)

    def test_walk_forward_windows_are_ordered_and_stay_inside_training_period(self):
        events = pd.DataFrame(
            {
                "time": list(range(1, 41)),
                "is_malicious": [0] * 40,
            }
        )
        training_mask, final_test_mask, _ = temporal_split(events)

        folds = chronological_validation_splits(events, training_mask, n_splits=3)

        self.assertEqual(len(folds), 3)
        for train_mask, validation_mask in folds:
            self.assertTrue((events.loc[train_mask, "time"].max()
                             < events.loc[validation_mask, "time"].min()))
            self.assertFalse((validation_mask & final_test_mask).any())

    def test_walk_forward_comparison_uses_identical_folds_for_each_model(self):
        events = pd.DataFrame(
            [
                [
                    "windows_auth", time, "u1", "C1", f"SRV{time % 3}",
                    "Kerberos/Network", "Success", int(time % 7 == 0),
                ]
                for time in range(1, 41)
            ],
            columns=[
                "source", "time", "user", "device", "target_device",
                "action", "result", "is_malicious",
            ],
        )

        with patch("src.model.AUTOENCODER_EPOCHS", 2):
            report = model.evaluate_walk_forward(events, n_splits=2)

        self.assertEqual(report["completed_folds"], 2)
        expected_models = {
            "first_seen_destination_baseline",
            "isolation_forest",
            "autoencoder",
        }
        for fold in report["folds"]:
            self.assertEqual(set(fold["models"]), expected_models)
            self.assertEqual(
                {metrics["test_events"] for metrics in fold["models"].values()},
                {fold["validation_events"]},
            )

    def test_risk_engine_emits_explainable_prioritization_factors(self):
        events = pd.DataFrame(
            {
                "anomaly_score_iforest": [0.1, 0.9],
                "is_new_target_device": [0, 1],
                "target_device": ["C1", "DB-CORE"],
                "is_failed": [0, 1],
                "rolling_count_1h": [1, 6],
            }
        )

        scored = compute_risk(events)

        self.assertEqual(scored.iloc[1]["risk_level"], "High")
        self.assertTrue(scored.iloc[1]["factor_new_device"])
        self.assertTrue(scored.iloc[1]["factor_sensitive_asset"])
        self.assertIn("sensitive asset", scored.iloc[1]["risk_reasons"])
        self.assertGreaterEqual(scored["risk_score"].max(), 0)
        self.assertLessEqual(scored["risk_score"].max(), 100)

    def test_observed_graph_paths_are_bounded_and_rank_critical_assets(self):
        events = pd.DataFrame(
            {
                "device": ["C1", "SRV1"],
                "target_device": ["SRV1", "DB-CORE"],
                "result": ["Success", "Success"],
            }
        )

        paths = get_attack_paths(build_graph(events), "C1", max_hops=2)

        self.assertEqual(paths[0]["target"], "DB-CORE")
        self.assertEqual(paths[0]["hops"], 2)
        self.assertEqual(paths[0]["path"], ["C1", "SRV1", "DB-CORE"])
        one_hop = get_attack_paths(build_graph(events), "C1", max_hops=1)
        self.assertEqual(one_hop[0]["target"], "SRV1")

    def test_failed_authentication_attempts_do_not_create_observed_paths(self):
        events = pd.DataFrame(
            {
                "device": ["C1", "SRV1", "C2"],
                "target_device": ["SRV1", "DB-CORE", "APP-HR"],
                "result": ["Failure", "Success", " failed "],
            }
        )

        graph = build_graph(events)

        self.assertNotIn("C1", graph)
        self.assertNotIn(("C1", "SRV1"), graph.edges)
        self.assertNotIn("C2", graph)
        self.assertEqual(list(graph.edges), [("SRV1", "DB-CORE")])

    def test_observed_graph_requires_authentication_outcomes(self):
        events = pd.DataFrame(
            {"device": ["C1"], "target_device": ["SRV1"]}
        )

        with self.assertRaisesRegex(ValueError, "Authentication outcomes are required"):
            build_graph(events)

    def test_encoded_features_have_analyst_friendly_names(self):
        self.assertEqual(
            format_feature_name("is_new_target_device"),
            "First-seen user-to-destination pairing",
        )
        self.assertEqual(
            format_feature_name("logon_Network"),
            "Logon category: Network",
        )


if __name__ == "__main__":
    unittest.main()
