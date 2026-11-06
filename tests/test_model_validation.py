import sys
import unittest
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from model import (  # noqa: E402
    action_categories,
    align_features,
    build_features,
    chronological_validation_splits,
    temporal_split,
)
from digital_twin import build_graph, format_feature_name, get_attack_paths  # noqa: E402
from risk_engine import compute_risk  # noqa: E402


class ModelValidationTests(unittest.TestCase):
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

    def test_align_features_uses_training_schema_and_zero_fills_missing_columns(self):
        features = pd.DataFrame({"extra": [9], "kept": [2]})

        aligned = align_features(features, ["kept", "unseen"])

        self.assertEqual(list(aligned.columns), ["kept", "unseen"])
        self.assertEqual(aligned.iloc[0].tolist(), [2, 0])

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
            }
        )

        paths = get_attack_paths(build_graph(events), "C1", max_hops=2)

        self.assertEqual(paths[0]["target"], "DB-CORE")
        self.assertEqual(paths[0]["hops"], 2)
        self.assertEqual(paths[0]["path"], ["C1", "SRV1", "DB-CORE"])
        one_hop = get_attack_paths(build_graph(events), "C1", max_hops=1)
        self.assertEqual(one_hop[0]["target"], "SRV1")

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
