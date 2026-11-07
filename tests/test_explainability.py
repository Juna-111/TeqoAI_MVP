import unittest

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler

from src.digital_twin import explain_event


class AdditiveInteractionModel:
    def decision_function(self, values):
        first = values[:, 0]
        second = values[:, 1]
        return -(first + second + 2 * first * second)


class ExactShapleyTests(unittest.TestCase):
    def test_exact_interventional_values_add_up_and_report_pair_interaction(self):
        background = pd.DataFrame(
            [[0, 0], [0, 0], [0, 0]],
            columns=["is_failed", "is_new_target_device"],
            index=[2, 4, 6],
        )
        features = pd.DataFrame(
            [[1, 1]],
            columns=background.columns,
            index=[9],
        )
        scaler = StandardScaler().fit(background)

        explanation = explain_event(
            AdditiveInteractionModel(),
            scaler,
            features,
            row_index=9,
            background=background,
        )

        self.assertAlmostEqual(explanation["base_value"], 0)
        self.assertAlmostEqual(explanation["prediction"], 4)
        self.assertAlmostEqual(explanation["drift"], 4)
        self.assertAlmostEqual(explanation["shap_sum"], 4)
        self.assertAlmostEqual(explanation["additivity_error"], 0, places=12)
        self.assertEqual(
            [item["feature"] for item in explanation["positive_drivers"]],
            ["is_failed", "is_new_target_device"],
        )
        for driver in explanation["positive_drivers"]:
            self.assertAlmostEqual(driver["shap_value"], 2)
        self.assertEqual(explanation["negative_drivers"], [])
        self.assertEqual(
            explanation["interactions"],
            [
                {
                    "feature_a": "Failed authentication activity",
                    "feature_b": "First-seen user-to-destination pairing",
                    "contribution": 2.0,
                }
            ],
        )

    def test_non_interacting_model_has_zero_pairwise_interaction(self):
        class LinearModel:
            def decision_function(self, values):
                return -values.sum(axis=1)

        background = pd.DataFrame(
            [[0, 0], [1, 0]],
            columns=["first", "second"],
        )
        features = pd.DataFrame(
            [[2, 3]],
            columns=background.columns,
        )

        explanation = explain_event(
            LinearModel(),
            StandardScaler().fit(background),
            features,
            row_index=0,
            background=background,
        )

        self.assertAlmostEqual(
            explanation["base_value"] + explanation["shap_sum"],
            explanation["prediction"],
        )
        self.assertEqual(explanation["interactions"], [])

    def test_rejects_feature_sets_too_large_for_exact_enumeration(self):
        features = pd.DataFrame(
            np.zeros((1, 13)),
            columns=[f"feature_{index}" for index in range(13)],
        )
        background = features.copy()

        with self.assertRaisesRegex(ValueError, "at most 12 semantic features"):
            explain_event(
                AdditiveInteractionModel(),
                StandardScaler().fit(background),
                features,
                row_index=0,
                background=background,
            )

    def test_rejects_non_finite_feature_values(self):
        features = pd.DataFrame({"feature": [np.nan]})
        background = pd.DataFrame({"feature": [0.0]})

        with self.assertRaisesRegex(ValueError, "must be finite"):
            explain_event(
                AdditiveInteractionModel(),
                StandardScaler().fit(background),
                features,
                row_index=0,
                background=background,
            )


if __name__ == "__main__":
    unittest.main()
