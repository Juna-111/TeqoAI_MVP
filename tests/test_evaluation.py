import unittest

import numpy as np

from src.evaluation import evaluate_ranked_scores


class EvaluateRankedScoresTests(unittest.TestCase):
    def test_computes_fixed_budget_and_ranking_metrics(self):
        result = evaluate_ranked_scores(
            [1, 0, 1, 0],
            [0.9, 0.8, 0.7, 0.1],
            alert_budgets=(2,),
        )

        self.assertEqual(result["status"], "evaluated")
        self.assertAlmostEqual(result["average_precision"], 5 / 6)
        self.assertAlmostEqual(result["roc_auc"], 0.75)
        self.assertEqual(result["alert_budgets"]["2"]["known_positive_hits"], 1)
        self.assertAlmostEqual(result["alert_budgets"]["2"]["precision"], 0.5)
        self.assertAlmostEqual(result["alert_budgets"]["2"]["recall"], 0.5)
        self.assertAlmostEqual(
            result["alert_budgets"]["2"]["random_baseline_expected_precision"], 0.5
        )

    def test_no_test_positives_is_explicitly_not_evaluable(self):
        result = evaluate_ranked_scores([0, 0], [0.9, 0.1])

        self.assertEqual(result["status"], "not_evaluable_no_test_positives")
        self.assertIsNone(result["average_precision"])
        self.assertIsNone(result["roc_auc"])
        self.assertEqual(result["alert_budgets"], {})

    def test_empty_test_period_is_explicitly_not_evaluable(self):
        result = evaluate_ranked_scores([], [])

        self.assertEqual(result["status"], "not_evaluable_no_test_events")
        self.assertEqual(result["test_events"], 0)

    def test_tied_scores_use_stable_input_order(self):
        result = evaluate_ranked_scores([1, 0], [0.5, 0.5], alert_budgets=(1,))

        self.assertEqual(result["alert_budgets"]["1"]["known_positive_hits"], 1)

    def test_budget_larger_than_test_set_selects_each_test_event_once(self):
        result = evaluate_ranked_scores([1, 0], [0.1, 0.9], alert_budgets=(30,))

        self.assertEqual(result["alert_budgets"]["30"]["selected_events"], 2)

    def test_invalid_inputs_are_rejected(self):
        with self.assertRaises(ValueError):
            evaluate_ranked_scores([1], [0.1, 0.2])
        with self.assertRaises(ValueError):
            evaluate_ranked_scores([np.nan], [0.1])
        with self.assertRaises(ValueError):
            evaluate_ranked_scores([2], [0.1])
        with self.assertRaises(ValueError):
            evaluate_ranked_scores([1], [np.inf])
        with self.assertRaises(ValueError):
            evaluate_ranked_scores([1], [0.1], alert_budgets=(True,))


if __name__ == "__main__":
    unittest.main()
