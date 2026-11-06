import unittest

from src.generate_synthetic_data import (
    NUM_DAYS,
    NUM_COMPROMISED_USERS,
    SECONDS_PER_DAY,
    TEST_DAY_START,
    generate_redteam_events,
)


class SyntheticEvaluationWindowTests(unittest.TestCase):
    def test_attack_chains_include_train_and_held_out_days(self):
        auth_events, labels = generate_redteam_events()
        heldout_start = TEST_DAY_START * SECONDS_PER_DAY
        heldout_labels = [row for row in labels if int(row[0]) >= heldout_start]
        training_labels = [row for row in labels if int(row[0]) < heldout_start]

        self.assertGreater(NUM_DAYS, TEST_DAY_START)
        self.assertGreater(NUM_COMPROMISED_USERS, 1)
        self.assertEqual(len(auth_events), len(labels))
        self.assertTrue(training_labels)
        self.assertTrue(heldout_labels)
        self.assertEqual(len(labels), NUM_COMPROMISED_USERS * 3)


if __name__ == "__main__":
    unittest.main()
