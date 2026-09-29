import unittest

from chronoguard.common import InputError
from chronoguard.statistics import loss_comparison


def row(target: float, model: float, baseline: float):
    return {"target": target, "prediction": model, "train_mean_baseline": baseline}


class LossComparisonTests(unittest.TestCase):
    def test_improvement_interval_and_determinism(self):
        rows = [row(float(index), float(index), float(index + 2)) for index in range(20)]
        first = loss_comparison(rows, repetitions=500, block_size=4, seed=23)
        self.assertEqual(first, loss_comparison(rows, repetitions=500, block_size=4, seed=23))
        self.assertEqual(first["mean_loss_delta"], -4)
        self.assertEqual(first["interval_95"], [-4, -4])

    def test_small_sample_does_not_claim_interval(self):
        result = loss_comparison([row(1, 1, 2), row(2, 2, 3)], repetitions=100, block_size=1)
        self.assertIsNone(result["interval_95"])
        self.assertEqual(result["interval_unavailable_reason"], "fewer_than_10_held_out_predictions")

    def test_rejects_bad_controls(self):
        rows = [row(1, 1, 2)]
        for options in ({"repetitions": 99}, {"block_size": 0}, {"block_size": 2}, {"seed": -1}):
            with self.subTest(options=options), self.assertRaises(InputError):
                loss_comparison(rows, **options)


if __name__ == "__main__":
    unittest.main()
