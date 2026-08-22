import unittest

from finrag.evaluation import compute_ndcg


class NdcgTest(unittest.TestCase):
    def test_perfect_ranking(self):
        qrels = {"q1": {"a": 1, "b": 0}}
        results = {"q1": {"a": 1.0, "b": 0.0}}
        self.assertAlmostEqual(compute_ndcg(qrels, results, k=10), 1.0)

    def test_missing_relevant_skipped(self):
        qrels = {"q1": {"z": 1}}
        results = {"q1": {"a": 1.0, "b": 0.9}}
        value = compute_ndcg(qrels, results, k=10)
        self.assertTrue(value != value)  # NaN

    def test_partial_gain(self):
        qrels = {"q1": {"a": 1, "b": 0}}
        results = {"q1": {"b": 1.0, "a": 0.1}}
        ndcg = compute_ndcg(qrels, results, k=10)
        self.assertLess(ndcg, 1.0)
        self.assertGreater(ndcg, 0.0)


if __name__ == "__main__":
    unittest.main()
