import unittest

from benchmarks.join_audit import benchmark


class BenchmarkSmokeTests(unittest.TestCase):
    def test_small_benchmark_checks_join_and_audit(self):
        result = benchmark(rows=30, entities=5)
        self.assertEqual(result["rows"], 30)
        self.assertGreaterEqual(result["join"]["seconds"], 0)
        self.assertGreaterEqual(result["audit"]["seconds"], 0)

    def test_invalid_dimensions(self):
        with self.assertRaises(ValueError):
            benchmark(rows=10, entities=11)


if __name__ == "__main__":
    unittest.main()
