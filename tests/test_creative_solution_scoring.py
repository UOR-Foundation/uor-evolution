import unittest

from modules.consciousness_validator import ConsciousnessValidator


class TestCreativeSolutionScoring(unittest.TestCase):
    def setUp(self):
        # Validator does not require a fully functional VM for this test
        self.validator = ConsciousnessValidator(None)

    def test_optimization_solution(self):
        problem = {
            "type": "optimization",
            "data": {"nodes": 10, "edges": 15},
        }
        solution = {
            "solution": list(range(10)),
            "confidence": 0.6,
            "approach": "naive_path",
            "iterations": 1,
        }
        eval = self.validator._evaluate_creative_solution(problem, solution)
        self.assertAlmostEqual(eval["novelty_score"], 0.4)
        self.assertAlmostEqual(eval["usefulness_score"], 0.8)
        self.assertAlmostEqual(eval["surprise_factor"], 0.4)

    def test_pattern_solution(self):
        problem = {
            "type": "pattern",
            "sequence": [1, 2, 3],
        }
        solution = {
            "solution": [1, 2, 3, 5, 7, 11],
            "confidence": 0.8,
            "approach": "prime_sequence",
            "iterations": 3,
        }
        eval = self.validator._evaluate_creative_solution(problem, solution)
        self.assertAlmostEqual(eval["novelty_score"], 0.5)
        self.assertAlmostEqual(eval["usefulness_score"], 1.0)
        self.assertAlmostEqual(eval["surprise_factor"], 0.6)

    def test_abstract_solution(self):
        problem = {
            "type": "abstract",
        }
        solution = {
            "solution": "Consciousness is awareness of experience",
            "confidence": 0.4,
            "approach": "simple_definition",
            "iterations": 1,
        }
        eval = self.validator._evaluate_creative_solution(problem, solution)
        self.assertAlmostEqual(eval["novelty_score"], 0.6)
        self.assertAlmostEqual(eval["usefulness_score"], 0.4)
        self.assertAlmostEqual(eval["surprise_factor"], 0.8)


if __name__ == "__main__":
    unittest.main()
