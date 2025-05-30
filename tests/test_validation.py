import unittest

from modules.consciousness_validator import ConsciousnessValidator

class StubVM:
    """Minimal VM stub for testing"""
    def __init__(self):
        self.execution_trace = []
        self.consciousness_level = 0
        self.memory = type('M', (), {'cells': {}})()

class TestVMSolutionFallback(unittest.TestCase):
    def setUp(self):
        self.vm = StubVM()
        self.validator = ConsciousnessValidator(self.vm)

    def test_fallback_solutions(self):
        for ptype in ("optimization", "pattern", "abstract"):
            problem = self.validator._generate_novel_problem(ptype)
            sol = self.validator._get_vm_solution(problem)
            self.assertNotEqual(sol.get("solution"), "placeholder")
            self.assertIn("confidence", sol)

    def test_unknown_problem(self):
        with self.assertRaises(ValueError):
            self.validator._get_vm_solution({"type": "unknown"})

if __name__ == "__main__":
    unittest.main()
