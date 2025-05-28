import unittest
from backend.adaptive_teacher import (
    PerformanceMonitor,
    SequenceGenerator,
    AdaptiveCurriculum,
    AdaptiveTeacher,
)

class TestSequenceGenerator(unittest.TestCase):
    def test_fibonacci(self):
        self.assertEqual(SequenceGenerator.fibonacci(6), [0, 1, 1, 2, 3, 5])

    def test_primes(self):
        self.assertEqual(SequenceGenerator.primes(5), [2, 3, 5, 7, 11])

class TestPerformanceMonitor(unittest.TestCase):
    def test_success_rate(self):
        mon = PerformanceMonitor()
        for _ in range(3):
            mon.record_attempt(True)
        self.assertEqual(mon.success_rate(), 1.0)
        mon.record_attempt(False)
        self.assertAlmostEqual(mon.success_rate(), 0.75)

class TestAdaptiveTeacher(unittest.TestCase):
    def test_adjust_difficulty(self):
        teacher = AdaptiveTeacher()
        for _ in range(5):
            teacher.record_attempt(0, True)
        self.assertEqual(teacher.difficulty, "HARD")
        # Simulate many failures
        teacher.monitor.successes = 0
        teacher.monitor.total_attempts = 10
        teacher.monitor.attempts_per_success = []
        teacher.monitor.current_attempts = 10
        teacher._adjust_difficulty()
        self.assertEqual(teacher.difficulty, "EASY")

if __name__ == '__main__':
    unittest.main()
