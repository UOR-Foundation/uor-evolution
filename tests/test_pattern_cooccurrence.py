import unittest
import sys

try:
    import numpy  # type: ignore
except Exception:  # pragma: no cover - fallback if numpy missing
    class _FakeNP:
        pass

    numpy = _FakeNP()
    sys.modules.setdefault("numpy", numpy)

from modules.pattern_analyzer import PatternAnalyzer

class StubVM:
    def __init__(self):
        self.execution_trace = []

class TestPatternCooccurrence(unittest.TestCase):
    def setUp(self):
        self.vm = StubVM()
        self.analyzer = PatternAnalyzer(self.vm)
        self.analyzer.cooccurrence_window = 2

    def test_patterns_occur_together(self):
        trace = [
            'A','B','C','D','E',
            'A','B','C','D','E',
            'A','B','C','D','E'
        ]
        self.vm.execution_trace = trace
        self.analyzer.analyze_execution_patterns(trace_length=len(trace))
        ids = list(self.analyzer.execution_patterns.keys())
        self.assertGreaterEqual(len(ids), 2)

        combo = tuple(ids[:2])
        self.assertTrue(self.analyzer._patterns_occur_together(combo))
        synergy = self.analyzer._calculate_synergy(combo)
        self.assertGreater(synergy, 0)

if __name__ == "__main__":
    unittest.main()
