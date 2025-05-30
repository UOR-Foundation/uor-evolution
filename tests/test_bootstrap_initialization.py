import unittest
import sys

try:
    import networkx  # type: ignore
except Exception:  # pragma: no cover - fallback if dependency missing
    class _FakeNX:  # minimal stub with DiGraph
        class DiGraph:
            def __init__(self, *a, **kw):
                pass

    networkx = _FakeNX()
    sys.modules.setdefault("networkx", networkx)

try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover - fallback if dependency missing
    class _FakeNP:
        pass

    np = _FakeNP()
    sys.modules.setdefault("numpy", np)

# Provide missing PrimeInstruction for Gödel loop imports if absent
import core.instruction_set as _instruction_set
if not hasattr(_instruction_set, "PrimeInstruction"):
    class PrimeInstruction:  # minimal placeholder
        pass

    _instruction_set.PrimeInstruction = PrimeInstruction

import builtins
if not hasattr(builtins, "StateTransitionManager"):
    class StateTransitionManager:  # minimal placeholder
        def __init__(self, *a, **kw):
            pass

        def get_possible_transitions(self, state):
            return []

        def execute_transition(self, transition):
            return None

    builtins.StateTransitionManager = StateTransitionManager

from core.prime_vm import ConsciousPrimeVM
from consciousness.consciousness_integration import ConsciousnessIntegrator
from modules.strange_loops.loop_detector import StrangeLoopDetector
from consciousness.perspective_engine import PerspectiveEngine


class TestBootstrapInitialization(unittest.TestCase):
    """Verify bootstrap initialization stage."""

    def test_initialize_components(self):
        vm = ConsciousPrimeVM()
        integrator = ConsciousnessIntegrator(vm)

        # Remove components to force bootstrap logic
        integrator.loop_detector = None
        integrator.loop_factory = None
        integrator.emergence_monitor = None
        integrator.multi_level_awareness = None
        integrator.recursive_self_model = None
        integrator.perspective_engine = None
        integrator.consciousness_core.loop_detector = None
        integrator.consciousness_core.emergence_monitor = None
        integrator.consciousness_core.multi_level_awareness = None
        integrator.consciousness_core.perspective_engine = None

        # Execute initialization stage
        integrator._execute_bootstrap_stage()

        self.assertIsInstance(integrator.loop_detector, StrangeLoopDetector)
        self.assertIsInstance(integrator.perspective_engine, PerspectiveEngine)
        self.assertIs(integrator.consciousness_core.loop_detector, integrator.loop_detector)
        self.assertIs(integrator.consciousness_core.perspective_engine, integrator.perspective_engine)


if __name__ == "__main__":
    unittest.main()
