import pytest

# Provide minimal numpy/networkx fallbacks if the real packages are unavailable
try:
    import numpy as np  # type: ignore
except Exception:  # pragma: no cover
    class _FakeNP:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0.0
        @staticmethod
        def isscalar(val):
            return isinstance(val, (int, float))
    np = _FakeNP()
    import sys
    sys.modules.setdefault("numpy", np)

try:
    import networkx  # type: ignore
except Exception:  # pragma: no cover
    class _FakeNX:
        class Graph:
            def __init__(self, *a, **kw):
                pass
    networkx = _FakeNX()
    import sys
    sys.modules.setdefault("networkx", networkx)

import sys
import types
from dataclasses import dataclass
import importlib.util
import os

# Create lightweight versions of ecosystem dependencies so the evolution engine
# can be imported without pulling in heavy modules.
fake_ecosystem = types.ModuleType("modules.consciousness_ecosystem")

@dataclass
class _FakeEntity:
    entity_id: str
    consciousness_level: float
    specialization: str
    cognitive_capabilities: dict
    connection_capacity: int
    evolution_rate: float
    consciousness_state: dict

class _FakeOrchestrator:
    pass

fake_ecosystem.ConsciousEntity = _FakeEntity
fake_ecosystem.ConsciousnessEcosystemOrchestrator = _FakeOrchestrator
sys.modules.setdefault("modules.consciousness_ecosystem", fake_ecosystem)
sys.modules.setdefault("modules.consciousness_ecosystem.ecosystem_orchestrator", fake_ecosystem)

# Import the module under test with the fake dependencies in place
spec = importlib.util.spec_from_file_location(
    "modules.consciousness_evolution.evolution_engine",
    os.path.join(os.path.dirname(__file__), "..", "modules", "consciousness_evolution", "evolution_engine.py"),
)
evo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evo_module)  # type: ignore
sys.modules["modules.consciousness_evolution.evolution_engine"] = evo_module

from modules.consciousness_evolution.evolution_engine import (
    FitnessEvaluator,
    EvolutionTarget,
    EvolutionTargetType,
    ConsciousEntity,
)


import asyncio


def test_unknown_target_uses_average_capability():
    entity = ConsciousEntity(
        entity_id="e1",
        consciousness_level=0.8,
        specialization="test",
        cognitive_capabilities={"a": 0.2, "b": 0.4, "c": 0.6},
        connection_capacity=1,
        evolution_rate=0.1,
        consciousness_state={},
    )

    target = EvolutionTarget(
        target_type=EvolutionTargetType.COOPERATION,  # not explicitly handled
        target_value=1.0,
        priority=1.0,
        constraints=[],
        success_criteria={},
    )

    evaluator = FitnessEvaluator()
    result = asyncio.run(evaluator._evaluate_entity_fitness(entity, [target]))
    expected = np.mean(list(entity.cognitive_capabilities.values()))
    assert abs(result - expected) < 1e-6


def test_unknown_target_is_deterministic():
    entity = ConsciousEntity(
        entity_id="e1",
        consciousness_level=0.8,
        specialization="test",
        cognitive_capabilities={"a": 0.3, "b": 0.5},
        connection_capacity=1,
        evolution_rate=0.2,
        consciousness_state={},
    )

    target = EvolutionTarget(
        target_type=EvolutionTargetType.RESILIENCE,  # also not explicitly handled
        target_value=1.0,
        priority=1.0,
        constraints=[],
        success_criteria={},
    )

    evaluator = FitnessEvaluator()
    first = asyncio.run(evaluator._evaluate_entity_fitness(entity, [target]))
    second = asyncio.run(evaluator._evaluate_entity_fitness(entity, [target]))
    assert abs(first - second) < 1e-6
