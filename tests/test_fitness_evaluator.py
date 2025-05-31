import pytest
import numpy as np
import importlib.util
import os

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
