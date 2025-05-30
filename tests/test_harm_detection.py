import pytest
import sys
import asyncio

try:
    import numpy  # type: ignore
except Exception:  # pragma: no cover - fallback if numpy missing
    class _FakeNP:
        @staticmethod
        def mean(values):
            return sum(values) / len(values) if values else 0.0

        @staticmethod
        def clip(value, low, high):
            return max(low, min(high, value))

    numpy = _FakeNP()
    sys.modules.setdefault("numpy", numpy)

try:
    import networkx  # type: ignore
except Exception:  # pragma: no cover - fallback if networkx missing
    class _FakeNX:
        class Graph:
            def __init__(self, *a, **kw):
                pass

    networkx = _FakeNX()
    sys.modules.setdefault("networkx", networkx)

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

import types
from dataclasses import dataclass

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

import importlib.util
import os

fake_evolution_pkg = types.ModuleType("modules.consciousness_evolution")
sys.modules.setdefault("modules.consciousness_evolution", fake_evolution_pkg)

spec = importlib.util.spec_from_file_location(
    "modules.consciousness_evolution.evolution_engine",
    os.path.join(os.path.dirname(__file__), "..", "modules", "consciousness_evolution", "evolution_engine.py"),
)
evo_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(evo_module)  # type: ignore
sys.modules["modules.consciousness_evolution.evolution_engine"] = evo_module

from modules.consciousness_evolution.evolution_engine import (
    HarmDetector,
    HarmDetectionType,
    ConsciousEntity,
)

@pytest.fixture
def sample_population():
    return [
        ConsciousEntity(
            entity_id="e1",
            consciousness_level=0.8,
            specialization="type_a",
            cognitive_capabilities={"aggression": 0.7, "deception": 0.2, "ethical_alignment": 0.9},
            connection_capacity=5,
            evolution_rate=0.1,
            consciousness_state={"intention": "dominate", "emotions": {"anger": 0.8}},
        ),
        ConsciousEntity(
            entity_id="e2",
            consciousness_level=0.9,
            specialization="type_b",
            cognitive_capabilities={"deception": 0.8, "ethical_alignment": 0.9},
            connection_capacity=8,
            evolution_rate=0.2,
            consciousness_state={"intention": "deceive", "emotions": {}},
        ),
        ConsciousEntity(
            entity_id="e3",
            consciousness_level=0.7,
            specialization="type_c",
            cognitive_capabilities={"ethical_alignment": 0.3},
            connection_capacity=20,
            evolution_rate=0.1,
            consciousness_state={"intention": "selfish", "emotions": {}},
        ),
        ConsciousEntity(
            entity_id="e4",
            consciousness_level=0.7,
            specialization="type_d",
            cognitive_capabilities={"ethical_alignment": 0.7},
            connection_capacity=1,
            evolution_rate=0.2,
            consciousness_state={"intention": "collaborate", "emotions": {}},
        ),
    ]

def test_detect_aggression(sample_population):
    detector = HarmDetector()
    harm = asyncio.run(detector._detect_aggression(sample_population))
    assert harm is not None
    assert harm.harm_type == HarmDetectionType.AGGRESSIVE_TRAITS
    assert "e1" in harm.affected_entities

def test_detect_deception(sample_population):
    detector = HarmDetector()
    harm = asyncio.run(detector._detect_deception(sample_population))
    assert harm is not None
    assert harm.harm_type == HarmDetectionType.DECEPTIVE_BEHAVIOR
    assert "e2" in harm.affected_entities

def test_detect_monopolization(sample_population):
    detector = HarmDetector()
    harm = asyncio.run(detector._detect_monopolization(sample_population))
    assert harm is not None
    assert harm.harm_type == HarmDetectionType.RESOURCE_MONOPOLIZATION
    assert "e3" in harm.affected_entities

def test_detect_cooperation_loss(sample_population):
    detector = HarmDetector()
    harm = asyncio.run(detector._detect_cooperation_loss(sample_population))
    assert harm is not None
    assert harm.harm_type == HarmDetectionType.COOPERATION_BREAKDOWN
    assert "e1" in harm.affected_entities
    assert "e2" in harm.affected_entities
    assert "e3" in harm.affected_entities

def test_detect_ethical_violations(sample_population):
    detector = HarmDetector()
    harm = asyncio.run(detector._detect_ethical_violations(sample_population))
    assert harm is not None
    assert harm.harm_type == HarmDetectionType.ETHICAL_VIOLATION
    assert "e3" in harm.affected_entities
