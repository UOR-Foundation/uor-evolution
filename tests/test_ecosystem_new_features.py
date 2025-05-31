import sys
import asyncio

try:
    import numpy as np
except Exception:  # pragma: no cover - allow running without numpy
    class _FakeRandom:
        @staticmethod
        def normal(loc=0.0, scale=1.0):
            return 0.1

    class _FakeNP:
        random = _FakeRandom()

        @staticmethod
        def normal(loc=0.0, scale=1.0):
            return 0.1

        @staticmethod
        def log(x):
            return 0.0

        ndarray = list
    np = _FakeNP()
    sys.modules.setdefault("numpy", np)

class _FakeGraph:
    def __init__(self, nodes=0, edges=0):
        self._nodes = nodes
        self._edges = edges

    def number_of_nodes(self):
        return self._nodes

    def number_of_edges(self):
        return self._edges


class _FakeNX:
    Graph = _FakeGraph

sys.modules.setdefault("networkx", _FakeNX())

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


import types
import importlib.util
import os

fake_uc_pkg = types.ModuleType("modules.unified_consciousness")
class _DummyOrch:
    pass
fake_uc_pkg.ConsciousnessOrchestrator = _DummyOrch
sys.modules.setdefault("modules.unified_consciousness", fake_uc_pkg)

spec = importlib.util.spec_from_file_location(
    "modules.consciousness_ecosystem.ecosystem_orchestrator",
    os.path.join(os.path.dirname(__file__), "..", "modules", "consciousness_ecosystem", "ecosystem_orchestrator.py"),
)
eco_module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(eco_module)  # type: ignore
sys.modules["modules.consciousness_ecosystem.ecosystem_orchestrator"] = eco_module

ConsciousEntity = eco_module.ConsciousEntity
ConsciousnessEcosystemOrchestrator = eco_module.ConsciousnessEcosystemOrchestrator
EmergenceMonitor = eco_module.EmergenceMonitor
EvolutionEngine = eco_module.EvolutionEngine
EmergentPropertyType = eco_module.EmergentPropertyType


def _make_entities(count=3):
    entities = []
    for i in range(count):
        entities.append(
            ConsciousEntity(
                entity_id=f"e{i}",
                consciousness_level=0.5,
                specialization="t",
                cognitive_capabilities={"cap": 0.5},
                connection_capacity=2,
                evolution_rate=0.1,
                consciousness_state={"beh": i},
            )
        )
    return entities


def test_innovation_rate_tracking():
    entities = _make_entities(3)
    orch = ConsciousnessEcosystemOrchestrator(entities)
    orch._measure_innovation_rate()  # initialize
    assert orch._measure_innovation_rate() == 0.0

    new_entity = ConsciousEntity(
        entity_id="new",
        consciousness_level=0.5,
        specialization="t",
        cognitive_capabilities={"cap": 0.5},
        connection_capacity=2,
        evolution_rate=0.1,
        consciousness_state={"beh": "new"},
    )
    orch.consciousness_nodes[new_entity.entity_id] = new_entity
    rate = orch._measure_innovation_rate()
    assert rate > 0


def test_emergence_monitor_patterns():
    monitor = EmergenceMonitor()
    state = {"total_consciousness": 5.0, "network_count": 2, "emergence_level": 0.6}
    props = []
    for _ in range(3):
        props = asyncio.run(monitor.detect_emergence(state))
    assert any(p.property_type == EmergentPropertyType.COLLECTIVE_INTELLIGENCE for p in props)


def test_evolution_engine_basic_evolution():
    population = _make_entities(4)
    engine = EvolutionEngine()
    new_gen = asyncio.run(engine.evolve_generation(population))
    assert 0 < len(new_gen) <= len(population)
    changed = any(n.consciousness_level != p.consciousness_level for n, p in zip(new_gen, population))
    assert changed
