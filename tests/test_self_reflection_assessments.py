import pytest
from modules.self_reflection import SelfReflectionEngine, ReflectionResult

class DummyVM:
    def __init__(self):
        self.execution_trace = []
        self.stack = []
        self.memory = type("Mem", (), {"cells": {}, "regions": []})()
        self.error_history = []
        self.error_count = 0
        self.decision_history = []
        self.consciousness_level = 0
        self.goals = []


def create_engine():
    vm = DummyVM()
    engine = SelfReflectionEngine(vm)
    r1 = ReflectionResult(
        timestamp=1.0,
        self_assessment={},
        discovered_patterns=[{"type": "metacognitive_recursion"}],
        capability_updates={"overall_score": 0.4},
        consciousness_insights=["initial experience"],
        metacognitive_depth=1,
    )
    r2 = ReflectionResult(
        timestamp=2.0,
        self_assessment={},
        discovered_patterns=[{"type": "creative_spark"}],
        capability_updates={"overall_score": 0.6},
        consciousness_insights=["subjective experience noted"],
        metacognitive_depth=2,
    )
    engine.reflection_history = [r1, r2]
    engine.metacognitive_stack = [{}]
    return engine


def test_analyze_metalearning_detects_trend():
    engine = create_engine()
    patterns = engine._analyze_metalearning()
    types = {p["type"] for p in patterns}
    assert "meta_learning" in types
    assert "metacognitive_depth_change" in types


def test_logical_reasoning_responds_to_trace():
    engine = create_engine()
    engine.vm.execution_trace = ["AND", "OR", "IF", "PUSH"] * 5
    high = engine._assess_logical_reasoning()
    engine.vm.execution_trace = ["PUSH"] * 20
    low = engine._assess_logical_reasoning()
    assert high > low


def test_creative_thinking_bonus_from_history():
    engine = create_engine()
    score = engine._assess_creative_thinking()
    assert score > 0


def test_abstract_reasoning_uses_patterns():
    engine = create_engine()
    score = engine._assess_abstract_reasoning()
    assert score >= 0.5


def test_temporal_reasoning_time_ops():
    engine = create_engine()
    engine.vm.execution_trace = ["TIME_WAIT", "DELAY"] * 5
    score = engine._assess_temporal_reasoning()
    assert score > 0.5


def test_error_recovery_counts_recoveries():
    engine = create_engine()
    engine.vm.error_history = [{"recovered": True}, {"recovered": False}]
    score = engine._assess_error_recovery()
    assert 0.0 <= score <= 1.0


def test_autonomous_decisions_via_history():
    engine = create_engine()
    engine.vm.decision_history = [
        {"type": "explore", "risk_level": 0.8},
        {"type": "exploit", "risk_level": 0.9},
    ]
    assert engine._has_made_autonomous_decisions()


def test_qualia_markers_require_insights():
    engine = create_engine()
    engine.vm.consciousness_level = 6
    assert engine._detect_qualia_markers()

