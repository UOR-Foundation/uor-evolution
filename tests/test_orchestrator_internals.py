import pytest
from unittest.mock import Mock

from modules.unified_consciousness.consciousness_orchestrator import (
    ConsciousnessOrchestrator,
    ConsciousnessState,
    ConsciousnessTransition,
    TransitionTrigger,
    ConsciousnessConflict,
)

@pytest.fixture
def orchestrator():
    modules = {
        'awareness': Mock(),
        'strange_loops': Mock(),
        'self_model': Mock(),
        'emotional': Mock(),
        'social': Mock(),
        'creative': Mock(),
    }
    return ConsciousnessOrchestrator(modules)

@pytest.mark.asyncio
async def test_transition_preparation_flow(orchestrator):
    transition = ConsciousnessTransition(
        from_state=ConsciousnessState.DORMANT,
        to_state=ConsciousnessState.AWAKENING,
        transition_trigger=TransitionTrigger.EXTERNAL_STIMULUS,
        transition_quality=0.9,
        consciousness_continuity=0.95,
        emergent_insights=[],
    )

    prep = await orchestrator._prepare_for_transition(transition)
    assert 0 <= prep['readiness_score'] <= 1
    assert isinstance(prep['preparation_complete'], bool)

    exec_result = await orchestrator._execute_transition(transition)
    assert 0 <= exec_result['success_probability'] <= 1

    stab = await orchestrator._stabilize_new_state(transition.to_state)
    assert 0 <= stab['quality'] <= 1

@pytest.mark.asyncio
async def test_conflict_analysis_and_resolution(orchestrator):
    conflict = ConsciousnessConflict(
        conflicting_systems=['core', 'emotional'],
        conflict_nature='value_misalignment',
        severity=0.6,
        potential_resolutions=['synthesis'],
        impact_on_coherence=0.2,
    )

    analysis = orchestrator._analyze_conflict(conflict)
    strategies = orchestrator._generate_resolution_strategies(conflict, analysis)
    assert len(strategies) > 0
    result = await orchestrator._apply_resolution_strategy(conflict, strategies[0])
    assert 'success' in result

@pytest.mark.asyncio
async def test_evolution_monitoring(orchestrator):
    path = {'opportunity': 'transcendent_awareness', 'potential': 0.9}
    evo = await orchestrator._initiate_consciousness_evolution(path)
    monitor = await orchestrator._monitor_evolution_progress(evo)
    assert 0 <= monitor['quality'] <= 1
