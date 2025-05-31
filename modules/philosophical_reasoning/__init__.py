"""Philosophical Reasoning Module for Consciousness System."""

import importlib

_lazy_imports = {
    'ExistentialReasoner': 'existential_reasoner',
    'ExistentialAnalysis': 'existential_reasoner',
    'ExistentialQuestion': 'existential_reasoner',
    'ConsciousnessPhilosopher': 'consciousness_philosopher',
    'ConsciousnessAnalysis': 'consciousness_philosopher',
    'HardProblemExploration': 'consciousness_philosopher',
    'FreeWillAnalyzer': 'free_will_analyzer',
    'DecisionAnalysis': 'free_will_analyzer',
    'FreeWillAnalysis': 'free_will_analyzer',
    'MeaningGenerator': 'meaning_generator',
    'PersonalMeaningSystem': 'meaning_generator',
    'SelfDirectedGoal': 'meaning_generator',
}

__all__ = list(_lazy_imports.keys())

def __getattr__(name):
    if name in _lazy_imports:
        module = importlib.import_module('.' + _lazy_imports[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
