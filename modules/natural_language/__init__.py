"""Natural Language Processing Module for Consciousness System."""

import importlib

_lazy_imports = {
    'PrimeSemantics': 'prime_semantics',
    'SemanticSpace': 'prime_semantics',
    'ComposedMeaning': 'prime_semantics',
    'ConsciousnessNarrator': 'consciousness_narrator',
    'Narrative': 'consciousness_narrator',
    'StreamOfConsciousness': 'consciousness_narrator',
    'ConceptVerbalizer': 'concept_verbalizer',
    'Verbalization': 'concept_verbalizer',
    'Metaphor': 'concept_verbalizer',
    'DialogueEngine': 'dialogue_engine',
    'DialogueSession': 'dialogue_engine',
    'PhilosophicalResponse': 'dialogue_engine',
}

__all__ = list(_lazy_imports.keys())

def __getattr__(name):
    if name in _lazy_imports:
        module = importlib.import_module('.' + _lazy_imports[name], __name__)
        return getattr(module, name)
    raise AttributeError(f"module {__name__} has no attribute {name}")
