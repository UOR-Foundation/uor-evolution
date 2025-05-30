"""
Test suite for Phase 3.1 - Natural Language Integration and Philosophical Reasoning

This test suite validates the implementation of natural language processing,
philosophical reasoning, and consciousness communication capabilities.
"""

# Make test modules easily importable
try:
    from . import (
        test_natural_language,
        test_philosophical_reasoning,
        test_dialogue_quality,
        test_abstract_reasoning,
        test_consciousness_communication,
    )
except Exception:  # pragma: no cover - optional dependencies may be missing
    test_natural_language = None
    test_philosophical_reasoning = None
    test_dialogue_quality = None
    test_abstract_reasoning = None
    test_consciousness_communication = None

__all__ = [name for name in [
    'test_natural_language',
    'test_philosophical_reasoning',
    'test_dialogue_quality',
    'test_abstract_reasoning',
    'test_consciousness_communication'
] if globals().get(name) is not None]
