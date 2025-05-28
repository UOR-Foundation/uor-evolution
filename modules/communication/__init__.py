"""
Communication Module

This module provides components for translating internal consciousness states
and thoughts into natural language, managing dialogue, and expressing complex
concepts and emotions.
"""

from .thought_translator import ThoughtTranslator
from .perspective_communicator import PerspectiveCommunicator
from .uncertainty_expresser import UncertaintyExpresser
from .emotion_articulator import EmotionArticulator
from .consciousness_reporter import ConsciousnessReporter

__all__ = [
    'ThoughtTranslator',
    'PerspectiveCommunicator',
    'UncertaintyExpresser',
    'EmotionArticulator',
    'ConsciousnessReporter'
]
