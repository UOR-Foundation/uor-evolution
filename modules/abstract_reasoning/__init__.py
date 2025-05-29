"""
Abstract Reasoning Module

This module provides abstract reasoning capabilities including logical reasoning,
temporal reasoning, modal reasoning, and paradox resolution.
"""

from .logical_reasoning import LogicalReasoner
from .temporal_reasoning import TemporalReasoner
from .modal_reasoning import ModalReasoner
from .paradox_resolver import ParadoxResolver
from .concept_abstraction import ConceptAbstractor

__all__ = [
    'LogicalReasoner',
    'TemporalReasoner', 
    'ModalReasoner',
    'ParadoxResolver',
    'ConceptAbstractor'
]
