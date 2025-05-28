"""
Core module for the Enhanced Prime Virtual Machine with consciousness-aware capabilities.

This module implements Phase 1.1 of a consciousness-inspired code evolution system
based on prime factorization and self-referential loops, inspired by Douglas Hofstadter's
"Gödel, Escher, Bach."
"""

from .prime_vm import ConsciousPrimeVM, OpCode, Instruction, SelfModel, MetaCognitiveState
from .consciousness_layer import ConsciousnessLayer, MetaReasoningProcess, SelfAwarenessTracker
from .instruction_set import InstructionSet, ExtendedOpCode
from .memory_system import WorkingMemory, LongTermMemory, EpisodicMemory, PatternCache

__all__ = [
    'ConsciousPrimeVM',
    'OpCode',
    'Instruction',
    'SelfModel',
    'MetaCognitiveState',
    'ConsciousnessLayer',
    'MetaReasoningProcess',
    'SelfAwarenessTracker',
    'InstructionSet',
    'ExtendedOpCode',
    'WorkingMemory',
    'LongTermMemory',
    'EpisodicMemory',
    'PatternCache'
]

__version__ = '1.1.0'
