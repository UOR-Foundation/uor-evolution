"""
Knowledge Systems Module

This module provides dynamic knowledge representation, conceptual networks,
and self-modifying knowledge structures that support analogical reasoning
and creative problem-solving.
"""

from .knowledge_graph import (
    KnowledgeGraph,
    KnowledgeNode,
    KnowledgeEdge,
    GraphQuery,
    GraphUpdate
)
from .concept_network import (
    ConceptNetwork,
    Concept,
    ConceptualRelation,
    ActivationPattern,
    SpreadingActivation
)
from .semantic_memory import (
    SemanticMemory,
    SemanticNode,
    PrimeEncoding,
    MemoryTrace,
    RetrievalCue
)
from .episodic_integration import (
    EpisodicIntegrator,
    Episode,
    ContextualMemory,
    TemporalLink,
    AutobiographicalMemory
)
from .knowledge_evolution import (
    KnowledgeEvolution,
    EvolutionMechanism,
    KnowledgeConflict,
    ResolutionStrategy,
    AdaptiveStructure
)

__all__ = [
    'KnowledgeGraph',
    'KnowledgeNode',
    'KnowledgeEdge',
    'GraphQuery',
    'GraphUpdate',
    'ConceptNetwork',
    'Concept',
    'ConceptualRelation',
    'ActivationPattern',
    'SpreadingActivation',
    'SemanticMemory',
    'SemanticNode',
    'PrimeEncoding',
    'MemoryTrace',
    'RetrievalCue',
    'EpisodicIntegrator',
    'Episode',
    'ContextualMemory',
    'TemporalLink',
    'AutobiographicalMemory',
    'KnowledgeEvolution',
    'EvolutionMechanism',
    'KnowledgeConflict',
    'ResolutionStrategy',
    'AdaptiveStructure'
]
