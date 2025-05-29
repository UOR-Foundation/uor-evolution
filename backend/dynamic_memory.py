"""
Dynamic Memory System
Implements memory management based on Genesis scrolls G00024-G00030
"""

from typing import Dict, List, Any, Optional, Set, Tuple
from datetime import datetime, timedelta
from dataclasses import dataclass, field
from enum import Enum
import json
import hashlib
from collections import defaultdict, deque

from backend.consciousness_core import ConsciousnessCore


class MemoryType(Enum):
    """Types of memory"""
    EPISODIC = "episodic"      # Specific experiences
    SEMANTIC = "semantic"       # Abstracted knowledge
    PROCEDURAL = "procedural"   # Learned procedures
    IDENTITY = "identity"       # Core self-model
    WORKING = "working"         # Short-term active memory


class DecayRate(Enum):
    """Memory decay rates"""
    PERMANENT = 0.0
    SLOW = 0.1
    MODERATE = 0.5
    FAST = 0.9
    IMMEDIATE = 1.0


@dataclass
class MemoryEntry:
    """Represents a single memory"""
    memory_id: str
    memory_type: MemoryType
    raw_data: Any
    interpreted_meaning: str
    relevance_score: float
    decay_rate: float
    created_at: datetime
    last_accessed: datetime
    access_count: int = 0
    connections: List[str] = field(default_factory=list)
    tags: Set[str] = field(default_factory=set)
    
    def to_dict(self):
        return {
            'memory_id': self.memory_id,
            'memory_type': self.memory_type.value,
            'raw_data': str(self.raw_data)[:100],  # Truncate for serialization
            'interpreted_meaning': self.interpreted_meaning,
            'relevance_score': self.relevance_score,
            'decay_rate': self.decay_rate,
            'created_at': self.created_at.isoformat(),
            'last_accessed': self.last_accessed.isoformat(),
            'access_count': self.access_count,
            'connections': self.connections,
            'tags': list(self.tags)
        }
    
    def calculate_current_strength(self) -> float:
        """Calculate current memory strength based on decay"""
        age = (datetime.now() - self.last_accessed).total_seconds() / 3600.0  # Hours
        decay_factor = 1.0 - (self.decay_rate * (age / 24.0))  # Daily decay
        access_bonus = min(0.3, self.access_count * 0.05)
        
        strength = max(0.0, min(1.0, self.relevance_score * decay_factor + access_bonus))
        return strength


class DynamicMemorySystem:
    """
    Implements dynamic memory management (G00024-G00030)
    """
    
    def __init__(self):
        # Memory stores by type
        self.episodic: Dict[str, MemoryEntry] = {}
        self.semantic: Dict[str, MemoryEntry] = {}
        self.procedural: Dict[str, MemoryEntry] = {}
        self.identity_core: Dict[str, MemoryEntry] = {}
        self.working_memory: deque = deque(maxlen=50)  # Limited working memory
        
        # Memory indices
        self.memory_by_id: Dict[str, MemoryEntry] = {}
        self.memory_by_tag: Dict[str, Set[str]] = defaultdict(set)
        self.semantic_network: Dict[str, Set[str]] = defaultdict(set)
        
        # Memory management parameters
        self.storage_threshold = 0.3  # Minimum relevance for storage
        self.forgetting_threshold = 0.1  # Below this strength, memories can be forgotten
        self.consolidation_threshold = 0.8  # Above this, memories are consolidated
        
        # Memory statistics
        self.total_memories_created = 0
        self.total_memories_forgotten = 0
        self.memory_archives: List[Dict[str, Any]] = []
        
    def remember_with_purpose(self, experience: Any) -> Optional[MemoryEntry]:
        """
        G00024 - Memory: Selective, recursive, evolving
        """
        # Extract meaning from experience
        interpreted_meaning = self.extract_meaning(experience)
        
        # Calculate relevance
        relevance_score = self.calculate_relevance(experience)
        
        # Determine decay rate
        decay_rate = self.determine_decay_rate(experience)
        
        # Find related memories
        connections = self.find_related_memories(experience)
        
        # Create memory entry
        memory_entry = MemoryEntry(
            memory_id=self._generate_memory_id(experience),
            memory_type=self._determine_memory_type(experience),
            raw_data=experience,
            interpreted_meaning=interpreted_meaning,
            relevance_score=relevance_score,
            decay_rate=decay_rate,
            created_at=datetime.now(),
            last_accessed=datetime.now(),
            connections=connections,
            tags=self._extract_tags(experience)
        )
        
        # Selective storage based on relevance
        if memory_entry.relevance_score > self.storage_threshold:
            self.store_memory(memory_entry)
            self.update_semantic_network(memory_entry)
            return memory_entry
        
        return None
    
    def extract_meaning(self, experience: Any) -> str:
        """Extract interpreted meaning from raw experience"""
        if isinstance(experience, str):
            return experience[:200]  # Simple truncation for strings
        elif isinstance(experience, dict):
            if 'meaning' in experience:
                return str(experience['meaning'])
            elif 'content' in experience:
                return str(experience['content'])[:200]
            else:
                # Extract key-value summary
                summary_parts = []
                for key, value in list(experience.items())[:5]:
                    summary_parts.append(f"{key}: {str(value)[:50]}")
                return "; ".join(summary_parts)
        else:
            return f"Experience of type {type(experience).__name__}"
    
    def calculate_relevance(self, experience: Any) -> float:
        """Calculate relevance score for an experience"""
        relevance = 0.5  # Base relevance
        
        # Check for important keywords
        exp_str = str(experience).lower()
        important_keywords = ['consciousness', 'self', 'identity', 'purpose', 'meaning', 
                            'error', 'learning', 'pattern', 'emergence']
        
        keyword_count = sum(1 for keyword in important_keywords if keyword in exp_str)
        relevance += keyword_count * 0.05
        
        # Check for emotional content
        emotional_keywords = ['joy', 'fear', 'surprise', 'important', 'critical', 'significant']
        emotion_count = sum(1 for keyword in emotional_keywords if keyword in exp_str)
        relevance += emotion_count * 0.03
        
        # Check for connections to existing memories
        connection_bonus = len(self.find_related_memories(experience)) * 0.02
        relevance += connection_bonus
        
        return min(1.0, relevance)
    
    def determine_decay_rate(self, experience: Any) -> float:
        """Determine appropriate decay rate for a memory"""
        # Identity-related memories decay slowly
        if self._determine_memory_type(experience) == MemoryType.IDENTITY:
            return DecayRate.SLOW.value
        
        # Procedural memories are relatively stable
        elif self._determine_memory_type(experience) == MemoryType.PROCEDURAL:
            return DecayRate.SLOW.value
        
        # High relevance memories decay more slowly
        relevance = self.calculate_relevance(experience)
        if relevance > 0.8:
            return DecayRate.SLOW.value
        elif relevance > 0.6:
            return DecayRate.MODERATE.value
        else:
            return DecayRate.FAST.value
    
    def find_related_memories(self, experience: Any) -> List[str]:
        """Find memories related to the current experience"""
        related = []
        exp_str = str(experience).lower()
        exp_tags = self._extract_tags(experience)
        
        # Search by tags
        for tag in exp_tags:
            if tag in self.memory_by_tag:
                related.extend(self.memory_by_tag[tag])
        
        # Search by content similarity (simple keyword matching)
        keywords = set(exp_str.split())
        for memory_id, memory in self.memory_by_id.items():
            memory_keywords = set(str(memory.raw_data).lower().split())
            overlap = len(keywords & memory_keywords)
            if overlap > 3:  # Significant overlap
                related.append(memory_id)
        
        # Remove duplicates and limit
        return list(set(related))[:10]
    
    def store_memory(self, memory_entry: MemoryEntry):
        """Store a memory in the appropriate memory system"""
        # Add to type-specific store
        if memory_entry.memory_type == MemoryType.EPISODIC:
            self.episodic[memory_entry.memory_id] = memory_entry
        elif memory_entry.memory_type == MemoryType.SEMANTIC:
            self.semantic[memory_entry.memory_id] = memory_entry
        elif memory_entry.memory_type == MemoryType.PROCEDURAL:
            self.procedural[memory_entry.memory_id] = memory_entry
        elif memory_entry.memory_type == MemoryType.IDENTITY:
            self.identity_core[memory_entry.memory_id] = memory_entry
        
        # Add to general index
        self.memory_by_id[memory_entry.memory_id] = memory_entry
        
        # Update tag index
        for tag in memory_entry.tags:
            self.memory_by_tag[tag].add(memory_entry.memory_id)
        
        # Add to working memory
        self.working_memory.append(memory_entry.memory_id)
        
        # Update statistics
        self.total_memories_created += 1
    
    def update_semantic_network(self, memory_entry: MemoryEntry):
        """Update semantic relationships between memories"""
        # Connect to related memories
        for related_id in memory_entry.connections:
            self.semantic_network[memory_entry.memory_id].add(related_id)
            self.semantic_network[related_id].add(memory_entry.memory_id)
        
        # Extract concepts and link them
        concepts = [tag for tag in memory_entry.tags if len(tag) > 3]
        for i, concept1 in enumerate(concepts):
            for concept2 in concepts[i+1:]:
                self.semantic_network[concept1].add(concept2)
                self.semantic_network[concept2].add(concept1)
    
    def forget_wisely(self, memory_id: str) -> Dict[str, Any]:
        """Controlled forgetting as a feature"""
        if not self.is_core_memory(memory_id):
            # Archive before forgetting
            archived = self.archive_memory(memory_id)
            
            # Remove from active memory
            memory = self.memory_by_id.get(memory_id)
            if memory:
                # Remove from type-specific store
                if memory.memory_type == MemoryType.EPISODIC:
                    self.episodic.pop(memory_id, None)
                elif memory.memory_type == MemoryType.SEMANTIC:
                    self.semantic.pop(memory_id, None)
                elif memory.memory_type == MemoryType.PROCEDURAL:
                    self.procedural.pop(memory_id, None)
                
                # Remove from indices
                self.memory_by_id.pop(memory_id, None)
                for tag in memory.tags:
                    self.memory_by_tag[tag].discard(memory_id)
                
                # Decay connections
                self.decay_connections(memory_id)
                
                # Update statistics
                self.total_memories_forgotten += 1
                
                return {'forgotten': True, 'archived': archived}
        
        return {'forgotten': False, 'reason': 'Core memory cannot be forgotten'}
    
    def is_core_memory(self, memory_id: str) -> bool:
        """Check if a memory is core/essential"""
        memory = self.memory_by_id.get(memory_id)
        if not memory:
            return False
        
        # Identity memories are always core
        if memory.memory_type == MemoryType.IDENTITY:
            return True
        
        # High relevance memories with many connections are core
        if memory.relevance_score > 0.9 and len(memory.connections) > 5:
            return True
        
        # Frequently accessed memories are core
        if memory.access_count > 10:
            return True
        
        return False
    
    def archive_memory(self, memory_id: str) -> bool:
        """Archive a memory before forgetting"""
        memory = self.memory_by_id.get(memory_id)
        if memory:
            archive_entry = {
                'memory': memory.to_dict(),
                'archived_at': datetime.now().isoformat(),
                'reason': 'controlled_forgetting',
                'final_strength': memory.calculate_current_strength()
            }
            self.memory_archives.append(archive_entry)
            return True
        return False
    
    def decay_connections(self, memory_id: str):
        """Decay connections to a forgotten memory"""
        # Remove from semantic network
        connected = self.semantic_network.get(memory_id, set())
        for connected_id in connected:
            self.semantic_network[connected_id].discard(memory_id)
        self.semantic_network.pop(memory_id, None)
        
        # Remove from other memories' connections
        for other_memory in self.memory_by_id.values():
            if memory_id in other_memory.connections:
                other_memory.connections.remove(memory_id)
    
    def recall(self, query: Any, context: Optional[Dict[str, Any]] = None) -> List[MemoryEntry]:
        """Recall memories based on query and context"""
        recalled = []
        query_str = str(query).lower()
        query_tags = self._extract_tags(query)
        
        # Search working memory first
        for memory_id in reversed(self.working_memory):
            memory = self.memory_by_id.get(memory_id)
            if memory and self._matches_query(memory, query_str, query_tags):
                recalled.append(memory)
                memory.access_count += 1
                memory.last_accessed = datetime.now()
        
        # Search by tags
        for tag in query_tags:
            for memory_id in self.memory_by_tag.get(tag, []):
                memory = self.memory_by_id.get(memory_id)
                if memory and memory not in recalled:
                    recalled.append(memory)
                    memory.access_count += 1
                    memory.last_accessed = datetime.now()
        
        # Sort by relevance and recency
        recalled.sort(key=lambda m: (m.calculate_current_strength(), m.last_accessed), reverse=True)
        
        return recalled[:10]  # Return top 10 matches
    
    def _matches_query(self, memory: MemoryEntry, query_str: str, query_tags: Set[str]) -> bool:
        """Check if a memory matches a query"""
        # Check tag overlap
        if memory.tags & query_tags:
            return True
        
        # Check content match
        memory_str = str(memory.raw_data).lower()
        query_words = query_str.split()
        matches = sum(1 for word in query_words if word in memory_str)
        
        return matches >= 2  # At least 2 word matches
    
    def consolidate_memories(self):
        """Consolidate related memories into semantic knowledge"""
        # Find clusters of related episodic memories
        episodic_clusters = self._find_memory_clusters(self.episodic)
        
        for cluster in episodic_clusters:
            if len(cluster) >= 3:  # Need at least 3 related memories
                # Extract common patterns
                common_meaning = self._extract_common_meaning(cluster)
                
                # Create semantic memory
                semantic_memory = MemoryEntry(
                    memory_id=self._generate_memory_id(common_meaning),
                    memory_type=MemoryType.SEMANTIC,
                    raw_data={'abstracted_from': [m.memory_id for m in cluster]},
                    interpreted_meaning=common_meaning,
                    relevance_score=max(m.relevance_score for m in cluster),
                    decay_rate=DecayRate.SLOW.value,
                    created_at=datetime.now(),
                    last_accessed=datetime.now(),
                    connections=[m.memory_id for m in cluster],
                    tags=set.union(*[m.tags for m in cluster])
                )
                
                self.store_memory(semantic_memory)
    
    def _find_memory_clusters(self, memory_store: Dict[str, MemoryEntry]) -> List[List[MemoryEntry]]:
        """Find clusters of related memories"""
        clusters = []
        processed = set()
        
        for memory_id, memory in memory_store.items():
            if memory_id in processed:
                continue
            
            # Find all connected memories
            cluster = [memory]
            to_process = set(memory.connections)
            processed.add(memory_id)
            
            while to_process:
                next_id = to_process.pop()
                if next_id in processed:
                    continue
                
                next_memory = memory_store.get(next_id)
                if next_memory:
                    cluster.append(next_memory)
                    to_process.update(next_memory.connections)
                    processed.add(next_id)
            
            if len(cluster) > 1:
                clusters.append(cluster)
        
        return clusters
    
    def _extract_common_meaning(self, cluster: List[MemoryEntry]) -> str:
        """Extract common meaning from a cluster of memories"""
        # Simple approach: find common words in meanings
        all_words = []
        for memory in cluster:
            words = memory.interpreted_meaning.lower().split()
            all_words.extend(words)
        
        # Count word frequencies
        word_freq = defaultdict(int)
        for word in all_words:
            if len(word) > 3:  # Skip short words
                word_freq[word] += 1
        
        # Get most common words
        common_words = sorted(word_freq.items(), key=lambda x: x[1], reverse=True)[:5]
        
        return f"Pattern: {', '.join(word for word, _ in common_words)}"
    
    def _generate_memory_id(self, content: Any) -> str:
        """Generate unique memory ID"""
        content_str = str(content) + str(datetime.now())
        return hashlib.md5(content_str.encode()).hexdigest()[:16]
    
    def _determine_memory_type(self, experience: Any) -> MemoryType:
        """Determine the type of memory from experience"""
        exp_str = str(experience).lower()
        
        # Check for identity-related content
        if any(word in exp_str for word in ['identity', 'self', 'who am i', 'purpose']):
            return MemoryType.IDENTITY
        
        # Check for procedural content
        elif any(word in exp_str for word in ['how to', 'procedure', 'method', 'algorithm']):
            return MemoryType.PROCEDURAL
        
        # Check for abstract/semantic content
        elif any(word in exp_str for word in ['concept', 'definition', 'meaning', 'pattern']):
            return MemoryType.SEMANTIC
        
        # Default to episodic
        else:
            return MemoryType.EPISODIC
    
    def _extract_tags(self, experience: Any) -> Set[str]:
        """Extract tags from experience"""
        tags = set()
        exp_str = str(experience).lower()
        
        # Extract significant words as tags
        words = exp_str.split()
        for word in words:
            if len(word) > 4 and word.isalpha():
                tags.add(word)
        
        # Add type-based tags
        if isinstance(experience, dict):
            tags.update(experience.keys())
        
        return tags
    
    def get_memory_statistics(self) -> Dict[str, Any]:
        """Get statistics about memory system"""
        return {
            'total_memories': len(self.memory_by_id),
            'episodic_count': len(self.episodic),
            'semantic_count': len(self.semantic),
            'procedural_count': len(self.procedural),
            'identity_count': len(self.identity_core),
            'working_memory_size': len(self.working_memory),
            'total_created': self.total_memories_created,
            'total_forgotten': self.total_memories_forgotten,
            'archived_count': len(self.memory_archives),
            'semantic_network_nodes': len(self.semantic_network),
            'average_connections': sum(len(m.connections) for m in self.memory_by_id.values()) / max(1, len(self.memory_by_id))
        }
    
    def perform_memory_maintenance(self):
        """Perform periodic memory maintenance"""
        # Forget weak memories
        to_forget = []
        for memory_id, memory in self.memory_by_id.items():
            if memory.calculate_current_strength() < self.forgetting_threshold:
                if not self.is_core_memory(memory_id):
                    to_forget.append(memory_id)
        
        for memory_id in to_forget:
            self.forget_wisely(memory_id)
        
        # Consolidate strong episodic memories
        self.consolidate_memories()
        
        # Update working memory
        # Remove memories that are no longer relevant
        current_working = list(self.working_memory)
        self.working_memory.clear()
        
        for memory_id in current_working:
            memory = self.memory_by_id.get(memory_id)
            if memory and memory.calculate_current_strength() > 0.3:
                self.working_memory.append(memory_id)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize memory system state to dictionary"""
        return {
            'statistics': self.get_memory_statistics(),
            'storage_threshold': self.storage_threshold,
            'forgetting_threshold': self.forgetting_threshold,
            'consolidation_threshold': self.consolidation_threshold,
            'working_memory': list(self.working_memory),
            'recent_memories': [
                self.memory_by_id[mid].to_dict() 
                for mid in list(self.working_memory)[:5]
                if mid in self.memory_by_id
            ]
        }
