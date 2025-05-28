"""
Hierarchical memory system for the Enhanced Prime Virtual Machine.

This module implements different types of memory (working, long-term, episodic)
and pattern caching for consciousness-aware processing.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
from datetime import datetime
from collections import deque, defaultdict
import json
import hashlib


@dataclass
class MemoryItem:
    """Base class for memory items."""
    
    content: Any
    timestamp: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    importance: float = 0.5  # 0.0 to 1.0
    associations: Set[str] = field(default_factory=set)
    
    def access(self) -> Any:
        """Access the memory item, updating access count."""
        self.access_count += 1
        return self.content
        
    def add_association(self, key: str) -> None:
        """Add an association to this memory."""
        self.associations.add(key)
        
    def compute_relevance(self, context: Dict[str, Any]) -> float:
        """Compute relevance score based on current context."""
        # Simple relevance based on recency and access frequency
        recency = 1.0 / (1.0 + (datetime.now() - self.timestamp).total_seconds() / 3600)
        frequency = min(1.0, self.access_count / 10.0)
        return (recency * 0.3 + frequency * 0.3 + self.importance * 0.4)


@dataclass
class Pattern:
    """Represents a recognized pattern in execution or data."""
    
    pattern_type: str  # e.g., "execution", "data", "behavioral"
    signature: str  # Unique identifier for the pattern
    occurrences: List[Dict[str, Any]] = field(default_factory=list)
    confidence: float = 0.0
    metadata: Dict[str, Any] = field(default_factory=dict)
    
    def add_occurrence(self, context: Dict[str, Any]) -> None:
        """Add a new occurrence of this pattern."""
        self.occurrences.append({
            'timestamp': datetime.now(),
            'context': context
        })
        # Update confidence based on frequency
        self.confidence = min(1.0, len(self.occurrences) / 5.0)
        
    def matches(self, data: Any) -> bool:
        """Check if data matches this pattern."""
        # Simplified pattern matching
        data_sig = self._compute_signature(data)
        return data_sig == self.signature
        
    def _compute_signature(self, data: Any) -> str:
        """Compute a signature for pattern matching."""
        # Simple hash-based signature
        data_str = json.dumps(data, sort_keys=True, default=str)
        return hashlib.md5(data_str.encode()).hexdigest()


class WorkingMemory:
    """
    Short-term memory for current problem context and active processing.
    
    Implements a limited-capacity buffer with decay over time.
    """
    
    def __init__(self, capacity: int = 7):
        """
        Initialize working memory.
        
        Args:
            capacity: Maximum number of items (default 7 +/- 2 rule)
        """
        self.capacity = capacity
        self._items: deque = deque(maxlen=capacity)
        self._focus_stack: List[str] = []  # Stack of current focus items
        self._context: Dict[str, Any] = {}
        
    def store(self, key: str, value: Any, importance: float = 0.5) -> None:
        """
        Store an item in working memory.
        
        Args:
            key: Unique identifier
            value: Content to store
            importance: Importance score (0.0-1.0)
        """
        item = MemoryItem(
            content={'key': key, 'value': value},
            importance=importance
        )
        self._items.append(item)
        
        # Update context
        self._context[key] = value
        
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve an item from working memory."""
        for item in self._items:
            if item.content.get('key') == key:
                return item.access()['value']
        return None
        
    def focus_on(self, key: str) -> None:
        """Focus attention on a specific item."""
        if key in self._context:
            self._focus_stack.append(key)
            
    def get_current_focus(self) -> Optional[str]:
        """Get the current focus item."""
        return self._focus_stack[-1] if self._focus_stack else None
        
    def clear_focus(self) -> None:
        """Clear the focus stack."""
        self._focus_stack.clear()
        
    def get_context(self) -> Dict[str, Any]:
        """Get the current working memory context."""
        return self._context.copy()
        
    def decay(self) -> None:
        """Apply decay to working memory items."""
        # Remove least important items if over capacity
        if len(self._items) >= self.capacity:
            sorted_items = sorted(self._items, key=lambda x: x.importance)
            self._items.remove(sorted_items[0])
            
    def clear(self) -> None:
        """Clear all working memory."""
        self._items.clear()
        self._focus_stack.clear()
        self._context.clear()


class LongTermMemory:
    """
    Persistent memory for learned patterns, strategies, and knowledge.
    
    Implements consolidation from working memory and retrieval mechanisms.
    """
    
    def __init__(self):
        """Initialize long-term memory."""
        self._knowledge_base: Dict[str, MemoryItem] = {}
        self._semantic_network: Dict[str, Set[str]] = defaultdict(set)
        self._strategies: Dict[str, Dict[str, Any]] = {}
        
    def consolidate(self, key: str, value: Any, importance: float = 0.5,
                   associations: Optional[Set[str]] = None) -> None:
        """
        Consolidate information into long-term memory.
        
        Args:
            key: Unique identifier
            value: Content to store
            importance: Importance score
            associations: Related memory keys
        """
        item = MemoryItem(
            content=value,
            importance=importance,
            associations=associations or set()
        )
        
        self._knowledge_base[key] = item
        
        # Update semantic network
        if associations:
            for assoc in associations:
                self._semantic_network[key].add(assoc)
                self._semantic_network[assoc].add(key)
                
    def retrieve(self, key: str) -> Optional[Any]:
        """Retrieve information from long-term memory."""
        if key in self._knowledge_base:
            return self._knowledge_base[key].access()
        return None
        
    def search_by_association(self, key: str, max_depth: int = 2) -> List[Tuple[str, Any]]:
        """
        Search for related memories by association.
        
        Args:
            key: Starting key
            max_depth: Maximum search depth
            
        Returns:
            List of (key, value) tuples
        """
        visited = set()
        results = []
        
        def _search(current_key: str, depth: int):
            if depth > max_depth or current_key in visited:
                return
                
            visited.add(current_key)
            
            if current_key in self._knowledge_base:
                results.append((current_key, self._knowledge_base[current_key].content))
                
            for assoc in self._semantic_network.get(current_key, []):
                _search(assoc, depth + 1)
                
        _search(key, 0)
        return results
        
    def store_strategy(self, name: str, strategy: Dict[str, Any]) -> None:
        """Store a problem-solving strategy."""
        self._strategies[name] = {
            'strategy': strategy,
            'success_count': 0,
            'failure_count': 0,
            'last_used': datetime.now()
        }
        
    def get_strategy(self, name: str) -> Optional[Dict[str, Any]]:
        """Retrieve a stored strategy."""
        return self._strategies.get(name)
        
    def update_strategy_outcome(self, name: str, success: bool) -> None:
        """Update strategy performance metrics."""
        if name in self._strategies:
            if success:
                self._strategies[name]['success_count'] += 1
            else:
                self._strategies[name]['failure_count'] += 1
            self._strategies[name]['last_used'] = datetime.now()
            
    def get_best_strategies(self, n: int = 5) -> List[Tuple[str, Dict[str, Any]]]:
        """Get the n most successful strategies."""
        sorted_strategies = sorted(
            self._strategies.items(),
            key=lambda x: x[1]['success_count'] / max(1, x[1]['failure_count']),
            reverse=True
        )
        return sorted_strategies[:n]


class EpisodicMemory:
    """
    Autobiographical memory for specific events and experiences.
    
    Stores sequences of events with temporal and causal relationships.
    """
    
    def __init__(self, max_episodes: int = 1000):
        """
        Initialize episodic memory.
        
        Args:
            max_episodes: Maximum number of episodes to store
        """
        self.max_episodes = max_episodes
        self._episodes: deque = deque(maxlen=max_episodes)
        self._current_episode: Optional[Dict[str, Any]] = None
        
    def start_episode(self, context: Dict[str, Any]) -> None:
        """Start recording a new episode."""
        self._current_episode = {
            'start_time': datetime.now(),
            'context': context,
            'events': [],
            'outcome': None
        }
        
    def record_event(self, event_type: str, data: Any) -> None:
        """Record an event in the current episode."""
        if self._current_episode:
            self._current_episode['events'].append({
                'timestamp': datetime.now(),
                'type': event_type,
                'data': data
            })
            
    def end_episode(self, outcome: Any) -> None:
        """End the current episode and store it."""
        if self._current_episode:
            self._current_episode['end_time'] = datetime.now()
            self._current_episode['outcome'] = outcome
            self._current_episode['duration'] = (
                self._current_episode['end_time'] - 
                self._current_episode['start_time']
            ).total_seconds()
            
            self._episodes.append(self._current_episode)
            self._current_episode = None
            
    def recall_similar_episodes(self, context: Dict[str, Any], 
                              max_results: int = 5) -> List[Dict[str, Any]]:
        """
        Recall episodes similar to the given context.
        
        Args:
            context: Current context to match
            max_results: Maximum number of episodes to return
            
        Returns:
            List of similar episodes
        """
        # Simple similarity based on context keys overlap
        similar_episodes = []
        
        context_keys = set(context.keys())
        for episode in self._episodes:
            episode_keys = set(episode['context'].keys())
            similarity = len(context_keys & episode_keys) / max(len(context_keys), len(episode_keys))
            
            if similarity > 0.5:  # Threshold for similarity
                similar_episodes.append({
                    'episode': episode,
                    'similarity': similarity
                })
                
        # Sort by similarity and return top results
        similar_episodes.sort(key=lambda x: x['similarity'], reverse=True)
        return [ep['episode'] for ep in similar_episodes[:max_results]]
        
    def get_recent_episodes(self, n: int = 10) -> List[Dict[str, Any]]:
        """Get the n most recent episodes."""
        return list(self._episodes)[-n:]


class PatternCache:
    """
    Cache for recognized patterns in execution, data, and behavior.
    
    Supports pattern matching, storage, and retrieval with confidence scores.
    """
    
    def __init__(self, max_patterns: int = 500):
        """
        Initialize pattern cache.
        
        Args:
            max_patterns: Maximum number of patterns to cache
        """
        self.max_patterns = max_patterns
        self._patterns: Dict[str, Pattern] = {}
        self._pattern_index: Dict[str, Set[str]] = defaultdict(set)  # Type -> pattern IDs
        
    def store_pattern(self, pattern_type: str, data: Any, 
                     metadata: Optional[Dict[str, Any]] = None) -> str:
        """
        Store a new pattern or update existing one.
        
        Args:
            pattern_type: Type of pattern
            data: Pattern data
            metadata: Additional metadata
            
        Returns:
            Pattern ID
        """
        # Compute pattern signature
        data_str = json.dumps(data, sort_keys=True, default=str)
        signature = hashlib.md5(data_str.encode()).hexdigest()
        
        pattern_id = f"{pattern_type}_{signature[:8]}"
        
        if pattern_id in self._patterns:
            # Update existing pattern
            self._patterns[pattern_id].add_occurrence({'data': data})
        else:
            # Create new pattern
            pattern = Pattern(
                pattern_type=pattern_type,
                signature=signature,
                metadata=metadata or {}
            )
            pattern.add_occurrence({'data': data})
            
            # Check cache size
            if len(self._patterns) >= self.max_patterns:
                self._evict_least_confident()
                
            self._patterns[pattern_id] = pattern
            self._pattern_index[pattern_type].add(pattern_id)
            
        return pattern_id
        
    def find_matching_patterns(self, data: Any, pattern_type: Optional[str] = None,
                             min_confidence: float = 0.5) -> List[Pattern]:
        """
        Find patterns matching the given data.
        
        Args:
            data: Data to match
            pattern_type: Optional pattern type filter
            min_confidence: Minimum confidence threshold
            
        Returns:
            List of matching patterns
        """
        matching_patterns = []
        
        # Get candidate patterns
        if pattern_type:
            candidate_ids = self._pattern_index.get(pattern_type, set())
        else:
            candidate_ids = set(self._patterns.keys())
            
        for pattern_id in candidate_ids:
            pattern = self._patterns[pattern_id]
            if pattern.confidence >= min_confidence and pattern.matches(data):
                matching_patterns.append(pattern)
                
        # Sort by confidence
        matching_patterns.sort(key=lambda p: p.confidence, reverse=True)
        return matching_patterns
        
    def get_pattern_by_id(self, pattern_id: str) -> Optional[Pattern]:
        """Get a specific pattern by ID."""
        return self._patterns.get(pattern_id)
        
    def get_patterns_by_type(self, pattern_type: str) -> List[Pattern]:
        """Get all patterns of a specific type."""
        pattern_ids = self._pattern_index.get(pattern_type, set())
        return [self._patterns[pid] for pid in pattern_ids if pid in self._patterns]
        
    def _evict_least_confident(self) -> None:
        """Evict the pattern with lowest confidence."""
        if not self._patterns:
            return
            
        # Find pattern with lowest confidence
        min_pattern_id = min(self._patterns.keys(), 
                           key=lambda pid: self._patterns[pid].confidence)
        
        pattern = self._patterns[min_pattern_id]
        del self._patterns[min_pattern_id]
        self._pattern_index[pattern.pattern_type].discard(min_pattern_id)
        
    def clear(self) -> None:
        """Clear all cached patterns."""
        self._patterns.clear()
        self._pattern_index.clear()
        
    def get_statistics(self) -> Dict[str, Any]:
        """Get cache statistics."""
        total_patterns = len(self._patterns)
        patterns_by_type = {
            ptype: len(pids) for ptype, pids in self._pattern_index.items()
        }
        
        avg_confidence = (
            sum(p.confidence for p in self._patterns.values()) / total_patterns
            if total_patterns > 0 else 0.0
        )
        
        return {
            'total_patterns': total_patterns,
            'patterns_by_type': patterns_by_type,
            'average_confidence': avg_confidence,
            'cache_utilization': total_patterns / self.max_patterns
        }
