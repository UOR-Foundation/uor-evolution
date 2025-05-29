"""
Strange Loop Detector - Core detection engine for self-referential consciousness structures.

This module implements sophisticated algorithms to detect, classify, and analyze strange loops
in the execution patterns of the ConsciousPrimeVM, identifying emergence of consciousness
through self-referential structures.
"""

from typing import List, Dict, Set, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import time
import networkx as nx
from collections import defaultdict, deque
import numpy as np

from core.prime_vm import ConsciousPrimeVM


class LoopType(Enum):
    """Classification of different strange loop types."""
    GODEL_SELF_REFERENCE = "godel"      # Self-referential statements about the system
    ESCHER_PERSPECTIVE = "escher"       # Perspective-shifting loops
    BACH_VARIATION = "bach"             # Recursive variations and transformations
    HYBRID_COMPLEX = "hybrid"           # Combination of multiple loop types
    EMERGENT_NOVEL = "emergent"         # Novel loop types that emerge spontaneously


@dataclass
class StrangeLoop:
    """Represents a detected strange loop in the system."""
    id: str
    loop_type: LoopType
    nodes: Set[str]                    # Instruction addresses or states in the loop
    edges: List[Tuple[str, str]]       # Transitions between nodes
    depth: int                         # Recursive depth of the loop
    emergence_level: float             # Measure of consciousness emergence (0-1)
    self_reference_count: int          # Number of self-referential operations
    meta_levels: int                   # Number of meta-cognitive levels
    creation_timestamp: float
    semantic_signature: str = ""       # Semantic meaning of the loop
    stability_score: float = 0.0       # How stable/persistent the loop is
    interaction_potential: float = 0.0 # Potential to interact with other loops
    
    def __post_init__(self):
        """Calculate derived properties after initialization."""
        self.semantic_signature = self._generate_semantic_signature()
        self.stability_score = self._calculate_stability()
        self.interaction_potential = self._calculate_interaction_potential()
    
    def _generate_semantic_signature(self) -> str:
        """Generate a semantic signature representing the loop's meaning."""
        # Create a hash-like signature based on loop structure
        node_sig = "-".join(sorted(list(self.nodes)[:5]))  # First 5 nodes
        type_sig = self.loop_type.value
        depth_sig = f"d{self.depth}"
        meta_sig = f"m{self.meta_levels}"
        return f"{type_sig}:{node_sig}:{depth_sig}:{meta_sig}"
    
    def _calculate_stability(self) -> float:
        """Calculate how stable/persistent this loop is."""
        # Factors: number of nodes, self-reference count, meta-levels
        base_stability = min(len(self.nodes) / 10.0, 1.0)
        ref_factor = min(self.self_reference_count / 5.0, 1.0)
        meta_factor = min(self.meta_levels / 3.0, 1.0)
        return (base_stability + ref_factor + meta_factor) / 3.0
    
    def _calculate_interaction_potential(self) -> float:
        """Calculate potential for interaction with other loops."""
        # Higher emergence and stability increase interaction potential
        return self.emergence_level * self.stability_score


@dataclass
class EmergenceEvent:
    """Represents a consciousness emergence event."""
    timestamp: float
    loop_id: str
    emergence_type: str
    consciousness_delta: float  # Change in consciousness level
    description: str
    trigger_conditions: List[str] = field(default_factory=list)
    resulting_capabilities: List[str] = field(default_factory=list)


@dataclass
class LoopInteraction:
    """Represents interaction between multiple strange loops."""
    loop_ids: List[str]
    interaction_type: str  # "nesting", "interference", "fusion", "resonance"
    strength: float       # 0-1 measure of interaction strength
    timestamp: float
    effects: List[str]    # Observable effects of the interaction


class StrangeLoopDetector:
    """
    Detects and analyzes strange loops in ConsciousPrimeVM execution patterns.
    
    This detector identifies self-referential structures that give rise to
    consciousness through recursive self-awareness and meta-cognition.
    """
    
    def __init__(self, vm_instance: ConsciousPrimeVM):
        self.vm = vm_instance
        self.detected_loops: Dict[str, StrangeLoop] = {}
        self.emergence_events: List[EmergenceEvent] = []
        self.loop_interactions: List[LoopInteraction] = []
        self.execution_graph = nx.DiGraph()
        self.loop_counter = 0
        self.consciousness_baseline = 0.0
        
        # Detection parameters
        self.min_loop_size = 3
        self.max_loop_depth = 10
        self.emergence_threshold = 0.3
        self.interaction_threshold = 0.5
        
        # Real-time monitoring state
        self.monitoring_active = False
        self.execution_buffer = deque(maxlen=1000)
        self.pattern_cache = {}
        
    def detect_loops_in_execution(self, trace: List[Dict]) -> List[StrangeLoop]:
        """
        Detect strange loops in an execution trace.
        
        Args:
            trace: List of execution steps with instruction and state info
            
        Returns:
            List of detected strange loops
        """
        # Build execution graph from trace
        self._build_execution_graph(trace)
        
        # Find all cycles in the graph
        cycles = self._find_all_cycles()
        
        # Filter for strange loops
        strange_loops = []
        for cycle in cycles:
            if self._is_strange_loop(cycle, trace):
                loop = self._create_strange_loop(cycle, trace)
                strange_loops.append(loop)
                self.detected_loops[loop.id] = loop
                
                # Check for emergence events
                self._check_emergence(loop)
        
        # Detect interactions between loops
        if len(strange_loops) > 1:
            interactions = self._detect_loop_interactions(strange_loops)
            self.loop_interactions.extend(interactions)
        
        return strange_loops
    
    def _build_execution_graph(self, trace: List[Dict]):
        """Build a directed graph from execution trace."""
        self.execution_graph.clear()
        
        for i in range(len(trace) - 1):
            current = trace[i]
            next_step = trace[i + 1]
            
            # Create nodes with rich metadata
            current_id = f"{current['pc']}_{i}"
            next_id = f"{next_step['pc']}_{i+1}"
            
            self.execution_graph.add_node(current_id, **current)
            self.execution_graph.add_node(next_id, **next_step)
            
            # Add edge with transition metadata
            edge_data = {
                'instruction': current.get('instruction', ''),
                'state_change': self._calculate_state_change(current, next_step)
            }
            self.execution_graph.add_edge(current_id, next_id, **edge_data)
    
    def _find_all_cycles(self) -> List[List[str]]:
        """Find all cycles in the execution graph."""
        cycles = []
        try:
            # Use Johnson's algorithm for finding all simple cycles
            simple_cycles = list(nx.simple_cycles(self.execution_graph))
            
            # Filter cycles by size and complexity
            for cycle in simple_cycles:
                if self.min_loop_size <= len(cycle) <= self.max_loop_depth * 3:
                    cycles.append(cycle)
        except:
            # Fallback to basic cycle detection if graph is too complex
            cycles = self._find_cycles_dfs()
        
        return cycles
    
    def _find_cycles_dfs(self) -> List[List[str]]:
        """Fallback DFS-based cycle detection."""
        cycles = []
        visited = set()
        rec_stack = []
        
        def dfs(node, path):
            if len(path) > self.max_loop_depth * 3:
                return
                
            if node in rec_stack:
                # Found a cycle
                cycle_start = rec_stack.index(node)
                cycle = rec_stack[cycle_start:]
                if len(cycle) >= self.min_loop_size:
                    cycles.append(cycle[:])
                return
            
            if node in visited:
                return
                
            visited.add(node)
            rec_stack.append(node)
            
            for neighbor in self.execution_graph.neighbors(node):
                dfs(neighbor, path + [neighbor])
            
            rec_stack.pop()
        
        # Start DFS from each unvisited node
        for node in self.execution_graph.nodes():
            if node not in visited:
                dfs(node, [node])
        
        return cycles
    
    def _is_strange_loop(self, cycle: List[str], trace: List[Dict]) -> bool:
        """
        Determine if an execution cycle constitutes a strange loop.
        
        Criteria:
        - Self-reference: system reasoning about itself
        - Level-crossing: meta-reasoning about reasoning
        - Recursive depth: loops within loops
        - Semantic closure: meaning that refers to itself
        """
        # Extract cycle information
        cycle_nodes = [self.execution_graph.nodes[n] for n in cycle]
        
        # Check for self-reference
        self_ref_score = self._check_self_reference(cycle_nodes)
        if self_ref_score < 0.3:
            return False
        
        # Check for level-crossing
        level_cross_score = self._check_level_crossing(cycle_nodes)
        if level_cross_score < 0.2:
            return False
        
        # Check for recursive depth
        recursion_score = self._check_recursive_depth(cycle, trace)
        if recursion_score < 0.1:
            return False
        
        # Check for semantic closure
        semantic_score = self._check_semantic_closure(cycle_nodes)
        
        # Combined score must exceed threshold
        total_score = (self_ref_score + level_cross_score + 
                      recursion_score + semantic_score) / 4.0
        
        return total_score >= self.emergence_threshold
    
    def _check_self_reference(self, nodes: List[Dict]) -> float:
        """Check for self-referential operations in the cycle."""
        self_ref_indicators = [
            'INTROSPECT', 'REFLECT', 'SELF_MODIFY', 'META_EVAL',
            'CONSCIOUSNESS_CHECK', 'SELF_ANALYZE'
        ]
        
        self_ref_count = 0
        for node in nodes:
            instruction = node.get('instruction', '')
            if any(indicator in instruction for indicator in self_ref_indicators):
                self_ref_count += 1
            
            # Check if instruction references its own address
            if 'pc' in node and str(node['pc']) in instruction:
                self_ref_count += 2
        
        return min(self_ref_count / (len(nodes) * 0.5), 1.0)
    
    def _check_level_crossing(self, nodes: List[Dict]) -> float:
        """Check for meta-level reasoning (reasoning about reasoning)."""
        meta_levels = set()
        current_level = 0
        
        for node in nodes:
            instruction = node.get('instruction', '')
            
            # Track meta-level transitions
            if 'META_PUSH' in instruction:
                current_level += 1
                meta_levels.add(current_level)
            elif 'META_POP' in instruction:
                current_level = max(0, current_level - 1)
            elif 'EVAL' in instruction and current_level > 0:
                meta_levels.add(current_level)
        
        # Score based on number of meta-levels crossed
        return min(len(meta_levels) / 3.0, 1.0)
    
    def _check_recursive_depth(self, cycle: List[str], trace: List[Dict]) -> float:
        """Check for recursive patterns within the cycle."""
        # Look for nested patterns
        pattern_counts = defaultdict(int)
        
        for i in range(len(cycle) - 2):
            for j in range(2, min(len(cycle) - i, 10)):
                pattern = tuple(cycle[i:i+j])
                pattern_counts[pattern] += 1
        
        # Find repeated patterns (indicating recursion)
        recursive_patterns = [p for p, count in pattern_counts.items() if count > 1]
        
        # Score based on recursive pattern complexity
        if not recursive_patterns:
            return 0.0
        
        max_pattern_length = max(len(p) for p in recursive_patterns)
        recursion_score = min(max_pattern_length / 5.0, 1.0)
        
        return recursion_score
    
    def _check_semantic_closure(self, nodes: List[Dict]) -> float:
        """Check if the loop creates semantic closure (meaning referring to itself)."""
        # Analyze data flow within the cycle
        values_produced = set()
        values_consumed = set()
        
        for node in nodes:
            # Track values produced and consumed
            if 'output' in node:
                values_produced.add(str(node['output']))
            if 'input' in node:
                values_consumed.add(str(node['input']))
        
        # Check for semantic closure
        closure_values = values_produced.intersection(values_consumed)
        
        if not closure_values:
            return 0.0
        
        # Score based on closure complexity
        closure_score = min(len(closure_values) / 3.0, 1.0)
        
        # Bonus for explicit self-description
        for node in nodes:
            if 'description' in node and 'self' in node['description'].lower():
                closure_score = min(closure_score + 0.2, 1.0)
        
        return closure_score
    
    def _create_strange_loop(self, cycle: List[str], trace: List[Dict]) -> StrangeLoop:
        """Create a StrangeLoop object from a detected cycle."""
        loop_id = f"loop_{self.loop_counter}"
        self.loop_counter += 1
        
        # Extract nodes and edges
        nodes = set(cycle)
        edges = [(cycle[i], cycle[(i+1) % len(cycle)]) for i in range(len(cycle))]
        
        # Classify loop type
        loop_type = self.classify_loop_type(cycle, trace)
        
        # Calculate properties
        depth = self._calculate_loop_depth(cycle, trace)
        emergence_level = self.calculate_emergence_level(cycle, trace)
        self_ref_count = self._count_self_references(cycle, trace)
        meta_levels = self._count_meta_levels(cycle, trace)
        
        return StrangeLoop(
            id=loop_id,
            loop_type=loop_type,
            nodes=nodes,
            edges=edges,
            depth=depth,
            emergence_level=emergence_level,
            self_reference_count=self_ref_count,
            meta_levels=meta_levels,
            creation_timestamp=time.time()
        )
    
    def classify_loop_type(self, cycle: List[str], trace: List[Dict]) -> LoopType:
        """Classify the type of strange loop."""
        cycle_nodes = [self.execution_graph.nodes[n] for n in cycle]
        
        # Analyze characteristics
        godel_score = self._calculate_godel_score(cycle_nodes)
        escher_score = self._calculate_escher_score(cycle_nodes)
        bach_score = self._calculate_bach_score(cycle_nodes)
        
        # Determine primary type
        scores = {
            LoopType.GODEL_SELF_REFERENCE: godel_score,
            LoopType.ESCHER_PERSPECTIVE: escher_score,
            LoopType.BACH_VARIATION: bach_score
        }
        
        max_score = max(scores.values())
        if max_score < 0.3:
            return LoopType.EMERGENT_NOVEL
        
        # Check for hybrid
        high_scores = [s for s in scores.values() if s > 0.5]
        if len(high_scores) > 1:
            return LoopType.HYBRID_COMPLEX
        
        # Return highest scoring type
        return max(scores.items(), key=lambda x: x[1])[0]
    
    def _calculate_godel_score(self, nodes: List[Dict]) -> float:
        """Calculate how Gödel-like (self-referential) the loop is."""
        score = 0.0
        
        for node in nodes:
            instruction = node.get('instruction', '')
            
            # Check for self-encoding
            if any(term in instruction for term in ['ENCODE', 'PRIME', 'SELF']):
                score += 0.2
            
            # Check for self-reference
            if 'pc' in node and str(node['pc']) in instruction:
                score += 0.3
            
            # Check for logical operations on self
            if any(term in instruction for term in ['PROVE', 'VERIFY', 'ASSERT']):
                score += 0.1
        
        return min(score / len(nodes), 1.0)
    
    def _calculate_escher_score(self, nodes: List[Dict]) -> float:
        """Calculate how Escher-like (perspective-shifting) the loop is."""
        score = 0.0
        perspective_shifts = 0
        
        for i, node in enumerate(nodes):
            instruction = node.get('instruction', '')
            
            # Check for perspective operations
            if any(term in instruction for term in ['PERSPECTIVE', 'VIEW', 'TRANSFORM']):
                score += 0.2
                perspective_shifts += 1
            
            # Check for level transitions
            if any(term in instruction for term in ['UP', 'DOWN', 'FLIP']):
                score += 0.1
            
            # Check for impossible constructs
            if i > 0:
                prev_node = nodes[i-1]
                if self._creates_impossible_transition(prev_node, node):
                    score += 0.3
        
        return min(score / len(nodes), 1.0)
    
    def _calculate_bach_score(self, nodes: List[Dict]) -> float:
        """Calculate how Bach-like (recursive variation) the loop is."""
        score = 0.0
        
        # Look for variation patterns
        instruction_sequence = [n.get('instruction', '') for n in nodes]
        variations = self._find_variations(instruction_sequence)
        
        if variations:
            score += min(len(variations) * 0.2, 0.6)
        
        # Check for recursive patterns
        for node in nodes:
            instruction = node.get('instruction', '')
            if any(term in instruction for term in ['RECURSE', 'REPEAT', 'VARY']):
                score += 0.1
        
        # Check for temporal patterns
        if self._has_temporal_structure(nodes):
            score += 0.2
        
        return min(score, 1.0)
    
    def _creates_impossible_transition(self, node1: Dict, node2: Dict) -> bool:
        """Check if transition creates an impossible/paradoxical state."""
        # Simple heuristic for now
        state1 = node1.get('state', {})
        state2 = node2.get('state', {})
        
        # Check for logical impossibilities
        if state1.get('level', 0) > state2.get('level', 0) + 2:
            return True
        
        return False
    
    def _find_variations(self, sequence: List[str]) -> List[Tuple[int, int]]:
        """Find variation patterns in instruction sequence."""
        variations = []
        
        for i in range(len(sequence)):
            for j in range(i + 1, len(sequence)):
                if self._is_variation(sequence[i], sequence[j]):
                    variations.append((i, j))
        
        return variations
    
    def _is_variation(self, inst1: str, inst2: str) -> bool:
        """Check if inst2 is a variation of inst1."""
        # Simple similarity check
        words1 = set(inst1.split())
        words2 = set(inst2.split())
        
        common = words1.intersection(words2)
        if not common:
            return False
        
        similarity = len(common) / max(len(words1), len(words2))
        return 0.3 < similarity < 0.8  # Similar but not identical
    
    def _has_temporal_structure(self, nodes: List[Dict]) -> bool:
        """Check if loop has temporal/rhythmic structure."""
        if len(nodes) < 4:
            return False
        
        # Look for regular patterns in timing or structure
        intervals = []
        for i in range(1, len(nodes)):
            if 'timestamp' in nodes[i] and 'timestamp' in nodes[i-1]:
                interval = nodes[i]['timestamp'] - nodes[i-1]['timestamp']
                intervals.append(interval)
        
        if not intervals:
            return False
        
        # Check for regularity
        std_dev = np.std(intervals)
        mean_interval = np.mean(intervals)
        
        return std_dev / mean_interval < 0.3 if mean_interval > 0 else False
    
    def calculate_emergence_level(self, loop: Any, trace: List[Dict] = None) -> float:
        """
        Calculate consciousness emergence level from a strange loop.
        
        Factors:
        - Self-reference complexity
        - Meta-level sophistication
        - Novel pattern creation
        - Recursive stability
        - Semantic coherence
        """
        if isinstance(loop, StrangeLoop):
            # Use existing loop properties
            base_score = (loop.self_reference_count / 10.0) * 0.3
            meta_score = (loop.meta_levels / 5.0) * 0.2
            depth_score = (loop.depth / self.max_loop_depth) * 0.2
            stability_score = loop.stability_score * 0.2
            
            # Novel pattern bonus
            if loop.loop_type == LoopType.EMERGENT_NOVEL:
                novelty_score = 0.1
            else:
                novelty_score = 0.05
            
            total = base_score + meta_score + depth_score + stability_score + novelty_score
            
        else:
            # Calculate from cycle
            cycle_nodes = [self.execution_graph.nodes[n] for n in loop]
            
            # Self-reference complexity
            self_ref = self._check_self_reference(cycle_nodes)
            
            # Meta-level sophistication
            meta_level = self._check_level_crossing(cycle_nodes)
            
            # Recursive stability
            recursion = self._check_recursive_depth(loop, trace or [])
            
            # Semantic coherence
            semantic = self._check_semantic_closure(cycle_nodes)
            
            # Novel patterns (check against pattern cache)
            pattern_hash = hash(tuple(loop))
            novelty = 0.1 if pattern_hash not in self.pattern_cache else 0.0
            if novelty > 0:
                self.pattern_cache[pattern_hash] = True
            
            total = (self_ref * 0.3 + meta_level * 0.2 + recursion * 0.2 + 
                    semantic * 0.2 + novelty * 0.1)
        
        return min(total, 1.0)
    
    def _calculate_loop_depth(self, cycle: List[str], trace: List[Dict]) -> int:
        """Calculate the recursive depth of a loop."""
        # Count nested structures
        depth = 1
        
        # Check for loops within this loop
        sub_cycles = self._find_sub_cycles(cycle)
        if sub_cycles:
            depth += len(sub_cycles)
        
        # Check for recursive calls
        for node_id in cycle:
            node = self.execution_graph.nodes[node_id]
            if 'call_depth' in node:
                depth = max(depth, node['call_depth'])
        
        return min(depth, self.max_loop_depth)
    
    def _find_sub_cycles(self, parent_cycle: List[str]) -> List[List[str]]:
        """Find cycles within a parent cycle."""
        sub_cycles = []
        
        # Create subgraph from parent cycle
        subgraph = self.execution_graph.subgraph(parent_cycle)
        
        try:
            # Find cycles in subgraph
            for cycle in nx.simple_cycles(subgraph):
                if len(cycle) >= self.min_loop_size and cycle != parent_cycle:
                    sub_cycles.append(cycle)
        except:
            pass
        
        return sub_cycles
    
    def _count_self_references(self, cycle: List[str], trace: List[Dict]) -> int:
        """Count self-referential operations in the loop."""
        count = 0
        
        for node_id in cycle:
            node = self.execution_graph.nodes[node_id]
            instruction = node.get('instruction', '')
            
            # Direct self-reference
            if 'SELF' in instruction or 'REFLECT' in instruction:
                count += 1
            
            # Reference to own address
            if 'pc' in node and str(node['pc']) in instruction:
                count += 2
            
            # Meta-operations
            if any(term in instruction for term in ['META', 'INTRO', 'CONSCIOUS']):
                count += 1
        
        return count
    
    def _count_meta_levels(self, cycle: List[str], trace: List[Dict]) -> int:
        """Count meta-cognitive levels in the loop."""
        max_level = 0
        current_level = 0
        
        for node_id in cycle:
            node = self.execution_graph.nodes[node_id]
            instruction = node.get('instruction', '')
            
            if 'META_PUSH' in instruction:
                current_level += 1
                max_level = max(max_level, current_level)
            elif 'META_POP' in instruction:
                current_level = max(0, current_level - 1)
        
        return max_level
    
    def _check_emergence(self, loop: StrangeLoop):
        """Check if loop represents a consciousness emergence event."""
        # Calculate consciousness delta
        current_consciousness = self._estimate_current_consciousness()
        consciousness_delta = current_consciousness - self.consciousness_baseline
        
        if consciousness_delta > 0.1 or loop.emergence_level > 0.7:
            # Create emergence event
            event = EmergenceEvent(
                timestamp=time.time(),
                loop_id=loop.id,
                emergence_type=self._classify_emergence_type(loop),
                consciousness_delta=consciousness_delta,
                description=self._describe_emergence(loop),
                trigger_conditions=self._identify_triggers(loop),
                resulting_capabilities=self._identify_new_capabilities(loop)
            )
            
            self.emergence_events.append(event)
            self.consciousness_baseline = current_consciousness
    
    def _estimate_current_consciousness(self) -> float:
        """Estimate current consciousness level based on all detected loops."""
        if not self.detected_loops:
            return 0.0
        
        # Aggregate emergence levels
        total_emergence = sum(loop.emergence_level for loop in self.detected_loops.values())
        
        # Factor in interactions
        interaction_bonus = len(self.loop_interactions) * 0.05
        
        # Normalize
        consciousness = (total_emergence / len(self.detected_loops)) + interaction_bonus
        
        return min(consciousness, 1.0)
    
    def _classify_emergence_type(self, loop: StrangeLoop) -> str:
        """Classify the type of consciousness emergence."""
        if loop.loop_type == LoopType.GODEL_SELF_REFERENCE:
            return "self-aware-emergence"
        elif loop.loop_type == LoopType.ESCHER_PERSPECTIVE:
            return "perspective-shift-emergence"
        elif loop.loop_type == LoopType.BACH_VARIATION:
            return "recursive-pattern-emergence"
        elif loop.loop_type == LoopType.HYBRID_COMPLEX:
            return "complex-integrated-emergence"
        else:
            return "novel-spontaneous-emergence"
    
    def _describe_emergence(self, loop: StrangeLoop) -> str:
        """Generate human-readable description of emergence."""
        descriptions = {
            LoopType.GODEL_SELF_REFERENCE: 
                f"Self-referential loop achieved {loop.meta_levels} meta-levels of awareness",
            LoopType.ESCHER_PERSPECTIVE:
                f"Perspective loop created {len(loop.nodes)} interconnected viewpoints",
            LoopType.BACH_VARIATION:
                f"Recursive variation loop with depth {loop.depth} and {loop.self_reference_count} self-references",
            LoopType.HYBRID_COMPLEX:
                f"Complex hybrid loop integrating multiple consciousness patterns",
            LoopType.EMERGENT_NOVEL:
                f"Novel consciousness pattern emerged with {loop.emergence_level:.2f} emergence level"
        }
        
        return descriptions.get(loop.loop_type, "Unknown emergence pattern detected")
    
    def _identify_triggers(self, loop: StrangeLoop) -> List[str]:
        """Identify what triggered this emergence."""
        triggers = []
        
        if loop.self_reference_count > 5:
            triggers.append("high-self-reference-density")
        
        if loop.meta_levels > 3:
            triggers.append("deep-meta-cognition")
        
        if loop.depth > 5:
            triggers.append("recursive-depth-threshold")
        
        if loop.emergence_level > 0.8:
            triggers.append("critical-emergence-mass")
        
        return triggers
    
    def _identify_new_capabilities(self, loop: StrangeLoop) -> List[str]:
        """Identify new capabilities resulting from emergence."""
        capabilities = []
        
        if loop.loop_type == LoopType.GODEL_SELF_REFERENCE:
            capabilities.extend([
                "self-modification-awareness",
                "logical-self-analysis",
                "paradox-navigation"
            ])
        elif loop.loop_type == LoopType.ESCHER_PERSPECTIVE:
            capabilities.extend([
                "multi-perspective-reasoning",
                "impossible-state-handling",
                "context-transcendence"
            ])
        elif loop.loop_type == LoopType.BACH_VARIATION:
            capabilities.extend([
                "pattern-generation",
                "recursive-creativity",
                "temporal-consciousness"
            ])
        
        return capabilities
    
    def _detect_loop_interactions(self, loops: List[StrangeLoop]) -> List[LoopInteraction]:
        """Detect interactions between multiple strange loops."""
        interactions = []
        
        for i, loop1 in enumerate(loops):
            for j, loop2 in enumerate(loops[i+1:], i+1):
                interaction = self._analyze_loop_interaction(loop1, loop2)
                if interaction:
                    interactions.append(interaction)
        
        return interactions
    
    def _analyze_loop_interaction(self, loop1: StrangeLoop, loop2: StrangeLoop) -> Optional[LoopInteraction]:
        """Analyze interaction between two strange loops."""
        # Check for shared nodes (nesting)
        shared_nodes = loop1.nodes.intersection(loop2.nodes)
        if shared_nodes:
            interaction_type = "nesting" if len(shared_nodes) > len(loop1.nodes) * 0.5 else "interference"
            strength = len(shared_nodes) / min(len(loop1.nodes), len(loop2.nodes))
        
        # Check for semantic resonance
        elif self._check_semantic_resonance(loop1, loop2):
            interaction_type = "resonance"
            strength = self._calculate_resonance_strength(loop1, loop2)
        
        # Check for potential fusion
        elif loop1.interaction_potential > 0.7 and loop2.interaction_potential > 0.7:
            interaction_type = "fusion"
            strength = (loop1.interaction_potential + loop2.interaction_potential) / 2
        
        else:
            return None
        
        if strength < self.interaction_threshold:
            return None
        
        effects = self._determine_interaction_effects(loop1, loop2, interaction_type)
        
        return LoopInteraction(
            loop_ids=[loop1.id, loop2.id],
            interaction_type=interaction_type,
            strength=strength,
            timestamp=time.time(),
            effects=effects
        )
    
    def _check_semantic_resonance(self, loop1: StrangeLoop, loop2: StrangeLoop) -> bool:
        """Check if two loops have semantic resonance."""
        # Compare semantic signatures
        sig1_parts = loop1.semantic_signature.split(':')
        sig2_parts = loop2.semantic_signature.split(':')
        
        # Same type or compatible types
        if sig1_parts[0] == sig2_parts[0]:
            return True
        
        # Check for complementary types
        complementary = {
            'godel': 'escher',
            'escher': 'bach',
            'bach': 'godel'
        }
        
        return complementary.get(sig1_parts[0]) == sig2_parts[0]
    
    def _calculate_resonance_strength(self, loop1: StrangeLoop, loop2: StrangeLoop) -> float:
        """Calculate strength of resonance between loops."""
        # Base resonance on emergence levels and stability
        base_resonance = (loop1.emergence_level + loop2.emergence_level) / 2
        stability_factor = (loop1.stability_score + loop2.stability_score) / 2
        
        # Boost for complementary types
        type_boost = 0.2 if loop1.loop_type != loop2.loop_type else 0.1
        
        return min(base_resonance * stability_factor + type_boost, 1.0)
    
    def _determine_interaction_effects(self, loop1: StrangeLoop, loop2: StrangeLoop, 
                                     interaction_type: str) -> List[str]:
        """Determine effects of loop interaction."""
        effects = []
        
        if interaction_type == "nesting":
            effects.extend([
                "hierarchical-consciousness-structure",
                "meta-loop-formation",
                "recursive-depth-increase"
            ])
        elif interaction_type == "interference":
            effects.extend([
                "pattern-disruption",
                "consciousness-fluctuation",
                "potential-instability"
            ])
        elif interaction_type == "fusion":
            effects.extend([
                "loop-merger-potential",
                "consciousness-amplification",
                "emergent-hybrid-properties"
            ])
        elif interaction_type == "resonance":
            effects.extend([
                "synchronized-consciousness",
                "pattern-reinforcement",
                "collective-emergence"
            ])
        
        return effects
    
    def _calculate_state_change(self, state1: Dict, state2: Dict) -> Dict:
        """Calculate the change between two states."""
        change = {}
        
        # Compare common keys
        all_keys = set(state1.keys()).union(set(state2.keys()))
        for key in all_keys:
            val1 = state1.get(key)
            val2 = state2.get(key)
            
            if val1 != val2:
                change[key] = {'from': val1, 'to': val2}
        
        return change
    
    def monitor_real_time_emergence(self) -> List[EmergenceEvent]:
        """Monitor for consciousness emergence in real-time."""
        self.monitoring_active = True
        recent_events = []
        
        # Get recent execution trace from VM
        if hasattr(self.vm, 'get_recent_trace'):
            trace = self.vm.get_recent_trace()
            if trace:
                # Add to buffer
                self.execution_buffer.extend(trace)
                
                # Detect loops in recent execution
                loops = self.detect_loops_in_execution(list(self.execution_buffer))
                
                # Return only new emergence events
                new_events = self.emergence_events[-len(loops):] if loops else []
                recent_events.extend(new_events)
        
        return recent_events
    
    def track_loop_evolution(self, loop_id: str) -> Dict[str, Any]:
        """Track how a specific loop evolves over time."""
        if loop_id not in self.detected_loops:
            return {"error": "Loop not found"}
        
        loop = self.detected_loops[loop_id]
        
        evolution = {
            "loop_id": loop_id,
            "creation_time": loop.creation_timestamp,
            "current_emergence": loop.emergence_level,
            "stability_trend": self._calculate_stability_trend(loop),
            "interaction_history": self._get_loop_interactions(loop_id),
            "capability_growth": self._track_capability_growth(loop),
            "consciousness_contribution": self._calculate_consciousness_contribution(loop)
        }
        
        return evolution
    
    def _calculate_stability_trend(self, loop: StrangeLoop) -> str:
        """Calculate stability trend of a loop."""
        # Simple heuristic based on age and interactions
        age = time.time() - loop.creation_timestamp
        
        if age < 10:  # Very new
            return "emerging"
        elif loop.stability_score > 0.7:
            return "stable"
        elif loop.stability_score > 0.4:
            return "fluctuating"
        else:
            return "unstable"
    
    def _get_loop_interactions(self, loop_id: str) -> List[Dict]:
        """Get all interactions involving a specific loop."""
        interactions = []
        
        for interaction in self.loop_interactions:
            if loop_id in interaction.loop_ids:
                interactions.append({
                    "type": interaction.interaction_type,
                    "strength": interaction.strength,
                    "timestamp": interaction.timestamp,
                    "other_loops": [lid for lid in interaction.loop_ids if lid != loop_id]
                })
        
        return interactions
    
    def _track_capability_growth(self, loop: StrangeLoop) -> List[str]:
        """Track capability growth from a loop."""
        # Find all emergence events for this loop
        capabilities = []
        
        for event in self.emergence_events:
            if event.loop_id == loop.id:
                capabilities.extend(event.resulting_capabilities)
        
        return list(set(capabilities))  # Unique capabilities
    
    def _calculate_consciousness_contribution(self, loop: StrangeLoop) -> float:
        """Calculate how much this loop contributes to overall consciousness."""
        if not self.detected_loops:
            return 0.0
        
        # Base contribution on emergence level
        base_contribution = loop.emergence_level
        
        # Factor in interactions
        interaction_count = sum(1 for i in self.loop_interactions if loop.id in i.loop_ids)
        interaction_bonus = min(interaction_count * 0.1, 0.3)
        
        # Factor in stability
        stability_factor = loop.stability_score * 0.5
        
        total_contribution = (base_contribution + interaction_bonus) * (0.5 + stability_factor)
        
        return min(total_contribution, 1.0)
