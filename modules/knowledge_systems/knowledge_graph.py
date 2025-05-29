"""
Knowledge Graph

This module implements a dynamic knowledge graph that supports self-modification,
prime-based encoding, and relationship inference for intelligent reasoning.
"""

from typing import List, Dict, Optional, Set, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime
import networkx as nx

from core.prime_vm import PrimeNumbers


class NodeType(Enum):
    """Types of knowledge nodes"""
    CONCEPT = "concept"
    INSTANCE = "instance"
    PROPERTY = "property"
    RELATION = "relation"
    RULE = "rule"
    PATTERN = "pattern"


class EdgeType(Enum):
    """Types of knowledge edges"""
    IS_A = "is_a"
    HAS_PROPERTY = "has_property"
    RELATED_TO = "related_to"
    CAUSES = "causes"
    IMPLIES = "implies"
    PART_OF = "part_of"
    SIMILAR_TO = "similar_to"
    OPPOSITE_OF = "opposite_of"


class QueryType(Enum):
    """Types of graph queries"""
    NODE_LOOKUP = "node_lookup"
    PATH_FINDING = "path_finding"
    PATTERN_MATCHING = "pattern_matching"
    INFERENCE = "inference"
    SIMILARITY = "similarity"


class UpdateType(Enum):
    """Types of graph updates"""
    ADD_NODE = "add_node"
    ADD_EDGE = "add_edge"
    MODIFY_NODE = "modify_node"
    MODIFY_EDGE = "modify_edge"
    REMOVE_NODE = "remove_node"
    REMOVE_EDGE = "remove_edge"
    MERGE_NODES = "merge_nodes"


@dataclass
class KnowledgeNode:
    """A node in the knowledge graph"""
    node_id: str
    node_type: NodeType
    content: Dict[str, Any]
    prime_encoding: Optional[int] = None
    activation_level: float = 0.0
    creation_time: datetime = field(default_factory=datetime.now)
    last_accessed: datetime = field(default_factory=datetime.now)
    access_count: int = 0
    confidence: float = 1.0
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class KnowledgeEdge:
    """An edge in the knowledge graph"""
    edge_id: str
    source_id: str
    target_id: str
    edge_type: EdgeType
    weight: float = 1.0
    confidence: float = 1.0
    bidirectional: bool = False
    properties: Dict[str, Any] = field(default_factory=dict)
    creation_time: datetime = field(default_factory=datetime.now)


@dataclass
class GraphQuery:
    """A query on the knowledge graph"""
    query_id: str
    query_type: QueryType
    parameters: Dict[str, Any]
    constraints: List[Dict[str, Any]] = field(default_factory=list)
    max_results: Optional[int] = None


@dataclass
class GraphUpdate:
    """An update to the knowledge graph"""
    update_id: str
    update_type: UpdateType
    target_elements: List[str]
    update_data: Dict[str, Any]
    timestamp: datetime = field(default_factory=datetime.now)
    source: str = "system"


@dataclass
class InferenceResult:
    """Result of an inference operation"""
    inferred_nodes: List[KnowledgeNode]
    inferred_edges: List[KnowledgeEdge]
    confidence: float
    reasoning_path: List[str]
    supporting_evidence: List[str]


@dataclass
class PatternMatch:
    """A pattern match in the graph"""
    pattern_id: str
    matched_nodes: List[str]
    matched_edges: List[str]
    match_score: float
    substitutions: Dict[str, str]


class KnowledgeGraph:
    """
    Dynamic knowledge graph with self-modification capabilities,
    prime-based encoding, and intelligent relationship inference.
    """
    
    def __init__(self, prime_generator: Optional[PrimeNumbers] = None):
        self.graph = nx.MultiDiGraph()
        self.nodes: Dict[str, KnowledgeNode] = {}
        self.edges: Dict[str, KnowledgeEdge] = {}
        self.prime_generator = prime_generator or PrimeNumbers()
        
        # Indexes for efficient lookup
        self.type_index: Dict[NodeType, Set[str]] = {t: set() for t in NodeType}
        self.prime_index: Dict[int, str] = {}
        self.pattern_cache: Dict[str, PatternMatch] = {}
        
        # Learning and adaptation
        self.access_patterns: List[List[str]] = []
        self.inference_history: List[InferenceResult] = []
        self.update_history: List[GraphUpdate] = []
        
    def add_node(self, node: KnowledgeNode) -> bool:
        """Add a node to the knowledge graph"""
        if node.node_id in self.nodes:
            return False
        
        # Assign prime encoding if not provided
        if node.prime_encoding is None:
            node.prime_encoding = self._generate_prime_encoding(node)
        
        # Add to graph
        self.graph.add_node(node.node_id, **node.content)
        self.nodes[node.node_id] = node
        
        # Update indexes
        self.type_index[node.node_type].add(node.node_id)
        if node.prime_encoding:
            self.prime_index[node.prime_encoding] = node.node_id
        
        # Record update
        self._record_update(GraphUpdate(
            update_id=f"add_node_{node.node_id}",
            update_type=UpdateType.ADD_NODE,
            target_elements=[node.node_id],
            update_data={'node': node}
        ))
        
        return True
    
    def add_edge(self, edge: KnowledgeEdge) -> bool:
        """Add an edge to the knowledge graph"""
        if edge.edge_id in self.edges:
            return False
        
        # Verify nodes exist
        if edge.source_id not in self.nodes or edge.target_id not in self.nodes:
            return False
        
        # Add to graph
        self.graph.add_edge(
            edge.source_id, edge.target_id,
            key=edge.edge_id,
            weight=edge.weight,
            **edge.properties
        )
        self.edges[edge.edge_id] = edge
        
        # Add reverse edge if bidirectional
        if edge.bidirectional:
            self.graph.add_edge(
                edge.target_id, edge.source_id,
                key=f"{edge.edge_id}_reverse",
                weight=edge.weight,
                **edge.properties
            )
        
        # Update node relationships
        self._update_node_relationships(edge.source_id, edge.target_id)
        
        # Record update
        self._record_update(GraphUpdate(
            update_id=f"add_edge_{edge.edge_id}",
            update_type=UpdateType.ADD_EDGE,
            target_elements=[edge.edge_id],
            update_data={'edge': edge}
        ))
        
        return True
    
    def query(self, query: GraphQuery) -> List[Any]:
        """Execute a query on the knowledge graph"""
        if query.query_type == QueryType.NODE_LOOKUP:
            return self._query_nodes(query)
        elif query.query_type == QueryType.PATH_FINDING:
            return self._query_paths(query)
        elif query.query_type == QueryType.PATTERN_MATCHING:
            return self._query_patterns(query)
        elif query.query_type == QueryType.INFERENCE:
            return self._query_inference(query)
        elif query.query_type == QueryType.SIMILARITY:
            return self._query_similarity(query)
        else:
            return []
    
    def update(self, update: GraphUpdate) -> bool:
        """Apply an update to the knowledge graph"""
        success = False
        
        if update.update_type == UpdateType.ADD_NODE:
            node_data = update.update_data.get('node')
            if node_data:
                success = self.add_node(node_data)
                
        elif update.update_type == UpdateType.ADD_EDGE:
            edge_data = update.update_data.get('edge')
            if edge_data:
                success = self.add_edge(edge_data)
                
        elif update.update_type == UpdateType.MODIFY_NODE:
            for node_id in update.target_elements:
                success = self._modify_node(node_id, update.update_data)
                
        elif update.update_type == UpdateType.MODIFY_EDGE:
            for edge_id in update.target_elements:
                success = self._modify_edge(edge_id, update.update_data)
                
        elif update.update_type == UpdateType.REMOVE_NODE:
            for node_id in update.target_elements:
                success = self._remove_node(node_id)
                
        elif update.update_type == UpdateType.REMOVE_EDGE:
            for edge_id in update.target_elements:
                success = self._remove_edge(edge_id)
                
        elif update.update_type == UpdateType.MERGE_NODES:
            if len(update.target_elements) >= 2:
                success = self._merge_nodes(update.target_elements)
        
        if success:
            self._record_update(update)
        
        return success
    
    def infer_relationships(self, node_id: str) -> InferenceResult:
        """Infer new relationships for a node"""
        if node_id not in self.nodes:
            return InferenceResult([], [], 0.0, [], [])
        
        node = self.nodes[node_id]
        inferred_nodes = []
        inferred_edges = []
        reasoning_path = []
        supporting_evidence = []
        
        # Transitive inference
        transitive_results = self._infer_transitive_relations(node_id)
        inferred_edges.extend(transitive_results['edges'])
        reasoning_path.extend(transitive_results['reasoning'])
        
        # Analogical inference
        analogical_results = self._infer_analogical_relations(node_id)
        inferred_edges.extend(analogical_results['edges'])
        reasoning_path.extend(analogical_results['reasoning'])
        
        # Property inheritance
        inheritance_results = self._infer_inherited_properties(node_id)
        inferred_nodes.extend(inheritance_results['nodes'])
        reasoning_path.extend(inheritance_results['reasoning'])
        
        # Pattern-based inference
        pattern_results = self._infer_from_patterns(node_id)
        inferred_edges.extend(pattern_results['edges'])
        inferred_nodes.extend(pattern_results['nodes'])
        reasoning_path.extend(pattern_results['reasoning'])
        
        # Calculate overall confidence
        confidence = self._calculate_inference_confidence(
            inferred_nodes, inferred_edges, reasoning_path
        )
        
        result = InferenceResult(
            inferred_nodes=inferred_nodes,
            inferred_edges=inferred_edges,
            confidence=confidence,
            reasoning_path=reasoning_path,
            supporting_evidence=supporting_evidence
        )
        
        # Store inference history
        self.inference_history.append(result)
        
        return result
    
    def find_similar_nodes(self, node_id: str, threshold: float = 0.7) -> List[Tuple[str, float]]:
        """Find nodes similar to the given node"""
        if node_id not in self.nodes:
            return []
        
        source_node = self.nodes[node_id]
        similar_nodes = []
        
        # Compare with all nodes of the same type
        for candidate_id in self.type_index[source_node.node_type]:
            if candidate_id == node_id:
                continue
            
            similarity = self._calculate_node_similarity(source_node, self.nodes[candidate_id])
            if similarity >= threshold:
                similar_nodes.append((candidate_id, similarity))
        
        # Sort by similarity
        similar_nodes.sort(key=lambda x: x[1], reverse=True)
        
        return similar_nodes
    
    def activate_spreading(self, start_nodes: List[str], 
                         activation_strength: float = 1.0,
                         decay_factor: float = 0.8,
                         threshold: float = 0.1) -> Dict[str, float]:
        """Perform spreading activation from start nodes"""
        activation_levels = {node_id: 0.0 for node_id in self.nodes}
        
        # Initialize start nodes
        for node_id in start_nodes:
            if node_id in self.nodes:
                activation_levels[node_id] = activation_strength
                self.nodes[node_id].activation_level = activation_strength
        
        # Spread activation
        changed = True
        iterations = 0
        max_iterations = 10
        
        while changed and iterations < max_iterations:
            changed = False
            new_levels = activation_levels.copy()
            
            for node_id, current_level in activation_levels.items():
                if current_level > threshold:
                    # Spread to neighbors
                    for neighbor in self.graph.neighbors(node_id):
                        # Calculate spread amount
                        edge_data = self.graph.get_edge_data(node_id, neighbor)
                        if edge_data:
                            # Get the strongest edge if multiple exist
                            max_weight = max(
                                data.get('weight', 1.0) 
                                for data in edge_data.values()
                            )
                            spread = current_level * decay_factor * max_weight
                            
                            if spread > threshold:
                                new_level = new_levels[neighbor] + spread
                                if new_level > new_levels[neighbor]:
                                    new_levels[neighbor] = new_level
                                    changed = True
            
            activation_levels = new_levels
            iterations += 1
        
        # Update node activation levels
        for node_id, level in activation_levels.items():
            self.nodes[node_id].activation_level = level
        
        # Record access pattern
        activated_nodes = [
            node_id for node_id, level in activation_levels.items() 
            if level > threshold
        ]
        if activated_nodes:
            self.access_patterns.append(activated_nodes)
        
        return activation_levels
    
    def evolve_structure(self, performance_feedback: Dict[str, float]):
        """Evolve the graph structure based on performance feedback"""
        # Strengthen successful paths
        for path, score in performance_feedback.items():
            nodes_in_path = path.split('->')
            for i in range(len(nodes_in_path) - 1):
                self._strengthen_connection(nodes_in_path[i], nodes_in_path[i + 1], score)
        
        # Prune weak connections
        self._prune_weak_edges(threshold=0.1)
        
        # Discover new patterns
        new_patterns = self._discover_patterns()
        for pattern in new_patterns:
            self._encode_pattern(pattern)
        
        # Reorganize based on access patterns
        self._reorganize_by_access_patterns()
    
    def _generate_prime_encoding(self, node: KnowledgeNode) -> int:
        """Generate a prime encoding for a node"""
        # Use node type and content to generate encoding
        type_prime = self.prime_generator.get_nth_prime(node.node_type.value.__hash__() % 100)
        
        # Combine with content hash
        content_hash = hash(str(node.content)) % 1000
        content_prime = self.prime_generator.get_nth_prime(content_hash)
        
        # Create composite encoding
        encoding = type_prime * content_prime
        
        # Ensure uniqueness
        while encoding in self.prime_index:
            encoding = self.prime_generator.get_next_prime(encoding)
        
        return encoding
    
    def _query_nodes(self, query: GraphQuery) -> List[KnowledgeNode]:
        """Query nodes based on criteria"""
        results = []
        
        # Extract query parameters
        node_type = query.parameters.get('node_type')
        content_filter = query.parameters.get('content_filter', {})
        activation_threshold = query.parameters.get('activation_threshold', 0.0)
        
        # Filter nodes
        for node_id, node in self.nodes.items():
            # Type filter
            if node_type and node.node_type != node_type:
                continue
            
            # Content filter
            if content_filter:
                match = all(
                    node.content.get(key) == value 
                    for key, value in content_filter.items()
                )
                if not match:
                    continue
            
            # Activation filter
            if node.activation_level < activation_threshold:
                continue
            
            # Apply constraints
            if self._check_constraints(node, query.constraints):
                results.append(node)
                
                # Update access tracking
                node.last_accessed = datetime.now()
                node.access_count += 1
        
        # Limit results
        if query.max_results:
            results = results[:query.max_results]
        
        return results
    
    def _query_paths(self, query: GraphQuery) -> List[List[str]]:
        """Find paths between nodes"""
        source = query.parameters.get('source')
        target = query.parameters.get('target')
        max_length = query.parameters.get('max_length', 5)
        
        if not source or not target:
            return []
        
        if source not in self.nodes or target not in self.nodes:
            return []
        
        # Find all simple paths
        try:
            paths = list(nx.all_simple_paths(
                self.graph, source, target, cutoff=max_length
            ))
        except nx.NetworkXNoPath:
            paths = []
        
        # Apply constraints
        filtered_paths = []
        for path in paths:
            if self._check_path_constraints(path, query.constraints):
                filtered_paths.append(path)
        
        # Limit results
        if query.max_results:
            filtered_paths = filtered_paths[:query.max_results]
        
        return filtered_paths
    
    def _infer_transitive_relations(self, node_id: str) -> Dict[str, Any]:
        """Infer transitive relationships"""
        inferred_edges = []
        reasoning = []
        
        # Check IS_A transitivity
        is_a_chain = self._find_relation_chain(node_id, EdgeType.IS_A)
        for i in range(len(is_a_chain) - 2):
            # If A is_a B and B is_a C, then A is_a C
            new_edge = KnowledgeEdge(
                edge_id=f"inferred_{node_id}_is_a_{is_a_chain[i + 2]}",
                source_id=node_id,
                target_id=is_a_chain[i + 2],
                edge_type=EdgeType.IS_A,
                weight=0.9 ** (i + 2),  # Decay with distance
                confidence=0.8
            )
            inferred_edges.append(new_edge)
            reasoning.append(
                f"Transitive inference: {node_id} is_a {is_a_chain[i + 1]} "
                f"and {is_a_chain[i + 1]} is_a {is_a_chain[i + 2]}"
            )
        
        return {'edges': inferred_edges, 'reasoning': reasoning}
    
    def _calculate_node_similarity(self, node1: KnowledgeNode, 
                                 node2: KnowledgeNode) -> float:
        """Calculate similarity between two nodes"""
        # Type similarity
        type_sim = 1.0 if node1.node_type == node2.node_type else 0.5
        
        # Content similarity
        common_keys = set(node1.content.keys()) & set(node2.content.keys())
        if common_keys:
            matching_values = sum(
                1 for key in common_keys 
                if node1.content[key] == node2.content[key]
            )
            content_sim = matching_values / len(common_keys)
        else:
            content_sim = 0.0
        
        # Structural similarity (shared neighbors)
        neighbors1 = set(self.graph.neighbors(node1.node_id))
        neighbors2 = set(self.graph.neighbors(node2.node_id))
        if neighbors1 or neighbors2:
            structural_sim = len(neighbors1 & neighbors2) / len(neighbors1 | neighbors2)
        else:
            structural_sim = 0.0
        
        # Prime encoding similarity
        if node1.prime_encoding and node2.prime_encoding:
            # Use GCD as a measure of similarity
            gcd = np.gcd(node1.prime_encoding, node2.prime_encoding)
            prime_sim = gcd / max(node1.prime_encoding, node2.prime_encoding)
        else:
            prime_sim = 0.0
        
        # Weighted combination
        similarity = (
            type_sim * 0.2 +
            content_sim * 0.3 +
            structural_sim * 0.3 +
            prime_sim * 0.2
        )
        
        return similarity
