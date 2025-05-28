"""
Semantic Analyzer Utilities

This module provides tools for analyzing semantic representations,
relationships between concepts, and meaning structures.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import defaultdict


class SemanticRelationType(Enum):
    """Types of semantic relationships"""
    SYNONYM = "synonym"
    ANTONYM = "antonym"
    HYPERNYM = "hypernym"  # Is-a relationship
    HYPONYM = "hyponym"  # Type-of relationship
    MERONYM = "meronym"  # Part-of relationship
    HOLONYM = "holonym"  # Whole-of relationship
    CAUSAL = "causal"
    TEMPORAL = "temporal"
    ASSOCIATIVE = "associative"


class ConceptType(Enum):
    """Types of concepts"""
    CONCRETE = "concrete"
    ABSTRACT = "abstract"
    PROCESS = "process"
    STATE = "state"
    PROPERTY = "property"
    RELATION = "relation"
    EVENT = "event"


@dataclass
class SemanticConcept:
    """A semantic concept with properties"""
    concept_id: str
    name: str
    concept_type: ConceptType
    prime_encoding: int
    properties: Dict[str, Any] = field(default_factory=dict)
    semantic_features: List[str] = field(default_factory=list)
    context_embeddings: List[float] = field(default_factory=list)


@dataclass
class SemanticRelation:
    """A semantic relationship between concepts"""
    source_concept: str
    target_concept: str
    relation_type: SemanticRelationType
    strength: float = 1.0
    bidirectional: bool = False
    context: Optional[str] = None


@dataclass
class SemanticNetwork:
    """A network of semantic concepts and relations"""
    concepts: Dict[str, SemanticConcept]
    relations: List[SemanticRelation]
    hierarchies: Dict[str, List[str]]  # concept -> children
    clusters: List[Set[str]]  # Semantic clusters


@dataclass
class SemanticAnalysisReport:
    """Report from semantic analysis"""
    concept_count: int
    relation_count: int
    semantic_density: float
    coherence_score: float
    key_concepts: List[str]
    semantic_clusters: List[Dict[str, Any]]
    anomalies: List[Dict[str, Any]]


class SemanticAnalyzer:
    """
    Analyzes semantic representations and relationships.
    """
    
    def __init__(self):
        self.concept_cache = {}
        self.relation_cache = defaultdict(list)
        self.semantic_features = self._initialize_semantic_features()
        self.prime_semantic_map = self._initialize_prime_semantics()
        
    def _initialize_semantic_features(self) -> Dict[str, List[str]]:
        """Initialize semantic feature sets"""
        return {
            "animate": ["living", "conscious", "mobile", "growing"],
            "abstract": ["conceptual", "intangible", "theoretical", "mental"],
            "temporal": ["time-bound", "changing", "sequential", "durational"],
            "spatial": ["located", "extended", "dimensional", "positional"],
            "causal": ["causing", "effecting", "influencing", "determining"],
            "intentional": ["purposeful", "goal-directed", "deliberate", "planned"],
            "emotional": ["affective", "feeling", "evaluative", "experiential"],
            "cognitive": ["mental", "thinking", "reasoning", "knowing"]
        }
    
    def _initialize_prime_semantics(self) -> Dict[int, str]:
        """Initialize prime number to semantic mapping"""
        # First 20 primes mapped to fundamental concepts
        return {
            2: "existence",
            3: "negation",
            5: "identity",
            7: "difference",
            11: "relation",
            13: "property",
            17: "process",
            19: "state",
            23: "time",
            29: "space",
            31: "cause",
            37: "effect",
            41: "part",
            43: "whole",
            47: "possibility",
            53: "necessity",
            59: "knowledge",
            61: "belief",
            67: "intention",
            71: "action"
        }
    
    def analyze_semantic_network(self, network: SemanticNetwork) -> SemanticAnalysisReport:
        """Analyze a semantic network"""
        # Calculate basic metrics
        concept_count = len(network.concepts)
        relation_count = len(network.relations)
        
        # Calculate semantic density
        max_relations = concept_count * (concept_count - 1)
        density = relation_count / max_relations if max_relations > 0 else 0
        
        # Assess coherence
        coherence = self._calculate_network_coherence(network)
        
        # Identify key concepts
        key_concepts = self._identify_key_concepts(network)
        
        # Find semantic clusters
        clusters = self._find_semantic_clusters(network)
        
        # Detect anomalies
        anomalies = self._detect_semantic_anomalies(network)
        
        return SemanticAnalysisReport(
            concept_count=concept_count,
            relation_count=relation_count,
            semantic_density=density,
            coherence_score=coherence,
            key_concepts=key_concepts,
            semantic_clusters=clusters,
            anomalies=anomalies
        )
    
    def calculate_semantic_similarity(self, concept1: SemanticConcept,
                                    concept2: SemanticConcept) -> float:
        """Calculate semantic similarity between concepts"""
        # Prime-based similarity
        prime_sim = self._prime_similarity(concept1.prime_encoding, concept2.prime_encoding)
        
        # Feature-based similarity
        feature_sim = self._feature_similarity(concept1.semantic_features, 
                                             concept2.semantic_features)
        
        # Property-based similarity
        property_sim = self._property_similarity(concept1.properties, concept2.properties)
        
        # Type compatibility
        type_compat = 1.0 if concept1.concept_type == concept2.concept_type else 0.5
        
        # Weighted combination
        similarity = (
            0.3 * prime_sim +
            0.3 * feature_sim +
            0.2 * property_sim +
            0.2 * type_compat
        )
        
        return similarity
    
    def find_semantic_path(self, network: SemanticNetwork,
                         start_concept: str,
                         end_concept: str) -> Optional[List[str]]:
        """Find semantic path between concepts"""
        if start_concept not in network.concepts or end_concept not in network.concepts:
            return None
        
        # Build adjacency list
        adjacency = defaultdict(list)
        for relation in network.relations:
            adjacency[relation.source_concept].append(relation.target_concept)
            if relation.bidirectional:
                adjacency[relation.target_concept].append(relation.source_concept)
        
        # BFS to find shortest path
        queue = [(start_concept, [start_concept])]
        visited = {start_concept}
        
        while queue:
            current, path = queue.pop(0)
            
            if current == end_concept:
                return path
            
            for neighbor in adjacency[current]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    queue.append((neighbor, path + [neighbor]))
        
        return None
    
    def extract_semantic_features(self, text: str) -> List[str]:
        """Extract semantic features from text"""
        features = []
        text_lower = text.lower()
        
        # Check against feature categories
        for category, feature_words in self.semantic_features.items():
            if any(word in text_lower for word in feature_words):
                features.append(category)
        
        # Add specific features based on patterns
        if any(word in text_lower for word in ["is", "are", "exists"]):
            features.append("existential")
        
        if any(word in text_lower for word in ["if", "then", "because"]):
            features.append("conditional")
        
        if any(word in text_lower for word in ["all", "every", "some", "no"]):
            features.append("quantified")
        
        return features
    
    def decompose_compound_concept(self, concept: SemanticConcept) -> List[SemanticConcept]:
        """Decompose compound concept into components"""
        components = []
        
        # Factor prime encoding
        factors = self._factorize_prime(concept.prime_encoding)
        
        for factor in factors:
            if factor in self.prime_semantic_map:
                component = SemanticConcept(
                    concept_id=f"{concept.concept_id}_component_{factor}",
                    name=self.prime_semantic_map[factor],
                    concept_type=ConceptType.ABSTRACT,
                    prime_encoding=factor
                )
                components.append(component)
        
        return components
    
    def merge_concepts(self, concepts: List[SemanticConcept]) -> SemanticConcept:
        """Merge multiple concepts into compound concept"""
        # Multiply primes for compound encoding
        compound_prime = 1
        for concept in concepts:
            compound_prime *= concept.prime_encoding
        
        # Merge properties
        merged_properties = {}
        for concept in concepts:
            merged_properties.update(concept.properties)
        
        # Union of features
        merged_features = list(set().union(*[set(c.semantic_features) for c in concepts]))
        
        # Create compound name
        compound_name = "+".join(c.name for c in concepts)
        
        return SemanticConcept(
            concept_id=f"compound_{compound_prime}",
            name=compound_name,
            concept_type=ConceptType.ABSTRACT,
            prime_encoding=compound_prime,
            properties=merged_properties,
            semantic_features=merged_features
        )
    
    def analyze_semantic_field(self, central_concept: SemanticConcept,
                             network: SemanticNetwork,
                             depth: int = 2) -> Dict[str, Any]:
        """Analyze semantic field around concept"""
        field = {
            "center": central_concept.name,
            "layers": [],
            "total_concepts": 1,
            "field_coherence": 1.0
        }
        
        visited = {central_concept.concept_id}
        current_layer = [central_concept.concept_id]
        
        for layer_num in range(depth):
            next_layer = []
            layer_concepts = []
            
            for concept_id in current_layer:
                # Find related concepts
                for relation in network.relations:
                    neighbor_id = None
                    
                    if relation.source_concept == concept_id:
                        neighbor_id = relation.target_concept
                    elif relation.bidirectional and relation.target_concept == concept_id:
                        neighbor_id = relation.source_concept
                    
                    if neighbor_id and neighbor_id not in visited:
                        visited.add(neighbor_id)
                        next_layer.append(neighbor_id)
                        
                        if neighbor_id in network.concepts:
                            layer_concepts.append({
                                "concept": network.concepts[neighbor_id].name,
                                "relation": relation.relation_type.value,
                                "strength": relation.strength
                            })
            
            if layer_concepts:
                field["layers"].append({
                    "depth": layer_num + 1,
                    "concepts": layer_concepts,
                    "count": len(layer_concepts)
                })
                field["total_concepts"] += len(layer_concepts)
            
            current_layer = next_layer
            
            if not current_layer:
                break
        
        # Calculate field coherence
        field["field_coherence"] = self._calculate_field_coherence(field, network)
        
        return field
    
    def identify_semantic_roles(self, concept: SemanticConcept,
                              network: SemanticNetwork) -> Dict[str, List[str]]:
        """Identify semantic roles of concept in network"""
        roles = {
            "agent": [],
            "patient": [],
            "instrument": [],
            "location": [],
            "time": [],
            "cause": [],
            "effect": []
        }
        
        # Analyze relations to determine roles
        for relation in network.relations:
            if relation.source_concept == concept.concept_id:
                if relation.relation_type == SemanticRelationType.CAUSAL:
                    roles["cause"].append(relation.target_concept)
                elif relation.relation_type == SemanticRelationType.TEMPORAL:
                    roles["time"].append(relation.target_concept)
            
            elif relation.target_concept == concept.concept_id:
                if relation.relation_type == SemanticRelationType.CAUSAL:
                    roles["effect"].append(relation.source_concept)
        
        # Infer roles from concept type and features
        if concept.concept_type == ConceptType.PROCESS:
            if "intentional" in concept.semantic_features:
                roles["agent"].append(concept.name)
        
        return {k: v for k, v in roles.items() if v}  # Only non-empty roles
    
    def detect_semantic_patterns(self, network: SemanticNetwork) -> List[Dict[str, Any]]:
        """Detect patterns in semantic network"""
        patterns = []
        
        # Detect hierarchical patterns
        hierarchies = self._find_hierarchies(network)
        for root, hierarchy in hierarchies.items():
            patterns.append({
                "type": "hierarchy",
                "root": root,
                "depth": self._hierarchy_depth(hierarchy),
                "breadth": len(hierarchy)
            })
        
        # Detect circular patterns
        cycles = self._find_semantic_cycles(network)
        for cycle in cycles:
            patterns.append({
                "type": "cycle",
                "concepts": cycle,
                "length": len(cycle)
            })
        
        # Detect hub patterns
        hubs = self._find_semantic_hubs(network)
        for hub, connections in hubs.items():
            patterns.append({
                "type": "hub",
                "center": hub,
                "connections": connections,
                "centrality": connections / len(network.concepts)
            })
        
        return patterns
    
    # Private helper methods
    
    def _calculate_network_coherence(self, network: SemanticNetwork) -> float:
        """Calculate overall network coherence"""
        if not network.concepts:
            return 0.0
        
        # Average pairwise similarity
        total_similarity = 0
        pairs = 0
        
        concepts_list = list(network.concepts.values())
        for i in range(len(concepts_list)):
            for j in range(i + 1, len(concepts_list)):
                similarity = self.calculate_semantic_similarity(
                    concepts_list[i], 
                    concepts_list[j]
                )
                total_similarity += similarity
                pairs += 1
        
        avg_similarity = total_similarity / pairs if pairs > 0 else 0
        
        # Connectivity factor
        connectivity = len(network.relations) / (len(network.concepts) * 2)
        connectivity_factor = min(1.0, connectivity)
        
        # Combine factors
        coherence = (avg_similarity + connectivity_factor) / 2
        
        return coherence
    
    def _identify_key_concepts(self, network: SemanticNetwork) -> List[str]:
        """Identify most important concepts in network"""
        # Count connections per concept
        connection_counts = defaultdict(int)
        
        for relation in network.relations:
            connection_counts[relation.source_concept] += 1
            if relation.bidirectional:
                connection_counts[relation.target_concept] += 1
        
        # Sort by connection count
        sorted_concepts = sorted(
            connection_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )
        
        # Return top concepts
        return [concept_id for concept_id, _ in sorted_concepts[:5]]
    
    def _find_semantic_clusters(self, network: SemanticNetwork) -> List[Dict[str, Any]]:
        """Find clusters of related concepts"""
        clusters = []
        visited = set()
        
        # Simple connected components as clusters
        for concept_id in network.concepts:
            if concept_id not in visited:
                cluster = self._explore_cluster(concept_id, network, visited)
                
                if len(cluster) > 1:
                    # Calculate cluster properties
                    cluster_concepts = [network.concepts[cid] for cid in cluster]
                    
                    clusters.append({
                        "concepts": cluster,
                        "size": len(cluster),
                        "coherence": self._calculate_cluster_coherence(cluster_concepts),
                        "central_theme": self._identify_cluster_theme(cluster_concepts)
                    })
        
        return clusters
    
    def _explore_cluster(self, start_id: str, network: SemanticNetwork,
                        visited: Set[str]) -> List[str]:
        """Explore connected cluster from starting concept"""
        cluster = []
        queue = [start_id]
        
        while queue:
            current = queue.pop(0)
            if current in visited:
                continue
            
            visited.add(current)
            cluster.append(current)
            
            # Find neighbors
            for relation in network.relations:
                if relation.source_concept == current:
                    if relation.target_concept not in visited:
                        queue.append(relation.target_concept)
                elif relation.bidirectional and relation.target_concept == current:
                    if relation.source_concept not in visited:
                        queue.append(relation.source_concept)
        
        return cluster
    
    def _calculate_cluster_coherence(self, concepts: List[SemanticConcept]) -> float:
        """Calculate coherence within cluster"""
        if len(concepts) < 2:
            return 1.0
        
        total_similarity = 0
        pairs = 0
        
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                similarity = self.calculate_semantic_similarity(concepts[i], concepts[j])
                total_similarity += similarity
                pairs += 1
        
        return total_similarity / pairs if pairs > 0 else 0
    
    def _identify_cluster_theme(self, concepts: List[SemanticConcept]) -> str:
        """Identify central theme of cluster"""
        # Count feature frequencies
        feature_counts = defaultdict(int)
        
        for concept in concepts:
            for feature in concept.semantic_features:
                feature_counts[feature] += 1
        
        # Most common feature as theme
        if feature_counts:
            theme = max(feature_counts.items(), key=lambda x: x[1])[0]
            return theme
        
        # Fallback to most common concept type
        type_counts = defaultdict(int)
        for concept in concepts:
            type_counts[concept.concept_type.value] += 1
        
        if type_counts:
            return max(type_counts.items(), key=lambda x: x[1])[0]
        
        return "mixed"
    
    def _detect_semantic_anomalies(self, network: SemanticNetwork) -> List[Dict[str, Any]]:
        """Detect anomalies in semantic network"""
        anomalies = []
        
        # Isolated concepts
        connected = set()
        for relation in network.relations:
            connected.add(relation.source_concept)
            connected.add(relation.target_concept)
        
        for concept_id in network.concepts:
            if concept_id not in connected:
                anomalies.append({
                    "type": "isolated_concept",
                    "concept": concept_id,
                    "severity": "low"
                })
        
        # Type mismatches in relations
        for relation in network.relations:
            if (relation.source_concept in network.concepts and 
                relation.target_concept in network.concepts):
                
                source = network.concepts[relation.source_concept]
                target = network.concepts[relation.target_concept]
                
                # Check for incompatible relation types
                if (relation.relation_type == SemanticRelationType.MERONYM and
                    source.concept_type == ConceptType.ABSTRACT and
                    target.concept_type == ConceptType.ABSTRACT):
                    
                    anomalies.append({
                        "type": "relation_type_mismatch",
                        "relation": f"{source.name} -[{relation.relation_type.value}]-> {target.name}",
                        "severity": "medium"
                    })
        
        return anomalies
    
    def _prime_similarity(self, prime1: int, prime2: int) -> float:
        """Calculate similarity based on prime encodings"""
        # GCD-based similarity
        gcd = math.gcd(prime1, prime2)
        
        if gcd == 1:
            return 0.0  # No common factors
        
        # Similarity based on shared factors
        factors1 = self._factorize_prime(prime1)
        factors2 = self._factorize_prime(prime2)
        
        shared = set(factors1) & set(factors2)
        total = set(factors1) | set(factors2)
        
        return len(shared) / len(total) if total else 0
    
    def _feature_similarity(self, features1: List[str], features2: List[str]) -> float:
        """Calculate similarity based on semantic features"""
        if not features1 and not features2:
            return 1.0
        
        set1 = set(features1)
        set2 = set(features2)
        
        intersection = set1 & set2
        union = set1 | set2
        
        return len(intersection) / len(union) if union else 0
    
    def _property_similarity(self, props1: Dict[str, Any], props2: Dict[str, Any]) -> float:
        """Calculate similarity based on properties"""
        if not props1 and not props2:
            return 1.0
        
        # Compare shared properties
        shared_keys = set(props1.keys()) & set(props2.keys())
        
        if not shared_keys:
            return 0.0
        
        matches = sum(1 for k in shared_keys if props1[k] == props2[k])
        
        return matches / len(shared_keys)
    
    def _factorize_prime(self, n: int) -> List[int]:
        """Factorize number into prime factors"""
        factors = []
        d = 2
        
        while d * d <= n:
            while n % d == 0:
                factors.append(d)
                n //= d
            d += 1
        
        if n > 1:
            factors.append(n)
        
        return factors
    
    def _calculate_field_coherence(self, field: Dict[str, Any], 
                                 network: SemanticNetwork) -> float:
        """Calculate coherence of semantic field"""
        if field["total_concepts"] <= 1:
            return 1.0
        
        # Average strength of connections in field
        total_strength = 0
        connection_count = 0
        
        for layer in field["layers"]:
            for concept_info in layer["concepts"]:
                total_strength += concept_info["strength"]
                connection_count += 1
        
        avg_strength = total_strength / connection_count if connection_count > 0 else 0
        
        # Decay factor based on depth
        depth_factor = 1.0 / (1 + len(field["layers"]) * 0.1)
        
        return avg_strength * depth_factor
    
    def _find_hierarchies(self, network: SemanticNetwork) -> Dict[str, List[str]]:
        """Find hierarchical structures in network"""
        hierarchies = {}
        
        # Find potential roots (concepts with no hypernyms)
        potential_roots = set(network.concepts.keys())
        
        for relation in network.relations:
            if relation.relation_type == SemanticRelationType.HYPERNYM:
                potential_roots.discard(relation.source_concept)
        
        # Build hierarchy for each root
        for root in potential_roots:
            hierarchy = self._build_hierarchy(root, network)
            if hierarchy:
                hierarchies[root] = hierarchy
        
        return hierarchies
    
    def _build_hierarchy(self, root: str, network: SemanticNetwork) -> List[str]:
        """Build hierarchy from root concept"""
        hierarchy = []
        queue = [(root, 0)]
        visited = {root}
        
        while queue:
            current, level = queue.pop(0)
            hierarchy.append(current)
            
            # Find children
            for relation in network.relations:
                if (relation.source_concept == current and 
                    relation.relation_type == SemanticRelationType.HYPONYM and
                    relation.target_concept not in visited):
                    
                    visited.add(relation.target_concept)
                    queue.append((relation.target_concept, level + 1))
        
        return hierarchy if len(hierarchy) > 1 else []
    
    def _hierarchy_depth(self, hierarchy: List[str]) -> int:
        """Calculate depth of hierarchy"""
        # Simplified - would need proper tree structure
        return int(math.log2(len(hierarchy))) + 1
    
    def _find_semantic_cycles(self, network: SemanticNetwork) -> List[List[str]]:
        """Find cycles in semantic network"""
        cycles = []
        visited = set()
        
        def dfs(node: str, path: List[str], visiting: Set[str]):
            if node in visiting:
                # Found cycle
                cycle_start = path.index(node)
                cycle = path[cycle_start:]
                if len(cycle) > 2:  # Meaningful cycles only
                    cycles.append(cycle)
                return
            
            if node in visited:
                return
            
            visiting.add(node)
            
            for relation in network.relations:
                if relation.source_concept == node:
                    dfs(relation.target_concept, path + [relation.target_concept], visiting)
            
            visiting.remove(node)
            visited.add(node)
        
        # Start DFS from each unvisited node
        for concept_id in network.concepts:
            if concept_id not in visited:
                dfs(concept_id, [concept_id], set())
        
        return cycles
    
    def _find_semantic_hubs(self, network: SemanticNetwork) -> Dict[str, int]:
        """Find hub concepts with many connections"""
        connection_counts = defaultdict(int)
        
        for relation in network.relations:
            connection_counts[relation.source_concept] += 1
            connection_counts[relation.target_concept] += 1
        
        # Filter for hubs (more than average connections)
        avg_connections = sum(connection_counts.values()) / len(connection_counts) if connection_counts else 0
        
        hubs = {
            concept: count 
            for concept, count in connection_counts.items()
            if count > avg_connections * 1.5
        }
        
        return hubs
