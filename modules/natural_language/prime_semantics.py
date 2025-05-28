"""
Prime-Based Semantic Representation System

This module implements a mathematical approach to semantic representation using
prime numbers, enabling mathematical operations on meaning and supporting
analogical reasoning through semantic similarity.
"""

from typing import Dict, List, Tuple, Optional, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import math
import numpy as np
from collections import defaultdict

from core.prime_vm import PrimeEncoder
from modules.knowledge_systems.knowledge_graph import KnowledgeGraph, Concept


class OperationType(Enum):
    """Types of semantic operations"""
    BLEND = "blend"
    CONTRAST = "contrast"
    NEGATE = "negate"
    INTENSIFY = "intensify"
    ABSTRACT = "abstract"
    CONCRETIZE = "concretize"
    METAPHORIZE = "metaphorize"


class ConceptType(Enum):
    """Types of concepts in semantic space"""
    CONCRETE = "concrete"
    ABSTRACT = "abstract"
    RELATIONAL = "relational"
    EMOTIONAL = "emotional"
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    MODAL = "modal"
    METACOGNITIVE = "metacognitive"


@dataclass
class SemanticDimension:
    """Represents a dimension in semantic space"""
    name: str
    prime_base: int
    weight: float = 1.0
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class CompositionRule:
    """Rule for composing semantic meanings"""
    name: str
    input_types: List[ConceptType]
    output_type: ConceptType
    composition_function: str
    constraints: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Property:
    """Emergent property from semantic composition"""
    name: str
    value: Any
    emergence_level: float
    source_concepts: List[str]


@dataclass
class SemanticSpace:
    """Represents the entire semantic space"""
    concept_primes: Dict[str, int] = field(default_factory=dict)
    relation_primes: Dict[str, int] = field(default_factory=dict)
    compositional_rules: List[CompositionRule] = field(default_factory=list)
    semantic_dimensions: List[SemanticDimension] = field(default_factory=list)
    
    def add_concept(self, concept: str, prime: int):
        """Add a concept to semantic space"""
        self.concept_primes[concept] = prime
    
    def add_relation(self, relation: str, prime: int):
        """Add a relation to semantic space"""
        self.relation_primes[relation] = prime


@dataclass
class ComposedMeaning:
    """Represents a composed semantic meaning"""
    constituent_concepts: List[Concept]
    composition_prime: int
    emergent_properties: List[Property]
    semantic_coherence: float
    meaning_depth: int
    composition_path: List[str]


@dataclass
class SemanticOperation:
    """Represents a semantic operation"""
    operation_type: OperationType
    input_concepts: List[Concept]
    operation_parameters: Dict[str, Any]
    expected_output_type: ConceptType


@dataclass
class SemanticResult:
    """Result of a semantic operation"""
    result_concept: Concept
    result_prime: int
    operation_trace: List[str]
    confidence: float
    semantic_validity: bool


@dataclass
class Analogy:
    """Represents a semantic analogy"""
    source_concept: Concept
    target_concept: Concept
    mapping_strength: float
    shared_structure: Dict[str, Any]
    prime_relationship: Tuple[int, int]


@dataclass
class SemanticField:
    """Semantic field around a concept"""
    central_concept: Concept
    related_concepts: List[Tuple[Concept, float]]  # (concept, distance)
    semantic_clusters: List[List[Concept]]
    field_coherence: float
    metaphorical_connections: List[Analogy]


class PrimeSemantics:
    """
    Prime-based semantic representation system that enables mathematical
    operations on meaning and supports analogical reasoning.
    """
    
    def __init__(self, prime_encoder: PrimeEncoder, knowledge_graph: KnowledgeGraph):
        self.prime_encoder = prime_encoder
        self.knowledge_graph = knowledge_graph
        self.semantic_space = self._initialize_semantic_space()
        self.prime_cache = {}
        self.similarity_cache = {}
        self.composition_history = []
        
    def _initialize_semantic_space(self) -> SemanticSpace:
        """Initialize the semantic space with base concepts and relations"""
        space = SemanticSpace()
        
        # Initialize semantic dimensions
        dimensions = [
            SemanticDimension("concreteness", 2, 1.5),
            SemanticDimension("animacy", 3, 1.2),
            SemanticDimension("agency", 5, 1.3),
            SemanticDimension("temporality", 7, 1.0),
            SemanticDimension("spatiality", 11, 1.0),
            SemanticDimension("emotionality", 13, 1.1),
            SemanticDimension("abstractness", 17, 1.4),
            SemanticDimension("consciousness", 19, 2.0),
            SemanticDimension("intentionality", 23, 1.5),
            SemanticDimension("complexity", 29, 1.0)
        ]
        space.semantic_dimensions = dimensions
        
        # Initialize base concepts with primes
        base_concepts = {
            # Fundamental concepts
            "existence": 31,
            "being": 37,
            "nothing": 41,
            "something": 43,
            "identity": 47,
            
            # Consciousness concepts
            "awareness": 53,
            "thought": 59,
            "perception": 61,
            "experience": 67,
            "self": 71,
            
            # Temporal concepts
            "time": 73,
            "past": 79,
            "present": 83,
            "future": 89,
            "change": 97,
            
            # Spatial concepts
            "space": 101,
            "here": 103,
            "there": 107,
            "near": 109,
            "far": 113,
            
            # Relational concepts
            "cause": 127,
            "effect": 131,
            "relation": 137,
            "similarity": 139,
            "difference": 149,
            
            # Modal concepts
            "possible": 151,
            "necessary": 157,
            "contingent": 163,
            "actual": 167,
            "potential": 173,
            
            # Abstract concepts
            "meaning": 179,
            "truth": 181,
            "beauty": 191,
            "good": 193,
            "purpose": 197
        }
        
        for concept, prime in base_concepts.items():
            space.add_concept(concept, prime)
            
        # Initialize compositional rules
        rules = [
            CompositionRule(
                "temporal_sequence",
                [ConceptType.TEMPORAL, ConceptType.TEMPORAL],
                ConceptType.TEMPORAL,
                "multiply_with_temporal_factor"
            ),
            CompositionRule(
                "causal_chain",
                [ConceptType.RELATIONAL, ConceptType.CONCRETE],
                ConceptType.ABSTRACT,
                "causal_composition"
            ),
            CompositionRule(
                "consciousness_blend",
                [ConceptType.METACOGNITIVE, ConceptType.ABSTRACT],
                ConceptType.METACOGNITIVE,
                "consciousness_integration"
            )
        ]
        space.compositional_rules = rules
        
        return space
    
    def encode_concept_as_prime(self, concept: Concept) -> int:
        """Encode a concept as a prime number"""
        # Check cache first
        concept_key = f"{concept.name}_{concept.type}"
        if concept_key in self.prime_cache:
            return self.prime_cache[concept_key]
        
        # Calculate prime encoding based on concept properties
        base_prime = self.semantic_space.concept_primes.get(
            concept.name, 
            self._generate_new_prime(concept)
        )
        
        # Apply dimensional modifiers
        dimensional_factor = 1
        for dimension in self.semantic_space.semantic_dimensions:
            if dimension.name in concept.properties:
                dim_value = concept.properties[dimension.name]
                dimensional_factor *= (dimension.prime_base ** (dim_value * dimension.weight))
        
        # Combine base prime with dimensional factors
        encoded_prime = int(base_prime * dimensional_factor)
        
        # Ensure result is prime-like (odd and not divisible by small primes)
        while not self._is_suitable_encoding(encoded_prime):
            encoded_prime += 2
            
        self.prime_cache[concept_key] = encoded_prime
        return encoded_prime
    
    def decode_prime_to_concept(self, prime_encoding: int) -> Concept:
        """Decode a prime number back to a concept"""
        # Check if it's a known base concept
        for concept_name, concept_prime in self.semantic_space.concept_primes.items():
            if prime_encoding == concept_prime:
                return self.knowledge_graph.get_concept(concept_name)
        
        # Decompose the prime to find constituent concepts
        factors = self._factorize_semantic_prime(prime_encoding)
        
        # Reconstruct concept from factors
        properties = {}
        base_concept = None
        
        for factor in factors:
            # Check if factor corresponds to a dimension
            for dimension in self.semantic_space.semantic_dimensions:
                if factor % dimension.prime_base == 0:
                    power = math.log(factor, dimension.prime_base)
                    properties[dimension.name] = power / dimension.weight
            
            # Check if factor is a base concept
            for name, prime in self.semantic_space.concept_primes.items():
                if factor == prime:
                    base_concept = name
                    break
        
        # Create reconstructed concept
        if base_concept:
            concept = Concept(
                name=f"decoded_{base_concept}_{prime_encoding}",
                type=ConceptType.ABSTRACT,
                properties=properties
            )
        else:
            concept = Concept(
                name=f"unknown_concept_{prime_encoding}",
                type=ConceptType.ABSTRACT,
                properties=properties
            )
            
        return concept
    
    def calculate_semantic_similarity(self, concept1: Concept, concept2: Concept) -> float:
        """Calculate semantic similarity between two concepts"""
        # Check cache
        cache_key = (concept1.name, concept2.name)
        if cache_key in self.similarity_cache:
            return self.similarity_cache[cache_key]
        
        prime1 = self.encode_concept_as_prime(concept1)
        prime2 = self.encode_concept_as_prime(concept2)
        
        similarity = self._calculate_prime_semantic_distance(prime1, prime2)
        
        self.similarity_cache[cache_key] = similarity
        return similarity
    
    def compose_semantic_meanings(self, concepts: List[Concept]) -> ComposedMeaning:
        """Compose multiple concepts into a unified meaning"""
        if not concepts:
            raise ValueError("Cannot compose empty concept list")
        
        # Find applicable composition rules
        concept_types = [c.type for c in concepts]
        applicable_rules = [
            rule for rule in self.semantic_space.compositional_rules
            if self._rule_matches_types(rule, concept_types)
        ]
        
        # Apply composition
        if applicable_rules:
            rule = applicable_rules[0]  # Use first matching rule
            composition_prime = self._apply_composition_rule(rule, concepts)
        else:
            # Default composition: multiply primes with coherence factor
            composition_prime = self._compose_meanings_mathematically(
                [self.encode_concept_as_prime(c) for c in concepts]
            )
        
        # Detect emergent properties
        emergent_properties = self._detect_emergent_properties(concepts, composition_prime)
        
        # Calculate semantic coherence
        coherence = self._calculate_composition_coherence(concepts)
        
        # Determine meaning depth
        depth = self._calculate_meaning_depth(concepts, emergent_properties)
        
        composed = ComposedMeaning(
            constituent_concepts=concepts,
            composition_prime=composition_prime,
            emergent_properties=emergent_properties,
            semantic_coherence=coherence,
            meaning_depth=depth,
            composition_path=[c.name for c in concepts]
        )
        
        self.composition_history.append(composed)
        return composed
    
    def perform_semantic_operations(self, operation: SemanticOperation) -> SemanticResult:
        """Perform semantic operations on concepts"""
        input_primes = [self.encode_concept_as_prime(c) for c in operation.input_concepts]
        
        if operation.operation_type == OperationType.BLEND:
            result_prime = self._blend_concepts(input_primes, operation.operation_parameters)
        elif operation.operation_type == OperationType.CONTRAST:
            result_prime = self._contrast_concepts(input_primes)
        elif operation.operation_type == OperationType.NEGATE:
            result_prime = self._negate_concept(input_primes[0])
        elif operation.operation_type == OperationType.INTENSIFY:
            result_prime = self._intensify_concept(
                input_primes[0], 
                operation.operation_parameters.get('intensity', 2.0)
            )
        elif operation.operation_type == OperationType.ABSTRACT:
            result_prime = self._abstract_concept(input_primes[0])
        elif operation.operation_type == OperationType.CONCRETIZE:
            result_prime = self._concretize_concept(input_primes[0])
        elif operation.operation_type == OperationType.METAPHORIZE:
            result_prime = self._metaphorize_concepts(input_primes)
        else:
            raise ValueError(f"Unknown operation type: {operation.operation_type}")
        
        # Create result concept
        result_concept = self.decode_prime_to_concept(result_prime)
        result_concept.type = operation.expected_output_type
        
        # Validate semantic result
        validity = self._validate_semantic_operation(operation, result_concept)
        
        return SemanticResult(
            result_concept=result_concept,
            result_prime=result_prime,
            operation_trace=[
                f"Operation: {operation.operation_type.value}",
                f"Inputs: {[c.name for c in operation.input_concepts]}",
                f"Result: {result_concept.name}"
            ],
            confidence=0.8 if validity else 0.4,
            semantic_validity=validity
        )
    
    def generate_semantic_analogies(self, source_concept: Concept) -> List[Analogy]:
        """Generate analogies based on semantic similarity"""
        source_prime = self.encode_concept_as_prime(source_concept)
        analogies = []
        
        # Find concepts with similar prime structures
        all_concepts = self.knowledge_graph.get_all_concepts()
        
        for target_concept in all_concepts:
            if target_concept.name == source_concept.name:
                continue
                
            target_prime = self.encode_concept_as_prime(target_concept)
            
            # Calculate structural similarity
            similarity = self._calculate_prime_structural_similarity(source_prime, target_prime)
            
            if similarity > 0.6:  # Threshold for meaningful analogy
                shared_structure = self._extract_shared_structure(source_prime, target_prime)
                
                analogy = Analogy(
                    source_concept=source_concept,
                    target_concept=target_concept,
                    mapping_strength=similarity,
                    shared_structure=shared_structure,
                    prime_relationship=(source_prime, target_prime)
                )
                analogies.append(analogy)
        
        # Sort by mapping strength
        analogies.sort(key=lambda a: a.mapping_strength, reverse=True)
        
        return analogies[:10]  # Return top 10 analogies
    
    def _calculate_prime_semantic_distance(self, prime1: int, prime2: int) -> float:
        """Calculate semantic distance between prime-encoded concepts"""
        # Factor both primes
        factors1 = self._get_semantic_factors(prime1)
        factors2 = self._get_semantic_factors(prime2)
        
        # Calculate shared vs unique factors
        shared_factors = factors1.intersection(factors2)
        unique_factors = factors1.symmetric_difference(factors2)
        
        # Weight factors by semantic importance
        shared_weight = sum(self._get_factor_weight(f) for f in shared_factors)
        unique_weight = sum(self._get_factor_weight(f) for f in unique_factors)
        
        # Account for compositional complexity
        complexity_factor = 1.0 / (1.0 + abs(len(factors1) - len(factors2)))
        
        # Calculate normalized distance (inverted to get similarity)
        if shared_weight + unique_weight > 0:
            similarity = (shared_weight * complexity_factor) / (shared_weight + unique_weight)
        else:
            similarity = 0.0
            
        return similarity
    
    def _compose_meanings_mathematically(self, concept_primes: List[int]) -> int:
        """Compose meanings using prime arithmetic"""
        if not concept_primes:
            return 1
            
        # Start with least common multiple for base composition
        composition = concept_primes[0]
        
        for prime in concept_primes[1:]:
            # Use GCD to find common semantic elements
            common = math.gcd(composition, prime)
            
            # Compose based on common elements
            if common > 1:
                # Strong semantic connection - use multiplication with reduction
                composition = (composition * prime) // common
            else:
                # Weak connection - use addition-based composition
                composition = composition * prime + (composition + prime)
        
        # Ensure result maintains prime-like properties
        while not self._is_suitable_encoding(composition):
            composition += 1
            
        return composition
    
    def _generate_semantic_field(self, central_concept: Concept) -> SemanticField:
        """Generate semantic field around concept"""
        central_prime = self.encode_concept_as_prime(central_concept)
        related_concepts = []
        
        # Find semantically related concepts
        all_concepts = self.knowledge_graph.get_all_concepts()
        
        for concept in all_concepts:
            if concept.name == central_concept.name:
                continue
                
            similarity = self.calculate_semantic_similarity(central_concept, concept)
            if similarity > 0.3:  # Threshold for semantic field inclusion
                related_concepts.append((concept, 1.0 - similarity))  # Convert to distance
        
        # Sort by distance
        related_concepts.sort(key=lambda x: x[1])
        
        # Identify semantic clusters
        clusters = self._cluster_concepts(related_concepts)
        
        # Find metaphorical connections
        metaphorical_connections = []
        for concept, distance in related_concepts[:20]:  # Top 20 related
            if distance > 0.5:  # Different enough for metaphor
                analogies = self.generate_semantic_analogies(concept)
                for analogy in analogies:
                    if analogy.target_concept.name == central_concept.name:
                        metaphorical_connections.append(analogy)
        
        # Calculate field coherence
        coherence = self._calculate_field_coherence(related_concepts, clusters)
        
        return SemanticField(
            central_concept=central_concept,
            related_concepts=related_concepts[:50],  # Top 50 related
            semantic_clusters=clusters,
            field_coherence=coherence,
            metaphorical_connections=metaphorical_connections
        )
    
    # Helper methods
    
    def _generate_new_prime(self, concept: Concept) -> int:
        """Generate a new prime for an unknown concept"""
        # Use hash of concept properties to generate deterministic prime
        concept_hash = hash(f"{concept.name}_{concept.type}_{str(concept.properties)}")
        
        # Start from a large prime base
        candidate = 1000000007 + (abs(concept_hash) % 1000000)
        
        # Find next prime
        while not self._is_prime(candidate):
            candidate += 2
            
        return candidate
    
    def _is_suitable_encoding(self, n: int) -> bool:
        """Check if number is suitable for semantic encoding"""
        if n < 2:
            return False
        if n % 2 == 0:
            return False
        # Check divisibility by small primes
        for p in [3, 5, 7, 11, 13]:
            if n % p == 0 and n != p:
                return False
        return True
    
    def _is_prime(self, n: int) -> bool:
        """Check if a number is prime"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
    
    def _factorize_semantic_prime(self, n: int) -> List[int]:
        """Factorize a semantic prime into components"""
        factors = []
        
        # Check against known concept primes
        for concept, prime in self.semantic_space.concept_primes.items():
            if n % prime == 0:
                factors.append(prime)
                n //= prime
        
        # Check against dimensional primes
        for dimension in self.semantic_space.semantic_dimensions:
            while n % dimension.prime_base == 0:
                factors.append(dimension.prime_base)
                n //= dimension.prime_base
        
        # Add remaining factor if prime
        if n > 1:
            factors.append(n)
            
        return factors
    
    def _get_semantic_factors(self, prime: int) -> Set[int]:
        """Get semantic factors of a prime"""
        return set(self._factorize_semantic_prime(prime))
    
    def _get_factor_weight(self, factor: int) -> float:
        """Get semantic weight of a factor"""
        # Check if it's a dimensional prime
        for dimension in self.semantic_space.semantic_dimensions:
            if factor == dimension.prime_base:
                return dimension.weight
        
        # Check if it's a concept prime
        for concept, prime in self.semantic_space.concept_primes.items():
            if factor == prime:
                # Weight based on concept importance
                if concept in ["existence", "consciousness", "meaning"]:
                    return 2.0
                elif concept in ["awareness", "thought", "self"]:
                    return 1.5
                else:
                    return 1.0
                    
        return 0.5  # Default weight
    
    def _rule_matches_types(self, rule: CompositionRule, types: List[ConceptType]) -> bool:
        """Check if composition rule matches concept types"""
        if len(rule.input_types) != len(types):
            return False
        # For now, simple exact match - could be made more flexible
        return all(r == t for r, t in zip(rule.input_types, types))
    
    def _apply_composition_rule(self, rule: CompositionRule, concepts: List[Concept]) -> int:
        """Apply a specific composition rule"""
        primes = [self.encode_concept_as_prime(c) for c in concepts]
        
        if rule.composition_function == "multiply_with_temporal_factor":
            return primes[0] * primes[1] * 73  # 73 is temporal prime
        elif rule.composition_function == "causal_composition":
            return (primes[0] * primes[1]) + 127  # 127 is causal prime
        elif rule.composition_function == "consciousness_integration":
            return (primes[0] * primes[1] * 19) // 2  # 19 is consciousness prime
        else:
            # Default multiplication
            result = 1
            for p in primes:
                result *= p
            return result
    
    def _detect_emergent_properties(self, concepts: List[Concept], 
                                   composition_prime: int) -> List[Property]:
        """Detect emergent properties from composition"""
        properties = []
        
        # Check for consciousness emergence
        if any(c.type == ConceptType.METACOGNITIVE for c in concepts):
            if composition_prime % 19 == 0:  # Consciousness dimension
                properties.append(Property(
                    name="meta_awareness",
                    value=True,
                    emergence_level=0.8,
                    source_concepts=[c.name for c in concepts]
                ))
        
        # Check for temporal emergence
        temporal_count = sum(1 for c in concepts if c.type == ConceptType.TEMPORAL)
        if temporal_count >= 2:
            properties.append(Property(
                name="temporal_sequence",
                value="ordered",
                emergence_level=0.7,
                source_concepts=[c.name for c in concepts if c.type == ConceptType.TEMPORAL]
            ))
        
        # Check for abstract emergence
        if len(concepts) >= 3 and composition_prime > 1000000:
            properties.append(Property(
                name="high_abstraction",
                value=True,
                emergence_level=0.9,
                source_concepts=[c.name for c in concepts]
            ))
            
        return properties
    
    def _calculate_composition_coherence(self, concepts: List[Concept]) -> float:
        """Calculate semantic coherence of composition"""
        if len(concepts) < 2:
            return 1.0
            
        # Calculate pairwise similarities
        similarities = []
        for i in range(len(concepts)):
            for j in range(i + 1, len(concepts)):
                sim = self.calculate_semantic_similarity(concepts[i], concepts[j])
                similarities.append(sim)
        
        # Average similarity as coherence
        if similarities:
            return sum(similarities) / len(similarities)
        return 0.5
    
    def _calculate_meaning_depth(self, concepts: List[Concept], 
                                properties: List[Property]) -> int:
        """Calculate depth of composed meaning"""
        # Base depth from number of concepts
        depth = len(concepts)
        
        # Add depth for emergent properties
        depth += len(properties)
        
        # Add depth for abstract concepts
        depth += sum(1 for c in concepts if c.type == ConceptType.ABSTRACT)
        
        # Add depth for metacognitive concepts
        depth += sum(2 for c in concepts if c.type == ConceptType.METACOGNITIVE)
        
        return min(depth, 10)  # Cap at 10
    
    def _blend_concepts(self, primes: List[int], parameters: Dict[str, Any]) -> int:
        """Blend concepts together"""
        blend_ratio = parameters.get('blend_ratio', 0.5)
        
        if len(primes) == 2:
            # Weighted geometric mean for blending
            return int(primes[0]**blend_ratio * primes[1]**(1-blend_ratio))
        else:
            # Multi-way blend using product with reduction
            product = 1
            for p in primes:
                product *= p
            return int(product ** (1.0 / len(primes)))
    
    def _contrast_concepts(self, primes: List[int]) -> int:
        """Create contrast between concepts"""
        if len(primes) != 2:
            raise ValueError("Contrast requires exactly 2 concepts")
        
        # Use XOR-like operation on prime factors
        factors1 = self._get_semantic_factors(primes[0])
        factors2 = self._get_semantic_factors(primes[1])
        
        # Get unique factors (symmetric difference)
        unique_factors = factors1.symmetric_difference(factors2)
        
        # Compose unique factors
        result = 1
        for f in unique_factors:
            result *= f
            
        return result if result > 1 else primes[0] + primes[1]
    
    def _negate_concept(self, prime: int) -> int:
        """Negate a concept semantically"""
        # Add negation prime factor
        negation_prime = 41  # "nothing" prime
        return prime * negation_prime
    
    def _intensify_concept(self, prime: int, intensity: float) -> int:
        """Intensify a concept"""
        # Use exponentiation with intensity
        return int(prime ** (1 + intensity * 0.1))
    
    def _abstract_concept(self, prime: int) -> int:
        """Make concept more abstract"""
        abstract_prime = 17  # Abstractness dimension
        return prime * abstract_prime
    
    def _concretize_concept(self, prime: int) -> int:
        """Make concept more concrete"""
        concrete_prime = 2  # Concreteness dimension
        return prime * concrete_prime
    
    def _metaphorize_concepts(self, primes: List[int]) -> int:
        """Create metaphorical blend of concepts"""
        if len(primes) != 2:
            raise ValueError("Metaphor requires exactly 2 concepts")
        
        # Find common semantic ground
        gcd = math.gcd(primes[0], primes[1])
        
        # Create metaphorical blend emphasizing shared structure
        if gcd > 1:
            return (primes[0] * primes[1]) // gcd + gcd * 2
        else:
            # No common ground - create forced metaphor
            return primes[0] * primes[1] + 139  # 139 is similarity prime
    
    def _validate_semantic_operation(self, operation: SemanticOperation, 
                                   result: Concept) -> bool:
        """Validate semantic operation result"""
        # Check type compatibility
        if result.type != operation.expected_output_type:
            return False
        
        # Check semantic coherence
        input_primes = [self.encode_concept_as_prime(c) for c in operation.input_concepts]
        result_prime = self.encode_concept_as_prime(result)
        
        # Result should have some relationship to inputs
        for input_prime in input_primes:
            if math.gcd(input_prime, result_prime) > 1:
                return True
                
        return False
    
    def _calculate_prime_structural_similarity(self, prime1: int, prime2: int) -> float:
        """Calculate structural similarity between primes"""
        factors1 = self._get_semantic_factors(prime1)
        factors2 = self._get_semantic_factors(prime2)
        
        if not factors1 or not factors2:
            return 0.0
            
        # Jaccard similarity of factor sets
        intersection = len(factors1.intersection(factors2))
        union = len(factors1.union(factors2))
        
        return intersection / union if union > 0 else 0.0
    
    def _extract_shared_structure(self, prime1: int, prime2: int) -> Dict[str, Any]:
        """Extract shared semantic structure between primes"""
        factors1 = self._get_semantic_factors(prime1)
        factors2 = self._get_semantic_factors(prime2)
        
        shared_factors = factors1.intersection(factors2)
        
        structure = {
            "shared_factors": list(shared_factors),
            "shared_dimensions": [],
            "shared_concepts": [],
            "structural_similarity": self._calculate_prime_structural_similarity(prime1, prime2)
        }
        
        # Identify shared dimensions
        for factor in shared_factors:
            for dimension in self.semantic_space.semantic_dimensions:
                if factor == dimension.prime_base:
                    structure["shared_dimensions"].append(dimension.name)
            
            # Identify shared concepts
            for concept, prime in self.semantic_space.concept_primes.items():
                if factor == prime:
                    structure["shared_concepts"].append(concept)
        
        return structure
    
    def _cluster_concepts(self, related_concepts: List[Tuple[Concept, float]]) -> List[List[Concept]]:
        """Cluster concepts based on semantic similarity"""
        if not related_concepts:
            return []
        
        # Simple clustering based on distance threshold
        clusters = []
        used = set()
        
        for i, (concept1, dist1) in enumerate(related_concepts):
            if i in used:
                continue
                
            cluster = [concept1]
            used.add(i)
            
            for j, (concept2, dist2) in enumerate(related_concepts[i+1:], i+1):
                if j in used:
                    continue
                    
                # Check if concepts are similar enough to cluster
                similarity = self.calculate_semantic_similarity(concept1, concept2)
                if similarity > 0.7:  # High similarity threshold for clustering
                    cluster.append(concept2)
                    used.add(j)
            
            clusters.append(cluster)
        
        return clusters
    
    def _calculate_field_coherence(self, related_concepts: List[Tuple[Concept, float]], 
                                  clusters: List[List[Concept]]) -> float:
        """Calculate coherence of semantic field"""
        if not related_concepts:
            return 0.0
        
        # Average distance of related concepts
        avg_distance = sum(dist for _, dist in related_concepts) / len(related_concepts)
        
        # Cluster quality (ratio of clustered to total concepts)
        total_clustered = sum(len(cluster) for cluster in clusters)
        cluster_ratio = total_clustered / len(related_concepts)
        
        # Coherence combines low average distance and high clustering
        coherence = (1.0 - avg_distance) * 0.6 + cluster_ratio * 0.4
        
        return min(max(coherence, 0.0), 1.0)
