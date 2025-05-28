"""
Concept Abstraction Module

This module handles the abstraction and manipulation of concepts,
including hierarchical concept organization, abstraction levels,
and concept transformation.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

from consciousness.consciousness_integration import ConsciousnessIntegrator
from modules.knowledge_systems.knowledge_graph import KnowledgeGraph, Concept


class AbstractionLevel(Enum):
    """Levels of abstraction"""
    CONCRETE = 0  # Specific instances
    BASIC = 1  # Basic-level categories
    SUBORDINATE = 2  # More specific categories
    SUPERORDINATE = 3  # More general categories
    ABSTRACT = 4  # Abstract concepts
    META = 5  # Meta-concepts


class ConceptRelation(Enum):
    """Types of concept relations"""
    IS_A = "is_a"  # Taxonomic relation
    PART_OF = "part_of"  # Mereological relation
    SIMILAR_TO = "similar_to"  # Similarity relation
    OPPOSITE_OF = "opposite_of"  # Opposition relation
    CAUSES = "causes"  # Causal relation
    PRECEDES = "precedes"  # Temporal relation
    INSTANCE_OF = "instance_of"  # Instance relation


@dataclass
class AbstractConcept:
    """An abstract concept with properties and relations"""
    concept_id: str
    name: str
    abstraction_level: AbstractionLevel
    properties: Dict[str, Any] = field(default_factory=dict)
    relations: Dict[ConceptRelation, List[str]] = field(default_factory=dict)
    prototypes: List[str] = field(default_factory=list)  # Prototypical examples
    boundaries: Dict[str, float] = field(default_factory=dict)  # Fuzzy boundaries
    
    def similarity_to(self, other: 'AbstractConcept') -> float:
        """Calculate similarity to another concept"""
        # Property overlap
        common_props = set(self.properties.keys()) & set(other.properties.keys())
        prop_similarity = len(common_props) / max(
            len(self.properties), len(other.properties), 1
        )
        
        # Relation overlap
        common_relations = set(self.relations.keys()) & set(other.relations.keys())
        rel_similarity = len(common_relations) / max(
            len(self.relations), len(other.relations), 1
        )
        
        # Level distance
        level_distance = abs(self.abstraction_level.value - other.abstraction_level.value)
        level_similarity = 1.0 / (1.0 + level_distance)
        
        # Weighted combination
        return 0.4 * prop_similarity + 0.4 * rel_similarity + 0.2 * level_similarity


@dataclass
class ConceptHierarchy:
    """Hierarchical organization of concepts"""
    root_concepts: List[str]
    parent_child_map: Dict[str, List[str]]
    child_parent_map: Dict[str, str]
    level_map: Dict[str, int]  # Concept -> hierarchy level
    
    def get_ancestors(self, concept_id: str) -> List[str]:
        """Get all ancestors of a concept"""
        ancestors = []
        current = self.child_parent_map.get(concept_id)
        
        while current:
            ancestors.append(current)
            current = self.child_parent_map.get(current)
        
        return ancestors
    
    def get_descendants(self, concept_id: str) -> List[str]:
        """Get all descendants of a concept"""
        descendants = []
        to_process = [concept_id]
        
        while to_process:
            current = to_process.pop(0)
            children = self.parent_child_map.get(current, [])
            descendants.extend(children)
            to_process.extend(children)
        
        return descendants
    
    def lowest_common_ancestor(self, concept1: str, concept2: str) -> Optional[str]:
        """Find lowest common ancestor of two concepts"""
        ancestors1 = set(self.get_ancestors(concept1))
        ancestors2 = set(self.get_ancestors(concept2))
        
        common = ancestors1 & ancestors2
        if not common:
            return None
        
        # Find the one with highest level (lowest in tree)
        return max(common, key=lambda c: self.level_map.get(c, 0))


@dataclass
class AbstractionOperation:
    """An operation that transforms concepts"""
    operation_type: str  # "generalize", "specialize", "analogize", etc.
    input_concepts: List[str]
    output_concept: str
    transformation_rules: Dict[str, Any]
    confidence: float


@dataclass
class ConceptSpace:
    """A space of concepts with distance metric"""
    concepts: Dict[str, AbstractConcept]
    dimensions: List[str]  # Semantic dimensions
    embeddings: Dict[str, List[float]]  # Concept -> vector embedding
    
    def distance(self, concept1: str, concept2: str) -> float:
        """Calculate distance between concepts in space"""
        if concept1 not in self.embeddings or concept2 not in self.embeddings:
            return float('inf')
        
        vec1 = self.embeddings[concept1]
        vec2 = self.embeddings[concept2]
        
        # Euclidean distance
        return math.sqrt(sum((a - b) ** 2 for a, b in zip(vec1, vec2)))
    
    def nearest_neighbors(self, concept_id: str, k: int = 5) -> List[Tuple[str, float]]:
        """Find k nearest neighbors of a concept"""
        distances = []
        
        for other_id in self.concepts:
            if other_id != concept_id:
                dist = self.distance(concept_id, other_id)
                distances.append((other_id, dist))
        
        distances.sort(key=lambda x: x[1])
        return distances[:k]


@dataclass
class ConceptBlend:
    """A blend of multiple concepts"""
    source_concepts: List[str]
    blended_concept: AbstractConcept
    mapping_structure: Dict[str, Dict[str, str]]  # Source -> target mappings
    emergent_properties: Dict[str, Any]
    blend_quality: float


class ConceptAbstractor:
    """
    Handles concept abstraction, manipulation, and transformation.
    """
    
    def __init__(self, consciousness_integrator: ConsciousnessIntegrator,
                 knowledge_graph: KnowledgeGraph):
        self.consciousness_integrator = consciousness_integrator
        self.knowledge_graph = knowledge_graph
        self.concept_registry = {}
        self.hierarchies = {}
        self.concept_spaces = {}
        self.abstraction_rules = self._initialize_abstraction_rules()
        
    def _initialize_abstraction_rules(self) -> Dict[str, Any]:
        """Initialize rules for abstraction operations"""
        return {
            "generalization": {
                "drop_specific_features": True,
                "keep_common_features": True,
                "increase_abstraction_level": True
            },
            "specialization": {
                "add_specific_features": True,
                "narrow_scope": True,
                "decrease_abstraction_level": True
            },
            "analogical_abstraction": {
                "preserve_relations": True,
                "map_properties": True,
                "maintain_structure": True
            }
        }
    
    def create_abstract_concept(self, name: str, 
                              properties: Dict[str, Any],
                              abstraction_level: AbstractionLevel) -> AbstractConcept:
        """Create a new abstract concept"""
        concept_id = f"concept_{len(self.concept_registry)}"
        
        concept = AbstractConcept(
            concept_id=concept_id,
            name=name,
            abstraction_level=abstraction_level,
            properties=properties
        )
        
        # Register concept
        self.concept_registry[concept_id] = concept
        
        # Add to knowledge graph
        kg_concept = Concept(
            concept_id=concept_id,
            name=name,
            properties=properties
        )
        self.knowledge_graph.add_concept(kg_concept)
        
        return concept
    
    def abstract_from_instances(self, instances: List[AbstractConcept]) -> AbstractConcept:
        """Create abstract concept from concrete instances"""
        if not instances:
            raise ValueError("Cannot abstract from empty instance list")
        
        # Find common properties
        common_properties = {}
        if instances:
            # Start with first instance's properties
            property_sets = [set(inst.properties.keys()) for inst in instances]
            common_keys = set.intersection(*property_sets)
            
            for key in common_keys:
                # Check if values are similar enough
                values = [inst.properties[key] for inst in instances]
                if self._values_similar(values):
                    common_properties[key] = self._abstract_value(values)
        
        # Determine abstraction level
        avg_level = sum(inst.abstraction_level.value for inst in instances) / len(instances)
        new_level = AbstractionLevel(min(int(avg_level) + 1, AbstractionLevel.META.value))
        
        # Create abstract concept
        abstract_name = f"Abstract_{instances[0].name}"
        abstract_concept = self.create_abstract_concept(
            name=abstract_name,
            properties=common_properties,
            abstraction_level=new_level
        )
        
        # Set instances as prototypes
        abstract_concept.prototypes = [inst.concept_id for inst in instances]
        
        # Create IS_A relations
        for instance in instances:
            if ConceptRelation.IS_A not in instance.relations:
                instance.relations[ConceptRelation.IS_A] = []
            instance.relations[ConceptRelation.IS_A].append(abstract_concept.concept_id)
        
        return abstract_concept
    
    def specialize_concept(self, abstract_concept: AbstractConcept,
                         additional_properties: Dict[str, Any]) -> AbstractConcept:
        """Create more specific concept from abstract one"""
        # Combine properties
        specialized_properties = abstract_concept.properties.copy()
        specialized_properties.update(additional_properties)
        
        # Lower abstraction level
        new_level = AbstractionLevel(
            max(abstract_concept.abstraction_level.value - 1, 
                AbstractionLevel.CONCRETE.value)
        )
        
        # Create specialized concept
        specialized = self.create_abstract_concept(
            name=f"{abstract_concept.name}_specialized",
            properties=specialized_properties,
            abstraction_level=new_level
        )
        
        # Create IS_A relation
        specialized.relations[ConceptRelation.IS_A] = [abstract_concept.concept_id]
        
        return specialized
    
    def build_concept_hierarchy(self, concepts: List[AbstractConcept]) -> ConceptHierarchy:
        """Build hierarchical organization of concepts"""
        parent_child_map = {}
        child_parent_map = {}
        level_map = {}
        
        # Build maps from IS_A relations
        for concept in concepts:
            concept_id = concept.concept_id
            level_map[concept_id] = 0  # Initialize
            
            if ConceptRelation.IS_A in concept.relations:
                for parent_id in concept.relations[ConceptRelation.IS_A]:
                    # Update parent-child map
                    if parent_id not in parent_child_map:
                        parent_child_map[parent_id] = []
                    parent_child_map[parent_id].append(concept_id)
                    
                    # Update child-parent map (assuming single inheritance)
                    child_parent_map[concept_id] = parent_id
        
        # Calculate levels
        def calculate_level(concept_id: str) -> int:
            if concept_id in level_map and level_map[concept_id] > 0:
                return level_map[concept_id]
            
            parent = child_parent_map.get(concept_id)
            if parent:
                level_map[concept_id] = calculate_level(parent) + 1
            else:
                level_map[concept_id] = 0
            
            return level_map[concept_id]
        
        for concept_id in level_map:
            calculate_level(concept_id)
        
        # Find root concepts
        root_concepts = [c for c in level_map if level_map[c] == 0]
        
        hierarchy = ConceptHierarchy(
            root_concepts=root_concepts,
            parent_child_map=parent_child_map,
            child_parent_map=child_parent_map,
            level_map=level_map
        )
        
        # Store hierarchy
        hierarchy_id = f"hierarchy_{len(self.hierarchies)}"
        self.hierarchies[hierarchy_id] = hierarchy
        
        return hierarchy
    
    def create_concept_space(self, concepts: List[AbstractConcept],
                           dimensions: List[str]) -> ConceptSpace:
        """Create a concept space with embeddings"""
        concept_dict = {c.concept_id: c for c in concepts}
        embeddings = {}
        
        # Create embeddings based on properties
        for concept in concepts:
            embedding = []
            for dim in dimensions:
                if dim in concept.properties:
                    # Normalize property value to [0, 1]
                    value = concept.properties[dim]
                    if isinstance(value, bool):
                        embedding.append(1.0 if value else 0.0)
                    elif isinstance(value, (int, float)):
                        embedding.append(float(value))
                    else:
                        embedding.append(0.5)  # Default for non-numeric
                else:
                    embedding.append(0.0)  # Missing dimension
            
            embeddings[concept.concept_id] = embedding
        
        space = ConceptSpace(
            concepts=concept_dict,
            dimensions=dimensions,
            embeddings=embeddings
        )
        
        # Store space
        space_id = f"space_{len(self.concept_spaces)}"
        self.concept_spaces[space_id] = space
        
        return space
    
    def blend_concepts(self, concepts: List[AbstractConcept],
                     blend_type: str = "property_union") -> ConceptBlend:
        """Blend multiple concepts into new concept"""
        if len(concepts) < 2:
            raise ValueError("Need at least 2 concepts to blend")
        
        # Different blending strategies
        if blend_type == "property_union":
            blended_properties = self._blend_by_union(concepts)
        elif blend_type == "property_intersection":
            blended_properties = self._blend_by_intersection(concepts)
        elif blend_type == "weighted_average":
            blended_properties = self._blend_by_average(concepts)
        else:
            blended_properties = {}
        
        # Determine abstraction level
        avg_level = sum(c.abstraction_level.value for c in concepts) / len(concepts)
        blend_level = AbstractionLevel(int(avg_level))
        
        # Create blended concept
        blend_name = "_".join([c.name for c in concepts])
        blended_concept = self.create_abstract_concept(
            name=f"Blend_{blend_name}",
            properties=blended_properties,
            abstraction_level=blend_level
        )
        
        # Identify emergent properties
        emergent = self._identify_emergent_properties(concepts, blended_concept)
        
        # Create blend structure
        blend = ConceptBlend(
            source_concepts=[c.concept_id for c in concepts],
            blended_concept=blended_concept,
            mapping_structure=self._create_blend_mappings(concepts, blended_concept),
            emergent_properties=emergent,
            blend_quality=self._assess_blend_quality(concepts, blended_concept)
        )
        
        return blend
    
    def find_analogical_abstractions(self, source_concept: AbstractConcept,
                                   target_domain: List[AbstractConcept]) -> List[Tuple[AbstractConcept, float]]:
        """Find analogical abstractions in target domain"""
        analogies = []
        
        for target in target_domain:
            # Calculate structural similarity
            structural_sim = self._structural_similarity(source_concept, target)
            
            # Calculate relational similarity
            relational_sim = self._relational_similarity(source_concept, target)
            
            # Combined score
            analogy_score = 0.6 * structural_sim + 0.4 * relational_sim
            
            if analogy_score > 0.5:  # Threshold
                analogies.append((target, analogy_score))
        
        # Sort by score
        analogies.sort(key=lambda x: x[1], reverse=True)
        
        return analogies
    
    def extract_abstraction_pattern(self, concepts: List[AbstractConcept]) -> Dict[str, Any]:
        """Extract common abstraction pattern from concepts"""
        pattern = {
            "common_properties": {},
            "common_relations": {},
            "structural_pattern": {},
            "abstraction_trajectory": []
        }
        
        # Find common properties
        if concepts:
            property_sets = [set(c.properties.keys()) for c in concepts]
            common_props = set.intersection(*property_sets)
            
            for prop in common_props:
                values = [c.properties[prop] for c in concepts]
                if self._values_similar(values):
                    pattern["common_properties"][prop] = self._abstract_value(values)
        
        # Find common relations
        if concepts:
            relation_sets = [set(c.relations.keys()) for c in concepts]
            common_rels = set.intersection(*relation_sets)
            pattern["common_relations"] = list(common_rels)
        
        # Analyze abstraction trajectory
        levels = [c.abstraction_level.value for c in concepts]
        pattern["abstraction_trajectory"] = {
            "min_level": min(levels) if levels else 0,
            "max_level": max(levels) if levels else 0,
            "average_level": sum(levels) / len(levels) if levels else 0
        }
        
        return pattern
    
    # Private helper methods
    
    def _values_similar(self, values: List[Any], threshold: float = 0.8) -> bool:
        """Check if values are similar enough to abstract"""
        if not values:
            return False
        
        # All same type
        if not all(isinstance(v, type(values[0])) for v in values):
            return False
        
        # For booleans
        if isinstance(values[0], bool):
            return all(v == values[0] for v in values)
        
        # For numbers
        if isinstance(values[0], (int, float)):
            mean = sum(values) / len(values)
            variance = sum((v - mean) ** 2 for v in values) / len(values)
            cv = math.sqrt(variance) / (mean + 1e-10)  # Coefficient of variation
            return cv < (1 - threshold)
        
        # For strings
        if isinstance(values[0], str):
            # Simple check - all same
            return len(set(values)) == 1
        
        return False
    
    def _abstract_value(self, values: List[Any]) -> Any:
        """Create abstract value from concrete values"""
        if not values:
            return None
        
        # For booleans - take majority
        if isinstance(values[0], bool):
            return sum(values) > len(values) / 2
        
        # For numbers - take mean
        if isinstance(values[0], (int, float)):
            return sum(values) / len(values)
        
        # For strings - take most common or create abstract
        if isinstance(values[0], str):
            # Could implement more sophisticated string abstraction
            return values[0]  # Simple: take first
        
        return values[0]
    
    def _blend_by_union(self, concepts: List[AbstractConcept]) -> Dict[str, Any]:
        """Blend by taking union of properties"""
        blended = {}
        
        for concept in concepts:
            for key, value in concept.properties.items():
                if key not in blended:
                    blended[key] = value
                else:
                    # Handle conflicts - could be more sophisticated
                    if isinstance(value, (int, float)) and isinstance(blended[key], (int, float)):
                        blended[key] = (blended[key] + value) / 2
        
        return blended
    
    def _blend_by_intersection(self, concepts: List[AbstractConcept]) -> Dict[str, Any]:
        """Blend by taking intersection of properties"""
        if not concepts:
            return {}
        
        # Start with first concept's properties
        blended = concepts[0].properties.copy()
        
        # Keep only common properties
        for concept in concepts[1:]:
            blended = {k: v for k, v in blended.items() if k in concept.properties}
        
        return blended
    
    def _blend_by_average(self, concepts: List[AbstractConcept]) -> Dict[str, Any]:
        """Blend by averaging numeric properties"""
        blended = {}
        property_counts = {}
        
        # Accumulate values
        for concept in concepts:
            for key, value in concept.properties.items():
                if isinstance(value, (int, float)):
                    if key not in blended:
                        blended[key] = 0
                        property_counts[key] = 0
                    blended[key] += value
                    property_counts[key] += 1
        
        # Average
        for key in blended:
            blended[key] /= property_counts[key]
        
        return blended
    
    def _identify_emergent_properties(self, sources: List[AbstractConcept],
                                    blend: AbstractConcept) -> Dict[str, Any]:
        """Identify emergent properties in blend"""
        emergent = {}
        
        # Properties in blend but not in any source
        source_props = set()
        for source in sources:
            source_props.update(source.properties.keys())
        
        for prop in blend.properties:
            if prop not in source_props:
                emergent[prop] = blend.properties[prop]
        
        return emergent
    
    def _create_blend_mappings(self, sources: List[AbstractConcept],
                             blend: AbstractConcept) -> Dict[str, Dict[str, str]]:
        """Create mappings from source concepts to blend"""
        mappings = {}
        
        for source in sources:
            source_map = {}
            for prop in source.properties:
                if prop in blend.properties:
                    source_map[prop] = prop  # Direct mapping
            mappings[source.concept_id] = source_map
        
        return mappings
    
    def _assess_blend_quality(self, sources: List[AbstractConcept],
                            blend: AbstractConcept) -> float:
        """Assess quality of concept blend"""
        # Factors: property preservation, coherence, usefulness
        
        # Property preservation
        total_props = sum(len(s.properties) for s in sources)
        preserved_props = len(blend.properties)
        preservation = preserved_props / max(total_props, 1)
        
        # Coherence (simplified - check for contradictions)
        coherence = 1.0  # Would implement contradiction checking
        
        # Combine factors
        quality = 0.6 * preservation + 0.4 * coherence
        
        return quality
    
    def _structural_similarity(self, concept1: AbstractConcept,
                             concept2: AbstractConcept) -> float:
        """Calculate structural similarity between concepts"""
        # Compare property structure
        props1 = set(concept1.properties.keys())
        props2 = set(concept2.properties.keys())
        
        if not props1 and not props2:
            return 1.0
        
        intersection = props1 & props2
        union = props1 | props2
        
        return len(intersection) / len(union)
    
    def _relational_similarity(self, concept1: AbstractConcept,
                             concept2: AbstractConcept) -> float:
        """Calculate relational similarity between concepts"""
        # Compare relation types
        rels1 = set(concept1.relations.keys())
        rels2 = set(concept2.relations.keys())
        
        if not rels1 and not rels2:
            return 1.0
        
        intersection = rels1 & rels2
        union = rels1 | rels2
        
        return len(intersection) / len(union)
