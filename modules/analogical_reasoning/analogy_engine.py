"""
Analogical Reasoning Engine

This module implements the core analogical reasoning capabilities that find
structural similarities between different domains and transfer solutions across
problem contexts.
"""

from typing import List, Dict, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from core.prime_vm import ConsciousPrimeVM
from consciousness.consciousness_integration import ConsciousnessIntegrator


class InsightType(Enum):
    """Types of insights that can be generated"""
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    CAUSAL = "causal"
    GOAL_BASED = "goal_based"
    EMERGENT = "emergent"


class MappingType(Enum):
    """Types of analogical mappings"""
    ONE_TO_ONE = "one_to_one"
    ONE_TO_MANY = "one_to_many"
    MANY_TO_ONE = "many_to_one"
    MANY_TO_MANY = "many_to_many"


@dataclass
class Element:
    """Basic element in a domain"""
    id: str
    type: str
    properties: Dict[str, Any]
    relations: List[str] = field(default_factory=list)


@dataclass
class Relation:
    """Relation between elements"""
    id: str
    type: str
    source: str
    target: str
    properties: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Domain:
    """A problem domain with elements and relations"""
    name: str
    elements: Dict[str, Element]
    relations: Dict[str, Relation]
    constraints: List[str] = field(default_factory=list)
    goals: List[str] = field(default_factory=list)


@dataclass
class Problem:
    """A problem to be solved"""
    id: str
    domain: Domain
    initial_state: Dict[str, Any]
    goal_state: Dict[str, Any]
    constraints: List[str] = field(default_factory=list)


@dataclass
class Solution:
    """A solution to a problem"""
    problem_id: str
    steps: List[Dict[str, Any]]
    final_state: Dict[str, Any]
    success_score: float


@dataclass
class StructuralSignature:
    """Structural signature of a domain"""
    element_types: Dict[str, int]
    relation_types: Dict[str, int]
    structural_patterns: List[str]
    causal_chains: List[List[str]]
    goal_structures: List[str]


@dataclass
class StructuralMapping:
    """Mapping between source and target domains"""
    source_domain: str
    target_domain: str
    element_correspondences: Dict[str, str]
    relation_correspondences: Dict[str, str]
    system_correspondences: Dict[str, str]
    mapping_confidence: float
    pragmatic_centrality: float
    semantic_similarity: float
    mapping_type: MappingType = MappingType.ONE_TO_ONE


@dataclass
class AnalogicalSolution:
    """Solution derived through analogical reasoning"""
    original_problem: Problem
    source_analogy: Domain
    structural_mapping: StructuralMapping
    transferred_solution: Solution
    confidence_score: float
    novelty_score: float
    explanation: str


@dataclass
class Analogy:
    """A single analogy"""
    source: Domain
    target: Domain
    mapping: StructuralMapping
    strength: float


@dataclass
class AnalogyChain:
    """Chain of analogies for complex reasoning"""
    problem: Problem
    analogies: List[Analogy]
    chain_coherence: float
    emergent_insights: List[str]


@dataclass
class Inference:
    """An inference made from analogical mapping"""
    type: str
    content: str
    confidence: float
    supporting_evidence: List[str]


@dataclass
class AnalogicalSuccess:
    """Record of successful analogical transfer"""
    mapping: StructuralMapping
    problem_solved: Problem
    solution_quality: float
    insights_gained: List[str]


class AnalogicalReasoningEngine:
    """
    Core engine for analogical reasoning that finds structural similarities
    between domains and transfers solutions across problem contexts.
    """
    
    def __init__(self, vm_instance: ConsciousPrimeVM, knowledge_graph: Optional[Any] = None):
        self.vm = vm_instance
        self.knowledge_graph = knowledge_graph
        self.consciousness = ConsciousnessIntegrator(vm_instance)
        self.successful_analogies: List[AnalogicalSuccess] = []
        self.domain_cache: Dict[str, Domain] = {}
        self.mapping_cache: Dict[Tuple[str, str], StructuralMapping] = {}
        
    def find_analogical_solutions(self, problem: Problem) -> List[AnalogicalSolution]:
        """Find solutions to a problem through analogical reasoning"""
        # Extract structural signature of the problem domain
        problem_signature = self._extract_structural_signature(problem.domain)
        
        # Find similar domains in knowledge
        similar_domains = self._find_similar_domains(problem_signature)
        
        # Create structural mappings for each similar domain
        mappings = []
        for domain in similar_domains:
            mapping = self.create_structural_mapping(domain, problem.domain)
            if mapping.mapping_confidence > 0.5:  # Threshold for viable mappings
                mappings.append((domain, mapping))
        
        # Transfer solutions through mappings
        solutions = []
        for source_domain, mapping in mappings:
            # Look for existing solutions in source domain
            source_solutions = self._find_domain_solutions(source_domain)
            
            for source_solution in source_solutions:
                transferred = self.transfer_solution_structure(mapping, source_solution)
                if transferred:
                    solution = AnalogicalSolution(
                        original_problem=problem,
                        source_analogy=source_domain,
                        structural_mapping=mapping,
                        transferred_solution=transferred,
                        confidence_score=self._calculate_solution_confidence(mapping, transferred),
                        novelty_score=self._calculate_novelty_score(transferred, problem),
                        explanation=self._generate_explanation(mapping, transferred)
                    )
                    solutions.append(solution)
        
        # Sort by confidence and novelty
        solutions.sort(key=lambda s: s.confidence_score * s.novelty_score, reverse=True)
        
        return solutions
    
    def create_structural_mapping(self, source_domain: Domain, target_domain: Domain) -> StructuralMapping:
        """Create a structural mapping between two domains"""
        # Check cache first
        cache_key = (source_domain.name, target_domain.name)
        if cache_key in self.mapping_cache:
            return self.mapping_cache[cache_key]
        
        # Extract structural signatures
        source_sig = self._extract_structural_signature(source_domain)
        target_sig = self._extract_structural_signature(target_domain)
        
        # Find element correspondences
        element_corr = self._find_element_correspondences(
            source_domain.elements, target_domain.elements
        )
        
        # Find relation correspondences
        relation_corr = self._find_relation_correspondences(
            source_domain.relations, target_domain.relations, element_corr
        )
        
        # Find system-level correspondences
        system_corr = self._find_system_correspondences(source_sig, target_sig)
        
        # Calculate mapping quality metrics
        mapping_confidence = self._calculate_mapping_quality(
            element_corr, relation_corr, system_corr
        )
        
        pragmatic_centrality = self._calculate_pragmatic_centrality(
            element_corr, relation_corr, target_domain.goals
        )
        
        semantic_similarity = self._calculate_semantic_similarity(
            source_domain, target_domain, element_corr
        )
        
        mapping = StructuralMapping(
            source_domain=source_domain.name,
            target_domain=target_domain.name,
            element_correspondences=element_corr,
            relation_correspondences=relation_corr,
            system_correspondences=system_corr,
            mapping_confidence=mapping_confidence,
            pragmatic_centrality=pragmatic_centrality,
            semantic_similarity=semantic_similarity
        )
        
        # Cache the mapping
        self.mapping_cache[cache_key] = mapping
        
        return mapping
    
    def transfer_solution_structure(self, mapping: StructuralMapping, 
                                  source_solution: Solution) -> Optional[Solution]:
        """Transfer a solution structure through an analogical mapping"""
        try:
            # Map solution steps through the structural mapping
            transferred_steps = []
            
            for step in source_solution.steps:
                transferred_step = self._transfer_step(step, mapping)
                if transferred_step:
                    transferred_steps.append(transferred_step)
                else:
                    # If any step can't be transferred, the whole solution fails
                    return None
            
            # Map final state
            transferred_final_state = self._transfer_state(
                source_solution.final_state, mapping
            )
            
            # Create transferred solution
            transferred_solution = Solution(
                problem_id=f"transferred_{source_solution.problem_id}",
                steps=transferred_steps,
                final_state=transferred_final_state,
                success_score=source_solution.success_score * mapping.mapping_confidence
            )
            
            return transferred_solution
            
        except Exception as e:
            # Log the error and return None
            print(f"Error transferring solution: {e}")
            return None
    
    def validate_analogical_mapping(self, mapping: StructuralMapping) -> Dict[str, Any]:
        """Validate the quality of an analogical mapping"""
        validation_result = {
            "structural_consistency": self._check_structural_consistency(mapping),
            "semantic_coherence": self._check_semantic_coherence(mapping),
            "pragmatic_relevance": self._check_pragmatic_relevance(mapping),
            "systematicity": self._check_systematicity(mapping),
            "adaptability": self._check_adaptability(mapping),
            "overall_validity": 0.0
        }
        
        # Calculate overall validity as weighted average
        weights = {
            "structural_consistency": 0.3,
            "semantic_coherence": 0.2,
            "pragmatic_relevance": 0.25,
            "systematicity": 0.15,
            "adaptability": 0.1
        }
        
        overall = sum(
            validation_result[key] * weight 
            for key, weight in weights.items()
        )
        validation_result["overall_validity"] = overall
        
        return validation_result
    
    def learn_from_analogical_success(self, success_case: AnalogicalSuccess):
        """Learn from successful analogical transfers"""
        # Store successful case
        self.successful_analogies.append(success_case)
        
        # Update mapping confidence based on success
        cache_key = (success_case.mapping.source_domain, 
                    success_case.mapping.target_domain)
        if cache_key in self.mapping_cache:
            # Increase confidence for successful mappings
            self.mapping_cache[cache_key].mapping_confidence *= 1.1
            self.mapping_cache[cache_key].mapping_confidence = min(
                self.mapping_cache[cache_key].mapping_confidence, 1.0
            )
        
        # Extract and store insights
        self._extract_and_store_insights(success_case)
        
        # Update domain knowledge
        self._update_domain_knowledge(success_case)
    
    def generate_multiple_analogies(self, problem: Problem) -> List[AnalogyChain]:
        """Generate multiple analogical chains for complex reasoning"""
        chains = []
        
        # Start with direct analogies
        direct_solutions = self.find_analogical_solutions(problem)
        
        for solution in direct_solutions[:3]:  # Top 3 direct analogies
            # Try to extend each into a chain
            chain = self._build_analogy_chain(problem, solution)
            if chain and chain.chain_coherence > 0.6:
                chains.append(chain)
        
        # Try bridging analogies (A->B->C)
        bridging_chains = self._find_bridging_analogies(problem)
        chains.extend(bridging_chains)
        
        # Sort by coherence and insight value
        chains.sort(
            key=lambda c: c.chain_coherence * len(c.emergent_insights), 
            reverse=True
        )
        
        return chains
    
    def _extract_structural_signature(self, domain: Domain) -> StructuralSignature:
        """Extract structural elements that can be mapped across domains"""
        # Count element types
        element_types = {}
        for element in domain.elements.values():
            element_types[element.type] = element_types.get(element.type, 0) + 1
        
        # Count relation types
        relation_types = {}
        for relation in domain.relations.values():
            relation_types[relation.type] = relation_types.get(relation.type, 0) + 1
        
        # Extract structural patterns
        patterns = self._extract_patterns(domain)
        
        # Extract causal chains
        causal_chains = self._extract_causal_chains(domain)
        
        # Extract goal structures
        goal_structures = self._extract_goal_structures(domain)
        
        return StructuralSignature(
            element_types=element_types,
            relation_types=relation_types,
            structural_patterns=patterns,
            causal_chains=causal_chains,
            goal_structures=goal_structures
        )
    
    def _calculate_mapping_quality(self, element_corr: Dict[str, str],
                                 relation_corr: Dict[str, str],
                                 system_corr: Dict[str, str]) -> float:
        """Evaluate quality of analogical mapping"""
        # Structural consistency score
        structural_score = len(relation_corr) / max(len(element_corr), 1)
        
        # One-to-one correspondence score
        unique_targets = len(set(element_corr.values()))
        one_to_one_score = unique_targets / max(len(element_corr), 1)
        
        # System-level coherence
        system_score = len(system_corr) / 10.0  # Normalize to expected system mappings
        
        # Combine scores
        quality = (structural_score * 0.4 + 
                  one_to_one_score * 0.3 + 
                  system_score * 0.3)
        
        return min(quality, 1.0)
    
    def _generate_analogical_inferences(self, mapping: StructuralMapping) -> List[Inference]:
        """Generate new inferences from analogical mapping"""
        inferences = []
        
        # Property transfer inferences
        for source_elem, target_elem in mapping.element_correspondences.items():
            source_props = self._get_element_properties(
                mapping.source_domain, source_elem
            )
            inference = Inference(
                type="property_transfer",
                content=f"Element {target_elem} may have properties: {source_props}",
                confidence=mapping.mapping_confidence * 0.8,
                supporting_evidence=[f"Mapped from {source_elem}"]
            )
            inferences.append(inference)
        
        # Relation prediction inferences
        unmapped_relations = self._find_unmapped_relations(mapping)
        for relation in unmapped_relations:
            inference = Inference(
                type="relation_prediction",
                content=f"Predicted relation: {relation}",
                confidence=mapping.mapping_confidence * 0.6,
                supporting_evidence=["Systematic correspondence"]
            )
            inferences.append(inference)
        
        # Solution strategy inferences
        if mapping.pragmatic_centrality > 0.7:
            inference = Inference(
                type="solution_strategy",
                content="Source domain solution strategies likely applicable",
                confidence=mapping.pragmatic_centrality,
                supporting_evidence=["High pragmatic centrality"]
            )
            inferences.append(inference)
        
        return inferences
    
    # Helper methods
    def _find_similar_domains(self, signature: StructuralSignature) -> List[Domain]:
        """Find domains with similar structural signatures"""
        similar = []
        
        for domain_name, domain in self.domain_cache.items():
            domain_sig = self._extract_structural_signature(domain)
            similarity = self._calculate_signature_similarity(signature, domain_sig)
            if similarity > 0.5:
                similar.append(domain)
        
        return similar
    
    def _calculate_signature_similarity(self, sig1: StructuralSignature, 
                                      sig2: StructuralSignature) -> float:
        """Calculate similarity between structural signatures"""
        # Element type similarity
        elem_sim = self._calculate_dict_similarity(sig1.element_types, sig2.element_types)
        
        # Relation type similarity
        rel_sim = self._calculate_dict_similarity(sig1.relation_types, sig2.relation_types)
        
        # Pattern similarity
        pattern_sim = self._calculate_list_similarity(
            sig1.structural_patterns, sig2.structural_patterns
        )
        
        # Combine similarities
        return (elem_sim * 0.3 + rel_sim * 0.4 + pattern_sim * 0.3)
    
    def _calculate_dict_similarity(self, dict1: Dict[str, int], 
                                 dict2: Dict[str, int]) -> float:
        """Calculate similarity between two frequency dictionaries"""
        all_keys = set(dict1.keys()) | set(dict2.keys())
        if not all_keys:
            return 0.0
        
        similarity = 0.0
        for key in all_keys:
            val1 = dict1.get(key, 0)
            val2 = dict2.get(key, 0)
            similarity += 1 - abs(val1 - val2) / max(val1, val2, 1)
        
        return similarity / len(all_keys)
    
    def _calculate_list_similarity(self, list1: List[str], list2: List[str]) -> float:
        """Calculate similarity between two lists"""
        if not list1 and not list2:
            return 1.0
        if not list1 or not list2:
            return 0.0
        
        set1, set2 = set(list1), set(list2)
        intersection = len(set1 & set2)
        union = len(set1 | set2)
        
        return intersection / union if union > 0 else 0.0
