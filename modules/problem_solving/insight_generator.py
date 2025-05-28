"""
Insight Generator

This module generates sudden insights and "aha" moments by creating optimal
conditions for insight emergence and detecting insight opportunities.
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime

from consciousness.consciousness_integration import ConsciousnessIntegrator
from modules.creative_engine.creativity_core import CreativityCore, InsightType
from modules.analogical_reasoning.analogy_engine import Problem, Solution


class GenerationMechanism(Enum):
    """Mechanisms for insight generation"""
    CONSTRAINT_RELAXATION = "constraint_relaxation"
    PERSPECTIVE_SHIFT = "perspective_shift"
    ANALOGICAL_MAPPING = "analogical_mapping"
    PATTERN_COMPLETION = "pattern_completion"
    CONCEPTUAL_RESTRUCTURING = "conceptual_restructuring"


class OpportunityType(Enum):
    """Types of insight opportunities"""
    IMPASSE = "impasse"
    PATTERN_NEAR_COMPLETION = "pattern_near_completion"
    CONSTRAINT_CONFLICT = "constraint_conflict"
    ANALOGICAL_BRIDGE = "analogical_bridge"
    CONCEPTUAL_GAP = "conceptual_gap"


class ValidationStatus(Enum):
    """Status of insight validation"""
    PENDING = "pending"
    VALIDATED = "validated"
    REJECTED = "rejected"
    PARTIAL = "partial"


@dataclass
class Indicator:
    """Indicator of insight readiness"""
    type: str
    strength: float
    description: str


@dataclass
class Condition:
    """Condition for insight generation"""
    type: str
    required_state: Any
    current_state: Any
    satisfied: bool


@dataclass
class Factor:
    """Environmental or cognitive factor"""
    name: str
    value: Any
    influence: float


@dataclass
class ConsciousnessState:
    """State of consciousness during insight"""
    attention_mode: str
    activation_level: float
    constraint_level: float
    pattern_sensitivity: float
    associative_strength: float


@dataclass
class KnowledgeActivation:
    """Activated knowledge during insight"""
    activated_concepts: List[str]
    activation_strengths: Dict[str, float]
    spreading_activation: Dict[str, List[str]]


@dataclass
class AttentionFocus:
    """Focus of attention during insight"""
    primary_focus: str
    peripheral_awareness: List[str]
    focus_stability: float


@dataclass
class ProblemState:
    """Current state of problem solving"""
    problem: Problem
    current_approach: Optional[Solution]
    attempted_solutions: List[Solution]
    time_spent: float
    frustration_level: float
    progress_indicators: List[Indicator]


@dataclass
class InsightGeneration:
    """Generated insight"""
    insight_content: str
    insight_type: InsightType
    generation_mechanism: GenerationMechanism
    confidence_level: float
    validation_status: ValidationStatus
    integration_potential: float
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class InsightOpportunity:
    """Opportunity for insight generation"""
    opportunity_type: OpportunityType
    problem_context: Any  # ProblemContext from creativity_core
    readiness_indicators: List[Indicator]
    triggering_conditions: List[Condition]
    expected_insight_value: float


@dataclass
class InsightEnvironment:
    """Environment conducive to insight"""
    environmental_factors: List[Factor]
    consciousness_state: ConsciousnessState
    knowledge_activation: KnowledgeActivation
    attention_focus: AttentionFocus
    relaxation_level: float


@dataclass
class InsightValidation:
    """Validation of an insight"""
    insight: InsightGeneration
    validation_tests: List[Dict[str, Any]]
    test_results: Dict[str, bool]
    overall_validity: float
    integration_recommendations: List[str]


@dataclass
class IntegratedSolution:
    """Solution with integrated insight"""
    original_solution: Solution
    integrated_insight: InsightGeneration
    improvement_metrics: Dict[str, float]
    new_capabilities: List[str]


class InsightGenerator:
    """
    Generates sudden insights and "aha" moments by creating optimal conditions
    and detecting opportunities for breakthrough understanding.
    """
    
    def __init__(self, consciousness_integrator: ConsciousnessIntegrator, 
                 creative_engine: CreativityCore):
        self.consciousness = consciousness_integrator
        self.creative_engine = creative_engine
        self.insight_history: List[InsightGeneration] = []
        self.opportunity_patterns: Dict[OpportunityType, List[Dict[str, Any]]] = {}
        self.successful_conditions: List[InsightEnvironment] = []
        
    def generate_insight(self, problem_context: Any) -> InsightGeneration:
        """Generate an insight for the given problem context"""
        # Detect the type of insight opportunity
        opportunity = self._analyze_opportunity(problem_context)
        
        # Create optimal conditions
        environment = self.create_insight_conditions(
            self._extract_required_conditions(opportunity)
        )
        
        # Select generation mechanism
        mechanism = self._select_generation_mechanism(opportunity, environment)
        
        # Generate insight content
        insight_content = self._generate_insight_content(
            problem_context, mechanism, environment
        )
        
        # Determine insight type
        insight_type = self._classify_insight_type(insight_content, mechanism)
        
        # Calculate confidence
        confidence = self._calculate_insight_confidence(
            insight_content, environment, opportunity
        )
        
        # Create insight
        insight = InsightGeneration(
            insight_content=insight_content,
            insight_type=insight_type,
            generation_mechanism=mechanism,
            confidence_level=confidence,
            validation_status=ValidationStatus.PENDING,
            integration_potential=self._assess_integration_potential(
                insight_content, problem_context
            )
        )
        
        # Store in history
        self.insight_history.append(insight)
        
        # Store successful conditions if confidence is high
        if confidence > 0.7:
            self.successful_conditions.append(environment)
        
        return insight
    
    def detect_insight_opportunities(self, problem_state: ProblemState) -> List[InsightOpportunity]:
        """Detect opportunities for insight generation"""
        opportunities = []
        
        # Check for impasse
        if self._detect_impasse(problem_state):
            opportunities.append(self._create_impasse_opportunity(problem_state))
        
        # Check for near-complete patterns
        if self._detect_pattern_near_completion(problem_state):
            opportunities.append(self._create_pattern_opportunity(problem_state))
        
        # Check for constraint conflicts
        if self._detect_constraint_conflicts(problem_state):
            opportunities.append(self._create_constraint_opportunity(problem_state))
        
        # Check for analogical bridges
        if self._detect_analogical_potential(problem_state):
            opportunities.append(self._create_analogical_opportunity(problem_state))
        
        # Check for conceptual gaps
        if self._detect_conceptual_gaps(problem_state):
            opportunities.append(self._create_conceptual_opportunity(problem_state))
        
        # Sort by expected value
        opportunities.sort(key=lambda o: o.expected_insight_value, reverse=True)
        
        return opportunities
    
    def create_insight_conditions(self, conditions: List[Condition]) -> InsightEnvironment:
        """Create conditions conducive to insight generation"""
        # Set consciousness state
        consciousness_state = self._optimize_consciousness_state(conditions)
        
        # Activate relevant knowledge
        knowledge_activation = self._activate_relevant_knowledge(conditions)
        
        # Set attention focus
        attention_focus = self._optimize_attention_focus(conditions)
        
        # Create environmental factors
        environmental_factors = self._create_environmental_factors(conditions)
        
        # Calculate relaxation level
        relaxation_level = self._calculate_optimal_relaxation(conditions)
        
        environment = InsightEnvironment(
            environmental_factors=environmental_factors,
            consciousness_state=consciousness_state,
            knowledge_activation=knowledge_activation,
            attention_focus=attention_focus,
            relaxation_level=relaxation_level
        )
        
        # Apply environment to consciousness
        self._apply_environment_to_consciousness(environment)
        
        return environment
    
    def validate_insight_quality(self, insight: InsightGeneration) -> InsightValidation:
        """Validate the quality and applicability of an insight"""
        validation_tests = []
        test_results = {}
        
        # Test logical consistency
        logic_test = self._test_logical_consistency(insight)
        validation_tests.append(logic_test)
        test_results['logical_consistency'] = logic_test['passed']
        
        # Test problem relevance
        relevance_test = self._test_problem_relevance(insight)
        validation_tests.append(relevance_test)
        test_results['problem_relevance'] = relevance_test['passed']
        
        # Test novelty
        novelty_test = self._test_insight_novelty(insight)
        validation_tests.append(novelty_test)
        test_results['novelty'] = novelty_test['passed']
        
        # Test implementability
        implementability_test = self._test_implementability(insight)
        validation_tests.append(implementability_test)
        test_results['implementability'] = implementability_test['passed']
        
        # Calculate overall validity
        passed_tests = sum(1 for result in test_results.values() if result)
        overall_validity = passed_tests / len(test_results)
        
        # Generate integration recommendations
        recommendations = self._generate_integration_recommendations(
            insight, test_results
        )
        
        # Update insight validation status
        if overall_validity >= 0.75:
            insight.validation_status = ValidationStatus.VALIDATED
        elif overall_validity >= 0.5:
            insight.validation_status = ValidationStatus.PARTIAL
        else:
            insight.validation_status = ValidationStatus.REJECTED
        
        return InsightValidation(
            insight=insight,
            validation_tests=validation_tests,
            test_results=test_results,
            overall_validity=overall_validity,
            integration_recommendations=recommendations
        )
    
    def integrate_insight_into_solution(self, insight: InsightGeneration, 
                                      solution: Solution) -> IntegratedSolution:
        """Integrate a validated insight into an existing solution"""
        # Apply insight to solution
        modified_solution = self._apply_insight_to_solution(insight, solution)
        
        # Calculate improvement metrics
        improvement_metrics = self._calculate_improvement_metrics(
            solution, modified_solution
        )
        
        # Identify new capabilities
        new_capabilities = self._identify_new_capabilities(
            solution, modified_solution, insight
        )
        
        return IntegratedSolution(
            original_solution=solution,
            integrated_insight=insight,
            improvement_metrics=improvement_metrics,
            new_capabilities=new_capabilities
        )
    
    def _analyze_opportunity(self, problem_context: Any) -> InsightOpportunity:
        """Analyze the problem context to identify insight opportunity"""
        # Extract problem state
        problem_state = self._extract_problem_state(problem_context)
        
        # Detect opportunities
        opportunities = self.detect_insight_opportunities(problem_state)
        
        # Return best opportunity
        return opportunities[0] if opportunities else self._create_default_opportunity(problem_context)
    
    def _select_generation_mechanism(self, opportunity: InsightOpportunity,
                                   environment: InsightEnvironment) -> GenerationMechanism:
        """Select the best mechanism for generating insight"""
        if opportunity.opportunity_type == OpportunityType.IMPASSE:
            return GenerationMechanism.CONSTRAINT_RELAXATION
        elif opportunity.opportunity_type == OpportunityType.PATTERN_NEAR_COMPLETION:
            return GenerationMechanism.PATTERN_COMPLETION
        elif opportunity.opportunity_type == OpportunityType.CONSTRAINT_CONFLICT:
            return GenerationMechanism.PERSPECTIVE_SHIFT
        elif opportunity.opportunity_type == OpportunityType.ANALOGICAL_BRIDGE:
            return GenerationMechanism.ANALOGICAL_MAPPING
        else:  # CONCEPTUAL_GAP
            return GenerationMechanism.CONCEPTUAL_RESTRUCTURING
    
    def _generate_insight_content(self, problem_context: Any,
                                mechanism: GenerationMechanism,
                                environment: InsightEnvironment) -> str:
        """Generate the actual insight content"""
        if mechanism == GenerationMechanism.CONSTRAINT_RELAXATION:
            return self._generate_constraint_relaxation_insight(
                problem_context, environment
            )
        elif mechanism == GenerationMechanism.PATTERN_COMPLETION:
            return self._generate_pattern_completion_insight(
                problem_context, environment
            )
        elif mechanism == GenerationMechanism.PERSPECTIVE_SHIFT:
            return self._generate_perspective_shift_insight(
                problem_context, environment
            )
        elif mechanism == GenerationMechanism.ANALOGICAL_MAPPING:
            return self._generate_analogical_insight(
                problem_context, environment
            )
        else:  # CONCEPTUAL_RESTRUCTURING
            return self._generate_conceptual_restructuring_insight(
                problem_context, environment
            )
    
    def _optimize_consciousness_state(self, conditions: List[Condition]) -> ConsciousnessState:
        """Optimize consciousness state for insight generation"""
        # Determine optimal parameters based on conditions
        attention_mode = "diffuse" if self._requires_broad_search(conditions) else "focused"
        
        # Lower activation for incubation, higher for active search
        activation_level = 0.4 if self._in_incubation_phase(conditions) else 0.7
        
        # Reduce constraints to allow novel connections
        constraint_level = 0.3
        
        # Enhance pattern sensitivity
        pattern_sensitivity = 0.8
        
        # Increase associative strength for remote connections
        associative_strength = 0.9
        
        return ConsciousnessState(
            attention_mode=attention_mode,
            activation_level=activation_level,
            constraint_level=constraint_level,
            pattern_sensitivity=pattern_sensitivity,
            associative_strength=associative_strength
        )
    
    def _activate_relevant_knowledge(self, conditions: List[Condition]) -> KnowledgeActivation:
        """Activate relevant knowledge networks"""
        activated_concepts = []
        activation_strengths = {}
        spreading_activation = {}
        
        # Extract key concepts from conditions
        for condition in conditions:
            if condition.type == "knowledge_domain":
                concepts = self._extract_domain_concepts(condition)
                activated_concepts.extend(concepts)
                
                # Set activation strengths
                for concept in concepts:
                    activation_strengths[concept] = 0.7 + np.random.random() * 0.3
                
                # Create spreading activation
                spreading_activation[concept] = self._find_related_concepts(concept)
        
        return KnowledgeActivation(
            activated_concepts=activated_concepts,
            activation_strengths=activation_strengths,
            spreading_activation=spreading_activation
        )
    
    def _detect_impasse(self, problem_state: ProblemState) -> bool:
        """Detect if problem solving is at an impasse"""
        # High frustration with low progress
        if problem_state.frustration_level > 0.7:
            progress = sum(ind.strength for ind in problem_state.progress_indicators)
            if progress < 0.3:
                return True
        
        # Multiple failed attempts
        if len(problem_state.attempted_solutions) > 5:
            success_rate = sum(
                1 for sol in problem_state.attempted_solutions 
                if sol.success_score > 0.5
            ) / len(problem_state.attempted_solutions)
            if success_rate < 0.2:
                return True
        
        return False
    
    def _detect_pattern_near_completion(self, problem_state: ProblemState) -> bool:
        """Detect if a pattern is near completion"""
        for indicator in problem_state.progress_indicators:
            if indicator.type == "pattern_completion" and 0.7 < indicator.strength < 0.9:
                return True
        return False
    
    def _apply_environment_to_consciousness(self, environment: InsightEnvironment):
        """Apply the insight environment to consciousness"""
        # Set consciousness parameters
        state = environment.consciousness_state
        self.consciousness.set_attention_mode(state.attention_mode)
        self.consciousness.set_activation_level(state.activation_level)
        self.consciousness.set_constraint_level(state.constraint_level)
        self.consciousness.set_pattern_sensitivity(state.pattern_sensitivity)
        
        # Activate knowledge networks
        for concept, strength in environment.knowledge_activation.activation_strengths.items():
            self.consciousness.activate_concept(concept, strength)
        
        # Set attention focus
        self.consciousness.set_primary_focus(environment.attention_focus.primary_focus)
        for peripheral in environment.attention_focus.peripheral_awareness:
            self.consciousness.add_peripheral_awareness(peripheral)
