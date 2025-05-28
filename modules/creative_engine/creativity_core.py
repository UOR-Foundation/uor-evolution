"""
Creative Engine Core

This module orchestrates different types of creativity to generate novel solutions
and ideas, leveraging consciousness and analogical reasoning.
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime

from core.prime_vm import ConsciousPrimeVM
from consciousness.consciousness_integration import ConsciousnessIntegrator
from modules.analogical_reasoning.analogy_engine import Problem, Solution


class CreativityType(Enum):
    """Types of creative processes"""
    COMBINATORIAL = "combinatorial"
    EXPLORATORY = "exploratory"
    TRANSFORMATIONAL = "transformational"
    ANALOGICAL = "analogical"
    EMERGENT = "emergent"


class InsightType(Enum):
    """Types of creative insights"""
    STRUCTURAL = "structural"
    FUNCTIONAL = "functional"
    CONCEPTUAL = "conceptual"
    PERCEPTUAL = "perceptual"
    STRATEGIC = "strategic"


class GenerationMethod(Enum):
    """Methods for generating creative solutions"""
    RANDOM_COMBINATION = "random_combination"
    GUIDED_SEARCH = "guided_search"
    CONSTRAINT_RELAXATION = "constraint_relaxation"
    ANALOGICAL_TRANSFER = "analogical_transfer"
    EMERGENT_SYNTHESIS = "emergent_synthesis"


@dataclass
class CreativityConstraints:
    """Constraints on the creative process"""
    allowed_types: List[CreativityType] = field(default_factory=list)
    time_limit: Optional[float] = None
    novelty_threshold: float = 0.5
    utility_threshold: float = 0.3
    resource_limits: Dict[str, Any] = field(default_factory=dict)


@dataclass
class SolutionApproach:
    """An approach to solving a problem"""
    method: str
    steps: List[Dict[str, Any]]
    resources_required: Dict[str, Any]
    estimated_success: float


@dataclass
class GenerationProcess:
    """Record of how a creative solution was generated"""
    method: GenerationMethod
    steps_taken: List[Dict[str, Any]]
    time_elapsed: float
    consciousness_states: List[str]
    key_insights: List[str]


@dataclass
class NoveltyAssessment:
    """Assessment of solution novelty"""
    overall_novelty: float
    structural_novelty: float
    functional_novelty: float
    conceptual_novelty: float
    comparison_base: List[str]


@dataclass
class UtilityAssessment:
    """Assessment of solution utility"""
    overall_utility: float
    problem_solving_effectiveness: float
    resource_efficiency: float
    generalizability: float
    side_benefits: List[str]


@dataclass
class SurpriseAssessment:
    """Assessment of solution surprise value"""
    overall_surprise: float
    expectation_violation: float
    paradigm_shift: float
    emergent_properties: List[str]


@dataclass
class EleganceAssessment:
    """Assessment of solution elegance"""
    overall_elegance: float
    simplicity: float
    symmetry: float
    parsimony: float
    aesthetic_value: float


@dataclass
class CreativeSolution:
    """A creative solution to a problem"""
    original_problem: Problem
    solution_approach: SolutionApproach
    creativity_type: CreativityType
    novelty_score: float
    utility_score: float
    surprise_score: float
    elegance_score: float
    generation_process: GenerationProcess


@dataclass
class CreativityEvaluation:
    """Comprehensive evaluation of creativity"""
    novelty_assessment: NoveltyAssessment
    utility_assessment: UtilityAssessment
    surprise_assessment: SurpriseAssessment
    elegance_assessment: EleganceAssessment
    overall_creativity_score: float


@dataclass
class Evidence:
    """Evidence supporting an insight"""
    type: str
    content: str
    confidence: float
    source: str


@dataclass
class Implication:
    """Implication of an insight"""
    type: str
    description: str
    impact_level: float
    affected_areas: List[str]


@dataclass
class ProblemContext:
    """Context surrounding a problem"""
    problem: Problem
    related_problems: List[Problem]
    previous_attempts: List[Solution]
    constraints: List[str]
    available_resources: Dict[str, Any]


@dataclass
class InsightEvent:
    """A creative insight event"""
    insight_type: InsightType
    problem_context: ProblemContext
    insight_content: str
    confidence_level: float
    supporting_evidence: List[Evidence]
    implications: List[Implication]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class CreativeProcess:
    """A managed creative process"""
    process_id: str
    creativity_type: CreativityType
    start_time: datetime
    end_time: Optional[datetime]
    stages_completed: List[str]
    current_stage: str
    intermediate_results: List[Any]
    final_result: Optional[CreativeSolution]


class CreativityCore:
    """
    Core engine for creative problem-solving that orchestrates different types
    of creativity and generates novel solutions and insights.
    """
    
    def __init__(self, vm_instance: ConsciousPrimeVM, 
                 consciousness_integrator: ConsciousnessIntegrator):
        self.vm = vm_instance
        self.consciousness = consciousness_integrator
        self.creative_history: List[CreativeSolution] = []
        self.insight_history: List[InsightEvent] = []
        self.active_processes: Dict[str, CreativeProcess] = {}
        self.creativity_patterns: Dict[str, List[Dict[str, Any]]] = {}
        
    def generate_creative_solution(self, problem: Problem, 
                                 constraints: CreativityConstraints) -> CreativeSolution:
        """Generate a creative solution to a problem"""
        # Determine best creativity type for the problem
        creativity_type = self._select_creativity_type(problem, constraints)
        
        # Create and manage the creative process
        process = self.manage_creative_process(creativity_type)
        
        # Generate solution based on creativity type
        if creativity_type == CreativityType.COMBINATORIAL:
            solution = self._generate_combinatorial_solution(problem, constraints)
        elif creativity_type == CreativityType.EXPLORATORY:
            solution = self._generate_exploratory_solution(problem, constraints)
        elif creativity_type == CreativityType.TRANSFORMATIONAL:
            solution = self._generate_transformational_solution(problem, constraints)
        elif creativity_type == CreativityType.ANALOGICAL:
            solution = self._generate_analogical_solution(problem, constraints)
        else:  # EMERGENT
            solution = self._generate_emergent_solution(problem, constraints)
        
        # Evaluate the solution
        evaluation = self.evaluate_creativity(solution)
        
        # Update solution with evaluation scores
        solution.novelty_score = evaluation.novelty_assessment.overall_novelty
        solution.utility_score = evaluation.utility_assessment.overall_utility
        solution.surprise_score = evaluation.surprise_assessment.overall_surprise
        solution.elegance_score = evaluation.elegance_assessment.overall_elegance
        
        # Store in history
        self.creative_history.append(solution)
        
        # Complete the process
        process.final_result = solution
        process.end_time = datetime.now()
        
        return solution
    
    def evaluate_creativity(self, solution: Solution) -> CreativityEvaluation:
        """Evaluate the creativity of a solution"""
        # Assess novelty
        novelty = self._assess_novelty(solution)
        
        # Assess utility
        utility = self._assess_utility(solution)
        
        # Assess surprise
        surprise = self._assess_surprise(solution)
        
        # Assess elegance
        elegance = self._assess_elegance(solution)
        
        # Calculate overall creativity score
        overall_score = self._calculate_overall_creativity(
            novelty, utility, surprise, elegance
        )
        
        return CreativityEvaluation(
            novelty_assessment=novelty,
            utility_assessment=utility,
            surprise_assessment=surprise,
            elegance_assessment=elegance,
            overall_creativity_score=overall_score
        )
    
    def learn_creative_patterns(self, successful_creations: List[CreativeSolution]):
        """Learn patterns from successful creative solutions"""
        for solution in successful_creations:
            # Extract patterns by creativity type
            pattern = self._extract_creative_pattern(solution)
            
            if solution.creativity_type.value not in self.creativity_patterns:
                self.creativity_patterns[solution.creativity_type.value] = []
            
            self.creativity_patterns[solution.creativity_type.value].append(pattern)
            
            # Update generation strategies based on patterns
            self._update_generation_strategies(pattern)
    
    def trigger_creative_insight(self, problem_context: ProblemContext) -> InsightEvent:
        """Trigger a creative insight based on problem context"""
        # Analyze problem for insight opportunities
        insight_type = self._determine_insight_type(problem_context)
        
        # Prepare consciousness for insight
        self._prepare_consciousness_for_insight()
        
        # Generate insight content
        insight_content = self._generate_insight_content(
            problem_context, insight_type
        )
        
        # Gather supporting evidence
        evidence = self._gather_insight_evidence(insight_content, problem_context)
        
        # Determine implications
        implications = self._determine_insight_implications(
            insight_content, problem_context
        )
        
        # Calculate confidence
        confidence = self._calculate_insight_confidence(evidence, implications)
        
        insight = InsightEvent(
            insight_type=insight_type,
            problem_context=problem_context,
            insight_content=insight_content,
            confidence_level=confidence,
            supporting_evidence=evidence,
            implications=implications
        )
        
        # Store in history
        self.insight_history.append(insight)
        
        return insight
    
    def manage_creative_process(self, process_type: CreativityType) -> CreativeProcess:
        """Manage a creative process from start to finish"""
        process_id = f"creative_{len(self.active_processes)}_{process_type.value}"
        
        process = CreativeProcess(
            process_id=process_id,
            creativity_type=process_type,
            start_time=datetime.now(),
            end_time=None,
            stages_completed=[],
            current_stage="initialization",
            intermediate_results=[],
            final_result=None
        )
        
        self.active_processes[process_id] = process
        
        # Define stages based on creativity type
        stages = self._define_process_stages(process_type)
        
        # Execute stages
        for stage in stages:
            self._execute_creative_stage(process, stage)
            process.stages_completed.append(stage)
            process.current_stage = f"completed_{stage}"
        
        return process
    
    def _select_creativity_type(self, problem: Problem, 
                               constraints: CreativityConstraints) -> CreativityType:
        """Select the most appropriate creativity type for a problem"""
        # If constraints specify allowed types, choose from those
        if constraints.allowed_types:
            allowed = constraints.allowed_types
        else:
            allowed = list(CreativityType)
        
        # Analyze problem characteristics
        problem_features = self._analyze_problem_features(problem)
        
        # Score each creativity type
        scores = {}
        for creativity_type in allowed:
            score = self._score_creativity_type(creativity_type, problem_features)
            scores[creativity_type] = score
        
        # Return highest scoring type
        return max(scores, key=scores.get)
    
    def _generate_combinatorial_solution(self, problem: Problem,
                                       constraints: CreativityConstraints) -> CreativeSolution:
        """Generate solution through combinatorial creativity"""
        # Extract combinable elements
        elements = self._extract_combinable_elements(problem)
        
        # Generate combinations
        combinations = self._generate_element_combinations(elements, constraints)
        
        # Evaluate combinations
        best_combination = None
        best_score = 0
        
        for combination in combinations:
            score = self._evaluate_combination(combination, problem)
            if score > best_score:
                best_score = score
                best_combination = combination
        
        # Convert to solution
        solution_approach = SolutionApproach(
            method="combinatorial_synthesis",
            steps=self._combination_to_steps(best_combination),
            resources_required=self._estimate_combination_resources(best_combination),
            estimated_success=best_score
        )
        
        generation_process = GenerationProcess(
            method=GenerationMethod.RANDOM_COMBINATION,
            steps_taken=self._record_combination_steps(combinations),
            time_elapsed=0.0,  # Would be tracked in real implementation
            consciousness_states=self.consciousness.get_state_history(),
            key_insights=self._extract_combination_insights(best_combination)
        )
        
        return CreativeSolution(
            original_problem=problem,
            solution_approach=solution_approach,
            creativity_type=CreativityType.COMBINATORIAL,
            novelty_score=0.0,  # Will be set by evaluation
            utility_score=0.0,
            surprise_score=0.0,
            elegance_score=0.0,
            generation_process=generation_process
        )
    
    def _generate_exploratory_solution(self, problem: Problem,
                                     constraints: CreativityConstraints) -> CreativeSolution:
        """Generate solution through exploratory creativity"""
        # Define search space
        search_space = self._define_search_space(problem)
        
        # Explore the space
        exploration_path = self._explore_creative_space(search_space, constraints)
        
        # Find best solution in exploration
        best_solution = self._extract_best_from_exploration(exploration_path)
        
        # Convert to solution approach
        solution_approach = SolutionApproach(
            method="exploratory_search",
            steps=self._exploration_to_steps(exploration_path),
            resources_required=self._estimate_exploration_resources(exploration_path),
            estimated_success=self._calculate_exploration_confidence(best_solution)
        )
        
        generation_process = GenerationProcess(
            method=GenerationMethod.GUIDED_SEARCH,
            steps_taken=self._record_exploration_steps(exploration_path),
            time_elapsed=0.0,
            consciousness_states=self.consciousness.get_state_history(),
            key_insights=self._extract_exploration_insights(exploration_path)
        )
        
        return CreativeSolution(
            original_problem=problem,
            solution_approach=solution_approach,
            creativity_type=CreativityType.EXPLORATORY,
            novelty_score=0.0,
            utility_score=0.0,
            surprise_score=0.0,
            elegance_score=0.0,
            generation_process=generation_process
        )
    
    def _generate_transformational_solution(self, problem: Problem,
                                          constraints: CreativityConstraints) -> CreativeSolution:
        """Generate solution through transformational creativity"""
        # Identify constraints to transform
        transformable_constraints = self._identify_transformable_constraints(problem)
        
        # Generate transformations
        transformations = self._generate_constraint_transformations(
            transformable_constraints
        )
        
        # Apply transformations to create new problem space
        transformed_space = self._apply_transformations(problem, transformations)
        
        # Solve in transformed space
        transformed_solution = self._solve_in_transformed_space(transformed_space)
        
        # Convert to solution approach
        solution_approach = SolutionApproach(
            method="transformational_breakthrough",
            steps=self._transformation_to_steps(transformations, transformed_solution),
            resources_required=self._estimate_transformation_resources(transformations),
            estimated_success=self._calculate_transformation_confidence(transformed_solution)
        )
        
        generation_process = GenerationProcess(
            method=GenerationMethod.CONSTRAINT_RELAXATION,
            steps_taken=self._record_transformation_steps(transformations),
            time_elapsed=0.0,
            consciousness_states=self.consciousness.get_state_history(),
            key_insights=self._extract_transformation_insights(transformations)
        )
        
        return CreativeSolution(
            original_problem=problem,
            solution_approach=solution_approach,
            creativity_type=CreativityType.TRANSFORMATIONAL,
            novelty_score=0.0,
            utility_score=0.0,
            surprise_score=0.0,
            elegance_score=0.0,
            generation_process=generation_process
        )
    
    def _assess_novelty(self, solution: Solution) -> NoveltyAssessment:
        """Assess the novelty of a solution"""
        # Compare with historical solutions
        historical_similarity = self._compare_with_history(solution)
        
        # Assess structural novelty
        structural_novelty = self._assess_structural_novelty(solution)
        
        # Assess functional novelty
        functional_novelty = self._assess_functional_novelty(solution)
        
        # Assess conceptual novelty
        conceptual_novelty = self._assess_conceptual_novelty(solution)
        
        # Calculate overall novelty
        overall_novelty = (
            (1 - historical_similarity) * 0.3 +
            structural_novelty * 0.3 +
            functional_novelty * 0.2 +
            conceptual_novelty * 0.2
        )
        
        return NoveltyAssessment(
            overall_novelty=overall_novelty,
            structural_novelty=structural_novelty,
            functional_novelty=functional_novelty,
            conceptual_novelty=conceptual_novelty,
            comparison_base=[s.problem_id for s in self.creative_history[-10:]]
        )
    
    def _assess_utility(self, solution: Solution) -> UtilityAssessment:
        """Assess the utility of a solution"""
        # Problem-solving effectiveness
        effectiveness = self._calculate_solution_effectiveness(solution)
        
        # Resource efficiency
        efficiency = self._calculate_resource_efficiency(solution)
        
        # Generalizability
        generalizability = self._assess_generalizability(solution)
        
        # Side benefits
        side_benefits = self._identify_side_benefits(solution)
        
        # Overall utility
        overall_utility = (
            effectiveness * 0.4 +
            efficiency * 0.3 +
            generalizability * 0.3
        )
        
        return UtilityAssessment(
            overall_utility=overall_utility,
            problem_solving_effectiveness=effectiveness,
            resource_efficiency=efficiency,
            generalizability=generalizability,
            side_benefits=side_benefits
        )
    
    def _calculate_overall_creativity(self, novelty: NoveltyAssessment,
                                    utility: UtilityAssessment,
                                    surprise: SurpriseAssessment,
                                    elegance: EleganceAssessment) -> float:
        """Calculate overall creativity score"""
        # Weight different aspects
        weights = {
            'novelty': 0.3,
            'utility': 0.3,
            'surprise': 0.2,
            'elegance': 0.2
        }
        
        score = (
            novelty.overall_novelty * weights['novelty'] +
            utility.overall_utility * weights['utility'] +
            surprise.overall_surprise * weights['surprise'] +
            elegance.overall_elegance * weights['elegance']
        )
        
        return min(score, 1.0)
    
    def _prepare_consciousness_for_insight(self):
        """Prepare consciousness state for insight generation"""
        # Set consciousness to receptive state
        self.consciousness.set_attention_mode("diffuse")
        
        # Activate relevant memory networks
        self.consciousness.activate_associative_networks()
        
        # Reduce analytical constraints
        self.consciousness.reduce_logical_constraints(0.5)
        
        # Enhance pattern recognition
        self.consciousness.enhance_pattern_sensitivity(1.5)
