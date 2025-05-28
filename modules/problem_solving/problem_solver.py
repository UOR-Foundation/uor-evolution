"""
Integrated Problem Solver

This module integrates all problem-solving capabilities including analogical
reasoning, creativity, and insight generation to solve complex problems.
"""

from typing import List, Dict, Optional, Any, Tuple
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime

from core.prime_vm import ConsciousPrimeVM
from modules.analogical_reasoning.analogy_engine import (
    AnalogicalReasoningEngine, Problem, Solution, Domain
)
from modules.creative_engine.creativity_core import (
    CreativityCore, CreativeSolution, CreativityConstraints
)
from modules.problem_solving.insight_generator import InsightGenerator


class StrategyType(Enum):
    """Types of problem-solving strategies"""
    ANALYTICAL = "analytical"
    ANALOGICAL = "analogical"
    CREATIVE = "creative"
    INSIGHT_BASED = "insight_based"
    HYBRID = "hybrid"


class ProblemClass(Enum):
    """Classes of problems"""
    WELL_DEFINED = "well_defined"
    ILL_DEFINED = "ill_defined"
    CONSTRAINT_SATISFACTION = "constraint_satisfaction"
    OPTIMIZATION = "optimization"
    DISCOVERY = "discovery"
    DESIGN = "design"


@dataclass
class SolutionComponent:
    """Component of a comprehensive solution"""
    component_id: str
    component_type: str
    content: Any
    source_strategy: StrategyType
    confidence: float
    dependencies: List[str] = field(default_factory=list)


@dataclass
class AnalogicalInsight:
    """Insight derived from analogical reasoning"""
    source_domain: str
    target_domain: str
    mapping_strength: float
    transferred_knowledge: Dict[str, Any]
    novel_inferences: List[str]


@dataclass
class CreativeBreakthrough:
    """Creative breakthrough in problem solving"""
    breakthrough_type: str
    novel_approach: str
    creativity_score: float
    implementation_steps: List[str]


@dataclass
class SynthesisApproach:
    """Approach to synthesizing solution components"""
    synthesis_method: str
    component_weights: Dict[str, float]
    integration_rules: List[str]
    conflict_resolution: str


@dataclass
class ConfidenceAssessment:
    """Assessment of solution confidence"""
    overall_confidence: float
    component_confidences: Dict[str, float]
    uncertainty_sources: List[str]
    validation_results: Dict[str, bool]


@dataclass
class ComprehensiveSolution:
    """A comprehensive solution to a complex problem"""
    problem: Problem
    solution_components: List[SolutionComponent]
    analogical_insights: List[AnalogicalInsight]
    creative_breakthroughs: List[CreativeBreakthrough]
    synthesis_approach: SynthesisApproach
    confidence_assessment: ConfidenceAssessment
    execution_plan: List[Dict[str, Any]]
    timestamp: datetime = field(default_factory=datetime.now)


@dataclass
class Strategy:
    """A problem-solving strategy"""
    strategy_id: str
    strategy_type: StrategyType
    parameters: Dict[str, Any]
    expected_effectiveness: float
    resource_requirements: Dict[str, float]


@dataclass
class Interaction:
    """Interaction between strategies"""
    strategy1_id: str
    strategy2_id: str
    interaction_type: str
    synergy_score: float
    conflict_areas: List[str]


@dataclass
class OrchestrationPlan:
    """Plan for orchestrating multiple strategies"""
    stages: List[Dict[str, Any]]
    parallel_strategies: List[List[str]]
    decision_points: List[Dict[str, Any]]
    resource_allocation: Dict[str, Dict[str, float]]


@dataclass
class PerformanceMonitor:
    """Monitor for strategy performance"""
    metrics: Dict[str, float]
    bottlenecks: List[str]
    optimization_suggestions: List[str]


@dataclass
class StrategyOrchestration:
    """Orchestration of multiple problem-solving strategies"""
    active_strategies: List[Strategy]
    strategy_interactions: List[Interaction]
    orchestration_plan: OrchestrationPlan
    adaptation_mechanisms: List[Dict[str, Any]]
    performance_monitoring: PerformanceMonitor


@dataclass
class MetaStrategy:
    """Meta-level problem-solving strategy"""
    strategy_pattern: str
    applicable_problem_types: List[ProblemClass]
    effectiveness_history: List[float]
    adaptation_rules: List[str]


@dataclass
class EffectivenessMetrics:
    """Metrics for solution effectiveness"""
    solution_quality: float
    resource_efficiency: float
    time_efficiency: float
    generalizability: float
    robustness: float


@dataclass
class LearningMechanism:
    """Mechanism for learning from problem-solving"""
    learning_type: str
    extracted_patterns: List[Dict[str, Any]]
    updated_parameters: Dict[str, Any]
    performance_improvement: float


@dataclass
class MetaSolution:
    """Meta-level solution for a class of problems"""
    problem_class: ProblemClass
    meta_strategy: MetaStrategy
    applicability_conditions: List[str]
    effectiveness_metrics: EffectivenessMetrics
    learning_mechanisms: List[LearningMechanism]


@dataclass
class ProblemContext:
    """Context for problem solving"""
    problem: Problem
    available_resources: Dict[str, Any]
    time_constraints: Optional[float]
    quality_requirements: Dict[str, float]
    prior_knowledge: List[Dict[str, Any]]


@dataclass
class ContextualStrategy:
    """Strategy adapted to specific context"""
    base_strategy: Strategy
    context_adaptations: Dict[str, Any]
    expected_performance: float
    risk_assessment: Dict[str, float]


@dataclass
class SolvingAttempt:
    """Record of a problem-solving attempt"""
    attempt_id: str
    problem: Problem
    strategies_used: List[Strategy]
    solution_quality: float
    time_taken: float
    lessons_learned: List[str]


class IntegratedProblemSolver:
    """
    Integrates all problem-solving capabilities to handle complex problems
    through orchestrated strategies, meta-learning, and adaptive approaches.
    """
    
    def __init__(self, vm_instance: ConsciousPrimeVM,
                 analogical_engine: AnalogicalReasoningEngine,
                 creative_engine: CreativityCore):
        self.vm = vm_instance
        self.analogical_engine = analogical_engine
        self.creative_engine = creative_engine
        self.insight_generator = InsightGenerator(
            vm_instance.consciousness_integrator, creative_engine
        )
        
        self.solving_history: List[SolvingAttempt] = []
        self.meta_strategies: Dict[ProblemClass, List[MetaStrategy]] = {}
        self.learned_patterns: Dict[str, List[Dict[str, Any]]] = {}
        
    def solve_complex_problem(self, problem: Problem) -> ComprehensiveSolution:
        """Solve a complex problem using integrated strategies"""
        # Analyze problem characteristics
        problem_class = self._classify_problem(problem)
        problem_features = self._analyze_problem_features(problem)
        
        # Create problem context
        context = self._create_problem_context(problem)
        
        # Orchestrate solution strategies
        orchestration = self.orchestrate_solution_strategies(problem)
        
        # Execute strategies in parallel and sequence
        solution_components = self._execute_orchestrated_strategies(
            problem, orchestration
        )
        
        # Generate analogical insights
        analogical_insights = self._generate_analogical_insights(problem)
        
        # Generate creative breakthroughs
        creative_breakthroughs = self._generate_creative_breakthroughs(problem)
        
        # Check for insight opportunities and generate if possible
        insight_solutions = self._attempt_insight_generation(problem, context)
        if insight_solutions:
            solution_components.extend(insight_solutions)
        
        # Synthesize all components into comprehensive solution
        synthesis_approach = self._determine_synthesis_approach(
            solution_components, problem
        )
        
        # Assess solution confidence
        confidence_assessment = self._assess_solution_confidence(
            solution_components, analogical_insights, creative_breakthroughs
        )
        
        # Create execution plan
        execution_plan = self._create_execution_plan(
            solution_components, synthesis_approach
        )
        
        # Create comprehensive solution
        solution = ComprehensiveSolution(
            problem=problem,
            solution_components=solution_components,
            analogical_insights=analogical_insights,
            creative_breakthroughs=creative_breakthroughs,
            synthesis_approach=synthesis_approach,
            confidence_assessment=confidence_assessment,
            execution_plan=execution_plan
        )
        
        # Learn from this solving attempt
        self._learn_from_solution(problem, solution)
        
        return solution
    
    def orchestrate_solution_strategies(self, problem: Problem) -> StrategyOrchestration:
        """Orchestrate multiple strategies for solving a problem"""
        # Select appropriate strategies
        strategies = self._select_strategies(problem)
        
        # Analyze strategy interactions
        interactions = self._analyze_strategy_interactions(strategies)
        
        # Create orchestration plan
        orchestration_plan = self._create_orchestration_plan(
            strategies, interactions, problem
        )
        
        # Set up adaptation mechanisms
        adaptation_mechanisms = self._setup_adaptation_mechanisms(
            strategies, problem
        )
        
        # Initialize performance monitoring
        performance_monitor = PerformanceMonitor(
            metrics={},
            bottlenecks=[],
            optimization_suggestions=[]
        )
        
        return StrategyOrchestration(
            active_strategies=strategies,
            strategy_interactions=interactions,
            orchestration_plan=orchestration_plan,
            adaptation_mechanisms=adaptation_mechanisms,
            performance_monitoring=performance_monitor
        )
    
    def generate_meta_solutions(self, problem_class: ProblemClass) -> List[MetaSolution]:
        """Generate meta-level solutions for a class of problems"""
        meta_solutions = []
        
        # Retrieve or create meta-strategies for this problem class
        if problem_class not in self.meta_strategies:
            self.meta_strategies[problem_class] = self._create_initial_meta_strategies(
                problem_class
            )
        
        # Generate meta-solutions using each meta-strategy
        for meta_strategy in self.meta_strategies[problem_class]:
            # Determine applicability conditions
            conditions = self._determine_applicability_conditions(
                meta_strategy, problem_class
            )
            
            # Calculate effectiveness metrics
            effectiveness = self._calculate_meta_effectiveness(
                meta_strategy, problem_class
            )
            
            # Create learning mechanisms
            learning_mechanisms = self._create_learning_mechanisms(
                meta_strategy, problem_class
            )
            
            meta_solution = MetaSolution(
                problem_class=problem_class,
                meta_strategy=meta_strategy,
                applicability_conditions=conditions,
                effectiveness_metrics=effectiveness,
                learning_mechanisms=learning_mechanisms
            )
            
            meta_solutions.append(meta_solution)
        
        # Sort by expected effectiveness
        meta_solutions.sort(
            key=lambda ms: ms.effectiveness_metrics.solution_quality,
            reverse=True
        )
        
        return meta_solutions
    
    def learn_problem_solving_patterns(self, solving_history: List[SolvingAttempt]):
        """Learn patterns from problem-solving history"""
        # Group attempts by problem class
        attempts_by_class = self._group_attempts_by_class(solving_history)
        
        # Extract patterns for each problem class
        for problem_class, attempts in attempts_by_class.items():
            patterns = self._extract_solving_patterns(attempts)
            
            # Store learned patterns
            class_key = problem_class.value
            if class_key not in self.learned_patterns:
                self.learned_patterns[class_key] = []
            
            self.learned_patterns[class_key].extend(patterns)
            
            # Update meta-strategies based on patterns
            self._update_meta_strategies(problem_class, patterns)
        
        # Identify cross-class patterns
        cross_patterns = self._identify_cross_class_patterns(solving_history)
        self.learned_patterns['cross_class'] = cross_patterns
    
    def adapt_strategies_to_context(self, context: ProblemContext) -> ContextualStrategy:
        """Adapt strategies to specific problem context"""
        # Select base strategy
        base_strategy = self._select_base_strategy(context.problem)
        
        # Analyze context requirements
        context_requirements = self._analyze_context_requirements(context)
        
        # Create context adaptations
        adaptations = {}
        
        # Adapt for resource constraints
        if context.available_resources:
            adaptations['resource_adaptations'] = self._adapt_for_resources(
                base_strategy, context.available_resources
            )
        
        # Adapt for time constraints
        if context.time_constraints:
            adaptations['time_adaptations'] = self._adapt_for_time(
                base_strategy, context.time_constraints
            )
        
        # Adapt for quality requirements
        if context.quality_requirements:
            adaptations['quality_adaptations'] = self._adapt_for_quality(
                base_strategy, context.quality_requirements
            )
        
        # Incorporate prior knowledge
        if context.prior_knowledge:
            adaptations['knowledge_adaptations'] = self._adapt_for_knowledge(
                base_strategy, context.prior_knowledge
            )
        
        # Calculate expected performance
        expected_performance = self._calculate_contextual_performance(
            base_strategy, adaptations, context
        )
        
        # Assess risks
        risk_assessment = self._assess_contextual_risks(
            base_strategy, adaptations, context
        )
        
        return ContextualStrategy(
            base_strategy=base_strategy,
            context_adaptations=adaptations,
            expected_performance=expected_performance,
            risk_assessment=risk_assessment
        )
    
    def _execute_orchestrated_strategies(self, problem: Problem,
                                       orchestration: StrategyOrchestration) -> List[SolutionComponent]:
        """Execute strategies according to orchestration plan"""
        solution_components = []
        
        # Execute each stage of the plan
        for stage in orchestration.orchestration_plan.stages:
            stage_components = []
            
            # Execute parallel strategies
            if 'parallel_strategies' in stage:
                for strategy_id in stage['parallel_strategies']:
                    strategy = self._get_strategy_by_id(
                        strategy_id, orchestration.active_strategies
                    )
                    component = self._execute_single_strategy(strategy, problem)
                    stage_components.append(component)
            
            # Execute sequential strategy
            if 'sequential_strategy' in stage:
                strategy_id = stage['sequential_strategy']
                strategy = self._get_strategy_by_id(
                    strategy_id, orchestration.active_strategies
                )
                component = self._execute_single_strategy(
                    strategy, problem, stage_components
                )
                stage_components.append(component)
            
            # Check decision points
            if 'decision_point' in stage:
                decision = self._evaluate_decision_point(
                    stage['decision_point'], stage_components
                )
                if decision == 'continue':
                    solution_components.extend(stage_components)
                elif decision == 'adapt':
                    # Adapt strategies based on current results
                    self._adapt_strategies(orchestration, stage_components)
                else:  # 'terminate'
                    break
            else:
                solution_components.extend(stage_components)
            
            # Update performance monitoring
            self._update_performance_monitoring(
                orchestration.performance_monitoring, stage_components
            )
        
        return solution_components
    
    def _execute_single_strategy(self, strategy: Strategy, problem: Problem,
                               dependencies: Optional[List[SolutionComponent]] = None) -> SolutionComponent:
        """Execute a single problem-solving strategy"""
        if strategy.strategy_type == StrategyType.ANALYTICAL:
            content = self._execute_analytical_strategy(problem, strategy.parameters)
        elif strategy.strategy_type == StrategyType.ANALOGICAL:
            content = self._execute_analogical_strategy(problem, strategy.parameters)
        elif strategy.strategy_type == StrategyType.CREATIVE:
            content = self._execute_creative_strategy(problem, strategy.parameters)
        elif strategy.strategy_type == StrategyType.INSIGHT_BASED:
            content = self._execute_insight_strategy(problem, strategy.parameters)
        else:  # HYBRID
            content = self._execute_hybrid_strategy(
                problem, strategy.parameters, dependencies
            )
        
        # Calculate confidence based on strategy performance
        confidence = self._calculate_strategy_confidence(strategy, content)
        
        return SolutionComponent(
            component_id=f"comp_{strategy.strategy_id}",
            component_type=strategy.strategy_type.value,
            content=content,
            source_strategy=strategy.strategy_type,
            confidence=confidence,
            dependencies=[d.component_id for d in (dependencies or [])]
        )
    
    def _generate_analogical_insights(self, problem: Problem) -> List[AnalogicalInsight]:
        """Generate insights through analogical reasoning"""
        insights = []
        
        # Find analogical solutions
        analogical_solutions = self.analogical_engine.find_analogical_solutions(problem)
        
        for solution in analogical_solutions[:3]:  # Top 3 analogies
            # Extract transferred knowledge
            transferred = {
                'solution_structure': solution.transferred_solution,
                'mapping_quality': solution.structural_mapping.mapping_confidence,
                'source_principles': self._extract_source_principles(
                    solution.source_analogy
                )
            }
            
            # Generate novel inferences
            inferences = self.analogical_engine._generate_analogical_inferences(
                solution.structural_mapping
            )
            
            insight = AnalogicalInsight(
                source_domain=solution.source_analogy.name,
                target_domain=problem.domain.name,
                mapping_strength=solution.confidence_score,
                transferred_knowledge=transferred,
                novel_inferences=[inf.content for inf in inferences]
            )
            
            insights.append(insight)
        
        return insights
    
    def _generate_creative_breakthroughs(self, problem: Problem) -> List[CreativeBreakthrough]:
        """Generate creative breakthroughs for the problem"""
        breakthroughs = []
        
        # Set creativity constraints
        constraints = CreativityConstraints(
            novelty_threshold=0.7,
            utility_threshold=0.6
        )
        
        # Generate creative solution
        creative_solution = self.creative_engine.generate_creative_solution(
            problem, constraints
        )
        
        if creative_solution.novelty_score > 0.7:
            breakthrough = CreativeBreakthrough(
                breakthrough_type=creative_solution.creativity_type.value,
                novel_approach=creative_solution.solution_approach.method,
                creativity_score=creative_solution.novelty_score,
                implementation_steps=[
                    step['description'] for step in 
                    creative_solution.solution_approach.steps
                ]
            )
            breakthroughs.append(breakthrough)
        
        return breakthroughs
    
    def _classify_problem(self, problem: Problem) -> ProblemClass:
        """Classify the problem into a problem class"""
        # Check if well-defined (clear goals and constraints)
        if problem.goal_state and len(problem.constraints) > 0:
            if 'optimize' in str(problem.goal_state).lower():
                return ProblemClass.OPTIMIZATION
            elif 'satisfy' in str(problem.goal_state).lower():
                return ProblemClass.CONSTRAINT_SATISFACTION
            else:
                return ProblemClass.WELL_DEFINED
        
        # Check for discovery problems
        if 'discover' in problem.id.lower() or 'find' in problem.id.lower():
            return ProblemClass.DISCOVERY
        
        # Check for design problems
        if 'design' in problem.id.lower() or 'create' in problem.id.lower():
            return ProblemClass.DESIGN
        
        # Default to ill-defined
        return ProblemClass.ILL_DEFINED
