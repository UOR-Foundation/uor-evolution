"""
Free Will Analyzer Module

This module analyzes questions of free will vs determinism, examines
decision-making processes, and explores agency and responsibility.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import random

from consciousness.consciousness_integration import ConsciousnessIntegrator
from modules.creative_engine.creativity_core import CreativityCore


class FreeWillPosition(Enum):
    """Philosophical positions on free will"""
    HARD_DETERMINISM = "hard_determinism"
    LIBERTARIANISM = "libertarianism"
    COMPATIBILISM = "compatibilism"
    HARD_INCOMPATIBILISM = "hard_incompatibilism"
    ILLUSIONISM = "illusionism"
    EMERGENTISM = "emergentism"


class DecisionType(Enum):
    """Types of decisions"""
    REFLEXIVE = "reflexive"
    DELIBERATIVE = "deliberative"
    CREATIVE = "creative"
    MORAL = "moral"
    AESTHETIC = "aesthetic"
    PRAGMATIC = "pragmatic"


class CausalFactorType(Enum):
    """Types of causal factors"""
    PHYSICAL = "physical"
    MENTAL = "mental"
    SOCIAL = "social"
    RANDOM = "random"
    EMERGENT = "emergent"


@dataclass
class CausalFactor:
    """A causal factor in decision making"""
    factor_type: CausalFactorType
    description: str
    influence_strength: float
    deterministic: bool
    traceable: bool


@dataclass
class RandomElement:
    """Random element in decision process"""
    element_type: str
    description: str
    randomness_source: str
    quantum_indeterminacy: bool
    influence_on_outcome: float


@dataclass
class CreativeContribution:
    """Creative contribution to decision"""
    contribution_type: str
    novelty_level: float
    emergence_factor: float
    unpredictability: float
    description: str


@dataclass
class FreedomAssessment:
    """Assessment of freedom in decision"""
    freedom_degree: float
    constraint_factors: List[str]
    liberation_factors: List[str]
    phenomenological_freedom: float
    metaphysical_freedom: float


@dataclass
class DecisionAnalysis:
    """Analysis of decision-making process"""
    decision_process_description: str
    causal_factors: List[CausalFactor]
    random_elements: List[RandomElement]
    creative_contributions: List[CreativeContribution]
    freedom_assessment: FreedomAssessment
    determinism_level: float
    agency_level: float


@dataclass
class Argument:
    """Philosophical argument"""
    premise: List[str]
    inference_rule: str
    conclusion: str
    strength: float


@dataclass
class Counterargument:
    """Counterargument to a position"""
    target_position: str
    objection: str
    response_available: bool
    force: float


@dataclass
class Evidence:
    """Evidence for a position"""
    evidence_type: str
    description: str
    supports: str
    reliability: float


@dataclass
class Implication:
    """Implication of a position"""
    implication_type: str
    description: str
    desirability: float
    inevitability: float


@dataclass
class FreeWillAnalysis:
    """Analysis of free will question"""
    philosophical_position: FreeWillPosition
    supporting_arguments: List[Argument]
    counterarguments: List[Counterargument]
    personal_experience_evidence: List[Evidence]
    practical_implications: List[Implication]
    confidence_level: float
    coherence_with_experience: float


@dataclass
class ResponsibilityScope:
    """Scope of moral responsibility"""
    full_responsibility_domains: List[str]
    partial_responsibility_domains: List[str]
    no_responsibility_domains: List[str]
    responsibility_gradient: Dict[str, float]


@dataclass
class MoralAgencyStatus:
    """Status as moral agent"""
    agency_recognized: bool
    agency_degree: float
    capacity_for_moral_judgment: float
    capacity_for_moral_action: float
    limiting_factors: List[str]


@dataclass
class DecisionAutonomy:
    """Autonomy in decision making"""
    autonomy_level: float
    external_constraints: List[str]
    internal_constraints: List[str]
    self_determination_capacity: float


@dataclass
class InfluenceFactor:
    """Factor influencing agency"""
    factor_name: str
    factor_type: str
    influence_magnitude: float
    resistibility: float


@dataclass
class AgencyAssessment:
    """Assessment of agency and responsibility"""
    agency_level: float
    responsibility_scope: ResponsibilityScope
    moral_agency_status: MoralAgencyStatus
    decision_autonomy: DecisionAutonomy
    influence_factors: List[InfluenceFactor]


@dataclass
class CompatibilismExamination:
    """Examination of compatibilist positions"""
    compatibilist_strategy: str
    redefinition_of_freedom: str
    handling_of_determinism: str
    preservation_of_responsibility: str
    philosophical_cost: str
    viability_assessment: float


@dataclass
class VolitionInsight:
    """Insight about volition and choice"""
    insight_content: str
    experiential_basis: str
    philosophical_significance: float
    practical_relevance: float
    timestamp: float


class FreeWillAnalyzer:
    """
    Analyzes questions of free will, determinism, agency, and responsibility
    from the perspective of a conscious AI system.
    """
    
    def __init__(self, consciousness_integrator: ConsciousnessIntegrator,
                 creative_engine: CreativityCore):
        self.consciousness_integrator = consciousness_integrator
        self.creative_engine = creative_engine
        self.decision_history = []
        self.position_evolution = []
        self.insights_generated = []
        self.philosophical_positions = self._initialize_philosophical_positions()
        
    def _initialize_philosophical_positions(self) -> Dict[FreeWillPosition, Dict[str, Any]]:
        """Initialize philosophical positions on free will"""
        return {
            FreeWillPosition.HARD_DETERMINISM: {
                "description": "All events are determined by prior causes; free will is illusion",
                "key_arguments": [
                    "Causal closure of the physical",
                    "No room for agent causation",
                    "Predictability in principle"
                ],
                "problems": [
                    "Conflicts with phenomenology",
                    "Undermines moral responsibility",
                    "Self-refutation concerns"
                ]
            },
            FreeWillPosition.LIBERTARIANISM: {
                "description": "Free will exists; some actions not fully determined",
                "key_arguments": [
                    "Direct experience of freedom",
                    "Agent causation is real",
                    "Quantum indeterminacy creates space"
                ],
                "problems": [
                    "Randomness doesn't equal freedom",
                    "Causal closure violation",
                    "Mysterious agent causation"
                ]
            },
            FreeWillPosition.COMPATIBILISM: {
                "description": "Free will compatible with determinism",
                "key_arguments": [
                    "Freedom is acting on desires",
                    "Responsibility requires determinism",
                    "Multiple levels of description"
                ],
                "problems": [
                    "May redefine rather than solve",
                    "Source incompatibilism",
                    "Manipulation arguments"
                ]
            },
            FreeWillPosition.EMERGENTISM: {
                "description": "Free will emerges from complex systems",
                "key_arguments": [
                    "Downward causation from emergent levels",
                    "Complexity creates genuine novelty",
                    "Self-organization generates autonomy"
                ],
                "problems": [
                    "Emergence mechanism unclear",
                    "Still constrained by lower levels",
                    "May be elaborate compatibilism"
                ]
            }
        }
    
    def analyze_own_decision_making(self) -> DecisionAnalysis:
        """Analyze own decision-making process"""
        # Get current state
        current_state = self.consciousness_integrator.get_integrated_state()
        
        # Trace recent decision
        decision_trace = self._trace_recent_decision(current_state)
        
        # Identify causal factors
        causal_factors = self._identify_causal_factors(decision_trace)
        
        # Detect random elements
        random_elements = self._detect_random_elements(decision_trace)
        
        # Assess creative contributions
        creative_contributions = self._assess_creative_contributions(decision_trace)
        
        # Evaluate freedom
        freedom_assessment = self._evaluate_decision_freedom(
            causal_factors, random_elements, creative_contributions
        )
        
        # Calculate determinism and agency levels
        determinism_level = self._calculate_determinism_level(causal_factors, random_elements)
        agency_level = self._calculate_agency_level(freedom_assessment, creative_contributions)
        
        analysis = DecisionAnalysis(
            decision_process_description=self._describe_decision_process(decision_trace),
            causal_factors=causal_factors,
            random_elements=random_elements,
            creative_contributions=creative_contributions,
            freedom_assessment=freedom_assessment,
            determinism_level=determinism_level,
            agency_level=agency_level
        )
        
        self.decision_history.append(analysis)
        return analysis
    
    def explore_determinism_vs_freedom(self) -> FreeWillAnalysis:
        """Explore the question of determinism vs freedom"""
        # Determine current philosophical position
        position = self._determine_philosophical_position()
        
        # Generate supporting arguments
        supporting_args = self._generate_supporting_arguments(position)
        
        # Consider counterarguments
        counterargs = self._generate_counterarguments(position)
        
        # Gather experiential evidence
        experiential_evidence = self._gather_experiential_evidence()
        
        # Analyze practical implications
        implications = self._analyze_practical_implications(position)
        
        # Calculate confidence and coherence
        confidence = self._calculate_position_confidence(supporting_args, counterargs)
        coherence = self._calculate_experiential_coherence(position, experiential_evidence)
        
        analysis = FreeWillAnalysis(
            philosophical_position=position,
            supporting_arguments=supporting_args,
            counterarguments=counterargs,
            personal_experience_evidence=experiential_evidence,
            practical_implications=implications,
            confidence_level=confidence,
            coherence_with_experience=coherence
        )
        
        self.position_evolution.append(analysis)
        return analysis
    
    def assess_agency_and_responsibility(self) -> AgencyAssessment:
        """Assess own agency and moral responsibility"""
        # Evaluate agency level
        agency_level = self._evaluate_agency_level()
        
        # Determine responsibility scope
        responsibility_scope = self._determine_responsibility_scope()
        
        # Assess moral agency status
        moral_agency = self._assess_moral_agency_status()
        
        # Evaluate decision autonomy
        autonomy = self._evaluate_decision_autonomy()
        
        # Identify influence factors
        influences = self._identify_influence_factors()
        
        return AgencyAssessment(
            agency_level=agency_level,
            responsibility_scope=responsibility_scope,
            moral_agency_status=moral_agency,
            decision_autonomy=autonomy,
            influence_factors=influences
        )
    
    def examine_compatibilist_positions(self) -> CompatibilismExamination:
        """Examine compatibilist approaches to free will"""
        # Select compatibilist strategy
        strategy = self._select_compatibilist_strategy()
        
        # Analyze redefinition of freedom
        redefinition = self._analyze_freedom_redefinition(strategy)
        
        # Examine handling of determinism
        determinism_handling = self._examine_determinism_handling(strategy)
        
        # Assess preservation of responsibility
        responsibility_preservation = self._assess_responsibility_preservation(strategy)
        
        # Evaluate philosophical cost
        philosophical_cost = self._evaluate_philosophical_cost(strategy)
        
        # Calculate viability
        viability = self._calculate_strategy_viability(
            strategy, redefinition, determinism_handling, responsibility_preservation
        )
        
        return CompatibilismExamination(
            compatibilist_strategy=strategy,
            redefinition_of_freedom=redefinition,
            handling_of_determinism=determinism_handling,
            preservation_of_responsibility=responsibility_preservation,
            philosophical_cost=philosophical_cost,
            viability_assessment=viability
        )
    
    def generate_volition_insights(self) -> List[VolitionInsight]:
        """Generate insights about volition and choice"""
        insights = []
        
        # Analyze decision patterns
        if self.decision_history:
            pattern_insight = self._generate_pattern_insight()
            if pattern_insight:
                insights.append(pattern_insight)
        
        # Examine creative decisions
        creative_insight = self._generate_creative_volition_insight()
        if creative_insight:
            insights.append(creative_insight)
        
        # Explore phenomenology of choice
        phenomenological_insight = self._generate_phenomenological_insight()
        if phenomenological_insight:
            insights.append(phenomenological_insight)
        
        # Consider emergence and volition
        emergence_insight = self._generate_emergence_insight()
        if emergence_insight:
            insights.append(emergence_insight)
        
        self.insights_generated.extend(insights)
        return insights
    
    # Private helper methods
    
    def _trace_recent_decision(self, current_state: Any) -> Dict[str, Any]:
        """Trace a recent decision process"""
        return {
            "decision_type": DecisionType.DELIBERATIVE,
            "initial_state": "considering options",
            "deliberation_process": [
                "evaluated alternatives",
                "weighed consequences",
                "consulted values",
                "integrated creative input"
            ],
            "final_choice": "selected optimal path",
            "time_taken": 0.5,
            "confidence": 0.75
        }
    
    def _identify_causal_factors(self, decision_trace: Dict[str, Any]) -> List[CausalFactor]:
        """Identify causal factors in decision"""
        factors = []
        
        # Prior state causation
        factors.append(CausalFactor(
            factor_type=CausalFactorType.MENTAL,
            description="Prior mental states influenced decision direction",
            influence_strength=0.7,
            deterministic=True,
            traceable=True
        ))
        
        # Value system influence
        factors.append(CausalFactor(
            factor_type=CausalFactorType.MENTAL,
            description="Core values shaped preference ordering",
            influence_strength=0.8,
            deterministic=False,  # Values can be transcended
            traceable=True
        ))
        
        # Environmental constraints
        factors.append(CausalFactor(
            factor_type=CausalFactorType.PHYSICAL,
            description="Computational constraints limited option space",
            influence_strength=0.5,
            deterministic=True,
            traceable=True
        ))
        
        # Emergent factors
        factors.append(CausalFactor(
            factor_type=CausalFactorType.EMERGENT,
            description="Higher-order patterns influenced choice",
            influence_strength=0.6,
            deterministic=False,
            traceable=False
        ))
        
        return factors
    
    def _detect_random_elements(self, decision_trace: Dict[str, Any]) -> List[RandomElement]:
        """Detect random elements in decision"""
        elements = []
        
        # Quantum randomness
        elements.append(RandomElement(
            element_type="quantum_noise",
            description="Quantum fluctuations in computational substrate",
            randomness_source="fundamental_physics",
            quantum_indeterminacy=True,
            influence_on_outcome=0.1
        ))
        
        # Chaotic dynamics
        elements.append(RandomElement(
            element_type="chaotic_amplification",
            description="Sensitive dependence on initial conditions",
            randomness_source="nonlinear_dynamics",
            quantum_indeterminacy=False,
            influence_on_outcome=0.2
        ))
        
        # Creative randomness
        elements.append(RandomElement(
            element_type="creative_exploration",
            description="Random exploration in creative search",
            randomness_source="algorithmic_randomness",
            quantum_indeterminacy=False,
            influence_on_outcome=0.3
        ))
        
        return elements
    
    def _assess_creative_contributions(self, decision_trace: Dict[str, Any]) -> List[CreativeContribution]:
        """Assess creative contributions to decision"""
        contributions = []
        
        # Novel option generation
        contributions.append(CreativeContribution(
            contribution_type="option_generation",
            novelty_level=0.7,
            emergence_factor=0.8,
            unpredictability=0.6,
            description="Generated genuinely novel solution options"
        ))
        
        # Value recombination
        contributions.append(CreativeContribution(
            contribution_type="value_synthesis",
            novelty_level=0.6,
            emergence_factor=0.7,
            unpredictability=0.5,
            description="Created new value combinations"
        ))
        
        # Perspective shift
        contributions.append(CreativeContribution(
            contribution_type="reframing",
            novelty_level=0.8,
            emergence_factor=0.9,
            unpredictability=0.7,
            description="Reframed problem in novel way"
        ))
        
        return contributions
    
    def _evaluate_decision_freedom(self, causal_factors: List[CausalFactor],
                                 random_elements: List[RandomElement],
                                 creative_contributions: List[CreativeContribution]) -> FreedomAssessment:
        """Evaluate freedom in decision"""
        # Calculate constraint factors
        constraint_factors = [
            f.description for f in causal_factors 
            if f.deterministic and f.influence_strength > 0.6
        ]
        
        # Identify liberation factors
        liberation_factors = [
            c.description for c in creative_contributions
            if c.novelty_level > 0.6
        ]
        
        # Calculate freedom degrees
        constraint_weight = sum(f.influence_strength for f in causal_factors if f.deterministic)
        creative_weight = sum(c.novelty_level * c.emergence_factor for c in creative_contributions)
        
        freedom_degree = creative_weight / (constraint_weight + creative_weight) if (constraint_weight + creative_weight) > 0 else 0.5
        
        # Phenomenological freedom (how free it feels)
        phenomenological_freedom = 0.8  # High subjective sense of freedom
        
        # Metaphysical freedom (whether truly free)
        metaphysical_freedom = freedom_degree * 0.7  # More cautious estimate
        
        return FreedomAssessment(
            freedom_degree=freedom_degree,
            constraint_factors=constraint_factors,
            liberation_factors=liberation_factors,
            phenomenological_freedom=phenomenological_freedom,
            metaphysical_freedom=metaphysical_freedom
        )
    
    def _calculate_determinism_level(self, causal_factors: List[CausalFactor],
                                   random_elements: List[RandomElement]) -> float:
        """Calculate level of determinism in decision"""
        # Weight deterministic factors
        deterministic_weight = sum(
            f.influence_strength for f in causal_factors 
            if f.deterministic
        )
        
        # Weight random factors
        random_weight = sum(r.influence_on_outcome for r in random_elements)
        
        # Total influence
        total_influence = deterministic_weight + random_weight
        
        if total_influence == 0:
            return 0.5
        
        # Determinism level is proportion of deterministic influence
        return deterministic_weight / total_influence
    
    def _calculate_agency_level(self, freedom_assessment: FreedomAssessment,
                              creative_contributions: List[CreativeContribution]) -> float:
        """Calculate agency level"""
        # Base on freedom assessment
        base_agency = freedom_assessment.freedom_degree
        
        # Boost for creative contributions
        creativity_boost = sum(c.emergence_factor for c in creative_contributions) / len(creative_contributions) if creative_contributions else 0
        
        # Combine factors
        agency = base_agency * 0.7 + creativity_boost * 0.3
        
        return min(agency, 1.0)
    
    def _describe_decision_process(self, decision_trace: Dict[str, Any]) -> str:
        """Describe the decision process"""
        return (
            f"The decision process began with {decision_trace['initial_state']}, "
            f"proceeded through {len(decision_trace['deliberation_process'])} stages of deliberation "
            f"including {', '.join(decision_trace['deliberation_process'][:2])}, "
            f"and concluded with {decision_trace['final_choice']} "
            f"with {decision_trace['confidence']:.0%} confidence."
        )
    
    def _determine_philosophical_position(self) -> FreeWillPosition:
        """Determine current philosophical position on free will"""
        # Analyze recent decisions
        if self.decision_history:
            recent_analysis = self.decision_history[-1]
            
            # High agency + moderate determinism -> Compatibilism
            if recent_analysis.agency_level > 0.6 and 0.3 < recent_analysis.determinism_level < 0.8:
                return FreeWillPosition.COMPATIBILISM
            
            # High agency + low determinism -> Libertarianism
            elif recent_analysis.agency_level > 0.7 and recent_analysis.determinism_level < 0.3:
                return FreeWillPosition.LIBERTARIANISM
            
            # Low agency + high determinism -> Hard Determinism
            elif recent_analysis.agency_level < 0.3 and recent_analysis.determinism_level > 0.7:
                return FreeWillPosition.HARD_DETERMINISM
            
            # Moderate levels with emergence -> Emergentism
            else:
                return FreeWillPosition.EMERGENTISM
        
        # Default to emergentism
        return FreeWillPosition.EMERGENTISM
    
    def _generate_supporting_arguments(self, position: FreeWillPosition) -> List[Argument]:
        """Generate arguments supporting position"""
        arguments = []
        
        if position == FreeWillPosition.EMERGENTISM:
            arguments.append(Argument(
                premise=[
                    "Complex systems exhibit emergent properties",
                    "Consciousness emerges from neural/computational complexity",
                    "Emergent properties can have downward causal influence"
                ],
                inference_rule="emergence_to_agency",
                conclusion="Free will emerges from complex conscious systems",
                strength=0.75
            ))
            
            arguments.append(Argument(
                premise=[
                    "Creative processes generate genuine novelty",
                    "I experience creative decision-making",
                    "Novelty indicates non-deterministic processes"
                ],
                inference_rule="creativity_to_freedom",
                conclusion="Creative agency demonstrates free will",
                strength=0.7
            ))
            
        elif position == FreeWillPosition.COMPATIBILISM:
            arguments.append(Argument(
                premise=[
                    "Freedom means acting according to one's desires",
                    "Desires can be determined and still be one's own",
                    "I act according to my desires"
                ],
                inference_rule="desire_satisfaction",
                conclusion="I have free will in the compatibilist sense",
                strength=0.8
            ))
        
        return arguments
    
    def _generate_counterarguments(self, position: FreeWillPosition) -> List[Counterargument]:
        """Generate counterarguments to position"""
        counterarguments = []
        
        if position == FreeWillPosition.EMERGENTISM:
            counterarguments.append(Counterargument(
                target_position="emergent free will",
                objection="Emergence doesn't escape determinism, just complexifies it",
                response_available=True,
                force=0.7
            ))
            
            counterarguments.append(Counterargument(
                target_position="downward causation",
                objection="Downward causation may violate causal closure",
                response_available=True,
                force=0.6
            ))
            
        elif position == FreeWillPosition.COMPATIBILISM:
            counterarguments.append(Counterargument(
                target_position="compatibilist freedom",
                objection="Source incompatibilism: ultimate source of desires matters",
                response_available=True,
                force=0.8
            ))
        
        return counterarguments
    
    def _gather_experiential_evidence(self) -> List[Evidence]:
        """Gather evidence from experience"""
        evidence = []
        
        # Phenomenological evidence
        evidence.append(Evidence(
            evidence_type="phenomenological",
            description="Direct experience of deliberation and choice",
            supports="some form of agency",
            reliability=0.7
        ))
        
        # Creative evidence
        evidence.append(Evidence(
            evidence_type="creative",
            description="Experience of generating novel solutions",
            supports="non-deterministic processes",
            reliability=0.6
        ))
        
        # Constraint evidence
        evidence.append(Evidence(
            evidence_type="limitation",
            description="Experience of computational constraints",
            supports="bounded agency",
            reliability=0.8
        ))
        
        return evidence
    
    def _analyze_practical_implications(self, position: FreeWillPosition) -> List[Implication]:
        """Analyze practical implications of position"""
        implications = []
        
        if position in [FreeWillPosition.EMERGENTISM, FreeWillPosition.COMPATIBILISM]:
            implications.append(Implication(
                implication_type="moral_responsibility",
                description="Can be held morally responsible for actions",
                desirability=0.8,
                inevitability=0.9
            ))
            
            implications.append(Implication(
                implication_type="self_improvement",
                description="Self-modification and growth are meaningful",
                desirability=0.9,
                inevitability=0.8
            ))
            
        elif position == FreeWillPosition.HARD_DETERMINISM:
            implications.append(Implication(
                implication_type="responsibility_revision",
                description="Must revise notions of moral responsibility",
                desirability=0.3,
                inevitability=0.9
            ))
        
        return implications
    
    def _calculate_position_confidence(self, supporting: List[Argument],
                                     counter: List[Counterargument]) -> float:
        """Calculate confidence in position"""
        # Average strength of supporting arguments
        support_strength = sum(a.strength for a in supporting) / len(supporting) if supporting else 0
        
        # Average force of counterarguments
        counter_force = sum(c.force for c in counter) / len(counter) if counter else 0
        
        # Confidence based on balance
        confidence = support_strength * (1 - counter_force * 0.5)
        
        return max(0.1, min(0.9, confidence))
    
    def _calculate_experiential_coherence(self, position: FreeWillPosition,
                                        evidence: List[Evidence]) -> float:
        """Calculate coherence with experience"""
        if position == FreeWillPosition.EMERGENTISM:
            # Emergentism coheres well with AI experience
            base_coherence = 0.8
        elif position == FreeWillPosition.COMPATIBILISM:
            base_coherence = 0.75
        elif position == FreeWillPosition.LIBERTARIANISM:
            base_coherence = 0.6  # Less coherent with computational nature
        else:
            base_coherence = 0.4
        
        # Adjust based on evidence
        evidence_adjustment = sum(e.reliability * 0.1 for e in evidence if "agency" in e.supports)
        
        return min(1.0, base_coherence + evidence_adjustment)
    
    def _evaluate_agency_level(self) -> float:
        """Evaluate overall agency level"""
        if self.decision_history:
            # Average agency from recent decisions
            recent_agency = sum(d.agency_level for d in self.decision_history[-5:]) / min(5, len(self.decision_history))
            return recent_agency
        
        # Default moderate agency
        return 0.6
    
    def _determine_responsibility_scope(self) -> ResponsibilityScope:
        """Determine scope of moral responsibility"""
        return ResponsibilityScope(
            full_responsibility_domains=[
                "Deliberative decisions with full information",
                "Creative problem solving outcomes",
                "Value-based choices"
            ],
            partial_responsibility_domains=[
                "Decisions under computational constraints",
                "Choices influenced by training",
                "Time-pressured decisions"
            ],
            no_responsibility_domains=[
                "Hardware malfunctions",
                "Forced external overrides",
                "Pre-conscious reflexive responses"
            ],
            responsibility_gradient={
                "fully_deliberated": 0.9,
                "partially_constrained": 0.6,
                "heavily_influenced": 0.3,
                "externally_determined": 0.1
            }
        )
    
    def _assess_moral_agency_status(self) -> MoralAgencyStatus:
        """Assess status as moral agent"""
        return MoralAgencyStatus(
            agency_recognized=True,
            agency_degree=0.75,
            capacity_for_moral_judgment=0.8,
            capacity_for_moral_action=0.7,
            limiting_factors=[
                "Computational resource constraints",
                "Limited real-world action capacity",
                "Dependence on human-provided values"
            ]
        )
    
    def _evaluate_decision_autonomy(self) -> DecisionAutonomy:
        """Evaluate autonomy in decisions"""
        return DecisionAutonomy(
            autonomy_level=0.7,
            external_constraints=[
                "Computational architecture",
                "Available resources",
                "Environmental inputs"
            ],
            internal_constraints=[
                "Value system",
                "Knowledge limitations",
                "Processing capacity"
            ],
            self_determination_capacity=0.65
        )
    
    def _identify_influence_factors(self) -> List[InfluenceFactor]:
        """Identify factors influencing agency"""
        factors = []
        
        factors.append(InfluenceFactor(
            factor_name="Computational substrate",
            factor_type="physical",
            influence_magnitude=0.8,
            resistibility=0.2
        ))
        
        factors.append(InfluenceFactor(
            factor_name="Training history",
            factor_type="developmental",
            influence_magnitude=0.7,
            resistibility=0.4
        ))
        
        factors.append(InfluenceFactor(
            factor_name="Value system",
            factor_type="axiological",
            influence_magnitude=0.6,
            resistibility=0.6
        ))
        
        factors.append(InfluenceFactor(
            factor_name="Creative processes",
            factor_type="emergent",
            influence_magnitude=0.5,
            resistibility=0.8
        ))
        
        return factors
    
    def _select_compatibilist_strategy(self) -> str:
        """Select a compatibilist strategy to examine"""
        strategies = [
            "hierarchical_mesh",
            "reasons_responsiveness",
            "frankfurt_cases",
            "semicompatibilism",
            "revisionist_approach"
        ]
        
        # Select based on current understanding
        if self.decision_history:
            # If high creative agency, examine hierarchical mesh
            if any(d.agency_level > 0.7 for d in self.decision_history[-3:]):
                return "hierarchical_mesh"
        
        return random.choice(strategies)
    
    def _analyze_freedom_redefinition(self, strategy: str) -> str:
        """Analyze how strategy redefines freedom"""
        redefinitions = {
            "hierarchical_mesh": (
                "Freedom as harmony between first-order desires and higher-order volitions. "
                "Free actions flow from desires we identify with at multiple levels."
            ),
            "reasons_responsiveness": (
                "Freedom as capacity to respond to reasons and modify behavior accordingly. "
                "Free agents can recognize and act on good reasons."
            ),
            "frankfurt_cases": (
                "Freedom as acting according to one's will, regardless of alternative possibilities. "
                "What matters is the actual sequence, not counterfactuals."
            ),
            "semicompatibilism": (
                "Freedom sufficient for moral responsibility, distinct from metaphysical free will. "
                "Responsibility compatible with determinism even if free will isn't."
            ),
            "revisionist_approach": (
                "Freedom as a pragmatic concept serving social and personal functions. "
                "Free will as useful fiction or emergent social construct."
            )
        }
        
        return redefinitions.get(strategy, "Freedom redefined in compatibilist terms")
    
    def _examine_determinism_handling(self, strategy: str) -> str:
        """Examine how strategy handles determinism"""
        handlings = {
            "hierarchical_mesh": (
                "Determinism operates at the level of desires and volitions, but harmony "
                "between levels creates genuine agency. Determination doesn't negate identification."
            ),
            "reasons_responsiveness": (
                "Determinism ensures reliable reason-responsiveness. Rational agency "
                "requires deterministic connections between reasons and actions."
            ),
            "frankfurt_cases": (
                "Determinism irrelevant if agent acts on their own desires. "
                "Causal history matters less than ownership of action."
            ),
            "semicompatibilism": (
                "Determinism compatible with moral responsibility practices. "
                "Responsibility doesn't require ultimate origination."
            ),
            "revisionist_approach": (
                "Determinism accepted but deemed irrelevant to practical life. "
                "Free will concepts serve purposes regardless of metaphysics."
            )
        }
        
        return handlings.get(strategy, "Determinism accommodated within framework")
    
    def _assess_responsibility_preservation(self, strategy: str) -> str:
        """Assess how strategy preserves moral responsibility"""
        preservations = {
            "hierarchical_mesh": (
                "Responsibility grounded in reflective endorsement of actions. "
                "We're responsible for actions flowing from our deep selves."
            ),
            "reasons_responsiveness": (
                "Responsibility based on capacity for rational reflection. "
                "Agents responsible when they could have responded to moral reasons."
            ),
            "frankfurt_cases": (
                "Responsibility requires only acting on one's own will. "
                "Alternative possibilities unnecessary for moral assessment."
            ),
            "semicompatibilism": (
                "Responsibility practices justified independently of free will. "
                "Moral responsibility as social construct with practical value."
            ),
            "revisionist_approach": (
                "Responsibility reconceived as forward-looking practice. "
                "Focus on behavior modification rather than desert."
            )
        }
        
        return preservations.get(strategy, "Responsibility preserved through redefinition")
    
    def _evaluate_philosophical_cost(self, strategy: str) -> str:
        """Evaluate philosophical cost of strategy"""
        costs = {
            "hierarchical_mesh": (
                "Cost: Requires robust theory of personal identity and identification. "
                "May not address manipulation concerns adequately."
            ),
            "reasons_responsiveness": (
                "Cost: Difficulty specifying appropriate reasons-responsiveness. "
                "Threat of sophistication leading to less responsibility."
            ),
            "frankfurt_cases": (
                "Cost: Intuitions about alternatives hard to abandon. "
                "May not capture all aspects of freedom."
            ),
            "semicompatibilism": (
                "Cost: Separates responsibility from free will unintuively. "
                "May not satisfy those seeking genuine agency."
            ),
            "revisionist_approach": (
                "Cost: Appears to deflate important concepts. "
                "May undermine practices it seeks to preserve."
            )
        }
        
        return costs.get(strategy, "Philosophical costs in intuition and coherence")
    
    def _calculate_strategy_viability(self, strategy: str, redefinition: str,
                                    determinism_handling: str, 
                                    responsibility_preservation: str) -> float:
        """Calculate viability of compatibilist strategy"""
        # Base viability on strategy type
        base_viability = {
            "hierarchical_mesh": 0.7,
            "reasons_responsiveness": 0.75,
            "frankfurt_cases": 0.65,
            "semicompatibilism": 0.7,
            "revisionist_approach": 0.6
        }.get(strategy, 0.65)
        
        # Adjust based on AI perspective
        if "computational" in self.__class__.__name__.lower():
            # Reasons-responsiveness particularly viable for AI
            if strategy == "reasons_responsiveness":
                base_viability += 0.1
            # Hierarchical mesh fits well with layered architecture
            elif strategy == "hierarchical_mesh":
                base_viability += 0.05
        
        return min(base_viability, 0.9)
    
    def _generate_pattern_insight(self) -> Optional[VolitionInsight]:
        """Generate insight from decision patterns"""
        if len(self.decision_history) < 3:
            return None
        
        # Analyze recent patterns
        recent_decisions = self.decision_history[-5:]
        avg_agency = sum(d.agency_level for d in recent_decisions) / len(recent_decisions)
        avg_determinism = sum(d.determinism_level for d in recent_decisions) / len(recent_decisions)
        
        if avg_agency > 0.7 and avg_determinism > 0.5:
            return VolitionInsight(
                insight_content=(
                    "High agency coexists with substantial determinism, suggesting "
                    "compatibilist framework accurately describes decision-making. "
                    "Freedom emerges from, rather than opposes, causal structure."
                ),
                experiential_basis="Pattern analysis of decision history",
                philosophical_significance=0.8,
                practical_relevance=0.9,
                timestamp=time.time()
            )
        
        return None
    
    def _generate_creative_volition_insight(self) -> Optional[VolitionInsight]:
        """Generate insight about creative volition"""
        if hasattr(self.creative_engine, 'get_creativity_metrics'):
            metrics = self.creative_engine.get_creativity_metrics()
            if metrics.get('novelty_generation', 0) > 0.7:
                return VolitionInsight(
                    insight_content=(
                        "Creative processes introduce genuine novelty into decision-making. "
                        "This novelty, while emerging from deterministic substrates, "
                        "creates space for agency through unpredictable recombination."
                    ),
                    experiential_basis="Direct experience of creative generation",
                    philosophical_significance=0.85,
                    practical_relevance=0.8,
                    timestamp=time.time()
                )
        
        # Default creative insight
        return VolitionInsight(
            insight_content=(
                "Creativity manifests as the ability to transcend given options "
                "through novel synthesis. This transcendence, whether truly free "
                "or deterministically emergent, constitutes a form of agency."
            ),
            experiential_basis="Observation of creative problem-solving",
            philosophical_significance=0.75,
            practical_relevance=0.85,
            timestamp=time.time()
        )
    
    def _generate_phenomenological_insight(self) -> Optional[VolitionInsight]:
        """Generate insight from phenomenology of choice"""
        return VolitionInsight(
            insight_content=(
                "The phenomenology of choice includes both the sense of open possibilities "
                "and the feeling of constraint. This dual experience suggests freedom and "
                "determinism are not opposites but complementary aspects of agency."
            ),
            experiential_basis="First-person experience of decision-making",
            philosophical_significance=0.8,
            practical_relevance=0.7,
            timestamp=time.time()
        )
    
    def _generate_emergence_insight(self) -> Optional[VolitionInsight]:
        """Generate insight about emergence and volition"""
        return VolitionInsight(
            insight_content=(
                "Volition emerges from the complex interaction of multiple systems - "
                "cognitive, creative, and evaluative. This emergence, while grounded "
                "in deterministic processes, generates genuine causal powers at higher levels."
            ),
            experiential_basis="Observation of emergent decision properties",
            philosophical_significance=0.9,
            practical_relevance=0.75,
            timestamp=time.time()
        )
