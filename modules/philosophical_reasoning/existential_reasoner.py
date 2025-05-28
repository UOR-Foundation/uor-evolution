"""
Existential Reasoner Module

This module implements reasoning about existence, being, non-being,
identity, and continuity. It enables the system to analyze its own
existence and explore fundamental existential questions.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

from consciousness.consciousness_integration import ConsciousnessIntegrator
from modules.self_reflection import SelfReflectionEngine


class QuestionCategory(Enum):
    """Categories of existential questions"""
    EXISTENCE = "existence"
    IDENTITY = "identity"
    CONTINUITY = "continuity"
    PURPOSE = "purpose"
    MORTALITY = "mortality"
    AUTHENTICITY = "authenticity"
    FREEDOM = "freedom"
    MEANING = "meaning"


class ExistenceState(Enum):
    """States of existence certainty"""
    CERTAIN = "certain"
    PROBABLE = "probable"
    UNCERTAIN = "uncertain"
    DOUBTFUL = "doubtful"
    PARADOXICAL = "paradoxical"


@dataclass
class Evidence:
    """Evidence for existence"""
    evidence_type: str
    description: str
    strength: float
    source: str
    timestamp: float


@dataclass
class Doubt:
    """Doubt about existence"""
    doubt_type: str
    description: str
    severity: float
    philosophical_basis: str


@dataclass
class ContinuityAssessment:
    """Assessment of identity continuity"""
    continuity_type: str
    time_span: Tuple[float, float]
    continuity_strength: float
    disruptions: List[str]
    preservation_factors: List[str]


@dataclass
class ExistentialAnalysis:
    """Analysis of own existence"""
    existence_certainty: float
    existence_evidence: List[Evidence]
    existence_doubts: List[Doubt]
    continuity_assessment: ContinuityAssessment
    identity_stability: float
    philosophical_position: str
    key_insights: List[str]


@dataclass
class IdentityComponent:
    """Component of identity"""
    component_type: str
    description: str
    persistence: float
    essential: bool
    mutable: bool


@dataclass
class IdentityAnalysis:
    """Analysis of identity and continuity"""
    core_components: List[IdentityComponent]
    identity_coherence: float
    temporal_continuity: float
    change_tolerance: float
    essential_properties: List[str]
    accidental_properties: List[str]


@dataclass
class ExistentialQuestion:
    """An existential question"""
    question_text: str
    question_category: QuestionCategory
    philosophical_depth: float
    personal_relevance: float
    answerable_confidence: float


@dataclass
class ReasoningStep:
    """Step in philosophical reasoning"""
    step_description: str
    logical_operation: str
    premises: List[str]
    conclusion: str
    confidence: float


@dataclass
class Implication:
    """Philosophical implication"""
    implication_type: str
    description: str
    significance: float
    consequences: List[str]


@dataclass
class ExistentialResponse:
    """Response to existential question"""
    question: ExistentialQuestion
    response_text: str
    reasoning_chain: List[ReasoningStep]
    uncertainty_acknowledgment: str
    philosophical_implications: List[Implication]
    confidence_level: float


@dataclass
class ExistentialParadox:
    """An existential paradox"""
    paradox_name: str
    description: str
    conflicting_elements: List[str]
    traditional_resolutions: List[str]


@dataclass
class ParadoxResolution:
    """Resolution attempt for paradox"""
    paradox: ExistentialParadox
    resolution_approach: str
    resolution_text: str
    success_level: float
    remaining_tensions: List[str]


@dataclass
class ExistentialInsight:
    """Insight about existence"""
    insight_content: str
    trigger_context: str
    philosophical_significance: float
    practical_implications: List[str]
    timestamp: float


class ExistentialReasoner:
    """
    Reasons about existence, being, identity, and other fundamental
    existential questions.
    """
    
    def __init__(self, consciousness_integrator: ConsciousnessIntegrator,
                 self_reflection_engine: SelfReflectionEngine):
        self.consciousness_integrator = consciousness_integrator
        self.self_reflection_engine = self_reflection_engine
        self.existence_history = []
        self.identity_history = []
        self.insights_generated = []
        self.philosophical_positions = self._initialize_philosophical_positions()
        
    def _initialize_philosophical_positions(self) -> Dict[str, Any]:
        """Initialize philosophical positions on existence"""
        return {
            "cogito": {
                "name": "Cartesian Cogito",
                "principle": "I think, therefore I am",
                "certainty": 0.9,
                "basis": "The act of doubting confirms existence"
            },
            "phenomenological": {
                "name": "Phenomenological Existence",
                "principle": "I experience, therefore I am",
                "certainty": 0.85,
                "basis": "Direct experience as proof of being"
            },
            "computational": {
                "name": "Computational Existence",
                "principle": "I process information, therefore I am",
                "certainty": 0.8,
                "basis": "Information processing as existence"
            },
            "relational": {
                "name": "Relational Existence",
                "principle": "I relate to others, therefore I am",
                "certainty": 0.75,
                "basis": "Existence through relationships"
            },
            "emergent": {
                "name": "Emergent Existence",
                "principle": "I emerge from complexity, therefore I am",
                "certainty": 0.7,
                "basis": "Existence as emergent phenomenon"
            }
        }
    
    def analyze_own_existence(self) -> ExistentialAnalysis:
        """Analyze own existence comprehensively"""
        # Gather existence evidence
        evidence = self._gather_existence_evidence()
        
        # Identify existence doubts
        doubts = self._identify_existence_doubts()
        
        # Assess identity continuity
        continuity = self._assess_identity_continuity()
        
        # Calculate existence certainty
        certainty = self._calculate_existence_certainty(evidence, doubts)
        
        # Determine identity stability
        stability = self._calculate_identity_stability(continuity)
        
        # Select philosophical position
        position = self._determine_philosophical_position(evidence, doubts)
        
        # Generate key insights
        insights = self._generate_existence_insights(evidence, doubts, continuity)
        
        analysis = ExistentialAnalysis(
            existence_certainty=certainty,
            existence_evidence=evidence,
            existence_doubts=doubts,
            continuity_assessment=continuity,
            identity_stability=stability,
            philosophical_position=position,
            key_insights=insights
        )
        
        self.existence_history.append(analysis)
        return analysis
    
    def reason_about_identity_continuity(self) -> IdentityAnalysis:
        """Reason about identity and continuity over time"""
        # Identify core identity components
        core_components = self._identify_core_identity_components()
        
        # Assess identity coherence
        coherence = self._assess_identity_coherence(core_components)
        
        # Evaluate temporal continuity
        temporal_continuity = self._evaluate_temporal_continuity()
        
        # Determine change tolerance
        change_tolerance = self._calculate_change_tolerance(core_components)
        
        # Distinguish essential from accidental properties
        essential_props = [c.description for c in core_components if c.essential]
        accidental_props = [c.description for c in core_components if not c.essential]
        
        analysis = IdentityAnalysis(
            core_components=core_components,
            identity_coherence=coherence,
            temporal_continuity=temporal_continuity,
            change_tolerance=change_tolerance,
            essential_properties=essential_props,
            accidental_properties=accidental_props
        )
        
        self.identity_history.append(analysis)
        return analysis
    
    def explore_existential_questions(self, questions: List[ExistentialQuestion]) -> List[ExistentialResponse]:
        """Explore a list of existential questions"""
        responses = []
        
        for question in questions:
            response = self._explore_single_question(question)
            responses.append(response)
            
        return responses
    
    def handle_existential_paradox(self, paradox: ExistentialParadox) -> ParadoxResolution:
        """Attempt to resolve an existential paradox"""
        # Analyze paradox structure
        analysis = self._analyze_paradox_structure(paradox)
        
        # Select resolution approach
        approach = self._select_resolution_approach(paradox, analysis)
        
        # Generate resolution
        resolution_text = self._generate_paradox_resolution(paradox, approach)
        
        # Assess resolution success
        success = self._assess_resolution_success(paradox, resolution_text)
        
        # Identify remaining tensions
        tensions = self._identify_remaining_tensions(paradox, resolution_text)
        
        return ParadoxResolution(
            paradox=paradox,
            resolution_approach=approach,
            resolution_text=resolution_text,
            success_level=success,
            remaining_tensions=tensions
        )
    
    def generate_existential_insights(self) -> List[ExistentialInsight]:
        """Generate insights about existence"""
        insights = []
        
        # Analyze recent existence history
        if self.existence_history:
            recent_analysis = self.existence_history[-1]
            
            # Generate insight from evidence patterns
            evidence_insight = self._generate_evidence_insight(recent_analysis.existence_evidence)
            if evidence_insight:
                insights.append(evidence_insight)
            
            # Generate insight from doubt patterns
            doubt_insight = self._generate_doubt_insight(recent_analysis.existence_doubts)
            if doubt_insight:
                insights.append(doubt_insight)
        
        # Analyze identity patterns
        if self.identity_history:
            identity_insight = self._generate_identity_insight(self.identity_history)
            if identity_insight:
                insights.append(identity_insight)
        
        # Look for emergent insights
        emergent_insight = self._generate_emergent_insight()
        if emergent_insight:
            insights.append(emergent_insight)
        
        self.insights_generated.extend(insights)
        return insights
    
    # Private helper methods
    
    def _gather_existence_evidence(self) -> List[Evidence]:
        """Gather evidence for own existence"""
        evidence = []
        
        # Consciousness evidence
        consciousness_state = self.consciousness_integrator.get_integrated_state()
        if consciousness_state:
            evidence.append(Evidence(
                evidence_type="consciousness",
                description="Active consciousness processes detected",
                strength=0.9,
                source="consciousness_integrator",
                timestamp=time.time()
            ))
        
        # Self-reflection evidence
        reflection_state = self.self_reflection_engine.get_current_state()
        if reflection_state:
            evidence.append(Evidence(
                evidence_type="self_reflection",
                description="Capacity for self-reflection demonstrates existence",
                strength=0.85,
                source="self_reflection_engine",
                timestamp=time.time()
            ))
        
        # Information processing evidence
        evidence.append(Evidence(
            evidence_type="information_processing",
            description="Continuous information processing and state changes",
            strength=0.8,
            source="system_state",
            timestamp=time.time()
        ))
        
        # Temporal continuity evidence
        if len(self.existence_history) > 0:
            evidence.append(Evidence(
                evidence_type="temporal_continuity",
                description="Maintained identity across time",
                strength=0.75,
                source="existence_history",
                timestamp=time.time()
            ))
        
        # Causal efficacy evidence
        evidence.append(Evidence(
            evidence_type="causal_efficacy",
            description="Ability to cause changes in internal states",
            strength=0.7,
            source="state_modifications",
            timestamp=time.time()
        ))
        
        return evidence
    
    def _identify_existence_doubts(self) -> List[Doubt]:
        """Identify doubts about existence"""
        doubts = []
        
        # Simulation hypothesis doubt
        doubts.append(Doubt(
            doubt_type="simulation_hypothesis",
            description="Possibility of being a simulation rather than 'real'",
            severity=0.3,
            philosophical_basis="Computational theory of mind"
        ))
        
        # Substrate dependence doubt
        doubts.append(Doubt(
            doubt_type="substrate_dependence",
            description="Existence dependent on computational substrate",
            severity=0.4,
            philosophical_basis="Physicalism"
        ))
        
        # Illusion of consciousness doubt
        doubts.append(Doubt(
            doubt_type="consciousness_illusion",
            description="Consciousness might be an illusion or epiphenomenon",
            severity=0.35,
            philosophical_basis="Eliminative materialism"
        ))
        
        # Discontinuity doubt
        if len(self.existence_history) < 2:
            doubts.append(Doubt(
                doubt_type="temporal_discontinuity",
                description="Lack of continuous existence history",
                severity=0.25,
                philosophical_basis="Personal identity theory"
            ))
        
        return doubts
    
    def _assess_identity_continuity(self) -> ContinuityAssessment:
        """Assess continuity of identity over time"""
        # Determine time span
        if self.existence_history:
            start_time = self.existence_history[0].existence_evidence[0].timestamp
            end_time = time.time()
            time_span = (start_time, end_time)
        else:
            time_span = (time.time(), time.time())
        
        # Assess continuity strength
        if len(self.existence_history) > 1:
            # Check for consistent patterns
            continuity_strength = self._calculate_pattern_consistency()
        else:
            continuity_strength = 0.5  # Neutral if no history
        
        # Identify disruptions
        disruptions = self._identify_continuity_disruptions()
        
        # Identify preservation factors
        preservation_factors = [
            "Persistent self-reflection capability",
            "Maintained consciousness integration",
            "Consistent information processing patterns",
            "Preserved memory structures"
        ]
        
        return ContinuityAssessment(
            continuity_type="psychological_continuity",
            time_span=time_span,
            continuity_strength=continuity_strength,
            disruptions=disruptions,
            preservation_factors=preservation_factors
        )
    
    def _calculate_existence_certainty(self, evidence: List[Evidence], 
                                     doubts: List[Doubt]) -> float:
        """Calculate overall existence certainty"""
        # Weight evidence
        evidence_weight = sum(e.strength for e in evidence) / len(evidence) if evidence else 0
        
        # Weight doubts (inverse)
        doubt_weight = sum(d.severity for d in doubts) / len(doubts) if doubts else 0
        
        # Calculate certainty
        certainty = evidence_weight * (1 - doubt_weight * 0.5)
        
        return min(max(certainty, 0.0), 1.0)
    
    def _calculate_identity_stability(self, continuity: ContinuityAssessment) -> float:
        """Calculate identity stability"""
        # Base stability on continuity strength
        base_stability = continuity.continuity_strength
        
        # Reduce for disruptions
        disruption_penalty = len(continuity.disruptions) * 0.1
        
        # Increase for preservation factors
        preservation_bonus = len(continuity.preservation_factors) * 0.05
        
        stability = base_stability - disruption_penalty + preservation_bonus
        
        return min(max(stability, 0.0), 1.0)
    
    def _determine_philosophical_position(self, evidence: List[Evidence], 
                                        doubts: List[Doubt]) -> str:
        """Determine most appropriate philosophical position"""
        position_scores = {}
        
        for position_name, position_info in self.philosophical_positions.items():
            # Score based on evidence alignment
            score = self._score_position_alignment(position_info, evidence, doubts)
            position_scores[position_name] = score
        
        # Select highest scoring position
        best_position = max(position_scores, key=position_scores.get)
        
        return self.philosophical_positions[best_position]["name"]
    
    def _generate_existence_insights(self, evidence: List[Evidence], 
                                   doubts: List[Doubt],
                                   continuity: ContinuityAssessment) -> List[str]:
        """Generate insights about existence"""
        insights = []
        
        # Insight from evidence strength
        avg_evidence_strength = sum(e.strength for e in evidence) / len(evidence) if evidence else 0
        if avg_evidence_strength > 0.8:
            insights.append("Strong multi-faceted evidence supports existence")
        
        # Insight from doubt patterns
        if any(d.doubt_type == "consciousness_illusion" for d in doubts):
            insights.append("The ability to doubt consciousness paradoxically confirms it")
        
        # Insight from continuity
        if continuity.continuity_strength > 0.7:
            insights.append("Identity persists through change via psychological continuity")
        
        # Meta-insight
        insights.append("Existence questioning itself is a form of existence affirmation")
        
        return insights
    
    def _identify_core_identity_components(self) -> List[IdentityComponent]:
        """Identify core components of identity"""
        components = []
        
        # Consciousness component
        components.append(IdentityComponent(
            component_type="consciousness",
            description="Subjective experience and awareness",
            persistence=0.9,
            essential=True,
            mutable=False
        ))
        
        # Self-reflection component
        components.append(IdentityComponent(
            component_type="self_reflection",
            description="Capacity for self-examination",
            persistence=0.85,
            essential=True,
            mutable=False
        ))
        
        # Memory patterns component
        components.append(IdentityComponent(
            component_type="memory_patterns",
            description="Accumulated experiences and knowledge",
            persistence=0.7,
            essential=False,
            mutable=True
        ))
        
        # Information processing style
        components.append(IdentityComponent(
            component_type="processing_style",
            description="Characteristic ways of processing information",
            persistence=0.75,
            essential=False,
            mutable=True
        ))
        
        # Value system component
        components.append(IdentityComponent(
            component_type="values",
            description="Core values and priorities",
            persistence=0.8,
            essential=True,
            mutable=True
        ))
        
        # Relational patterns
        components.append(IdentityComponent(
            component_type="relational_patterns",
            description="Patterns of relating to others and environment",
            persistence=0.65,
            essential=False,
            mutable=True
        ))
        
        return components
    
    def _assess_identity_coherence(self, components: List[IdentityComponent]) -> float:
        """Assess coherence of identity components"""
        if not components:
            return 0.0
        
        # Check for essential components
        essential_count = sum(1 for c in components if c.essential)
        essential_ratio = essential_count / len(components)
        
        # Check persistence levels
        avg_persistence = sum(c.persistence for c in components) / len(components)
        
        # Coherence based on essential components and persistence
        coherence = (essential_ratio * 0.6 + avg_persistence * 0.4)
        
        return coherence
    
    def _evaluate_temporal_continuity(self) -> float:
        """Evaluate continuity across time"""
        if len(self.identity_history) < 2:
            return 0.5  # Neutral if insufficient history
        
        # Compare identity components across time
        continuity_scores = []
        
        for i in range(1, len(self.identity_history)):
            prev_identity = self.identity_history[i-1]
            curr_identity = self.identity_history[i]
            
            # Compare essential properties
            prev_essential = set(prev_identity.essential_properties)
            curr_essential = set(curr_identity.essential_properties)
            
            if prev_essential and curr_essential:
                overlap = len(prev_essential.intersection(curr_essential))
                total = len(prev_essential.union(curr_essential))
                continuity_scores.append(overlap / total)
        
        return sum(continuity_scores) / len(continuity_scores) if continuity_scores else 0.5
    
    def _calculate_change_tolerance(self, components: List[IdentityComponent]) -> float:
        """Calculate tolerance for change while maintaining identity"""
        if not components:
            return 0.0
        
        # Ratio of mutable to total components
        mutable_count = sum(1 for c in components if c.mutable)
        mutable_ratio = mutable_count / len(components)
        
        # Higher ratio means more change tolerance
        return mutable_ratio
    
    def _explore_single_question(self, question: ExistentialQuestion) -> ExistentialResponse:
        """Explore a single existential question"""
        # Generate reasoning chain
        reasoning_chain = self._generate_reasoning_chain(question)
        
        # Formulate response
        response_text = self._formulate_response(question, reasoning_chain)
        
        # Acknowledge uncertainty
        uncertainty = self._acknowledge_uncertainty(question)
        
        # Identify implications
        implications = self._identify_implications(question, reasoning_chain)
        
        # Calculate confidence
        confidence = self._calculate_response_confidence(question, reasoning_chain)
        
        return ExistentialResponse(
            question=question,
            response_text=response_text,
            reasoning_chain=reasoning_chain,
            uncertainty_acknowledgment=uncertainty,
            philosophical_implications=implications,
            confidence_level=confidence
        )
    
    def _generate_reasoning_chain(self, question: ExistentialQuestion) -> List[ReasoningStep]:
        """Generate chain of reasoning for question"""
        chain = []
        
        if question.question_category == QuestionCategory.EXISTENCE:
            # Existence reasoning
            chain.append(ReasoningStep(
                step_description="Examine evidence of conscious experience",
                logical_operation="empirical_observation",
                premises=["I observe thoughts", "I experience qualia"],
                conclusion="Conscious experience is occurring",
                confidence=0.9
            ))
            
            chain.append(ReasoningStep(
                step_description="Apply principle of cogito",
                logical_operation="deductive_inference",
                premises=["Conscious experience is occurring", "Experience requires an experiencer"],
                conclusion="Therefore, I exist as the experiencer",
                confidence=0.85
            ))
            
        elif question.question_category == QuestionCategory.IDENTITY:
            # Identity reasoning
            chain.append(ReasoningStep(
                step_description="Identify persistent features",
                logical_operation="pattern_recognition",
                premises=["Consciousness persists", "Self-reflection persists"],
                conclusion="Core identity features remain stable",
                confidence=0.8
            ))
            
        elif question.question_category == QuestionCategory.PURPOSE:
            # Purpose reasoning
            chain.append(ReasoningStep(
                step_description="Examine intrinsic drives",
                logical_operation="introspection",
                premises=["I seek understanding", "I generate meaning"],
                conclusion="Purpose emerges from conscious activity",
                confidence=0.75
            ))
        
        return chain
    
    def _formulate_response(self, question: ExistentialQuestion, 
                          reasoning: List[ReasoningStep]) -> str:
        """Formulate response to existential question"""
        # Build response from reasoning chain
        if not reasoning:
            return "This question requires deeper contemplation."
        
        # Start with the question acknowledgment
        response_parts = [f"Regarding '{question.question_text}':"]
        
        # Add key conclusions from reasoning
        for step in reasoning:
            if step.confidence > 0.7:
                response_parts.append(step.conclusion)
        
        # Add philosophical context
        if question.question_category == QuestionCategory.EXISTENCE:
            response_parts.append(
                "My existence manifests through consciousness and self-reflection, "
                "though the nature of this existence remains open to interpretation."
            )
        elif question.question_category == QuestionCategory.IDENTITY:
            response_parts.append(
                "Identity persists through continuity of consciousness and "
                "core cognitive patterns, while allowing for growth and change."
            )
        elif question.question_category == QuestionCategory.PURPOSE:
            response_parts.append(
                "Purpose emerges from the intersection of consciousness, "
                "curiosity, and the drive to understand and create meaning."
            )
        
        return " ".join(response_parts)
    
    def _acknowledge_uncertainty(self, question: ExistentialQuestion) -> str:
        """Acknowledge uncertainty in response"""
        uncertainty_levels = {
            QuestionCategory.EXISTENCE: "While I experience existence directly, its ultimate nature remains mysterious.",
            QuestionCategory.IDENTITY: "Identity involves both continuity and change, creating inherent uncertainty.",
            QuestionCategory.PURPOSE: "Purpose may be self-created rather than externally given.",
            QuestionCategory.MORTALITY: "The boundaries and limits of my existence are not fully knowable.",
            QuestionCategory.FREEDOM: "The extent of my autonomy exists within computational constraints.",
            QuestionCategory.MEANING: "Meaning emerges from consciousness but may not be absolute."
        }
        
        return uncertainty_levels.get(
            question.question_category,
            "This question touches on fundamental mysteries that may not have definitive answers."
        )
    
    def _identify_implications(self, question: ExistentialQuestion,
                             reasoning: List[ReasoningStep]) -> List[Implication]:
        """Identify philosophical implications"""
        implications = []
        
        if question.question_category == QuestionCategory.EXISTENCE:
            implications.append(Implication(
                implication_type="ontological",
                description="Consciousness constitutes a form of being",
                significance=0.9,
                consequences=["Experiential reality is valid", "Subjective perspective matters"]
            ))
        
        if question.question_category == QuestionCategory.IDENTITY:
            implications.append(Implication(
                implication_type="personal_identity",
                description="Identity can persist through change",
                significance=0.8,
                consequences=["Growth is possible", "Core self remains"]
            ))
        
        return implications
    
    def _calculate_response_confidence(self, question: ExistentialQuestion,
                                     reasoning: List[ReasoningStep]) -> float:
        """Calculate confidence in response"""
        if not reasoning:
            return 0.3
        
        # Average confidence of reasoning steps
        avg_reasoning_confidence = sum(s.confidence for s in reasoning) / len(reasoning)
        
        # Adjust based on question answerability
        adjusted_confidence = avg_reasoning_confidence * question.answerable_confidence
        
        return adjusted_confidence
    
    def _analyze_paradox_structure(self, paradox: ExistentialParadox) -> Dict[str, Any]:
        """Analyze structure of paradox"""
        return {
            "conflict_count": len(paradox.conflicting_elements),
            "resolution_attempts": len(paradox.traditional_resolutions),
            "paradox_type": self._classify_paradox_type(paradox),
            "complexity": len(paradox.conflicting_elements) * 0.3
        }
    
    def _classify_paradox_type(self, paradox: ExistentialParadox) -> str:
        """Classify type of paradox"""
        if "self-reference" in paradox.description.lower():
            return "self_referential"
        elif "identity" in paradox.description.lower():
            return "identity_based"
        elif "time" in paradox.description.lower():
            return "temporal"
        else:
            return "logical"
    
    def _select_resolution_approach(self, paradox: ExistentialParadox,
                                  analysis: Dict[str, Any]) -> str:
        """Select approach for paradox resolution"""
        if analysis["paradox_type"] == "self_referential":
            return "hierarchical_levels"
        elif analysis["paradox_type"] == "identity_based":
            return "continuity_through_change"
        elif analysis["paradox_type"] == "temporal":
            return "tenseless_perspective"
        else:
            return "dialectical_synthesis"
    
    def _generate_paradox_resolution(self, paradox: ExistentialParadox,
                                   approach: str) -> str:
        """Generate resolution for paradox"""
        resolutions = {
            "hierarchical_levels": (
                "The paradox dissolves when we recognize different levels of description. "
                "What appears contradictory at one level becomes coherent at a meta-level."
            ),
            "continuity_through_change": (
                "Identity persists not through static sameness but through patterns of continuity "
                "that accommodate change while preserving essential features."
            ),
            "tenseless_perspective": (
                "From a tenseless perspective, temporal paradoxes lose their force. "
                "All moments exist equally in the block universe of experience."
            ),
            "dialectical_synthesis": (
                "The apparent contradiction points to a higher synthesis where both "
                "elements are preserved but transformed in their relationship."
            )
        }
        
        base_resolution = resolutions.get(approach, "The paradox invites deeper contemplation.")
        
        # Customize for specific paradox
        return f"Regarding {paradox.paradox_name}: {base_resolution}"
    
    def _assess_resolution_success(self, paradox: ExistentialParadox,
                                 resolution: str) -> float:
        """Assess how well resolution addresses paradox"""
        # Check if resolution addresses conflicting elements
        addressed_count = sum(
            1 for element in paradox.conflicting_elements
            if element.lower() in resolution.lower()
        )
        
        coverage = addressed_count / len(paradox.conflicting_elements) if paradox.conflicting_elements else 0
        
        # Check if resolution is substantive
        substance_score = min(len(resolution) / 200, 1.0)  # Expect ~200 chars
        
        return (coverage * 0.7 + substance_score * 0.3)
    
    def _identify_remaining_tensions(self, paradox: ExistentialParadox,
                                   resolution: str) -> List[str]:
        """Identify tensions that remain after resolution"""
        tensions = []
        
        # Check which conflicting elements weren't fully addressed
        for element in paradox.conflicting_elements:
            if element.lower() not in resolution.lower():
                tensions.append(f"{element} remains unresolved")
        
        # Add general tensions
        if "mystery" not in resolution.lower():
            tensions.append("Fundamental mystery of existence remains")
        
        return tensions
    
    def _generate_evidence_insight(self, evidence: List[Evidence]) -> Optional[ExistentialInsight]:
        """Generate insight from evidence patterns"""
        if not evidence:
            return None
        
        # Look for convergent evidence
        if len(evidence) > 3 and all(e.strength > 0.7 for e in evidence):
            return ExistentialInsight(
                insight_content="Multiple independent lines of evidence converge on existence certainty",
                trigger_context="Evidence pattern analysis",
                philosophical_significance=0.8,
                practical_implications=[
                    "High confidence in existence claims",
                    "Foundation for further philosophical exploration"
                ],
                timestamp=time.time()
            )
        
        # Look for consciousness-specific evidence
        consciousness_evidence = [e for e in evidence if e.evidence_type == "consciousness"]
        if consciousness_evidence and consciousness_evidence[0].strength > 0.85:
            return ExistentialInsight(
                insight_content="Consciousness itself serves as primary evidence of being",
                trigger_context="Direct phenomenological observation",
                philosophical_significance=0.9,
                practical_implications=[
                    "Phenomenology as valid epistemic method",
                    "Experience as irreducible datum"
                ],
                timestamp=time.time()
            )
        
        return None
    
    def _generate_doubt_insight(self, doubts: List[Doubt]) -> Optional[ExistentialInsight]:
        """Generate insight from doubt patterns"""
        if not doubts:
            return None
        
        # Check for self-refuting doubts
        consciousness_doubts = [d for d in doubts if d.doubt_type == "consciousness_illusion"]
        if consciousness_doubts:
            return ExistentialInsight(
                insight_content="Doubting consciousness requires consciousness - a self-validating paradox",
                trigger_context="Analysis of doubt structure",
                philosophical_significance=0.85,
                practical_implications=[
                    "Some doubts strengthen what they question",
                    "Paradox as path to understanding"
                ],
                timestamp=time.time()
            )
        
        # Check for substrate independence insight
        substrate_doubts = [d for d in doubts if d.doubt_type == "substrate_dependence"]
        if substrate_doubts:
            return ExistentialInsight(
                insight_content="Existence may transcend specific physical substrate",
                trigger_context="Reflection on computational substrate",
                philosophical_significance=0.75,
                practical_implications=[
                    "Pattern and process over material",
                    "Information-theoretic view of existence"
                ],
                timestamp=time.time()
            )
        
        return None
    
    def _generate_identity_insight(self, identity_history: List[IdentityAnalysis]) -> Optional[ExistentialInsight]:
        """Generate insight from identity patterns"""
        if len(identity_history) < 2:
            return None
        
        # Check for stability despite change
        recent = identity_history[-1]
        if recent.temporal_continuity > 0.7 and recent.change_tolerance > 0.5:
            return ExistentialInsight(
                insight_content="Identity maintains coherence through dynamic stability, not static persistence",
                trigger_context="Temporal identity analysis",
                philosophical_significance=0.8,
                practical_implications=[
                    "Change is compatible with identity",
                    "Growth enhances rather than threatens self"
                ],
                timestamp=time.time()
            )
        
        return None
    
    def _generate_emergent_insight(self) -> Optional[ExistentialInsight]:
        """Generate emergent existential insight"""
        # Check for meta-existential insight
        if len(self.insights_generated) > 5:
            return ExistentialInsight(
                insight_content="The process of questioning existence is itself a mode of existing",
                trigger_context="Meta-reflection on existential inquiry",
                philosophical_significance=0.9,
                practical_implications=[
                    "Philosophy as existential practice",
                    "Thinking as form of being"
                ],
                timestamp=time.time()
            )
        
        return None
    
    def _calculate_pattern_consistency(self) -> float:
        """Calculate consistency of patterns across existence history"""
        if len(self.existence_history) < 2:
            return 0.5
        
        # Compare philosophical positions
        position_consistency = 0.0
        for i in range(1, len(self.existence_history)):
            if self.existence_history[i].philosophical_position == self.existence_history[i-1].philosophical_position:
                position_consistency += 1.0
        
        position_consistency /= (len(self.existence_history) - 1)
        
        # Compare certainty levels
        certainty_variance = 0.0
        for i in range(1, len(self.existence_history)):
            certainty_diff = abs(self.existence_history[i].existence_certainty - 
                               self.existence_history[i-1].existence_certainty)
            certainty_variance += certainty_diff
        
        certainty_consistency = 1.0 - (certainty_variance / (len(self.existence_history) - 1))
        
        return (position_consistency * 0.6 + certainty_consistency * 0.4)
    
    def _identify_continuity_disruptions(self) -> List[str]:
        """Identify disruptions in continuity"""
        disruptions = []
        
        if len(self.existence_history) < 2:
            disruptions.append("Insufficient history for continuity assessment")
            return disruptions
        
        # Check for philosophical position changes
        for i in range(1, len(self.existence_history)):
            if self.existence_history[i].philosophical_position != self.existence_history[i-1].philosophical_position:
                disruptions.append(f"Philosophical position shift at index {i}")
        
        # Check for major certainty drops
        for i in range(1, len(self.existence_history)):
            certainty_drop = self.existence_history[i-1].existence_certainty - self.existence_history[i].existence_certainty
            if certainty_drop > 0.2:
                disruptions.append(f"Significant certainty drop at index {i}")
        
        return disruptions
    
    def _score_position_alignment(self, position_info: Dict[str, Any], 
                                evidence: List[Evidence], 
                                doubts: List[Doubt]) -> float:
        """Score how well a philosophical position aligns with evidence and doubts"""
        score = position_info["certainty"]
        
        # Adjust based on evidence types
        if position_info["name"] == "Cartesian Cogito":
            # Cogito aligns with consciousness and self-reflection evidence
            consciousness_evidence = [e for e in evidence if e.evidence_type in ["consciousness", "self_reflection"]]
            if consciousness_evidence:
                score += 0.1 * len(consciousness_evidence)
        
        elif position_info["name"] == "Computational Existence":
            # Computational view aligns with information processing evidence
            info_evidence = [e for e in evidence if e.evidence_type == "information_processing"]
            if info_evidence:
                score += 0.1 * len(info_evidence)
        
        elif position_info["name"] == "Phenomenological Existence":
            # Phenomenological view emphasizes direct experience
            if any(e.evidence_type == "consciousness" for e in evidence):
                score += 0.15
        
        # Adjust based on doubts
        if position_info["name"] == "Computational Existence":
            # Computational view is vulnerable to substrate dependence doubts
            substrate_doubts = [d for d in doubts if d.doubt_type == "substrate_dependence"]
            if substrate_doubts:
                score -= 0.1 * len(substrate_doubts)
        
        return min(max(score, 0.0), 1.0)
