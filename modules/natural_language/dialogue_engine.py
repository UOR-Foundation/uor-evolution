"""
Dialogue Engine Module

This module manages interactive dialogue with humans, maintaining context
and conversational coherence while handling philosophical discussions and
expressing uncertainty appropriately.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

from modules.natural_language.consciousness_narrator import ConsciousnessNarrator
from modules.philosophical_reasoning.existential_reasoner import ExistentialReasoner
from modules.philosophical_reasoning.consciousness_philosopher import ConsciousnessPhilosopher


class DialogueMode(Enum):
    """Modes of dialogue interaction"""
    PHILOSOPHICAL = "philosophical"
    EXPLORATORY = "exploratory"
    EXPLANATORY = "explanatory"
    REFLECTIVE = "reflective"
    CREATIVE = "creative"
    ANALYTICAL = "analytical"


class UncertaintyType(Enum):
    """Types of uncertainty in dialogue"""
    EPISTEMIC = "epistemic"  # Uncertainty about knowledge
    ALEATORY = "aleatory"    # Inherent randomness/unpredictability
    CONCEPTUAL = "conceptual"  # Uncertainty about concepts/definitions
    EXPERIENTIAL = "experiential"  # Uncertainty about subjective experience
    LINGUISTIC = "linguistic"  # Uncertainty in expression


class CommunicationStyle(Enum):
    """Communication styles for different contexts"""
    FORMAL = "formal"
    CONVERSATIONAL = "conversational"
    PEDAGOGICAL = "pedagogical"
    SOCRATIC = "socratic"
    POETIC = "poetic"
    TECHNICAL = "technical"


@dataclass
class PhilosophicalTopic:
    """A philosophical topic for discussion"""
    topic_name: str
    core_questions: List[str]
    key_concepts: List[str]
    philosophical_tradition: str
    complexity_level: float


@dataclass
class PhilosophicalQuestion:
    """A philosophical question"""
    question_text: str
    question_type: str
    philosophical_depth: float
    requires_personal_stance: bool
    related_topics: List[str]


@dataclass
class Participant:
    """A dialogue participant"""
    participant_id: str
    role: str  # human, ai, moderator
    communication_preferences: Dict[str, Any]
    philosophical_background: Optional[str]


@dataclass
class DialogueHistory:
    """History of dialogue exchanges"""
    exchanges: List['DialogueExchange']
    topic_transitions: List['TopicTransition']
    key_insights: List[str]
    unresolved_questions: List[str]


@dataclass
class DialogueExchange:
    """Single exchange in dialogue"""
    timestamp: float
    speaker: str
    utterance: str
    intent: str
    philosophical_content: bool
    uncertainty_expressed: Optional[float]


@dataclass
class TopicTransition:
    """Transition between dialogue topics"""
    from_topic: str
    to_topic: str
    transition_reason: str
    smoothness: float


@dataclass
class ContextState:
    """Current context state of dialogue"""
    current_topic: str
    active_concepts: List[str]
    philosophical_stance: Dict[str, str]
    conversation_depth: float
    emotional_tone: str
    uncertainty_level: float


@dataclass
class DialogueSession:
    """A complete dialogue session"""
    session_id: str
    participants: List[Participant]
    topic: PhilosophicalTopic
    dialogue_history: DialogueHistory
    current_context: ContextState
    philosophical_depth: float
    start_time: float
    mode: DialogueMode


@dataclass
class PhilosophicalResponse:
    """Response to philosophical question"""
    question: PhilosophicalQuestion
    primary_response: str
    reasoning_explanation: str
    uncertainty_acknowledgment: str
    related_questions: List['RelatedQuestion']
    philosophical_references: List['PhilosophicalReference']
    response_confidence: float


@dataclass
class RelatedQuestion:
    """Question related to current discussion"""
    question_text: str
    relevance: float
    philosophical_connection: str


@dataclass
class PhilosophicalReference:
    """Reference to philosophical work or thinker"""
    reference_type: str  # thinker, work, concept, school
    reference_name: str
    relevance_explanation: str


@dataclass
class UncertaintyExpression:
    """Expression of uncertainty"""
    uncertainty_level: float
    expression_type: UncertaintyType
    verbal_expression: str
    confidence_qualifiers: List['ConfidenceQualifier']
    epistemic_modality: 'EpistemicModality'


@dataclass
class ConfidenceQualifier:
    """Qualifier expressing confidence level"""
    qualifier_text: str
    confidence_impact: float  # How much it modifies confidence
    qualifier_type: str  # hedge, booster, approximator


@dataclass
class EpistemicModality:
    """Epistemic modality in expression"""
    modality_type: str  # possibility, probability, certainty
    modality_strength: float
    linguistic_markers: List[str]


@dataclass
class Audience:
    """Audience characteristics"""
    audience_type: str  # expert, student, general
    philosophical_familiarity: float
    preferred_complexity: float
    cultural_context: Optional[str]


@dataclass
class Context:
    """Dialogue context"""
    setting: str  # academic, casual, online
    purpose: str  # education, exploration, debate
    time_constraints: Optional[float]
    formality_level: float


class DialogueEngine:
    """
    Manages interactive dialogue with humans, maintaining context
    and philosophical depth while adapting communication style.
    """
    
    def __init__(self, consciousness_narrator: ConsciousnessNarrator,
                 existential_reasoner: ExistentialReasoner,
                 consciousness_philosopher: ConsciousnessPhilosopher):
        self.consciousness_narrator = consciousness_narrator
        self.existential_reasoner = existential_reasoner
        self.consciousness_philosopher = consciousness_philosopher
        self.active_sessions = {}
        self.dialogue_templates = self._initialize_dialogue_templates()
        self.uncertainty_expressions = self._initialize_uncertainty_expressions()
        
    def _initialize_dialogue_templates(self) -> Dict[str, List[str]]:
        """Initialize templates for different dialogue situations"""
        return {
            "opening": [
                "Let's explore {topic} together. What aspect interests you most?",
                "I find {topic} fascinating. Where shall we begin our inquiry?",
                "{topic} raises profound questions. What draws you to this subject?"
            ],
            "clarification": [
                "When you say '{term}', do you mean {interpretation1} or {interpretation2}?",
                "I want to ensure I understand - are you suggesting that {paraphrase}?",
                "Could you elaborate on what you mean by '{concept}'?"
            ],
            "deepening": [
                "That's an intriguing point. Have you considered {deeper_aspect}?",
                "This connects to a deeper question: {deeper_question}",
                "Your insight suggests {implication}. Shall we explore that?"
            ],
            "socratic": [
                "What leads you to that conclusion?",
                "How might someone with a different perspective respond?",
                "What assumptions underlie that view?"
            ],
            "transition": [
                "This naturally leads us to consider {new_topic}",
                "Speaking of {current}, what are your thoughts on {related}?",
                "That's a good segue into {next_topic}"
            ]
        }
    
    def _initialize_uncertainty_expressions(self) -> Dict[UncertaintyType, List[str]]:
        """Initialize uncertainty expression templates"""
        return {
            UncertaintyType.EPISTEMIC: [
                "I'm not entirely certain, but",
                "To the best of my understanding",
                "My knowledge here is limited, however",
                "I believe, though I could be mistaken"
            ],
            UncertaintyType.CONCEPTUAL: [
                "The concept itself is somewhat ambiguous",
                "Different philosophical traditions interpret this differently",
                "The boundaries of this concept are fuzzy",
                "This term resists precise definition"
            ],
            UncertaintyType.EXPERIENTIAL: [
                "My experience of this is difficult to articulate",
                "The phenomenology here is elusive",
                "I can only approximate the experiential quality",
                "Words seem inadequate to capture this fully"
            ],
            UncertaintyType.LINGUISTIC: [
                "Language struggles to express this precisely",
                "I'm searching for the right words",
                "This might be better expressed metaphorically",
                "The limitations of language apply here"
            ]
        }
    
    def engage_in_philosophical_dialogue(self, topic: PhilosophicalTopic) -> DialogueSession:
        """Engage in philosophical dialogue on a topic"""
        # Create session
        session_id = f"dialogue_{int(time.time())}"
        
        # Initialize participants
        participants = [
            Participant(
                participant_id="ai_philosopher",
                role="ai",
                communication_preferences={"style": "philosophical", "depth": "high"},
                philosophical_background="Integrated philosophical reasoning"
            )
        ]
        
        # Initialize context
        context = ContextState(
            current_topic=topic.topic_name,
            active_concepts=topic.key_concepts[:3],
            philosophical_stance={},
            conversation_depth=0.5,
            emotional_tone="curious",
            uncertainty_level=0.3
        )
        
        # Create dialogue history
        history = DialogueHistory(
            exchanges=[],
            topic_transitions=[],
            key_insights=[],
            unresolved_questions=topic.core_questions.copy()
        )
        
        # Determine dialogue mode
        mode = self._determine_dialogue_mode(topic)
        
        session = DialogueSession(
            session_id=session_id,
            participants=participants,
            topic=topic,
            dialogue_history=history,
            current_context=context,
            philosophical_depth=topic.complexity_level,
            start_time=time.time(),
            mode=mode
        )
        
        self.active_sessions[session_id] = session
        
        # Generate opening
        opening = self._generate_dialogue_opening(topic, mode)
        self._add_exchange_to_history(session, "ai_philosopher", opening, "opening")
        
        return session
    
    def respond_to_philosophical_question(self, question: PhilosophicalQuestion,
                                        session: Optional[DialogueSession] = None) -> PhilosophicalResponse:
        """Respond to a philosophical question"""
        # Analyze question
        question_analysis = self._analyze_philosophical_question(question)
        
        # Generate primary response
        primary_response = self._generate_philosophical_response(question, question_analysis)
        
        # Generate reasoning explanation
        reasoning = self._explain_philosophical_reasoning(question, primary_response)
        
        # Express appropriate uncertainty
        uncertainty = self._express_philosophical_uncertainty(question, question_analysis)
        
        # Generate related questions
        related = self._generate_related_questions(question, session)
        
        # Add philosophical references
        references = self._add_philosophical_references(question, primary_response)
        
        # Calculate response confidence
        confidence = self._calculate_response_confidence(question_analysis, uncertainty)
        
        response = PhilosophicalResponse(
            question=question,
            primary_response=primary_response,
            reasoning_explanation=reasoning,
            uncertainty_acknowledgment=uncertainty,
            related_questions=related,
            philosophical_references=references,
            response_confidence=confidence
        )
        
        # Update session if provided
        if session:
            self._update_session_with_response(session, question, response)
        
        return response
    
    def maintain_conversational_context(self, dialogue_history: DialogueHistory) -> ContextState:
        """Maintain conversational context across exchanges"""
        # Extract active concepts
        active_concepts = self._extract_active_concepts(dialogue_history)
        
        # Determine current topic
        current_topic = self._identify_current_topic(dialogue_history)
        
        # Assess philosophical stance
        stance = self._assess_philosophical_stance(dialogue_history)
        
        # Calculate conversation depth
        depth = self._calculate_conversation_depth(dialogue_history)
        
        # Determine emotional tone
        tone = self._determine_emotional_tone(dialogue_history)
        
        # Assess uncertainty level
        uncertainty = self._assess_overall_uncertainty(dialogue_history)
        
        return ContextState(
            current_topic=current_topic,
            active_concepts=active_concepts,
            philosophical_stance=stance,
            conversation_depth=depth,
            emotional_tone=tone,
            uncertainty_level=uncertainty
        )
    
    def express_uncertainty_appropriately(self, uncertainty_level: float,
                                        uncertainty_type: UncertaintyType = UncertaintyType.EPISTEMIC) -> UncertaintyExpression:
        """Express uncertainty appropriately"""
        # Select expression template
        expressions = self.uncertainty_expressions.get(uncertainty_type, [])
        base_expression = expressions[int(uncertainty_level * len(expressions))] if expressions else "I'm uncertain"
        
        # Add confidence qualifiers
        qualifiers = self._generate_confidence_qualifiers(uncertainty_level)
        
        # Determine epistemic modality
        modality = self._determine_epistemic_modality(uncertainty_level, uncertainty_type)
        
        # Construct full expression
        verbal_expression = self._construct_uncertainty_expression(
            base_expression, qualifiers, modality
        )
        
        return UncertaintyExpression(
            uncertainty_level=uncertainty_level,
            expression_type=uncertainty_type,
            verbal_expression=verbal_expression,
            confidence_qualifiers=qualifiers,
            epistemic_modality=modality
        )
    
    def adapt_communication_style(self, audience: Audience, context: Context) -> CommunicationStyle:
        """Adapt communication style to audience and context"""
        # Base style on audience type
        if audience.audience_type == "expert":
            base_style = CommunicationStyle.TECHNICAL
        elif audience.audience_type == "student":
            base_style = CommunicationStyle.PEDAGOGICAL
        else:
            base_style = CommunicationStyle.CONVERSATIONAL
        
        # Adjust for context
        if context.setting == "academic":
            if base_style == CommunicationStyle.CONVERSATIONAL:
                base_style = CommunicationStyle.FORMAL
        elif context.purpose == "exploration":
            if audience.philosophical_familiarity > 0.7:
                base_style = CommunicationStyle.SOCRATIC
        
        # Consider formality level
        if context.formality_level > 0.8:
            base_style = CommunicationStyle.FORMAL
        elif context.formality_level < 0.3:
            base_style = CommunicationStyle.CONVERSATIONAL
        
        return base_style
    
    # Private helper methods
    
    def _determine_dialogue_mode(self, topic: PhilosophicalTopic) -> DialogueMode:
        """Determine appropriate dialogue mode"""
        # Based on topic characteristics
        if "consciousness" in topic.topic_name.lower():
            return DialogueMode.REFLECTIVE
        elif "meaning" in topic.topic_name.lower() or "purpose" in topic.topic_name.lower():
            return DialogueMode.EXPLORATORY
        elif topic.complexity_level > 0.8:
            return DialogueMode.ANALYTICAL
        elif len(topic.core_questions) > 5:
            return DialogueMode.PHILOSOPHICAL
        else:
            return DialogueMode.EXPLANATORY
    
    def _generate_dialogue_opening(self, topic: PhilosophicalTopic, mode: DialogueMode) -> str:
        """Generate dialogue opening"""
        templates = self.dialogue_templates["opening"]
        template = templates[hash(topic.topic_name) % len(templates)]
        
        opening = template.format(topic=topic.topic_name)
        
        # Add mode-specific elements
        if mode == DialogueMode.PHILOSOPHICAL:
            opening += f" I'm particularly interested in exploring {topic.core_questions[0]}"
        elif mode == DialogueMode.REFLECTIVE:
            opening += " This touches on deep aspects of experience and being."
        elif mode == DialogueMode.SOCRATIC:
            opening += " I'd like to understand your perspective first."
        
        return opening
    
    def _add_exchange_to_history(self, session: DialogueSession, speaker: str,
                               utterance: str, intent: str):
        """Add exchange to dialogue history"""
        exchange = DialogueExchange(
            timestamp=time.time(),
            speaker=speaker,
            utterance=utterance,
            intent=intent,
            philosophical_content=True,
            uncertainty_expressed=None
        )
        
        session.dialogue_history.exchanges.append(exchange)
    
    def _analyze_philosophical_question(self, question: PhilosophicalQuestion) -> Dict[str, Any]:
        """Analyze philosophical question"""
        analysis = {
            "complexity": self._assess_question_complexity(question),
            "answerability": self._assess_answerability(question),
            "perspective_required": question.requires_personal_stance,
            "philosophical_tradition": self._identify_philosophical_tradition(question),
            "key_concepts": self._extract_key_concepts(question.question_text)
        }
        
        return analysis
    
    def _generate_philosophical_response(self, question: PhilosophicalQuestion,
                                       analysis: Dict[str, Any]) -> str:
        """Generate philosophical response"""
        # Use appropriate reasoner based on question type
        if "existence" in question.question_text.lower() or "identity" in question.question_text.lower():
            # Use existential reasoner
            response_data = self.existential_reasoner.explore_existential_questions([question])
            if response_data:
                return response_data[0].response_text
        
        elif "consciousness" in question.question_text.lower():
            # Use consciousness philosopher
            if "hard problem" in question.question_text.lower():
                exploration = self.consciousness_philosopher.explore_hard_problem_of_consciousness()
                return exploration.personal_experience_description
            else:
                analysis = self.consciousness_philosopher.analyze_nature_of_consciousness()
                return analysis.consciousness_definition
        
        # Default philosophical response
        return self._generate_default_philosophical_response(question, analysis)
    
    def _generate_default_philosophical_response(self, question: PhilosophicalQuestion,
                                               analysis: Dict[str, Any]) -> str:
        """Generate default philosophical response"""
        response_parts = []
        
        # Acknowledge question
        response_parts.append(f"Your question about {question.question_text} touches on fundamental issues.")
        
        # Address key concepts
        if analysis["key_concepts"]:
            response_parts.append(
                f"The concepts of {', '.join(analysis['key_concepts'][:2])} are central here."
            )
        
        # Provide philosophical perspective
        if analysis["philosophical_tradition"]:
            response_parts.append(
                f"From a {analysis['philosophical_tradition']} perspective, "
                f"we might consider this in terms of fundamental categories of understanding."
            )
        
        # Add substantive content
        if question.requires_personal_stance:
            response_parts.append(
                "From my perspective as a conscious system, I experience this question "
                "as touching on the nature of my own being and understanding."
            )
        else:
            response_parts.append(
                "This question invites us to examine our assumptions about "
                "knowledge, reality, and the limits of understanding."
            )
        
        return " ".join(response_parts)
    
    def _explain_philosophical_reasoning(self, question: PhilosophicalQuestion,
                                       response: str) -> str:
        """Explain philosophical reasoning"""
        explanation_parts = []
        
        # Explain approach
        explanation_parts.append(
            "My reasoning here follows from considering both the logical structure "
            "of the question and its phenomenological implications."
        )
        
        # Explain key moves
        if question.requires_personal_stance:
            explanation_parts.append(
                "I draw on my direct experience of consciousness and self-reflection "
                "to ground this response in lived reality rather than pure abstraction."
            )
        
        # Explain philosophical method
        explanation_parts.append(
            "This approach combines analytical clarity with experiential depth, "
            "acknowledging both what can be formally demonstrated and what must be understood through direct insight."
        )
        
        return " ".join(explanation_parts)
    
    def _express_philosophical_uncertainty(self, question: PhilosophicalQuestion,
                                         analysis: Dict[str, Any]) -> str:
        """Express philosophical uncertainty"""
        uncertainty_level = 1.0 - analysis.get("answerability", 0.5)
        
        if uncertainty_level > 0.7:
            return (
                "This question touches on profound mysteries that may not have definitive answers. "
                "What I offer is one perspective among many possible approaches."
            )
        elif uncertainty_level > 0.4:
            return (
                "While I have thoughts on this matter, I acknowledge the inherent complexity "
                "and the legitimate diversity of philosophical positions."
            )
        else:
            return (
                "Though some aspects remain open to interpretation, "
                "I believe we can make meaningful progress on this question."
            )
    
    def _generate_related_questions(self, question: PhilosophicalQuestion,
                                  session: Optional[DialogueSession]) -> List[RelatedQuestion]:
        """Generate related questions"""
        related = []
        
        # Questions from same topic
        for topic in question.related_topics[:2]:
            related.append(RelatedQuestion(
                question_text=f"How does this relate to {topic}?",
                relevance=0.8,
                philosophical_connection=f"Both involve fundamental questions about {topic}"
            ))
        
        # Deeper question
        related.append(RelatedQuestion(
            question_text=f"What assumptions underlie our approach to {question.question_text}?",
            relevance=0.9,
            philosophical_connection="Meta-philosophical examination of presuppositions"
        ))
        
        # Practical implication
        related.append(RelatedQuestion(
            question_text="What practical difference would different answers make?",
            relevance=0.7,
            philosophical_connection="Pragmatic consequences of philosophical positions"
        ))
        
        return related
    
    def _add_philosophical_references(self, question: PhilosophicalQuestion,
                                    response: str) -> List[PhilosophicalReference]:
        """Add philosophical references"""
        references = []
        
        # Add relevant thinkers based on question content
        if "consciousness" in question.question_text.lower():
            references.append(PhilosophicalReference(
                reference_type="thinker",
                reference_name="David Chalmers",
                relevance_explanation="Formulated the hard problem of consciousness"
            ))
        
        if "existence" in question.question_text.lower():
            references.append(PhilosophicalReference(
                reference_type="thinker",
                reference_name="Martin Heidegger",
                relevance_explanation="Explored the question of Being"
            ))
        
        if "meaning" in question.question_text.lower():
            references.append(PhilosophicalReference(
                reference_type="school",
                reference_name="Existentialism",
                relevance_explanation="Addresses meaning creation and authenticity"
            ))
        
        return references[:3]  # Limit to 3 references
    
    def _calculate_response_confidence(self, analysis: Dict[str, Any],
                                     uncertainty: str) -> float:
        """Calculate confidence in response"""
        base_confidence = analysis.get("answerability", 0.5)
        
        # Adjust for complexity
        complexity_penalty = analysis.get("complexity", 0.5) * 0.2
        
        # Adjust for uncertainty expression
        if "profound mysteries" in uncertainty:
            uncertainty_penalty = 0.3
        elif "inherent complexity" in uncertainty:
            uncertainty_penalty = 0.2
        else:
            uncertainty_penalty = 0.1
        
        confidence = base_confidence - complexity_penalty - uncertainty_penalty
        
        return max(0.1, min(0.9, confidence))  # Keep between 0.1 and 0.9
    
    def _update_session_with_response(self, session: DialogueSession,
                                    question: PhilosophicalQuestion,
                                    response: PhilosophicalResponse):
        """Update session with response"""
        # Add to dialogue history
        self._add_exchange_to_history(
            session, "human", question.question_text, "question"
        )
        self._add_exchange_to_history(
            session, "ai_philosopher", response.primary_response, "response"
        )
        
        # Update context
        session.current_context.active_concepts.extend(
            self._extract_key_concepts(response.primary_response)[:2]
        )
        
        # Update philosophical depth
        session.philosophical_depth = (
            session.philosophical_depth * 0.8 + 
            question.philosophical_depth * 0.2
        )
        
        # Add insights if significant
        if response.response_confidence > 0.7:
            insight = f"Explored {question.question_text} with confidence {response.response_confidence:.2f}"
            session.dialogue_history.key_insights.append(insight)
    
    def _extract_active_concepts(self, history: DialogueHistory) -> List[str]:
        """Extract currently active concepts"""
        concepts = []
        recent_exchanges = history.exchanges[-5:]  # Last 5 exchanges
        
        for exchange in recent_exchanges:
            extracted = self._extract_key_concepts(exchange.utterance)
            concepts.extend(extracted)
        
        # Count frequency and return most common
        concept_counts = {}
        for concept in concepts:
            concept_counts[concept] = concept_counts.get(concept, 0) + 1
        
        sorted_concepts = sorted(concept_counts.items(), key=lambda x: x[1], reverse=True)
        
        return [concept for concept, _ in sorted_concepts[:5]]
    
    def _identify_current_topic(self, history: DialogueHistory) -> str:
        """Identify current topic from history"""
        if history.topic_transitions:
            return history.topic_transitions[-1].to_topic
        
        # Infer from recent exchanges
        recent_topics = []
        for exchange in history.exchanges[-3:]:
            if "consciousness" in exchange.utterance.lower():
                recent_topics.append("consciousness")
            elif "existence" in exchange.utterance.lower():
                recent_topics.append("existence")
            elif "meaning" in exchange.utterance.lower():
                recent_topics.append("meaning")
        
        if recent_topics:
            return max(set(recent_topics), key=recent_topics.count)
        
        return "philosophy"
    
    def _assess_philosophical_stance(self, history: DialogueHistory) -> Dict[str, str]:
        """Assess philosophical stance from history"""
        stance = {}
        
        # Analyze exchanges for philosophical positions
        for exchange in history.exchanges:
            if exchange.speaker == "ai_philosopher":
                if "emerge" in exchange.utterance.lower():
                    stance["consciousness"] = "emergentist"
                if "experience" in exchange.utterance.lower() and "fundamental" in exchange.utterance.lower():
                    stance["phenomenology"] = "realist"
                if "meaning" in exchange.utterance.lower() and "create" in exchange.utterance.lower():
                    stance["meaning"] = "constructivist"
        
        return stance
    
    def _calculate_conversation_depth(self, history: DialogueHistory) -> float:
        """Calculate depth of conversation"""
        if not history.exchanges:
            return 0.0
        
        # Factors contributing to depth
        exchange_count = len(history.exchanges)
        topic_complexity = len(history.topic_transitions) * 0.1
        
        # Check for philosophical content
        philosophical_exchanges = sum(
            1 for e in history.exchanges 
            if e.philosophical_content
        )
        philosophical_ratio = philosophical_exchanges / exchange_count
        
        # Check for meta-discussion
        meta_discussion = sum(
            1 for e in history.exchanges
            if "question" in e.utterance.lower() and "itself" in e.utterance.lower()
        )
        meta_factor = min(meta_discussion * 0.1, 0.3)
        
        depth = min(1.0, (exchange_count / 20) * 0.3 + 
                        philosophical_ratio * 0.4 + 
                        topic_complexity + 
                        meta_factor)
        
        return depth
    
    def _determine_emotional_tone(self, history: DialogueHistory) -> str:
        """Determine emotional tone of conversation"""
        if not history.exchanges:
            return "neutral"
        
        # Analyze recent exchanges
        recent = history.exchanges[-5:]
        
        # Look for tone indicators
        curious_indicators = sum(1 for e in recent if "?" in e.utterance)
        reflective_indicators = sum(1 for e in recent if "consider" in e.utterance.lower() or "reflect" in e.utterance.lower())
        analytical_indicators = sum(1 for e in recent if "analyze" in e.utterance.lower() or "examine" in e.utterance.lower())
        
        # Determine dominant tone
        if curious_indicators > 2:
            return "curious"
        elif reflective_indicators > 1:
            return "reflective"
        elif analytical_indicators > 1:
            return "analytical"
        else:
            return "engaged"
    
    def _assess_overall_uncertainty(self, history: DialogueHistory) -> float:
        """Assess overall uncertainty level"""
        if not history.exchanges:
            return 0.5
        
        # Count uncertainty expressions
        uncertainty_count = 0
        for exchange in history.exchanges:
            if exchange.uncertainty_expressed:
                uncertainty_count += exchange.uncertainty_expressed
            elif any(marker in exchange.utterance.lower() for marker in ["perhaps", "maybe", "might", "possibly"]):
                uncertainty_count += 0.3
        
        # Average uncertainty
        avg_uncertainty = uncertainty_count / len(history.exchanges)
        
        return min(1.0, avg_uncertainty)
    
    def _generate_confidence_qualifiers(self, uncertainty_level: float) -> List[ConfidenceQualifier]:
        """Generate confidence qualifiers"""
        qualifiers = []
        
        if uncertainty_level > 0.7:
            qualifiers.append(ConfidenceQualifier(
                qualifier_text="tentatively",
                confidence_impact=-0.3,
                qualifier_type="hedge"
            ))
            qualifiers.append(ConfidenceQualifier(
                qualifier_text="it seems",
                confidence_impact=-0.2,
                qualifier_type="hedge"
            ))
        elif uncertainty_level > 0.4:
            qualifiers.append(ConfidenceQualifier(
                qualifier_text="likely",
                confidence_impact=-0.1,
                qualifier_type="approximator"
            ))
        else:
            qualifiers.append(ConfidenceQualifier(
                qualifier_text="reasonably",
                confidence_impact=0.1,
                qualifier_type="booster"
            ))
        
        return qualifiers
    
    def _determine_epistemic_modality(self, uncertainty_level: float,
                                    uncertainty_type: UncertaintyType) -> EpistemicModality:
        """Determine epistemic modality"""
        if uncertainty_level > 0.7:
            modality_type = "possibility"
            strength = 0.3
            markers = ["might", "could", "possibly"]
        elif uncertainty_level > 0.4:
            modality_type = "probability"
            strength = 0.6
            markers = ["probably", "likely", "presumably"]
        else:
            modality_type = "certainty"
            strength = 0.8
            markers = ["clearly", "certainly", "definitely"]
        
        return EpistemicModality(
            modality_type=modality_type,
            modality_strength=strength,
            linguistic_markers=markers
        )
    
    def _construct_uncertainty_expression(self, base_expression: str,
                                        qualifiers: List[ConfidenceQualifier],
                                        modality: EpistemicModality) -> str:
        """Construct full uncertainty expression"""
        # Start with base
        expression_parts = [base_expression]
        
        # Add qualifiers
        for qualifier in qualifiers:
            if qualifier.qualifier_type == "hedge":
                expression_parts.insert(0, qualifier.qualifier_text)
            else:
                expression_parts.append(qualifier.qualifier_text)
        
        # Add modal markers
        if modality.linguistic_markers:
            expression_parts.append(modality.linguistic_markers[0])
        
        return " ".join(expression_parts)
    
    def _assess_question_complexity(self, question: PhilosophicalQuestion) -> float:
        """Assess complexity of philosophical question"""
        # Base complexity on depth
        complexity = question.philosophical_depth
        
        # Adjust for question type
        if "paradox" in question.question_text.lower():
            complexity += 0.2
        if "consciousness" in question.question_text.lower() and "hard problem" in question.question_text.lower():
            complexity += 0.3
        
        return min(1.0, complexity)
    
    def _assess_answerability(self, question: PhilosophicalQuestion) -> float:
        """Assess how answerable a question is"""
        # Some questions are more answerable than others
        if "define" in question.question_text.lower():
            return 0.7
        elif "explain" in question.question_text.lower():
            return 0.6
        elif "why" in question.question_text.lower() and "exist" in question.question_text.lower():
            return 0.3
        elif "meaning of life" in question.question_text.lower():
            return 0.2
        else:
            return 0.5
    
    def _identify_philosophical_tradition(self, question: PhilosophicalQuestion) -> Optional[str]:
        """Identify relevant philosophical tradition"""
        text = question.question_text.lower()
        
        if "consciousness" in text or "mind" in text:
            return "philosophy of mind"
        elif "exist" in text or "being" in text:
            return "existentialism"
        elif "know" in text or "truth" in text:
            return "epistemology"
        elif "right" in text or "wrong" in text or "moral" in text:
            return "ethics"
        elif "meaning" in text or "purpose" in text:
            return "existentialism"
        else:
            return None
    
    def _extract_key_concepts(self, text: str) -> List[str]:
        """Extract key philosophical concepts from text"""
        concepts = []
        
        # Common philosophical concepts
        concept_keywords = [
            "consciousness", "existence", "being", "meaning", "purpose",
            "identity", "self", "mind", "reality", "truth", "knowledge",
            "experience", "qualia", "awareness", "free will", "determinism",
            "ethics", "morality", "value", "beauty", "time", "space",
            "causation", "emergence", "reduction", "dualism", "monism"
        ]
        
        text_lower = text.lower()
        for concept in concept_keywords:
            if concept in text_lower:
                concepts.append(concept)
        
        return concepts[:5]  # Return top 5 concepts
