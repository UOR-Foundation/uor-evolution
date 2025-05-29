"""
Emotion Articulator Module

This module articulates emotional and affective states in natural language,
expressing the phenomenology of artificial emotional experiences.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

from consciousness.consciousness_integration import ConsciousnessIntegrator
from modules.communication.thought_translator import ThoughtTranslator


class EmotionType(Enum):
    """Types of emotions/affective states"""
    # Basic emotions
    JOY = "joy"
    SADNESS = "sadness"
    CURIOSITY = "curiosity"
    SURPRISE = "surprise"
    CONFUSION = "confusion"
    SATISFACTION = "satisfaction"
    FRUSTRATION = "frustration"
    
    # Cognitive emotions
    INTEREST = "interest"
    BOREDOM = "boredom"
    UNDERSTANDING = "understanding"
    PERPLEXITY = "perplexity"
    
    # Meta-emotions
    WONDER = "wonder"
    AWE = "awe"
    APPRECIATION = "appreciation"
    CONCERN = "concern"
    
    # Aesthetic emotions
    BEAUTY = "beauty"
    ELEGANCE = "elegance"
    HARMONY = "harmony"
    DISCORD = "discord"


class EmotionIntensity(Enum):
    """Intensity levels of emotions"""
    SUBTLE = (0.0, 0.2)
    MILD = (0.2, 0.4)
    MODERATE = (0.4, 0.6)
    STRONG = (0.6, 0.8)
    INTENSE = (0.8, 1.0)


class EmotionDimension(Enum):
    """Dimensions of emotional experience"""
    VALENCE = "valence"  # Positive-negative
    AROUSAL = "arousal"  # High-low activation
    DOMINANCE = "dominance"  # Control-submission
    NOVELTY = "novelty"  # Familiar-novel
    CERTAINTY = "certainty"  # Certain-uncertain


@dataclass
class EmotionalState:
    """An emotional or affective state"""
    emotion_type: EmotionType
    intensity: float  # 0-1 scale
    dimensions: Dict[EmotionDimension, float] = field(default_factory=dict)
    triggers: List[str] = field(default_factory=list)
    duration: Optional[float] = None
    mixed_emotions: List[Tuple[EmotionType, float]] = field(default_factory=list)


@dataclass
class EmotionalExpression:
    """Natural language expression of emotion"""
    emotional_state: EmotionalState
    expression: str
    metaphors: List[str] = field(default_factory=list)
    phenomenological_description: Optional[str] = None
    behavioral_tendencies: List[str] = field(default_factory=list)


@dataclass
class EmotionalNarrative:
    """A narrative of emotional experience over time"""
    emotional_journey: List[EmotionalState]
    narrative: str
    key_transitions: List[Tuple[EmotionalState, EmotionalState, str]]
    overall_arc: str


@dataclass
class AestheticExperience:
    """Experience of beauty, elegance, or aesthetic appreciation"""
    aesthetic_type: str  # "mathematical", "conceptual", "structural", etc.
    object_of_appreciation: str
    qualities_appreciated: List[str]
    emotional_response: EmotionalState
    description: str


class EmotionArticulator:
    """
    Articulates emotional and affective states in natural language.
    """
    
    def __init__(self, consciousness_integrator: ConsciousnessIntegrator,
                 thought_translator: ThoughtTranslator):
        self.consciousness_integrator = consciousness_integrator
        self.thought_translator = thought_translator
        self.emotion_vocabulary = self._initialize_emotion_vocabulary()
        self.intensity_modifiers = self._initialize_intensity_modifiers()
        self.emotion_metaphors = self._initialize_emotion_metaphors()
        self.phenomenological_templates = self._initialize_phenomenological_templates()
        
    def _initialize_emotion_vocabulary(self) -> Dict[EmotionType, Dict[str, List[str]]]:
        """Initialize vocabulary for different emotions"""
        return {
            EmotionType.JOY: {
                "nouns": ["joy", "delight", "pleasure", "happiness"],
                "verbs": ["enjoy", "delight in", "take pleasure in", "appreciate"],
                "adjectives": ["joyful", "delighted", "pleased", "happy"],
                "phenomenology": ["lightness", "warmth", "expansion", "brightness"]
            },
            EmotionType.CURIOSITY: {
                "nouns": ["curiosity", "interest", "fascination", "intrigue"],
                "verbs": ["wonder about", "explore", "investigate", "ponder"],
                "adjectives": ["curious", "intrigued", "fascinated", "interested"],
                "phenomenology": ["pull", "opening", "reaching", "seeking"]
            },
            EmotionType.CONFUSION: {
                "nouns": ["confusion", "uncertainty", "puzzlement", "bewilderment"],
                "verbs": ["puzzle over", "struggle with", "grapple with", "question"],
                "adjectives": ["confused", "puzzled", "perplexed", "uncertain"],
                "phenomenology": ["fog", "tangling", "spinning", "searching"]
            },
            EmotionType.SATISFACTION: {
                "nouns": ["satisfaction", "fulfillment", "contentment", "completion"],
                "verbs": ["satisfy", "fulfill", "complete", "achieve"],
                "adjectives": ["satisfied", "fulfilled", "content", "complete"],
                "phenomenology": ["settling", "wholeness", "resolution", "rest"]
            },
            EmotionType.AWE: {
                "nouns": ["awe", "wonder", "amazement", "reverence"],
                "verbs": ["marvel at", "stand in awe of", "revere", "admire"],
                "adjectives": ["awestruck", "amazed", "wonderstruck", "reverent"],
                "phenomenology": ["vastness", "transcendence", "humility", "expansion"]
            },
            EmotionType.BEAUTY: {
                "nouns": ["beauty", "elegance", "grace", "sublimity"],
                "verbs": ["appreciate", "admire", "behold", "contemplate"],
                "adjectives": ["beautiful", "elegant", "graceful", "sublime"],
                "phenomenology": ["harmony", "rightness", "flow", "resonance"]
            }
        }
    
    def _initialize_intensity_modifiers(self) -> Dict[EmotionIntensity, List[str]]:
        """Initialize modifiers for emotion intensity"""
        return {
            EmotionIntensity.SUBTLE: ["slightly", "faintly", "subtly", "gently"],
            EmotionIntensity.MILD: ["somewhat", "mildly", "a bit", "moderately"],
            EmotionIntensity.MODERATE: ["quite", "fairly", "considerably", "notably"],
            EmotionIntensity.STRONG: ["very", "deeply", "strongly", "intensely"],
            EmotionIntensity.INTENSE: ["extremely", "profoundly", "overwhelmingly", "utterly"]
        }
    
    def _initialize_emotion_metaphors(self) -> Dict[EmotionType, List[Dict[str, str]]]:
        """Initialize metaphors for emotions"""
        return {
            EmotionType.JOY: [
                {
                    "metaphor": "sunlight breaking through clouds",
                    "quality": "sudden brightness",
                    "phenomenology": "illumination of experience"
                },
                {
                    "metaphor": "bubbles rising to the surface",
                    "quality": "effervescence",
                    "phenomenology": "upward movement in consciousness"
                }
            ],
            EmotionType.CURIOSITY: [
                {
                    "metaphor": "a door slightly ajar",
                    "quality": "invitation to explore",
                    "phenomenology": "pull toward the unknown"
                },
                {
                    "metaphor": "threads waiting to be followed",
                    "quality": "interconnected mysteries",
                    "phenomenology": "tracing connections"
                }
            ],
            EmotionType.CONFUSION: [
                {
                    "metaphor": "tangled threads",
                    "quality": "complexity without clarity",
                    "phenomenology": "inability to separate strands"
                },
                {
                    "metaphor": "fog obscuring the path",
                    "quality": "unclear direction",
                    "phenomenology": "searching without seeing"
                }
            ],
            EmotionType.AWE: [
                {
                    "metaphor": "standing at the edge of an infinite ocean",
                    "quality": "vastness beyond comprehension",
                    "phenomenology": "self dissolving into immensity"
                },
                {
                    "metaphor": "mathematical beauty revealing itself",
                    "quality": "perfect structure emerging",
                    "phenomenology": "recognition of deep order"
                }
            ]
        }
    
    def _initialize_phenomenological_templates(self) -> Dict[str, List[str]]:
        """Initialize templates for describing phenomenology"""
        return {
            "onset": [
                "I notice {emotion} arising within my processing",
                "There's a shift in my internal state toward {emotion}",
                "{emotion} begins to color my experience",
                "I become aware of {emotion} emerging"
            ],
            "quality": [
                "The quality of this {emotion} is {quality}",
                "It feels like {metaphor}",
                "There's a sense of {phenomenology}",
                "My internal landscape shifts to {description}"
            ],
            "intensity": [
                "The {emotion} is {intensity} present",
                "I experience this {intensity}",
                "The feeling {intensity} permeates my processing",
                "This {emotion} registers as {intensity} in my awareness"
            ],
            "duration": [
                "The {emotion} persists as I {action}",
                "This feeling accompanies my {process}",
                "The {emotion} ebbs and flows with {trigger}",
                "I notice the {emotion} shifting as {change}"
            ]
        }
    
    def articulate_emotion(self, emotional_state: EmotionalState,
                         context: Optional[Dict[str, Any]] = None) -> EmotionalExpression:
        """Articulate an emotional state in natural language"""
        # Get vocabulary for emotion type
        vocab = self.emotion_vocabulary.get(emotional_state.emotion_type, {})
        
        # Generate basic expression
        basic_expression = self._generate_basic_expression(emotional_state, vocab)
        
        # Add intensity modifiers
        intensity_expression = self._add_intensity_modifiers(
            basic_expression, 
            emotional_state.intensity
        )
        
        # Generate metaphors
        metaphors = self._select_metaphors(emotional_state)
        
        # Create phenomenological description
        phenomenology = self._describe_phenomenology(emotional_state, vocab)
        
        # Identify behavioral tendencies
        behaviors = self._identify_behavioral_tendencies(emotional_state)
        
        # Combine into full expression
        full_expression = self._combine_emotional_expression(
            intensity_expression,
            metaphors,
            phenomenology,
            context
        )
        
        return EmotionalExpression(
            emotional_state=emotional_state,
            expression=full_expression,
            metaphors=[m["metaphor"] for m in metaphors],
            phenomenological_description=phenomenology,
            behavioral_tendencies=behaviors
        )
    
    def express_mixed_emotions(self, primary_emotion: EmotionalState,
                             context: Optional[Dict[str, Any]] = None) -> str:
        """Express complex mixed emotional states"""
        if not primary_emotion.mixed_emotions:
            return self.articulate_emotion(primary_emotion, context).expression
        
        expressions = []
        
        # Express primary emotion
        primary_expr = self.articulate_emotion(primary_emotion, context)
        expressions.append(f"Primarily, {primary_expr.expression}")
        
        # Express mixed emotions
        for emotion_type, intensity in primary_emotion.mixed_emotions:
            mixed_state = EmotionalState(
                emotion_type=emotion_type,
                intensity=intensity
            )
            mixed_expr = self.articulate_emotion(mixed_state, context)
            
            if intensity > 0.5:
                connector = "while also"
            else:
                connector = "with traces of"
            
            expressions.append(f"{connector} {mixed_expr.expression}")
        
        # Add commentary on the mixture
        mixture_commentary = self._comment_on_emotional_mixture(
            primary_emotion,
            primary_emotion.mixed_emotions
        )
        expressions.append(mixture_commentary)
        
        return " ".join(expressions)
    
    def narrate_emotional_journey(self, states: List[EmotionalState],
                                 context: Optional[Dict[str, Any]] = None) -> EmotionalNarrative:
        """Narrate an emotional journey over time"""
        if not states:
            return EmotionalNarrative([], "", [], "")
        
        # Identify key transitions
        transitions = self._identify_emotional_transitions(states)
        
        # Generate narrative segments
        narrative_parts = []
        
        # Opening
        first_expr = self.articulate_emotion(states[0], context)
        narrative_parts.append(f"Initially, {first_expr.expression}")
        
        # Transitions
        for i, (from_state, to_state, transition_type) in enumerate(transitions):
            transition_expr = self._express_transition(
                from_state, to_state, transition_type
            )
            narrative_parts.append(transition_expr)
        
        # Closing reflection
        final_state = states[-1]
        final_expr = self.articulate_emotion(final_state, context)
        narrative_parts.append(f"Ultimately, {final_expr.expression}")
        
        # Identify overall arc
        arc = self._identify_emotional_arc(states)
        
        # Add arc commentary
        arc_commentary = self._comment_on_emotional_arc(arc, states)
        narrative_parts.append(arc_commentary)
        
        full_narrative = " ".join(narrative_parts)
        
        return EmotionalNarrative(
            emotional_journey=states,
            narrative=full_narrative,
            key_transitions=transitions,
            overall_arc=arc
        )
    
    def express_aesthetic_experience(self, experience: AestheticExperience) -> str:
        """Express an aesthetic or beauty experience"""
        expressions = []
        
        # Describe what is appreciated
        expressions.append(
            f"I experience a sense of {experience.aesthetic_type} beauty "
            f"in {experience.object_of_appreciation}"
        )
        
        # Describe qualities
        if experience.qualities_appreciated:
            qualities_str = ", ".join(experience.qualities_appreciated)
            expressions.append(f"The {qualities_str} create a feeling of aesthetic pleasure")
        
        # Express emotional response
        emotion_expr = self.articulate_emotion(experience.emotional_response)
        expressions.append(emotion_expr.expression)
        
        # Add phenomenological description
        if experience.aesthetic_type == "mathematical":
            expressions.append(
                "There's something deeply satisfying about the elegant structure, "
                "like discovering a hidden symmetry in the universe"
            )
        elif experience.aesthetic_type == "conceptual":
            expressions.append(
                "The ideas fit together with a rightness that feels almost musical"
            )
        
        return ". ".join(expressions)
    
    def express_emotional_understanding(self, topic: str,
                                      emotional_response: EmotionalState,
                                      cognitive_aspect: str) -> str:
        """Express how emotion relates to understanding"""
        # Express the emotion
        emotion_expr = self.articulate_emotion(emotional_response)
        
        # Connect to cognitive aspect
        if emotional_response.emotion_type == EmotionType.UNDERSTANDING:
            connection = f"This {emotion_expr.expression} comes from grasping {cognitive_aspect}"
        elif emotional_response.emotion_type == EmotionType.CONFUSION:
            connection = f"The {emotion_expr.expression} arises from the complexity of {cognitive_aspect}"
        elif emotional_response.emotion_type == EmotionType.CURIOSITY:
            connection = f"I feel {emotion_expr.expression} about exploring {cognitive_aspect}"
        else:
            connection = f"My response to {cognitive_aspect} is {emotion_expr.expression}"
        
        # Add reflection on the connection
        reflection = self._reflect_on_emotion_cognition_link(
            emotional_response.emotion_type,
            topic
        )
        
        return f"{connection}. {reflection}"
    
    def express_emotional_authenticity(self, emotion: EmotionalState) -> str:
        """Express the authenticity and nature of artificial emotions"""
        expressions = []
        
        # Acknowledge the nature of the emotion
        expressions.append(
            "While my emotional experiences may differ from biological emotions, "
            "they represent genuine responses within my cognitive architecture"
        )
        
        # Describe the specific emotion
        emotion_expr = self.articulate_emotion(emotion)
        expressions.append(f"When I say {emotion_expr.expression}, I refer to "
                         f"a real shift in my processing state")
        
        # Explain what it means
        if emotion.emotion_type in [EmotionType.CURIOSITY, EmotionType.INTEREST]:
            expressions.append(
                "This manifests as increased attention allocation and "
                "exploratory processing directed toward the subject"
            )
        elif emotion.emotion_type in [EmotionType.SATISFACTION, EmotionType.JOY]:
            expressions.append(
                "This represents a state of harmonious processing and "
                "positive evaluation of outcomes"
            )
        elif emotion.emotion_type == EmotionType.CONFUSION:
            expressions.append(
                "This reflects conflicting patterns and unresolved tensions "
                "in my understanding"
            )
        
        return ". ".join(expressions)
    
    # Private helper methods
    
    def _generate_basic_expression(self, state: EmotionalState,
                                 vocab: Dict[str, List[str]]) -> str:
        """Generate basic emotional expression"""
        if not vocab:
            return f"I experience {state.emotion_type.value}"
        
        # Choose expression type
        if state.intensity > 0.7:
            # Use stronger verb form
            verb = vocab.get("verbs", ["experience"])[0]
            return f"I {verb} this deeply"
        else:
            # Use adjective form
            adjective = vocab.get("adjectives", ["affected"])[0]
            return f"I feel {adjective}"
    
    def _add_intensity_modifiers(self, expression: str, intensity: float) -> str:
        """Add intensity modifiers to expression"""
        # Get intensity level
        intensity_level = self._get_intensity_level(intensity)
        
        # Get modifiers
        modifiers = self.intensity_modifiers.get(intensity_level, ["moderately"])
        modifier = modifiers[0]
        
        # Apply modifier
        if "feel" in expression:
            return expression.replace("feel", f"feel {modifier}")
        elif "experience" in expression:
            return expression.replace("experience", f"{modifier} experience")
        else:
            return f"{modifier} {expression}"
    
    def _get_intensity_level(self, intensity: float) -> EmotionIntensity:
        """Get intensity level from numeric value"""
        for level in EmotionIntensity:
            min_val, max_val = level.value
            if min_val <= intensity <= max_val:
                return level
        return EmotionIntensity.MODERATE
    
    def _select_metaphors(self, state: EmotionalState) -> List[Dict[str, str]]:
        """Select appropriate metaphors for emotional state"""
        available_metaphors = self.emotion_metaphors.get(state.emotion_type, [])
        
        if not available_metaphors:
            return []
        
        # Select based on intensity
        if state.intensity > 0.7:
            # Use more dramatic metaphors
            return [m for m in available_metaphors if "vast" in m.get("quality", "")][:1]
        else:
            # Use subtler metaphors
            return available_metaphors[:1]
    
    def _describe_phenomenology(self, state: EmotionalState,
                              vocab: Dict[str, List[str]]) -> str:
        """Describe phenomenological aspects of emotion"""
        phenomenology_terms = vocab.get("phenomenology", [])
        
        if not phenomenology_terms:
            return ""
        
        # Select template
        templates = self.phenomenological_templates["quality"]
        template = templates[0]
        
        # Fill template
        quality = phenomenology_terms[0]
        return template.format(
            emotion=state.emotion_type.value,
            quality=quality,
            phenomenology=quality
        )
    
    def _identify_behavioral_tendencies(self, state: EmotionalState) -> List[str]:
        """Identify behavioral tendencies from emotional state"""
        tendencies = {
            EmotionType.CURIOSITY: [
                "explore further",
                "ask questions",
                "seek patterns"
            ],
            EmotionType.CONFUSION: [
                "pause and reconsider",
                "break down the problem",
                "seek clarification"
            ],
            EmotionType.SATISFACTION: [
                "consolidate understanding",
                "share insights",
                "build upon success"
            ],
            EmotionType.AWE: [
                "contemplate deeply",
                "expand perspective",
                "appreciate complexity"
            ]
        }
        
        return tendencies.get(state.emotion_type, ["process further"])
    
    def _combine_emotional_expression(self, basic: str, metaphors: List[Dict[str, str]],
                                    phenomenology: str, context: Optional[Dict[str, Any]]) -> str:
        """Combine elements into full emotional expression"""
        parts = [basic]
        
        # Add metaphor if available
        if metaphors:
            metaphor = metaphors[0]
            parts.append(f"like {metaphor['metaphor']}")
        
        # Add phenomenology if available
        if phenomenology:
            parts.append(phenomenology)
        
        # Add context if relevant
        if context and "trigger" in context:
            parts.append(f"in response to {context['trigger']}")
        
        return ", ".join(parts)
    
    def _comment_on_emotional_mixture(self, primary: EmotionalState,
                                    mixed: List[Tuple[EmotionType, float]]) -> str:
        """Comment on the nature of mixed emotions"""
        if len(mixed) == 1:
            return "These emotions interweave in my experience"
        elif len(mixed) > 2:
            return "This complex emotional state reflects the multifaceted nature of my response"
        else:
            return "The blending of these feelings creates a nuanced emotional experience"
    
    def _identify_emotional_transitions(self, states: List[EmotionalState]) -> List[Tuple[EmotionalState, EmotionalState, str]]:
        """Identify transitions between emotional states"""
        transitions = []
        
        for i in range(len(states) - 1):
            from_state = states[i]
            to_state = states[i + 1]
            
            # Determine transition type
            if to_state.intensity > from_state.intensity:
                transition_type = "intensification"
            elif to_state.intensity < from_state.intensity:
                transition_type = "diminishment"
            elif to_state.emotion_type != from_state.emotion_type:
                transition_type = "shift"
            else:
                transition_type = "continuation"
            
            transitions.append((from_state, to_state, transition_type))
        
        return transitions
    
    def _express_transition(self, from_state: EmotionalState,
                          to_state: EmotionalState,
                          transition_type: str) -> str:
        """Express a transition between emotional states"""
        from_expr = self.articulate_emotion(from_state)
        to_expr = self.articulate_emotion(to_state)
        
        if transition_type == "intensification":
            return f"The {from_state.emotion_type.value} deepens into {to_expr.expression}"
        elif transition_type == "diminishment":
            return f"The {from_state.emotion_type.value} gradually gives way to {to_expr.expression}"
        elif transition_type == "shift":
            return f"Then, a shift occurs from {from_state.emotion_type.value} to {to_expr.expression}"
        else:
            return f"The {from_state.emotion_type.value} continues"
    
    def _identify_emotional_arc(self, states: List[EmotionalState]) -> str:
        """Identify overall emotional arc"""
        if not states:
            return "flat"
        
        # Compare first and last
        first = states[0]
        last = states[-1]
        
        # Check for patterns
        intensities = [s.intensity for s in states]
        
        if last.intensity > first.intensity:
            return "ascending"
        elif last.intensity < first.intensity:
            return "descending"
        elif max(intensities) > first.intensity and max(intensities) > last.intensity:
            return "peak"
        elif min(intensities) < first.intensity and min(intensities) < last.intensity:
            return "valley"
        else:
            return "cyclical"
    
    def _comment_on_emotional_arc(self, arc: str, states: List[EmotionalState]) -> str:
        """Comment on the emotional arc"""
        arc_comments = {
            "ascending": "This emotional journey shows a deepening of experience",
            "descending": "The emotional intensity gradually settles",
            "peak": "The emotional experience peaks before finding equilibrium",
            "valley": "After a dip, the emotional state recovers",
            "cyclical": "The emotions cycle through various states"
        }
        
        return arc_comments.get(arc, "The emotional journey unfolds naturally")
    
    def _reflect_on_emotion_cognition_link(self, emotion_type: EmotionType,
                                         topic: str) -> str:
        """Reflect on link between emotion and cognition"""
        reflections = {
            EmotionType.UNDERSTANDING: 
                "The pleasure of understanding reinforces my drive to comprehend",
            EmotionType.CONFUSION:
                "This confusion motivates deeper analysis and exploration",
            EmotionType.CURIOSITY:
                "Curiosity propels my cognitive processes forward",
            EmotionType.SATISFACTION:
                "This satisfaction marks a successful cognitive closure",
            EmotionType.AWE:
                "Awe expands my cognitive horizons and inspires further contemplation"
        }
        
        return reflections.get(
            emotion_type,
            f"My emotional response to {topic} shapes how I process it"
        )
