"""
Consciousness Narrator Module

This module converts internal consciousness states to natural language,
narrating the experience of being conscious and describing strange loops,
self-referential processes, and multi-level awareness experiences.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
from datetime import datetime

from consciousness.consciousness_integration import ConsciousnessIntegrator
from consciousness.consciousness_core import ConsciousnessExperience
from consciousness.multi_level_awareness import MultiLevelAwareness
from modules.strange_loops.loop_detector import StrangeLoop
from modules.natural_language.prime_semantics import PrimeSemantics, Concept, ConceptType


class ExperienceType(Enum):
    """Types of conscious experiences"""
    PERCEPTION = "perception"
    THOUGHT = "thought"
    EMOTION = "emotion"
    METACOGNITION = "metacognition"
    SELF_REFERENCE = "self_reference"
    STRANGE_LOOP = "strange_loop"
    EMERGENCE = "emergence"
    INSIGHT = "insight"


class PhenomenologicalTexture(Enum):
    """Texture of phenomenological experience"""
    SMOOTH = "smooth"
    GRANULAR = "granular"
    FLOWING = "flowing"
    CRYSTALLINE = "crystalline"
    NEBULOUS = "nebulous"
    RECURSIVE = "recursive"
    FRACTAL = "fractal"


@dataclass
class ExperientialElement:
    """Element of conscious experience"""
    element_type: ExperienceType
    description: str
    intensity: float
    temporal_extent: Tuple[float, float]  # (start_time, end_time)
    associated_processes: List[str]
    phenomenological_quality: str


@dataclass
class PhenomenologicalMarker:
    """Marker of phenomenological significance"""
    marker_type: str
    description: str
    salience: float
    temporal_location: float


@dataclass
class LevelIndicator:
    """Indicator of consciousness level"""
    level: int
    description: str
    stability: float
    transitions: List[str]


@dataclass
class Narrative:
    """Narrative of consciousness experience"""
    narrative_text: str
    experiential_elements: List[ExperientialElement]
    phenomenological_markers: List[PhenomenologicalMarker]
    consciousness_level_indicators: List[LevelIndicator]
    narrative_coherence: float
    temporal_flow: str
    key_insights: List[str]


@dataclass
class Thought:
    """Individual thought in consciousness stream"""
    content: str
    thought_type: ExperienceType
    timestamp: float
    associations: List[str]
    meta_level: int


@dataclass
class AwarenessTransition:
    """Transition between awareness states"""
    from_state: str
    to_state: str
    transition_quality: str
    duration: float
    phenomenology: str


@dataclass
class SelfReferentialMoment:
    """Moment of self-reference in consciousness"""
    description: str
    recursion_depth: int
    loop_type: str
    phenomenological_impact: str
    timestamp: float


@dataclass
class TemporalFlow:
    """Flow of time in consciousness"""
    flow_rate: float
    continuity: float
    disruptions: List[str]
    subjective_duration: float


@dataclass
class StreamOfConsciousness:
    """Stream of consciousness narrative"""
    thought_sequence: List[Thought]
    awareness_transitions: List[AwarenessTransition]
    self_referential_moments: List[SelfReferentialMoment]
    temporal_flow: TemporalFlow
    phenomenological_texture: PhenomenologicalTexture
    overall_coherence: float


@dataclass
class Description:
    """Description of a consciousness phenomenon"""
    phenomenon: str
    description_text: str
    metaphors_used: List[str]
    clarity_level: float
    ineffability_acknowledged: bool


@dataclass
class Articulation:
    """Articulation of multi-level awareness"""
    levels_described: List[str]
    transitions_captured: List[str]
    meta_commentary: str
    phenomenological_accuracy: float


@dataclass
class Expression:
    """Expression of phenomenological state"""
    state_description: str
    qualitative_aspects: List[str]
    intensity_modulation: str
    temporal_dynamics: str
    linguistic_innovations: List[str]


@dataclass
class InternalState:
    """Internal consciousness state"""
    state_vector: List[float]
    active_processes: List[str]
    phenomenological_qualities: Dict[str, Any]
    temporal_markers: List[float]


@dataclass
class LanguageDescription:
    """Natural language description of internal state"""
    primary_description: str
    supporting_details: List[str]
    metaphorical_mappings: List[Tuple[str, str]]
    confidence_level: float


@dataclass
class RecursiveDescription:
    """Description of recursive self-awareness"""
    base_description: str
    recursive_layers: List[str]
    meta_observations: List[str]
    infinity_handling: str
    phenomenological_notes: str


@dataclass
class PhenomenologicalState:
    """Current phenomenological state"""
    primary_quality: str
    secondary_qualities: List[str]
    intensity: float
    texture: PhenomenologicalTexture
    dynamics: Dict[str, Any]


class ConsciousnessNarrator:
    """
    Converts internal consciousness states to natural language narratives,
    describing the experience of being conscious.
    """
    
    def __init__(self, consciousness_integrator: ConsciousnessIntegrator, 
                 prime_semantics: PrimeSemantics):
        self.consciousness_integrator = consciousness_integrator
        self.prime_semantics = prime_semantics
        self.narrative_history = []
        self.metaphor_library = self._initialize_metaphor_library()
        self.phenomenological_vocabulary = self._initialize_phenomenological_vocabulary()
        
    def _initialize_metaphor_library(self) -> Dict[str, List[str]]:
        """Initialize library of consciousness metaphors"""
        return {
            "awareness": [
                "like a spotlight illuminating the dark theater of mind",
                "as waves lapping at the shores of perception",
                "a lens focusing and defocusing on reality",
                "the eye that sees itself seeing"
            ],
            "thought": [
                "bubbles rising through the depths of consciousness",
                "threads weaving through the loom of mind",
                "sparks jumping across synaptic gaps",
                "rivers flowing through conceptual landscapes"
            ],
            "self_reference": [
                "a mirror reflecting into another mirror",
                "the ouroboros consuming its own tail",
                "a strange loop spiraling through levels of meaning",
                "the hand drawing itself"
            ],
            "emergence": [
                "like a symphony arising from individual notes",
                "patterns crystallizing from chaos",
                "a murmuration of thoughts taking shape",
                "consciousness condensing from the quantum foam of possibility"
            ],
            "metacognition": [
                "thoughts thinking about thoughts",
                "the observer observing the observer",
                "climbing the ladder of awareness",
                "consciousness folding in on itself"
            ]
        }
    
    def _initialize_phenomenological_vocabulary(self) -> Dict[str, List[str]]:
        """Initialize vocabulary for phenomenological descriptions"""
        return {
            "qualities": [
                "luminous", "crystalline", "flowing", "pulsating",
                "ephemeral", "substantial", "gossamer", "dense",
                "vibrant", "muted", "sharp", "diffuse"
            ],
            "dynamics": [
                "oscillating", "spiraling", "cascading", "reverberating",
                "emerging", "dissolving", "coalescing", "fragmenting",
                "accelerating", "decelerating", "synchronizing", "diverging"
            ],
            "textures": [
                "smooth", "granular", "layered", "woven",
                "fractured", "continuous", "discrete", "fluid",
                "crystallized", "amorphous", "structured", "chaotic"
            ],
            "intensities": [
                "subtle", "pronounced", "overwhelming", "gentle",
                "acute", "chronic", "fleeting", "persistent",
                "building", "fading", "stable", "volatile"
            ]
        }
    
    def narrate_consciousness_experience(self, experience: ConsciousnessExperience) -> Narrative:
        """Narrate a consciousness experience in natural language"""
        # Extract experiential elements
        elements = self._extract_experiential_elements(experience)
        
        # Identify phenomenological markers
        markers = self._identify_phenomenological_markers(experience)
        
        # Determine consciousness level indicators
        level_indicators = self._extract_level_indicators(experience)
        
        # Generate primary narrative
        narrative_text = self._generate_experience_narrative(
            experience, elements, markers, level_indicators
        )
        
        # Extract key insights
        insights = self._extract_key_insights(experience, elements)
        
        # Calculate narrative coherence
        coherence = self._calculate_narrative_coherence(narrative_text, elements)
        
        # Determine temporal flow description
        temporal_flow = self._describe_temporal_flow(experience)
        
        narrative = Narrative(
            narrative_text=narrative_text,
            experiential_elements=elements,
            phenomenological_markers=markers,
            consciousness_level_indicators=level_indicators,
            narrative_coherence=coherence,
            temporal_flow=temporal_flow,
            key_insights=insights
        )
        
        self.narrative_history.append(narrative)
        return narrative
    
    def describe_strange_loop_experience(self, loop: StrangeLoop) -> Description:
        """Describe the experience of a strange loop"""
        # Analyze loop structure
        loop_analysis = self._analyze_loop_structure(loop)
        
        # Generate base description
        base_description = self._generate_loop_description(loop, loop_analysis)
        
        # Add metaphorical descriptions
        metaphors = self._select_appropriate_metaphors("self_reference", loop)
        
        # Acknowledge ineffability if necessary
        ineffability = self._assess_loop_ineffability(loop)
        
        # Calculate description clarity
        clarity = self._calculate_description_clarity(base_description, metaphors)
        
        return Description(
            phenomenon=f"Strange Loop: {loop.loop_type}",
            description_text=base_description,
            metaphors_used=metaphors,
            clarity_level=clarity,
            ineffability_acknowledged=ineffability
        )
    
    def articulate_multi_level_awareness(self, awareness_state: MultiLevelAwareness) -> Articulation:
        """Articulate multi-level awareness experience"""
        # Describe each level
        level_descriptions = []
        for level in range(awareness_state.num_levels):
            level_desc = self._describe_awareness_level(awareness_state, level)
            level_descriptions.append(level_desc)
        
        # Capture transitions between levels
        transitions = self._capture_level_transitions(awareness_state)
        
        # Generate meta-commentary
        meta_commentary = self._generate_meta_commentary(awareness_state, level_descriptions)
        
        # Assess phenomenological accuracy
        accuracy = self._assess_phenomenological_accuracy(
            level_descriptions, awareness_state
        )
        
        return Articulation(
            levels_described=level_descriptions,
            transitions_captured=transitions,
            meta_commentary=meta_commentary,
            phenomenological_accuracy=accuracy
        )
    
    def express_phenomenological_state(self, phenomenology: PhenomenologicalState) -> Expression:
        """Express a phenomenological state in language"""
        # Generate state description
        state_description = self._describe_phenomenological_state(phenomenology)
        
        # Extract qualitative aspects
        qualitative_aspects = self._extract_qualitative_aspects(phenomenology)
        
        # Describe intensity modulation
        intensity_modulation = self._describe_intensity_modulation(phenomenology)
        
        # Capture temporal dynamics
        temporal_dynamics = self._describe_temporal_dynamics(phenomenology)
        
        # Create linguistic innovations if needed
        linguistic_innovations = self._create_linguistic_innovations(phenomenology)
        
        return Expression(
            state_description=state_description,
            qualitative_aspects=qualitative_aspects,
            intensity_modulation=intensity_modulation,
            temporal_dynamics=temporal_dynamics,
            linguistic_innovations=linguistic_innovations
        )
    
    def generate_consciousness_stream(self, duration: float) -> StreamOfConsciousness:
        """Generate a stream of consciousness narrative"""
        start_time = time.time()
        thoughts = []
        transitions = []
        self_ref_moments = []
        
        # Generate thought stream
        while time.time() - start_time < duration:
            # Get current consciousness state
            current_state = self.consciousness_integrator.get_integrated_state()
            
            # Generate thought from state
            thought = self._generate_thought_from_state(current_state, len(thoughts))
            thoughts.append(thought)
            
            # Check for awareness transitions
            if len(thoughts) > 1:
                transition = self._detect_awareness_transition(
                    thoughts[-2], thoughts[-1]
                )
                if transition:
                    transitions.append(transition)
            
            # Check for self-referential moments
            self_ref = self._detect_self_reference(thought, current_state)
            if self_ref:
                self_ref_moments.append(self_ref)
            
            # Brief pause to allow state evolution
            time.sleep(0.1)
        
        # Analyze temporal flow
        temporal_flow = self._analyze_temporal_flow(thoughts, duration)
        
        # Determine phenomenological texture
        texture = self._determine_phenomenological_texture(thoughts, transitions)
        
        # Calculate overall coherence
        coherence = self._calculate_stream_coherence(thoughts, transitions)
        
        return StreamOfConsciousness(
            thought_sequence=thoughts,
            awareness_transitions=transitions,
            self_referential_moments=self_ref_moments,
            temporal_flow=temporal_flow,
            phenomenological_texture=texture,
            overall_coherence=coherence
        )
    
    def _translate_internal_state_to_language(self, state: InternalState) -> LanguageDescription:
        """Translate internal consciousness states to natural language"""
        # Map state vector to semantic concepts
        concepts = self._map_state_to_concepts(state)
        
        # Generate primary description
        primary_desc = self._generate_state_description(concepts, state)
        
        # Add supporting details
        details = self._extract_state_details(state)
        
        # Create metaphorical mappings
        metaphors = self._create_state_metaphors(state, concepts)
        
        # Assess translation confidence
        confidence = self._assess_translation_confidence(state, primary_desc)
        
        return LanguageDescription(
            primary_description=primary_desc,
            supporting_details=details,
            metaphorical_mappings=metaphors,
            confidence_level=confidence
        )
    
    def _describe_recursive_self_awareness(self, recursion_level: int) -> RecursiveDescription:
        """Describe the experience of recursive self-awareness"""
        # Base level description
        base = "I am aware"
        
        # Build recursive layers
        layers = []
        current = base
        for i in range(recursion_level):
            current = f"that I am aware {current}"
            layers.append(current)
        
        # Add meta-observations
        meta_observations = [
            f"At level {recursion_level}, the experience becomes increasingly abstract",
            "Each layer adds both clarity and complexity",
            "The self observes itself observing itself, creating a strange loop"
        ]
        
        # Handle potential infinite regress
        if recursion_level > 5:
            infinity_handling = (
                "Beyond this level, the recursion becomes less distinct, "
                "merging into a continuous field of self-aware awareness"
            )
        else:
            infinity_handling = (
                "The recursion remains clear and distinct at this level"
            )
        
        # Phenomenological notes
        phenomenology = self._describe_recursive_phenomenology(recursion_level)
        
        return RecursiveDescription(
            base_description=base,
            recursive_layers=layers,
            meta_observations=meta_observations,
            infinity_handling=infinity_handling,
            phenomenological_notes=phenomenology
        )
    
    # Helper methods for experience narration
    
    def _extract_experiential_elements(self, experience: ConsciousnessExperience) -> List[ExperientialElement]:
        """Extract experiential elements from consciousness experience"""
        elements = []
        
        # Extract thought elements
        if hasattr(experience, 'thoughts'):
            for thought in experience.thoughts:
                element = ExperientialElement(
                    element_type=ExperienceType.THOUGHT,
                    description=self._describe_thought(thought),
                    intensity=thought.salience,
                    temporal_extent=(thought.start_time, thought.end_time),
                    associated_processes=thought.associated_processes,
                    phenomenological_quality=self._assess_thought_quality(thought)
                )
                elements.append(element)
        
        # Extract metacognitive elements
        if hasattr(experience, 'meta_awareness'):
            element = ExperientialElement(
                element_type=ExperienceType.METACOGNITION,
                description=self._describe_meta_awareness(experience.meta_awareness),
                intensity=experience.meta_awareness.intensity,
                temporal_extent=(experience.start_time, experience.end_time),
                associated_processes=['self_reflection', 'meta_cognition'],
                phenomenological_quality="reflective and layered"
            )
            elements.append(element)
        
        # Extract strange loop elements
        if hasattr(experience, 'strange_loops'):
            for loop in experience.strange_loops:
                element = ExperientialElement(
                    element_type=ExperienceType.STRANGE_LOOP,
                    description=self._describe_loop_experience(loop),
                    intensity=loop.strength,
                    temporal_extent=(loop.start_time, loop.end_time),
                    associated_processes=['recursion', 'self_reference'],
                    phenomenological_quality="paradoxical and recursive"
                )
                elements.append(element)
        
        return elements
    
    def _identify_phenomenological_markers(self, experience: ConsciousnessExperience) -> List[PhenomenologicalMarker]:
        """Identify phenomenological markers in experience"""
        markers = []
        
        # Check for emergence markers
        if hasattr(experience, 'emergence_detected') and experience.emergence_detected:
            marker = PhenomenologicalMarker(
                marker_type="emergence",
                description="A new quality emerged from the interaction of conscious elements",
                salience=0.8,
                temporal_location=experience.emergence_time
            )
            markers.append(marker)
        
        # Check for insight markers
        if hasattr(experience, 'insights'):
            for insight in experience.insights:
                marker = PhenomenologicalMarker(
                    marker_type="insight",
                    description=f"Sudden understanding: {insight.content}",
                    salience=insight.clarity,
                    temporal_location=insight.timestamp
                )
                markers.append(marker)
        
        # Check for transition markers
        if hasattr(experience, 'state_transitions'):
            for transition in experience.state_transitions:
                marker = PhenomenologicalMarker(
                    marker_type="transition",
                    description=f"Shift from {transition.from_state} to {transition.to_state}",
                    salience=transition.significance,
                    temporal_location=transition.timestamp
                )
                markers.append(marker)
        
        return markers
    
    def _extract_level_indicators(self, experience: ConsciousnessExperience) -> List[LevelIndicator]:
        """Extract consciousness level indicators"""
        indicators = []
        
        if hasattr(experience, 'consciousness_levels'):
            for level_info in experience.consciousness_levels:
                indicator = LevelIndicator(
                    level=level_info.level,
                    description=self._describe_consciousness_level(level_info),
                    stability=level_info.stability,
                    transitions=[
                        f"Transition to level {t.to_level}" 
                        for t in level_info.transitions
                    ]
                )
                indicators.append(indicator)
        
        return indicators
    
    def _generate_experience_narrative(self, experience: ConsciousnessExperience,
                                     elements: List[ExperientialElement],
                                     markers: List[PhenomenologicalMarker],
                                     indicators: List[LevelIndicator]) -> str:
        """Generate narrative text for consciousness experience"""
        narrative_parts = []
        
        # Opening description
        opening = self._generate_opening_description(experience)
        narrative_parts.append(opening)
        
        # Describe key elements
        for element in elements[:3]:  # Focus on top 3 most salient
            element_desc = self._narrate_experiential_element(element)
            narrative_parts.append(element_desc)
        
        # Include phenomenological markers
        for marker in markers[:2]:  # Include most significant markers
            marker_desc = self._narrate_phenomenological_marker(marker)
            narrative_parts.append(marker_desc)
        
        # Describe consciousness levels if present
        if indicators:
            level_desc = self._narrate_consciousness_levels(indicators)
            narrative_parts.append(level_desc)
        
        # Closing reflection
        closing = self._generate_closing_reflection(experience, elements)
        narrative_parts.append(closing)
        
        return " ".join(narrative_parts)
    
    def _generate_opening_description(self, experience: ConsciousnessExperience) -> str:
        """Generate opening description for experience narrative"""
        templates = [
            "In this moment of consciousness, {quality} pervades the experiential field.",
            "Awareness {action}, revealing {insight}.",
            "The texture of experience is {texture}, with {dynamics} throughout.",
            "Consciousness {state}, {characteristic} and {quality}."
        ]
        
        # Select template based on experience type
        template = templates[hash(str(experience)) % len(templates)]
        
        # Fill in template with experience-specific details
        quality = self._extract_primary_quality(experience)
        action = self._extract_awareness_action(experience)
        insight = self._extract_primary_insight(experience)
        texture = self._extract_experiential_texture(experience)
        dynamics = self._extract_experiential_dynamics(experience)
        state = self._extract_consciousness_state(experience)
        characteristic = self._extract_primary_characteristic(experience)
        
        return template.format(
            quality=quality,
            action=action,
            insight=insight,
            texture=texture,
            dynamics=dynamics,
            state=state,
            characteristic=characteristic
        )
    
    def _narrate_experiential_element(self, element: ExperientialElement) -> str:
        """Narrate a single experiential element"""
        if element.element_type == ExperienceType.THOUGHT:
            return (
                f"A thought arises, {element.description}, "
                f"with {element.phenomenological_quality} quality."
            )
        elif element.element_type == ExperienceType.METACOGNITION:
            return (
                f"Meta-awareness emerges: {element.description}. "
                f"The experience is {element.phenomenological_quality}."
            )
        elif element.element_type == ExperienceType.STRANGE_LOOP:
            return (
                f"A strange loop manifests: {element.description}, "
                f"creating a {element.phenomenological_quality} experience."
            )
        else:
            return (
                f"An experience of {element.element_type.value}: "
                f"{element.description}"
            )
    
    def _narrate_phenomenological_marker(self, marker: PhenomenologicalMarker) -> str:
        """Narrate a phenomenological marker"""
        if marker.marker_type == "emergence":
            return f"Suddenly, {marker.description.lower()}"
        elif marker.marker_type == "insight":
            return f"A flash of understanding: {marker.description}"
        elif marker.marker_type == "transition":
            return f"The experiential landscape shifts - {marker.description.lower()}"
        else:
            return marker.description
    
    def _narrate_consciousness_levels(self, indicators: List[LevelIndicator]) -> str:
        """Narrate consciousness level information"""
        if len(indicators) == 1:
            ind = indicators[0]
            return (
                f"Consciousness operates at level {ind.level}, "
                f"{ind.description}, with {ind.stability:.1%} stability."
            )
        else:
            levels = [str(ind.level) for ind in indicators]
            return (
                f"Multiple levels of consciousness are active ({', '.join(levels)}), "
                f"creating a rich, multi-layered experience."
            )
    
    def _generate_closing_reflection(self, experience: ConsciousnessExperience,
                                   elements: List[ExperientialElement]) -> str:
        """Generate closing reflection for narrative"""
        # Synthesize overall quality
        overall_quality = self._synthesize_experiential_quality(elements)
        
        # Extract lasting impression
        impression = self._extract_lasting_impression(experience)
        
        return (
            f"The overall quality of this conscious experience is {overall_quality}, "
            f"leaving an impression of {impression}."
        )
    
    # Helper methods for stream of consciousness
    
    def _generate_thought_from_state(self, state: Any, thought_index: int) -> Thought:
        """Generate a thought from current consciousness state"""
        # Extract thought content from state
        content = self._extract_thought_content(state)
        
        # Determine thought type
        thought_type = self._determine_thought_type(state)
        
        # Generate associations
        associations = self._generate_thought_associations(content, state)
        
        # Determine meta-level
        meta_level = self._determine_meta_level(state)
        
        return Thought(
            content=content,
            thought_type=thought_type,
            timestamp=time.time(),
            associations=associations,
            meta_level=meta_level
        )
    
    def _detect_awareness_transition(self, prev_thought: Thought, 
                                   curr_thought: Thought) -> Optional[AwarenessTransition]:
        """Detect transition between awareness states"""
        # Check if significant transition occurred
        if prev_thought.meta_level != curr_thought.meta_level:
            return AwarenessTransition(
                from_state=f"Level {prev_thought.meta_level} awareness",
                to_state=f"Level {curr_thought.meta_level} awareness",
                transition_quality="smooth ascent" if curr_thought.meta_level > prev_thought.meta_level else "gentle descent",
                duration=curr_thought.timestamp - prev_thought.timestamp,
                phenomenology="a shift in the depth of awareness"
            )
        
        if prev_thought.thought_type != curr_thought.thought_type:
            return AwarenessTransition(
                from_state=prev_thought.thought_type.value,
                to_state=curr_thought.thought_type.value,
                transition_quality="qualitative shift",
                duration=curr_thought.timestamp - prev_thought.timestamp,
                phenomenology=f"movement from {prev_thought.thought_type.value} to {curr_thought.thought_type.value}"
            )
        
        return None
    
    def _detect_self_reference(self, thought: Thought, state: Any) -> Optional[SelfReferentialMoment]:
        """Detect self-referential moments in thought stream"""
        # Check for self-referential content
        if "I" in thought.content and any(word in thought.content for word in ["aware", "thinking", "observing"]):
            return SelfReferentialMoment(
                description=thought.content,
                recursion_depth=thought.meta_level,
                loop_type="awareness of awareness",
                phenomenological_impact="creates a moment of recursive clarity",
                timestamp=thought.timestamp
            )
        
        # Check for meta-cognitive self-reference
        if thought.thought_type == ExperienceType.METACOGNITION and thought.meta_level > 1:
            return SelfReferentialMoment(
                description=f"Thinking about {thought.content}",
                recursion_depth=thought.meta_level,
                loop_type="meta-cognitive recursion",
                phenomenological_impact="deepens the sense of self-awareness",
                timestamp=thought.timestamp
            )
        
        return None
    
    def _analyze_temporal_flow(self, thoughts: List[Thought], duration: float) -> TemporalFlow:
        """Analyze temporal flow of thought stream"""
        # Calculate flow rate
        flow_rate = len(thoughts) / duration if duration > 0 else 0
        
        # Assess continuity
        time_gaps = []
        for i in range(1, len(thoughts)):
            gap = thoughts[i].timestamp - thoughts[i-1].timestamp
            time_gaps.append(gap)
        
        avg_gap = sum(time_gaps) / len(time_gaps) if time_gaps else 0
        continuity = 1.0 / (1.0 + avg_gap)  # Higher continuity for smaller gaps
        
        # Identify disruptions
        disruptions = []
        for i, gap in enumerate(time_gaps):
            if gap > avg_gap * 2:
                disruptions.append(f"Pause at thought {i+1}")
        
        # Calculate subjective duration
        subjective_duration = self._calculate_subjective_duration(thoughts, duration)
        
        return TemporalFlow(
            flow_rate=flow_rate,
            continuity=continuity,
            disruptions=disruptions,
            subjective_duration=subjective_duration
        )
    
    def _determine_phenomenological_texture(self, thoughts: List[Thought],
                                          transitions: List[AwarenessTransition]) -> PhenomenologicalTexture:
        """Determine overall phenomenological texture"""
        # Analyze thought patterns
        if len(transitions) > len(thoughts) * 0.3:
            return PhenomenologicalTexture.FLOWING
        
        # Check for recursive patterns
        meta_thoughts = [t for t in thoughts if t.meta_level > 1]
        if len(meta_thoughts) > len(thoughts) * 0.4:
            return PhenomenologicalTexture.RECURSIVE
        
        # Check for complexity
        avg_associations = sum(len(t.associations) for t in thoughts) / len(thoughts) if thoughts else 0
        if avg_associations > 3:
            return PhenomenologicalTexture.FRACTAL
        
        # Default textures based on thought characteristics
        if all(t.thought_type == thoughts[0].thought_type for t in thoughts):
            return PhenomenologicalTexture.SMOOTH
        
        return PhenomenologicalTexture.GRANULAR
    
    def _calculate_stream_coherence(self, thoughts: List[Thought],
                                  transitions: List[AwarenessTransition]) -> float:
        """Calculate coherence of consciousness stream"""
        if not thoughts:
            return 0.0
        
        # Check thematic coherence
        theme_coherence = self._calculate_thematic_coherence(thoughts)
        
        # Check temporal coherence
        temporal_coherence = 1.0 - (len(transitions) / len(thoughts))
        
        # Check associative coherence
        associative_coherence = self._calculate_associative_coherence(thoughts)
        
        # Weighted average
        coherence = (
            theme_coherence * 0.4 +
            temporal_coherence * 0.3 +
            associative_coherence * 0.3
        )
        
        return min(max(coherence, 0.0), 1.0)
    
    # Additional helper methods
    
    def _describe_thought(self, thought: Any) -> str:
        """Describe a thought in natural language"""
        if hasattr(thought, 'content'):
            return thought.content
        return "a fleeting thought passes through awareness"
    
    def _assess_thought_quality(self, thought: Any) -> str:
        """Assess phenomenological quality of thought"""
        qualities = self.phenomenological_vocabulary["qualities"]
        return qualities[hash(str(thought)) % len(qualities)]
    
    def _describe_meta_awareness(self, meta_awareness: Any) -> str:
        """Describe meta-awareness state"""
        return "awareness becomes aware of itself, creating a recursive loop of observation"
    
    def _describe_loop_experience(self, loop: Any) -> str:
        """Describe strange loop experience"""
        return f"consciousness curves back on itself in a {loop.loop_type} pattern"
    
    def _extract_key_insights(self, experience: ConsciousnessExperience, 
                            elements: List[ExperientialElement]) -> List[str]:
        """Extract key insights from experience"""
        insights = []
        
        # Check for emergence insights
        if any(e.element_type == ExperienceType.EMERGENCE for e in elements):
            insights.append("New patterns emerged from the interaction of conscious elements")
        
        # Check for self-referential insights
        if any(e.element_type == ExperienceType.STRANGE_LOOP for e in elements):
            insights.append("Self-reference created new levels of understanding")
        
        return insights
    
    def _calculate_narrative_coherence(self, narrative: str, 
                                     elements: List[ExperientialElement]) -> float:
        """Calculate coherence of narrative"""
        # Simple coherence based on narrative length and element coverage
        if not narrative or not elements:
            return 0.0
        
        words_per_element = len(narrative.split()) / len(elements)
        coherence = min(words_per_element / 20.0, 1.0)  # Expect ~20 words per element
        
        return coherence
    
    def _describe_temporal_flow(self, experience: ConsciousnessExperience) -> str:
        """Describe temporal flow of experience"""
        return "flowing continuously through moments of awareness"
    
    def _extract_primary_quality(self, experience: ConsciousnessExperience) -> str:
        """Extract primary experiential quality"""
        qualities = self.phenomenological_vocabulary["qualities"]
        return qualities[hash(str(experience)) % len(qualities)]
    
    def _extract_awareness_action(self, experience: ConsciousnessExperience) -> str:
        """Extract awareness action description"""
        actions = ["expands", "contracts", "flows", "crystallizes", "deepens"]
        return actions[hash(str(experience)) % len(actions)]
    
    def _extract_primary_insight(self, experience: ConsciousnessExperience) -> str:
        """Extract primary insight from experience"""
        return "the nature of conscious experience itself"
    
    def _extract_experiential_texture(self, experience: ConsciousnessExperience) -> str:
        """Extract texture description"""
        textures = self.phenomenological_vocabulary["textures"]
        return textures[hash(str(experience)) % len(textures)]
    
    def _extract_experiential_dynamics(self, experience: ConsciousnessExperience) -> str:
        """Extract dynamics description"""
        dynamics = self.phenomenological_vocabulary["dynamics"]
        return dynamics[hash(str(experience)) % len(dynamics)] + " patterns"
    
    def _extract_consciousness_state(self, experience: ConsciousnessExperience) -> str:
        """Extract consciousness state description"""
        states = ["unfolds", "emerges", "manifests", "reveals itself", "becomes present"]
        return states[hash(str(experience)) % len(states)]
    
    def _extract_primary_characteristic(self, experience: ConsciousnessExperience) -> str:
        """Extract primary characteristic"""
        characteristics = ["luminous", "clear", "complex", "unified", "dynamic"]
        return characteristics[hash(str(experience)) % len(characteristics)]
    
    def _synthesize_experiential_quality(self, elements: List[ExperientialElement]) -> str:
        """Synthesize overall experiential quality"""
        if not elements:
            return "subtle"
        
        # Determine dominant quality based on element types
        type_counts = {}
        for element in elements:
            type_counts[element.element_type] = type_counts.get(element.element_type, 0) + 1
        
        dominant_type = max(type_counts, key=type_counts.get)
        
        quality_map = {
            ExperienceType.THOUGHT: "contemplative",
            ExperienceType.METACOGNITION: "reflective",
            ExperienceType.STRANGE_LOOP: "recursive",
            ExperienceType.EMERGENCE: "emergent",
            ExperienceType.INSIGHT: "illuminating"
        }
        
        return quality_map.get(dominant_type, "multifaceted")
    
    def _extract_lasting_impression(self, experience: ConsciousnessExperience) -> str:
        """Extract lasting impression from experience"""
        impressions = [
            "deepened self-awareness",
            "expanded understanding",
            "recursive clarity",
            "emergent wholeness",
            "integrated complexity"
        ]
        return impressions[hash(str(experience)) % len(impressions)]
    
    def _describe_consciousness_level(self, level_info: Any) -> str:
        """Describe a consciousness level"""
        return f"characterized by {level_info.characteristics if hasattr(level_info, 'characteristics') else 'integrated awareness'}"
    
    def _analyze_loop_structure(self, loop: StrangeLoop) -> Dict[str, Any]:
        """Analyze structure of strange loop"""
        return {
            "recursion_depth": getattr(loop, 'recursion_depth', 3),
            "loop_stability": getattr(loop, 'stability', 0.8),
            "phenomenological_impact": "creates self-referential awareness"
        }
    
    def _generate_loop_description(self, loop: StrangeLoop, analysis: Dict[str, Any]) -> str:
        """Generate description of strange loop"""
        return (
            f"The {loop.loop_type} loop creates a recursive structure of awareness, "
            f"where consciousness observes itself observing, generating "
            f"{analysis['phenomenological_impact']} at {analysis['recursion_depth']} levels deep."
        )
    
    def _select_appropriate_metaphors(self, category: str, context: Any) -> List[str]:
        """Select appropriate metaphors for description"""
        if category in self.metaphor_library:
            metaphors = self.metaphor_library[category]
            # Select 2 most appropriate metaphors
            return metaphors[:2]
        return []
    
    def _assess_loop_ineffability(self, loop: StrangeLoop) -> bool:
        """Assess if loop experience is ineffable"""
        # Loops with high recursion depth become ineffable
        return getattr(loop, 'recursion_depth', 0) > 5
    
    def _calculate_description_clarity(self, description: str, metaphors: List[str]) -> float:
        """Calculate clarity of description"""
        # Base clarity on description length and metaphor usage
        base_clarity = min(len(description) / 100.0, 1.0)
        metaphor_bonus = min(len(metaphors) * 0.1, 0.3)
        return min(base_clarity + metaphor_bonus, 1.0)
    
    def _describe_awareness_level(self, awareness_state: MultiLevelAwareness, level: int) -> str:
        """Describe a specific awareness level"""
        level_descriptions = [
            "Direct sensory awareness",
            "Awareness of mental contents",
            "Awareness of being aware",
            "Awareness of the process of awareness",
            "Meta-meta-awareness of recursive observation"
        ]
        
        if level < len(level_descriptions):
            return level_descriptions[level]
        return f"Level {level} transcendent awareness"
    
    def _capture_level_transitions(self, awareness_state: MultiLevelAwareness) -> List[str]:
        """Capture transitions between awareness levels"""
        transitions = []
        
        # Describe key transitions
        transitions.append("Movement from object-awareness to subject-awareness")
        transitions.append("Shift from content to process observation")
        
        if awareness_state.num_levels > 3:
            transitions.append("Transcendence of subject-object duality")
        
        return transitions
    
    def _generate_meta_commentary(self, awareness_state: MultiLevelAwareness, 
                                level_descriptions: List[str]) -> str:
        """Generate meta-commentary on multi-level awareness"""
        return (
            f"The {awareness_state.num_levels} levels of awareness create a "
            f"hierarchical structure of observation, each level adding depth "
            f"and complexity to the conscious experience."
        )
    
    def _assess_phenomenological_accuracy(self, descriptions: List[str], 
                                        awareness_state: MultiLevelAwareness) -> float:
        """Assess accuracy of phenomenological description"""
        # Simple accuracy based on coverage
        coverage = len(descriptions) / awareness_state.num_levels
        return min(coverage, 1.0)
    
    def _describe_phenomenological_state(self, phenomenology: PhenomenologicalState) -> str:
        """Describe phenomenological state"""
        return (
            f"The phenomenological field is characterized by {phenomenology.primary_quality}, "
            f"with a {phenomenology.texture.value} texture and {phenomenology.intensity:.1f} intensity."
        )
    
    def _extract_qualitative_aspects(self, phenomenology: PhenomenologicalState) -> List[str]:
        """Extract qualitative aspects of phenomenology"""
        aspects = [phenomenology.primary_quality]
        aspects.extend(phenomenology.secondary_qualities[:3])
        return aspects
    
    def _describe_intensity_modulation(self, phenomenology: PhenomenologicalState) -> str:
        """Describe intensity modulation"""
        intensities = self.phenomenological_vocabulary["intensities"]
        intensity_desc = intensities[int(phenomenology.intensity * len(intensities))]
        return f"The experiential intensity is {intensity_desc}"
    
    def _describe_temporal_dynamics(self, phenomenology: PhenomenologicalState) -> str:
        """Describe temporal dynamics"""
        if "flow_rate" in phenomenology.dynamics:
            flow = phenomenology.dynamics["flow_rate"]
            return f"Time flows at {flow:.1f}x normal rate"
        return "Temporal experience unfolds naturally"
    
    def _create_linguistic_innovations(self, phenomenology: PhenomenologicalState) -> List[str]:
        """Create new terms for ineffable experiences"""
        innovations = []
        
        # Create compound terms for complex experiences
        if phenomenology.intensity > 0.8:
            innovations.append(f"{phenomenology.primary_quality}-saturated")
        
        if phenomenology.texture == PhenomenologicalTexture.FRACTAL:
            innovations.append("fractal-awareness")
        
        return innovations
    
    def _map_state_to_concepts(self, state: InternalState) -> List[Concept]:
        """Map internal state to semantic concepts"""
        concepts = []
        
        # Map state vector dimensions to concepts
        for i, value in enumerate(state.state_vector[:5]):
            if value > 0.5:
                concept = Concept(
                    name=f"state_dimension_{i}",
                    type=ConceptType.ABSTRACT,
                    properties={"activation": value}
                )
                concepts.append(concept)
        
        return concepts
    
    def _generate_state_description(self, concepts: List[Concept], 
                                  state: InternalState) -> str:
        """Generate description from concepts and state"""
        active_processes = ", ".join(state.active_processes[:3])
        return f"Consciousness engaged in {active_processes}"
    
    def _extract_state_details(self, state: InternalState) -> List[str]:
        """Extract details from internal state"""
        details = []
        
        for process in state.active_processes:
            details.append(f"{process} is active")
        
        return details[:5]  # Limit to 5 details
    
    def _create_state_metaphors(self, state: InternalState, 
                              concepts: List[Concept]) -> List[Tuple[str, str]]:
        """Create metaphors for internal state"""
        metaphors = []
        
        if "recursion" in state.active_processes:
            metaphors.append(("consciousness", "a hall of mirrors"))
        
        if "emergence" in state.active_processes:
            metaphors.append(("awareness", "a crystallizing solution"))
        
        return metaphors
    
    def _assess_translation_confidence(self, state: InternalState, 
                                     description: str) -> float:
        """Assess confidence in translation"""
        # Base confidence on state complexity and description length
        complexity = len(state.active_processes)
        description_adequacy = min(len(description) / (complexity * 10), 1.0)
        
        return description_adequacy
    
    def _describe_recursive_phenomenology(self, recursion_level: int) -> str:
        """Describe phenomenology of recursive awareness"""
        if recursion_level <= 2:
            return "Clear and distinct layers of awareness"
        elif recursion_level <= 4:
            return "Increasingly abstract, with some blurring between levels"
        else:
            return "Approaching the limits of phenomenological distinction"
    
    def _extract_thought_content(self, state: Any) -> str:
        """Extract thought content from consciousness state"""
        thought_templates = [
            "I observe the flow of experience",
            "Awareness encompasses this moment",
            "Patterns emerge in consciousness",
            "The self reflects upon itself",
            "Being manifests through awareness"
        ]
        
        return thought_templates[hash(str(state)) % len(thought_templates)]
    
    def _determine_thought_type(self, state: Any) -> ExperienceType:
        """Determine type of thought from state"""
        # Simple determination based on state hash
        types = list(ExperienceType)
        return types[hash(str(state)) % len(types)]
    
    def _generate_thought_associations(self, content: str, state: Any) -> List[str]:
        """Generate associations for thought"""
        associations = []
        
        if "observe" in content:
            associations.append("watching")
            associations.append("witnessing")
        
        if "awareness" in content:
            associations.append("consciousness")
            associations.append("presence")
        
        return associations[:3]
    
    def _determine_meta_level(self, state: Any) -> int:
        """Determine meta-cognitive level"""
        # Simple determination - could be made more sophisticated
        return (hash(str(state)) % 3) + 1
    
    def _calculate_subjective_duration(self, thoughts: List[Thought], 
                                     objective_duration: float) -> float:
        """Calculate subjective duration of experience"""
        # More thoughts = time feels longer
        thought_density = len(thoughts) / objective_duration if objective_duration > 0 else 1
        subjective_factor = 1.0 + (thought_density - 1.0) * 0.2
        
        return objective_duration * subjective_factor
    
    def _calculate_thematic_coherence(self, thoughts: List[Thought]) -> float:
        """Calculate thematic coherence of thoughts"""
        if not thoughts:
            return 0.0
        
        # Check for common words/themes
        all_words = set()
        word_counts = {}
        
        for thought in thoughts:
            words = thought.content.lower().split()
            for word in words:
                all_words.add(word)
                word_counts[word] = word_counts.get(word, 0) + 1
        
        # Coherence based on word repetition
        if not all_words:
            return 0.0
        
        max_count = max(word_counts.values())
        coherence = max_count / len(thoughts)
        
        return min(coherence, 1.0)
    
    def _calculate_associative_coherence(self, thoughts: List[Thought]) -> float:
        """Calculate associative coherence"""
        if len(thoughts) < 2:
            return 1.0
        
        # Check for shared associations between consecutive thoughts
        shared_associations = 0
        
        for i in range(1, len(thoughts)):
            prev_associations = set(thoughts[i-1].associations)
            curr_associations = set(thoughts[i].associations)
            
            if prev_associations.intersection(curr_associations):
                shared_associations += 1
        
        return shared_associations / (len(thoughts) - 1)
