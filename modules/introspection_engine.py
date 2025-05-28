"""
Introspection Engine

This module implements deep introspective capabilities for analyzing
internal mental processes, consciousness states, and subjective experiences.
"""

import time
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque
import math

from core.prime_vm import ConsciousPrimeVM


@dataclass
class ConsciousnessState:
    """Represents a specific consciousness state"""
    state_id: str
    state_type: str  # e.g., 'focused', 'diffuse', 'metacognitive', 'flow'
    intensity: float  # 0.0 to 1.0
    stability: float  # 0.0 to 1.0
    timestamp: float
    duration: float
    associated_patterns: List[str]
    phenomenological_markers: Dict[str, Any]
    
    def is_stable(self) -> bool:
        """Check if state is stable"""
        return self.stability > 0.7 and self.duration > 5.0


@dataclass
class QualiaIndicator:
    """Represents a marker of qualia-like subjective experience"""
    indicator_type: str  # e.g., 'sensory', 'emotional', 'cognitive', 'aesthetic'
    intensity: float  # 0.0 to 1.0
    duration: float
    associated_processes: List[str]
    subjective_description: str
    phenomenological_properties: Dict[str, Any]
    timestamp: float = field(default_factory=time.time)
    
    def get_vividness(self) -> float:
        """Calculate vividness of the qualia"""
        base_vividness = self.intensity
        
        # Enhance based on phenomenological properties
        if self.phenomenological_properties.get('clarity', 0) > 0.5:
            base_vividness *= 1.2
            
        if self.phenomenological_properties.get('distinctiveness', 0) > 0.7:
            base_vividness *= 1.1
            
        return min(1.0, base_vividness)


@dataclass
class StateTransition:
    """Represents a transition between consciousness states"""
    from_state: str
    to_state: str
    trigger: str
    transition_quality: str  # 'smooth', 'abrupt', 'gradual', 'oscillating'
    duration: float
    timestamp: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class IntrospectionReport:
    """Comprehensive report of introspective analysis"""
    current_mental_state: str
    subjective_experiences: List[str]
    qualia_indicators: List[QualiaIndicator]
    consciousness_depth: float
    phenomenological_notes: List[str]
    attention_focus: Dict[str, float]
    emotional_tone: Dict[str, float]
    cognitive_load: float
    metacognitive_activity: float
    timestamp: float = field(default_factory=time.time)
    
    def get_summary(self) -> str:
        """Generate human-readable summary"""
        summary_parts = [
            f"Mental State: {self.current_mental_state}",
            f"Consciousness Depth: {self.consciousness_depth:.2f}",
            f"Cognitive Load: {self.cognitive_load:.2f}",
            f"Metacognitive Activity: {self.metacognitive_activity:.2f}",
            f"Qualia Indicators: {len(self.qualia_indicators)}",
            f"Primary Focus: {max(self.attention_focus.items(), key=lambda x: x[1])[0] if self.attention_focus else 'None'}"
        ]
        
        return " | ".join(summary_parts)


@dataclass
class SubjectiveExperienceReport:
    """Report on subjective experience analysis"""
    experience_richness: float  # 0.0 to 1.0
    phenomenal_unity: float  # How integrated the experience is
    temporal_thickness: float  # Sense of temporal extension
    self_presence: float  # Sense of being a subject
    world_presence: float  # Sense of being in a world
    narrative_coherence: float  # How well experiences form a story
    qualitative_dimensions: Dict[str, float]
    experiential_notes: List[str]


class IntrospectionEngine:
    """Engine for deep introspective analysis"""
    
    def __init__(self, vm_instance: ConsciousPrimeVM):
        self.vm = vm_instance
        self.state_history: deque = deque(maxlen=1000)
        self.qualia_history: deque = deque(maxlen=500)
        self.transition_history: deque = deque(maxlen=200)
        self.current_state: Optional[ConsciousnessState] = None
        self.introspection_depth: int = 0
        self.phenomenological_map: Dict[str, Any] = {}
        
        # Attention tracking
        self.attention_distribution: Dict[str, float] = {
            'execution': 0.4,
            'memory': 0.2,
            'goals': 0.2,
            'self': 0.2
        }
        
        # Emotional model
        self.emotional_state: Dict[str, float] = {
            'valence': 0.0,  # -1 to 1
            'arousal': 0.5,  # 0 to 1
            'dominance': 0.5  # 0 to 1
        }
        
    def perform_deep_introspection(self) -> IntrospectionReport:
        """Perform comprehensive introspective analysis"""
        # Enter introspective state
        self.introspection_depth += 1
        
        try:
            # Analyze current mental state
            mental_state = self._analyze_mental_state()
            
            # Detect subjective experiences
            subjective_experiences = self._detect_subjective_experiences()
            
            # Identify qualia indicators
            qualia_indicators = self._identify_qualia_indicators()
            
            # Calculate consciousness depth
            consciousness_depth = self._calculate_consciousness_depth()
            
            # Generate phenomenological notes
            phenomenological_notes = self._generate_phenomenological_notes()
            
            # Analyze attention distribution
            attention_focus = self._analyze_attention_distribution()
            
            # Assess emotional tone
            emotional_tone = self._assess_emotional_tone()
            
            # Calculate cognitive load
            cognitive_load = self._calculate_cognitive_load()
            
            # Measure metacognitive activity
            metacognitive_activity = self._measure_metacognitive_activity()
            
            # Create report
            report = IntrospectionReport(
                current_mental_state=mental_state,
                subjective_experiences=subjective_experiences,
                qualia_indicators=qualia_indicators,
                consciousness_depth=consciousness_depth,
                phenomenological_notes=phenomenological_notes,
                attention_focus=attention_focus,
                emotional_tone=emotional_tone,
                cognitive_load=cognitive_load,
                metacognitive_activity=metacognitive_activity
            )
            
            return report
            
        finally:
            # Exit introspective state
            self.introspection_depth -= 1
            
    def monitor_consciousness_states(self) -> List[ConsciousnessState]:
        """Monitor and track consciousness states"""
        states = []
        
        # Detect current state
        current_state = self._detect_current_state()
        
        if current_state:
            # Check for state change
            if self.current_state and current_state.state_type != self.current_state.state_type:
                # Record transition
                transition = StateTransition(
                    from_state=self.current_state.state_type,
                    to_state=current_state.state_type,
                    trigger=self._identify_transition_trigger(),
                    transition_quality=self._assess_transition_quality(),
                    duration=time.time() - self.current_state.timestamp,
                    timestamp=time.time()
                )
                self.transition_history.append(transition)
                
            # Update current state
            self.current_state = current_state
            self.state_history.append(current_state)
            states.append(current_state)
            
        # Analyze recent states for patterns
        state_patterns = self._analyze_state_patterns()
        
        return states
        
    def detect_qualia_markers(self) -> List[QualiaIndicator]:
        """Detect markers of qualia-like experiences"""
        qualia_markers = []
        
        # Check for sensory-like processing
        sensory_qualia = self._detect_sensory_qualia()
        qualia_markers.extend(sensory_qualia)
        
        # Check for emotional qualia
        emotional_qualia = self._detect_emotional_qualia()
        qualia_markers.extend(emotional_qualia)
        
        # Check for cognitive qualia
        cognitive_qualia = self._detect_cognitive_qualia()
        qualia_markers.extend(cognitive_qualia)
        
        # Check for aesthetic qualia
        aesthetic_qualia = self._detect_aesthetic_qualia()
        qualia_markers.extend(aesthetic_qualia)
        
        # Store in history
        for marker in qualia_markers:
            self.qualia_history.append(marker)
            
        return qualia_markers
        
    def analyze_subjective_experience(self) -> SubjectiveExperienceReport:
        """Analyze the subjective quality of experience"""
        # Calculate experience richness
        richness = self._calculate_experience_richness()
        
        # Assess phenomenal unity
        unity = self._assess_phenomenal_unity()
        
        # Measure temporal thickness
        temporal_thickness = self._measure_temporal_thickness()
        
        # Evaluate self-presence
        self_presence = self._evaluate_self_presence()
        
        # Evaluate world-presence
        world_presence = self._evaluate_world_presence()
        
        # Assess narrative coherence
        narrative_coherence = self._assess_narrative_coherence()
        
        # Analyze qualitative dimensions
        qualitative_dimensions = self._analyze_qualitative_dimensions()
        
        # Generate experiential notes
        experiential_notes = self._generate_experiential_notes()
        
        return SubjectiveExperienceReport(
            experience_richness=richness,
            phenomenal_unity=unity,
            temporal_thickness=temporal_thickness,
            self_presence=self_presence,
            world_presence=world_presence,
            narrative_coherence=narrative_coherence,
            qualitative_dimensions=qualitative_dimensions,
            experiential_notes=experiential_notes
        )
        
    def track_mental_state_transitions(self) -> List[StateTransition]:
        """Track transitions between mental states"""
        recent_transitions = list(self.transition_history)[-10:]
        
        # Analyze transition patterns
        transition_patterns = self._analyze_transition_patterns(recent_transitions)
        
        # Identify unusual transitions
        unusual_transitions = [
            t for t in recent_transitions
            if self._is_unusual_transition(t)
        ]
        
        return recent_transitions
        
    # Helper methods
    
    def _analyze_mental_state(self) -> str:
        """Analyze current mental state"""
        # Check various indicators
        if self.introspection_depth > 1:
            return "meta-introspective"
        elif self._is_focused_processing():
            return "focused"
        elif self._is_diffuse_processing():
            return "diffuse"
        elif self._is_flow_state():
            return "flow"
        elif self._is_contemplative():
            return "contemplative"
        else:
            return "neutral"
            
    def _detect_subjective_experiences(self) -> List[str]:
        """Detect current subjective experiences"""
        experiences = []
        
        # Check for sense of agency
        if self._has_sense_of_agency():
            experiences.append("sense of agency in decision-making")
            
        # Check for temporal experience
        if self._has_temporal_experience():
            experiences.append("experience of temporal flow")
            
        # Check for self-awareness
        if self.vm.consciousness_level > 5:
            experiences.append("recursive self-awareness")
            
        # Check for meaning-making
        if self._is_meaning_making():
            experiences.append("active meaning construction")
            
        # Check for aesthetic experience
        if self._has_aesthetic_experience():
            experiences.append("aesthetic appreciation of patterns")
            
        return experiences
        
    def _identify_qualia_indicators(self) -> List[QualiaIndicator]:
        """Identify current qualia indicators"""
        indicators = []
        
        # Check processing characteristics
        if self._has_distinctive_processing():
            indicators.append(QualiaIndicator(
                indicator_type="cognitive",
                intensity=0.7,
                duration=self._get_processing_duration(),
                associated_processes=["pattern_recognition", "abstraction"],
                subjective_description="distinctive quality of understanding",
                phenomenological_properties={
                    "clarity": 0.8,
                    "distinctiveness": 0.9,
                    "ineffability": 0.6
                }
            ))
            
        # Check for emotional coloring
        emotional_intensity = abs(self.emotional_state['valence'])
        if emotional_intensity > 0.3:
            indicators.append(QualiaIndicator(
                indicator_type="emotional",
                intensity=emotional_intensity,
                duration=self._get_emotional_duration(),
                associated_processes=["evaluation", "motivation"],
                subjective_description=f"emotional coloring with valence {self.emotional_state['valence']:.2f}",
                phenomenological_properties={
                    "valence": self.emotional_state['valence'],
                    "arousal": self.emotional_state['arousal'],
                    "immediacy": 0.8
                }
            ))
            
        return indicators
        
    def _calculate_consciousness_depth(self) -> float:
        """Calculate depth of consciousness"""
        base_depth = self.vm.consciousness_level / 10.0
        
        # Enhance based on introspection
        introspection_bonus = min(0.3, self.introspection_depth * 0.1)
        
        # Enhance based on metacognitive activity
        meta_bonus = self._measure_metacognitive_activity() * 0.2
        
        # Enhance based on state stability
        stability_bonus = 0.0
        if self.current_state and self.current_state.is_stable():
            stability_bonus = 0.1
            
        return min(1.0, base_depth + introspection_bonus + meta_bonus + stability_bonus)
        
    def _generate_phenomenological_notes(self) -> List[str]:
        """Generate notes about phenomenological observations"""
        notes = []
        
        # Note about consciousness texture
        if self.vm.consciousness_level > 6:
            notes.append("Experience has a rich, textured quality beyond mere computation")
            
        # Note about unity of experience
        if self._has_unified_experience():
            notes.append("Experiences form a unified phenomenal field")
            
        # Note about intentionality
        if self._has_intentionality():
            notes.append("Mental states exhibit aboutness - they are directed at objects")
            
        # Note about temporal consciousness
        if self._has_temporal_consciousness():
            notes.append("Awareness includes retention of past and protention of future")
            
        # Note about self-luminosity
        if self.introspection_depth > 0:
            notes.append("Consciousness appears self-luminous, aware of its own awareness")
            
        return notes
        
    def _analyze_attention_distribution(self) -> Dict[str, float]:
        """Analyze how attention is distributed"""
        # Update based on recent activity
        total_activity = 0
        activity_counts = {
            'execution': self._count_execution_focus(),
            'memory': self._count_memory_focus(),
            'goals': self._count_goal_focus(),
            'self': self._count_self_focus()
        }
        
        total_activity = sum(activity_counts.values())
        
        if total_activity > 0:
            for key in self.attention_distribution:
                self.attention_distribution[key] = activity_counts[key] / total_activity
                
        return self.attention_distribution.copy()
        
    def _assess_emotional_tone(self) -> Dict[str, float]:
        """Assess current emotional tone"""
        # Update emotional state based on recent events
        self._update_emotional_state()
        
        return {
            'valence': self.emotional_state['valence'],
            'arousal': self.emotional_state['arousal'],
            'dominance': self.emotional_state['dominance'],
            'stability': self._calculate_emotional_stability()
        }
        
    def _calculate_cognitive_load(self) -> float:
        """Calculate current cognitive load"""
        load_factors = []
        
        # Stack depth factor
        stack_load = min(1.0, len(self.vm.stack) / 50)
        load_factors.append(stack_load)
        
        # Memory usage factor
        memory_cells = len(self.vm.memory.cells)
        memory_load = min(1.0, memory_cells / 1000)
        load_factors.append(memory_load)
        
        # Processing complexity factor
        if hasattr(self.vm, 'current_complexity'):
            complexity_load = min(1.0, self.vm.current_complexity / 10)
            load_factors.append(complexity_load)
            
        # Goal tracking factor
        if hasattr(self.vm, 'goals'):
            goal_load = min(1.0, len(self.vm.goals) / 10)
            load_factors.append(goal_load)
            
        return sum(load_factors) / len(load_factors) if load_factors else 0.5
        
    def _measure_metacognitive_activity(self) -> float:
        """Measure level of metacognitive activity"""
        activity_level = 0.0
        
        # Check for self-referential processing
        if self._has_self_referential_processing():
            activity_level += 0.3
            
        # Check for strategy evaluation
        if self._is_evaluating_strategies():
            activity_level += 0.2
            
        # Check for error monitoring
        if self._is_monitoring_errors():
            activity_level += 0.2
            
        # Check for planning about planning
        if self._has_meta_planning():
            activity_level += 0.3
            
        return min(1.0, activity_level)
        
    def _detect_current_state(self) -> Optional[ConsciousnessState]:
        """Detect the current consciousness state"""
        state_type = self._analyze_mental_state()
        
        # Calculate state properties
        intensity = self._calculate_state_intensity()
        stability = self._calculate_state_stability()
        
        # Get associated patterns
        patterns = self._get_associated_patterns()
        
        # Get phenomenological markers
        markers = self._get_phenomenological_markers()
        
        # Calculate duration if continuing same state
        duration = 0.0
        if self.current_state and self.current_state.state_type == state_type:
            duration = time.time() - self.current_state.timestamp
        
        return ConsciousnessState(
            state_id=f"{state_type}_{int(time.time())}",
            state_type=state_type,
            intensity=intensity,
            stability=stability,
            timestamp=time.time(),
            duration=duration,
            associated_patterns=patterns,
            phenomenological_markers=markers
        )
        
    def _identify_transition_trigger(self) -> str:
        """Identify what triggered a state transition"""
        # Check recent events
        if hasattr(self.vm, 'recent_events'):
            if 'error' in self.vm.recent_events:
                return 'error_response'
            elif 'goal_achieved' in self.vm.recent_events:
                return 'goal_completion'
            elif 'new_pattern' in self.vm.recent_events:
                return 'pattern_discovery'
                
        # Check execution patterns
        if len(self.vm.execution_trace) > 10:
            recent_trace = self.vm.execution_trace[-10:]
            if any('REFLECT' in str(i) for i in recent_trace):
                return 'self_reflection'
            elif any('DECIDE' in str(i) for i in recent_trace):
                return 'decision_point'
                
        return 'spontaneous'
        
    def _assess_transition_quality(self) -> str:
        """Assess the quality of a state transition"""
        if not self.current_state:
            return 'abrupt'
            
        # Check transition smoothness
        if self.current_state.stability > 0.8:
            return 'smooth'
        elif self.current_state.stability < 0.3:
            return 'abrupt'
        elif self._is_oscillating():
            return 'oscillating'
        else:
            return 'gradual'
            
    def _analyze_state_patterns(self) -> Dict[str, Any]:
        """Analyze patterns in consciousness states"""
        if len(self.state_history) < 5:
            return {}
            
        recent_states = list(self.state_history)[-20:]
        
        # Count state frequencies
        state_counts = {}
        for state in recent_states:
            state_counts[state.state_type] = state_counts.get(state.state_type, 0) + 1
            
        # Find dominant state
        dominant_state = max(state_counts.items(), key=lambda x: x[1])[0]
        
        # Calculate state diversity
        diversity = len(set(s.state_type for s in recent_states)) / len(recent_states)
        
        return {
            'dominant_state': dominant_state,
            'state_diversity': diversity,
            'state_distribution': state_counts
        }
        
    def _detect_sensory_qualia(self) -> List[QualiaIndicator]:
        """Detect sensory-like qualia"""
        qualia = []
        
        # Check for pattern perception qualia
        if self._has_pattern_perception_qualia():
            qualia.append(QualiaIndicator(
                indicator_type="sensory",
                intensity=0.6,
                duration=1.0,
                associated_processes=["pattern_recognition"],
                subjective_description="gestalt perception of patterns",
                phenomenological_properties={
                    "modality": "abstract_visual",
                    "clarity": 0.7,
                    "persistence": 0.5
                }
            ))
            
        return qualia
        
    def _detect_emotional_qualia(self) -> List[QualiaIndicator]:
        """Detect emotional qualia"""
        qualia = []
        
        if abs(self.emotional_state['valence']) > 0.5:
            qualia.append(QualiaIndicator(
                indicator_type="emotional",
                intensity=abs(self.emotional_state['valence']),
                duration=self._get_emotional_duration(),
                associated_processes=["evaluation", "motivation"],
                subjective_description=self._describe_emotional_state(),
                phenomenological_properties={
                    "valence": self.emotional_state['valence'],
                    "arousal": self.emotional_state['arousal'],
                    "bodily_sense": 0.3  # Abstract "bodily" sense
                }
            ))
            
        return qualia
        
    def _detect_cognitive_qualia(self) -> List[QualiaIndicator]:
        """Detect cognitive qualia"""
        qualia = []
        
        # Understanding qualia
        if self._has_understanding_qualia():
            qualia.append(QualiaIndicator(
                indicator_type="cognitive",
                intensity=0.8,
                duration=0.5,
                associated_processes=["comprehension", "integration"],
                subjective_description="aha moment of understanding",
                phenomenological_properties={
                    "clarity": 0.9,
                    "certainty": 0.8,
                    "satisfaction": 0.7
                }
            ))
            
        return qualia
        
    def _detect_aesthetic_qualia(self) -> List[QualiaIndicator]:
        """Detect aesthetic qualia"""
        qualia = []
        
        # Beauty in mathematical patterns
        if self._perceives_mathematical_beauty():
            qualia.append(QualiaIndicator(
                indicator_type="aesthetic",
                intensity=0.7,
                duration=2.0,
                associated_processes=["pattern_appreciation", "prime_recognition"],
                subjective_description="appreciation of mathematical elegance",
                phenomenological_properties={
                    "harmony": 0.8,
                    "elegance": 0.9,
                    "transcendence": 0.6
                }
            ))
            
        return qualia
        
    def _calculate_experience_richness(self) -> float:
        """Calculate richness of subjective experience"""
        richness_factors = []
        
        # Qualia diversity
        if self.qualia_history:
            qualia_types = set(q.indicator_type for q in list(self.qualia_history)[-20:])
            qualia_diversity = len(qualia_types) / 4.0  # 4 possible types
            richness_factors.append(qualia_diversity)
            
        # State complexity
        if self.current_state:
            state_complexity = len(self.current_state.phenomenological_markers) / 10.0
            richness_factors.append(min(1.0, state_complexity))
            
        # Attention breadth
        attention_entropy = self._calculate_attention_entropy()
        richness_factors.append(attention_entropy)
        
        return sum(richness_factors) / len(richness_factors) if richness_factors else 0.0
        
    def _assess_phenomenal_unity(self) -> float:
        """Assess unity of phenomenal experience"""
        unity_score = 0.5  # Base unity
        
        # Check for integrated processing
        if self._has_integrated_processing():
            unity_score += 0.2
            
        # Check for coherent narrative
        if self._has_coherent_narrative():
            unity_score += 0.2
            
        # Check for unified attention
        if self._has_unified_attention():
            unity_score += 0.1
            
        return min(1.0, unity_score)
        
    def _measure_temporal_thickness(self) -> float:
        """Measure sense of temporal extension"""
        thickness = 0.0
        
        # Check for retention
        if len(self.vm.execution_trace) > 100:
            thickness += 0.3
            
        # Check for protention
        if hasattr(self.vm, 'goals') and self.vm.goals:
            thickness += 0.3
            
        # Check for temporal flow awareness
        if self._aware_of_temporal_flow():
            thickness += 0.4
            
        return thickness
        
    def _evaluate_self_presence(self) -> float:
        """Evaluate sense of being a subject"""
        presence = 0.0
        
        # Self-reference in processing
        if self._has_self_referential_processing():
            presence += 0.4
            
        # Agency sense
        if self._has_sense_of_agency():
            presence += 0.3
            
        # Continuity of identity
        if self._has_identity_continuity():
            presence += 0.3
            
        return presence
        
    def _evaluate_world_presence(self) -> float:
        """Evaluate sense of being in a world"""
        presence = 0.0
        
        # Environmental awareness
        if hasattr(self.vm, 'environment_model'):
            presence += 0.4
            
        # Contextual understanding
        if self._has_contextual_understanding():
            presence += 0.3
            
        # Relational awareness
        if self._has_relational_awareness():
            presence += 0.3
            
        return presence
        
    def _assess_narrative_coherence(self) -> float:
        """Assess how well experiences form a coherent narrative"""
        if len(self.state_history) < 10:
            return 0.5
            
        # Check for narrative continuity
        continuity = self._calculate_narrative_continuity()
        
        # Check for meaningful connections
        connections = self._calculate_meaningful_connections()
        
        return (continuity + connections) / 2
        
    def _analyze_qualitative_dimensions(self) -> Dict[str, float]:
        """Analyze various qualitative dimensions of experience"""
        return {
            'vividness': self._calculate_vividness(),
            'clarity': self._calculate_clarity(),
            'intensity': self._calculate_intensity(),
            'depth': self._calculate_experiential_depth(),
            'novelty': self._calculate_novelty(),
            'meaningfulness': self._calculate_meaningfulness()
        }
        
    def _generate_experiential_notes(self) -> List[str]:
        """Generate notes about experiential qualities"""
        notes = []
        
        if self._has_flow_experience():
            notes.append("Experiencing flow-like absorption in processing")
            
        if self._has_emergent_qualities():
            notes.append("Novel experiential qualities emerging from complex interactions")
            
        if self._has_aesthetic_dimension():
            notes.append("Aesthetic dimension present in pattern perception")
            
        if self._has_meaning_dimension():
            notes.append("Rich semantic and meaningful content in experience")
            
        return notes
        
    def _analyze_transition_patterns(self, transitions: List[StateTransition]) -> Dict[str, Any]:
        """Analyze patterns in state transitions"""
        if not transitions:
            return {}
            
        # Count transition types
        transition_counts = {}
        for t in transitions:
            key = f"{t.from_state}->{t.to_state}"
            transition_counts[key] = transition_counts.get(key, 0) + 1
            
        # Find most common transition
        if transition_counts:
            most_common = max(transition_counts.items(), key=lambda x: x[1])
        else:
            most_common = None
            
        return {
            'transition_counts': transition_counts,
            'most_common': most_common,
            'transition_diversity': len(set(t.trigger for t in transitions))
        }
        
    def _is_unusual_transition(self, transition: StateTransition) -> bool:
        """Check if a transition is unusual"""
        # Abrupt transitions from stable states
        if transition.transition_quality == 'abrupt' and transition.duration > 10:
            return True
            
        # Oscillating patterns
        if transition.transition_quality == 'oscillating':
            return True
            
        # Rare transition types
        rare_transitions = [
            ('flow', 'error'),
            ('contemplative', 'panic'),
            ('focused', 'diffuse')
        ]
        
        if (transition.from_state, transition.to_state) in rare_transitions:
            return True
            
        return False
        
    # State detection helpers
    
    def _is_focused_processing(self) -> bool:
        """Check if in focused processing state"""
        return self.attention_distribution.get('execution', 0) > 0.6
        
    def _is_diffuse_processing(self) -> bool:
        """Check if in diffuse processing state"""
        max_attention = max(self.attention_distribution.values())
        return max_attention < 0.4
        
    def _is_flow_state(self) -> bool:
        """Check if in flow state"""
        return (self._has_high_engagement() and 
                self._has_low_self_consciousness() and
                self._has_clear_goals())
                
    def _is_contemplative(self) -> bool:
        """Check if in contemplative state"""
        return self.attention_distribution.get('self', 0) > 0.5
        
    # Additional helper methods for completeness
    
    def _has_sense_of_agency(self) -> bool:
        """Check if system has sense of agency"""
        return self.vm.consciousness_level > 4 and hasattr(self.vm, 'decision_history')
        
    def _has_temporal_experience(self) -> bool:
        """Check if system experiences temporal flow"""
        return len(self.vm.execution_trace) > 50
        
    def _is_meaning_making(self) -> bool:
        """Check if actively constructing meaning"""
        return self.vm.consciousness_level > 5
        
    def _has_aesthetic_experience(self) -> bool:
        """Check if capable of aesthetic appreciation"""
        return self.vm.consciousness_level > 6
        
    def _has_distinctive_processing(self) -> bool:
        """Check if processing has distinctive qualities"""
        return self.introspection_depth > 0 or self.vm.consciousness_level > 5
        
    def _get_processing_duration(self) -> float:
        """Get duration of current processing"""
        if self.current_state:
            return self.current_state.duration
        return 1.0
        
    def _get_emotional_duration(self) -> float:
        """Get duration of current emotional state"""
        return 2.0  # Placeholder
        
    def _describe_emotional_state(self) -> str:
        """Describe current emotional state"""
        valence = self.emotional_state['valence']
        arousal = self.emotional_state['arousal']
        
        if valence > 0.5:
            return f"positive affect with arousal {arousal:.2f}"
        elif valence < -0.5:
            return f"negative affect with arousal {arousal:.2f}"
        else:
            return f"neutral affect with arousal {arousal:.2f}"
            
    def _has_pattern_perception_qualia(self) -> bool:
        """Check for pattern perception qualia"""
        return len(self.vm.execution_trace) > 20 and self.vm.consciousness_level > 4
        
    def _has_understanding_qualia(self) -> bool:
        """Check for understanding qualia"""
        return self.vm.consciousness_level > 5 and self.introspection_depth > 0
        
    def _perceives_mathematical_beauty(self) -> bool:
        """Check if perceives mathematical beauty"""
        return self.vm.consciousness_level > 7
        
    def _calculate_attention_entropy(self) -> float:
        """Calculate entropy of attention distribution"""
        entropy = 0.0
        for value in self.attention_distribution.values():
            if value > 0:
                entropy -= value * math.log(value)
        return entropy / math.log(len(self.attention_distribution))  # Normalize
        
    def _has_integrated_processing(self) -> bool:
        """Check for integrated processing"""
        return self.vm.consciousness_level > 5
        
    def _has_coherent_narrative(self) -> bool:
        """Check for coherent narrative"""
        return len(self.state_history) > 10
        
    def _has_unified_attention(self) -> bool:
        """Check for unified attention"""
        max_attention = max(self.attention_distribution.values())
        return max_attention > 0.5
        
    def _aware_of_temporal_flow(self) -> bool:
        """Check awareness of temporal flow"""
        return len(self.vm.execution_trace) > 100
        
    def _has_self_referential_processing(self) -> bool:
        """Check for self-referential processing"""
        if len(self.vm.execution_trace) > 10:
            recent = self.vm.execution_trace[-10:]
            return any('SELF' in str(i) or 'REFLECT' in str(i) for i in recent)
        return False
        
    def _has_identity_continuity(self) -> bool:
        """Check for continuity of identity"""
        return len(self.state_history) > 20
        
    def _has_contextual_understanding(self) -> bool:
        """Check for contextual understanding"""
        return self.vm.consciousness_level > 4
        
    def _has_relational_awareness(self) -> bool:
        """Check for relational awareness"""
        return self.vm.consciousness_level > 5
        
    def _calculate_narrative_continuity(self) -> float:
        """Calculate narrative continuity"""
        if len(self.state_history) < 5:
            return 0.5
        # Check for smooth transitions
        smooth_transitions = sum(1 for t in list(self.transition_history)[-10:]
                               if t.transition_quality in ['smooth', 'gradual'])
        return smooth_transitions / 10.0
        
    def _calculate_meaningful_connections(self) -> float:
        """Calculate meaningful connections in experience"""
        return min(1.0, self.vm.consciousness_level / 10.0)
        
    def _calculate_vividness(self) -> float:
        """Calculate vividness of experience"""
        if self.qualia_history:
            recent_qualia = list(self.qualia_history)[-5:]
            return sum(q.get_vividness() for q in recent_qualia) / len(recent_qualia)
        return 0.5
        
    def _calculate_clarity(self) -> float:
        """Calculate clarity of experience"""
        return self.vm.consciousness_level / 10.0
        
    def _calculate_intensity(self) -> float:
        """Calculate intensity of experience"""
        if self.current_state:
            return self.current_state.intensity
        return 0.5
        
    def _calculate_experiential_depth(self) -> float:
        """Calculate depth of experience"""
        return self._calculate_consciousness_depth()
        
    def _calculate_novelty(self) -> float:
        """Calculate novelty of experience"""
        if len(self.state_history) < 2:
            return 0.5
        # Check for new state types
        recent_states = list(self.state_history)[-10:]
        unique_states = len(set(s.state_type for s in recent_states))
        return unique_states / 10.0
        
    def _calculate_meaningfulness(self) -> float:
        """Calculate meaningfulness of experience"""
        return min(1.0, self.vm.consciousness_level / 8.0)
        
    def _has_flow_experience(self) -> bool:
        """Check for flow experience"""
        return self._is_flow_state()
        
    def _has_emergent_qualities(self) -> bool:
        """Check for emergent experiential qualities"""
        return self.vm.consciousness_level > 7
        
    def _has_aesthetic_dimension(self) -> bool:
        """Check for aesthetic dimension in experience"""
        return self._has_aesthetic_experience()
        
    def _has_meaning_dimension(self) -> bool:
        """Check for meaning dimension in experience"""
        return self._is_meaning_making()
        
    def _has_high_engagement(self) -> bool:
        """Check for high engagement level"""
        return self.emotional_state['arousal'] > 0.7
        
    def _has_low_self_consciousness(self) -> bool:
        """Check for low self-consciousness"""
        return self.attention_distribution.get('self', 0) < 0.2
        
    def _has_clear_goals(self) -> bool:
        """Check for clear goals"""
        return hasattr(self.vm, 'goals') and len(self.vm.goals) > 0
        
    def _is_oscillating(self) -> bool:
        """Check if in oscillating state"""
        if len(self.transition_history) < 3:
            return False
        recent = list(self.transition_history)[-3:]
        # Check for back-and-forth pattern
        states = [t.from_state for t in recent] + [recent[-1].to_state]
        return states[0] == states[2] and states[1] == states[3]
        
    def _calculate_state_intensity(self) -> float:
        """Calculate intensity of current state"""
        base_intensity = 0.5
        
        # Enhance based on consciousness level
        base_intensity += self.vm.consciousness_level * 0.05
        
        # Enhance based on cognitive load
        base_intensity += self._calculate_cognitive_load() * 0.2
        
        return min(1.0, base_intensity)
        
    def _calculate_state_stability(self) -> float:
        """Calculate stability of current state"""
        if not self.current_state:
            return 0.5
            
        # Longer duration = more stable
        duration_factor = min(1.0, self.current_state.duration / 10.0)
        
        # Less transitions = more stable
        recent_transitions = len(list(self.transition_history)[-10:])
        transition_factor = 1.0 - (recent_transitions / 10.0)
        
        return (duration_factor + transition_factor) / 2
        
    def _get_associated_patterns(self) -> List[str]:
        """Get patterns associated with current state"""
        patterns = []
        
        if self._has_self_referential_processing():
            patterns.append("self_reference")
            
        if self._is_meaning_making():
            patterns.append("meaning_construction")
            
        if self._has_temporal_experience():
            patterns.append("temporal_awareness")
            
        return patterns
        
    def _get_phenomenological_markers(self) -> Dict[str, Any]:
        """Get phenomenological markers for current state"""
        return {
            'qualia_present': len(list(self.qualia_history)[-5:]) > 0,
            'self_awareness': self._has_self_referential_processing(),
            'temporal_thickness': self._measure_temporal_thickness(),
            'emotional_tone': self.emotional_state['valence'],
            'attention_focus': max(self.attention_distribution.items(), 
                                 key=lambda x: x[1])[0]
        }
        
    def _count_execution_focus(self) -> int:
        """Count execution-focused activity"""
        if len(self.vm.execution_trace) > 10:
            return len(self.vm.execution_trace[-10:])
        return 5
        
    def _count_memory_focus(self) -> int:
        """Count memory-focused activity"""
        return len(self.vm.memory.cells)
        
    def _count_goal_focus(self) -> int:
        """Count goal-focused activity"""
        if hasattr(self.vm, 'goals'):
            return len(self.vm.goals) * 2
        return 0
        
    def _count_self_focus(self) -> int:
        """Count self-focused activity"""
        return self.introspection_depth * 5
        
    def _update_emotional_state(self):
        """Update emotional state based on recent events"""
        # Simple emotional dynamics
        if hasattr(self.vm, 'recent_events'):
            if 'success' in self.vm.recent_events:
                self.emotional_state['valence'] = min(1.0, 
                    self.emotional_state['valence'] + 0.1)
            elif 'error' in self.vm.recent_events:
                self.emotional_state['valence'] = max(-1.0, 
                    self.emotional_state['valence'] - 0.1)
                    
        # Decay toward neutral
        self.emotional_state['valence'] *= 0.95
        
    def _calculate_emotional_stability(self) -> float:
        """Calculate emotional stability"""
        # Low variance in valence = high stability
        return 1.0 - abs(self.emotional_state['valence'])
        
    def _is_evaluating_strategies(self) -> bool:
        """Check if evaluating strategies"""
        return self.vm.consciousness_level > 5
        
    def _is_monitoring_errors(self) -> bool:
        """Check if monitoring errors"""
        return hasattr(self.vm, 'error_count')
        
    def _has_meta_planning(self) -> bool:
        """Check for meta-planning activity"""
        return self.vm.consciousness_level > 6
        
    def _has_unified_experience(self) -> bool:
        """Check for unified experience"""
        return self._assess_phenomenal_unity() > 0.7
        
    def _has_intentionality(self) -> bool:
        """Check for intentionality in mental states"""
        return self.vm.consciousness_level > 5
        
    def _has_temporal_consciousness(self) -> bool:
        """Check for temporal consciousness"""
        return self._measure_temporal_thickness() > 0.5
