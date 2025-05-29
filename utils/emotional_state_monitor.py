"""
Emotional State Monitor - Monitors and analyzes emotional states and transitions

This module provides tools for monitoring emotional states, tracking mood changes,
analyzing emotional patterns, and detecting emotional anomalies in conscious agents.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass, field
from enum import Enum
import numpy as np
from datetime import datetime, timedelta
from collections import deque, defaultdict
import logging

logger = logging.getLogger(__name__)


class EmotionalCategory(Enum):
    """Primary emotional categories based on psychological models"""
    JOY = "joy"
    SADNESS = "sadness"
    ANGER = "anger"
    FEAR = "fear"
    SURPRISE = "surprise"
    DISGUST = "disgust"
    TRUST = "trust"
    ANTICIPATION = "anticipation"
    NEUTRAL = "neutral"


@dataclass
class EmotionalState:
    """Represents an emotional state at a point in time"""
    timestamp: datetime
    agent_id: str
    primary_emotion: EmotionalCategory
    emotion_vector: Dict[str, float]  # Emotion -> intensity (0-1)
    arousal: float  # Low to high activation
    valence: float  # Negative to positive
    dominance: float  # Submissive to dominant
    stability: float  # How stable the emotional state is
    context: Dict[str, Any]
    
    def get_intensity(self) -> float:
        """Get overall emotional intensity"""
        return np.mean(list(self.emotion_vector.values()))
    
    def get_complexity(self) -> float:
        """Calculate emotional complexity (mixed emotions)"""
        # Higher entropy = more complex emotional state
        values = np.array(list(self.emotion_vector.values()))
        values = values[values > 0]  # Only consider present emotions
        if len(values) <= 1:
            return 0.0
        
        # Normalize and calculate entropy
        values = values / values.sum()
        entropy = -np.sum(values * np.log(values + 1e-10))
        max_entropy = np.log(len(values))
        
        return entropy / max_entropy if max_entropy > 0 else 0.0


@dataclass
class EmotionalTransition:
    """Represents a transition between emotional states"""
    from_state: EmotionalState
    to_state: EmotionalState
    trigger: Optional[str]
    transition_time: timedelta
    smoothness: float  # How smooth vs abrupt the transition was
    
    def get_emotional_distance(self) -> float:
        """Calculate distance between emotional states"""
        from_vector = np.array(list(self.from_state.emotion_vector.values()))
        to_vector = np.array(list(self.to_state.emotion_vector.values()))
        return np.linalg.norm(to_vector - from_vector)


@dataclass
class EmotionalPattern:
    """Detected pattern in emotional states"""
    pattern_type: str
    description: str
    frequency: float
    agents_affected: List[str]
    time_range: Tuple[datetime, datetime]
    confidence: float
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class MoodProfile:
    """Long-term mood profile for an agent"""
    agent_id: str
    baseline_emotions: Dict[str, float]
    mood_stability: float
    emotional_range: float
    typical_transitions: List[Tuple[str, str, float]]  # (from, to, probability)
    stress_indicators: Dict[str, float]
    resilience_score: float
    last_updated: datetime


class EmotionalStateMonitor:
    """Monitor and analyze emotional states across agents"""
    
    def __init__(self, window_size: int = 100):
        # State tracking
        self.current_states: Dict[str, EmotionalState] = {}
        self.state_history: Dict[str, deque] = defaultdict(lambda: deque(maxlen=window_size))
        self.transition_history: List[EmotionalTransition] = []
        
        # Mood profiles
        self.mood_profiles: Dict[str, MoodProfile] = {}
        
        # Pattern detection
        self.detected_patterns: List[EmotionalPattern] = []
        self.pattern_detectors = {
            'emotional_contagion': self._detect_emotional_contagion,
            'mood_cycles': self._detect_mood_cycles,
            'stress_patterns': self._detect_stress_patterns,
            'emotional_regulation': self._detect_regulation_patterns
        }
        
        # Monitoring parameters
        self.anomaly_threshold = 2.5  # Standard deviations
        self.contagion_threshold = 0.7
        self.stability_window = 10  # States to consider for stability
        
        # Alert system
        self.alert_callbacks: List[callable] = []
        self.alert_conditions = {
            'extreme_emotion': lambda state: state.get_intensity() > 0.9,
            'rapid_cycling': lambda history: self._check_rapid_cycling(history),
            'emotional_flatness': lambda state: state.get_intensity() < 0.1,
            'prolonged_negative': lambda history: self._check_prolonged_negative(history)
        }
    
    def update_emotional_state(self, agent_id: str, emotion_data: Dict[str, Any]) -> EmotionalState:
        """Update the emotional state for an agent"""
        # Create new state
        new_state = EmotionalState(
            timestamp=datetime.now(),
            agent_id=agent_id,
            primary_emotion=self._determine_primary_emotion(emotion_data),
            emotion_vector=emotion_data.get('emotions', {}),
            arousal=emotion_data.get('arousal', 0.5),
            valence=emotion_data.get('valence', 0.0),
            dominance=emotion_data.get('dominance', 0.5),
            stability=self._calculate_stability(agent_id, emotion_data),
            context=emotion_data.get('context', {})
        )
        
        # Check for transition
        if agent_id in self.current_states:
            old_state = self.current_states[agent_id]
            transition = self._create_transition(old_state, new_state, emotion_data.get('trigger'))
            self.transition_history.append(transition)
        
        # Update current state and history
        self.current_states[agent_id] = new_state
        self.state_history[agent_id].append(new_state)
        
        # Update mood profile
        self._update_mood_profile(agent_id)
        
        # Check for alerts
        self._check_alerts(agent_id, new_state)
        
        # Detect patterns periodically
        if len(self.state_history[agent_id]) % 10 == 0:
            self._run_pattern_detection()
        
        return new_state
    
    def get_emotional_summary(self, agent_id: str, 
                            time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Get comprehensive emotional summary for an agent"""
        if agent_id not in self.state_history:
            return {'error': 'No emotional data for agent'}
        
        # Filter states by time window
        states = list(self.state_history[agent_id])
        if time_window:
            cutoff_time = datetime.now() - time_window
            states = [s for s in states if s.timestamp > cutoff_time]
        
        if not states:
            return {'error': 'No states in specified time window'}
        
        # Calculate summary statistics
        emotion_averages = defaultdict(float)
        emotion_counts = defaultdict(int)
        
        for state in states:
            for emotion, intensity in state.emotion_vector.items():
                emotion_averages[emotion] += intensity
                if intensity > 0.1:
                    emotion_counts[emotion] += 1
        
        # Normalize averages
        for emotion in emotion_averages:
            emotion_averages[emotion] /= len(states)
        
        # Calculate other metrics
        avg_arousal = np.mean([s.arousal for s in states])
        avg_valence = np.mean([s.valence for s in states])
        emotional_variability = np.std([s.get_intensity() for s in states])
        
        # Find dominant emotions
        dominant_emotions = sorted(emotion_averages.items(), 
                                 key=lambda x: x[1], reverse=True)[:3]
        
        return {
            'agent_id': agent_id,
            'time_period': {
                'start': states[0].timestamp,
                'end': states[-1].timestamp,
                'duration': states[-1].timestamp - states[0].timestamp,
                'state_count': len(states)
            },
            'emotion_averages': dict(emotion_averages),
            'dominant_emotions': dominant_emotions,
            'average_arousal': avg_arousal,
            'average_valence': avg_valence,
            'emotional_variability': emotional_variability,
            'current_state': self._get_current_emotional_description(agent_id),
            'mood_profile': self.mood_profiles.get(agent_id)
        }
    
    def analyze_emotional_dynamics(self, agent_ids: Optional[List[str]] = None) -> Dict[str, Any]:
        """Analyze emotional dynamics across multiple agents"""
        if agent_ids is None:
            agent_ids = list(self.current_states.keys())
        
        if not agent_ids:
            return {'error': 'No agents to analyze'}
        
        # Collect current states
        current_emotions = {}
        for agent_id in agent_ids:
            if agent_id in self.current_states:
                state = self.current_states[agent_id]
                current_emotions[agent_id] = {
                    'primary': state.primary_emotion.value,
                    'intensity': state.get_intensity(),
                    'valence': state.valence
                }
        
        # Calculate group metrics
        if current_emotions:
            avg_valence = np.mean([e['valence'] for e in current_emotions.values()])
            emotional_synchrony = self._calculate_emotional_synchrony(agent_ids)
            emotional_diversity = self._calculate_emotional_diversity(current_emotions)
        else:
            avg_valence = 0.0
            emotional_synchrony = 0.0
            emotional_diversity = 0.0
        
        # Identify emotional clusters
        clusters = self._identify_emotional_clusters(agent_ids)
        
        # Recent patterns
        recent_patterns = [p for p in self.detected_patterns 
                          if p.time_range[1] > datetime.now() - timedelta(hours=1)]
        
        return {
            'timestamp': datetime.now(),
            'agent_count': len(agent_ids),
            'current_emotions': current_emotions,
            'group_metrics': {
                'average_valence': avg_valence,
                'emotional_synchrony': emotional_synchrony,
                'emotional_diversity': emotional_diversity,
                'collective_mood': self._determine_collective_mood(avg_valence)
            },
            'emotional_clusters': clusters,
            'recent_patterns': recent_patterns,
            'alerts': self._get_active_alerts(agent_ids)
        }
    
    def detect_emotional_anomalies(self, agent_id: str) -> List[Dict[str, Any]]:
        """Detect anomalous emotional states or transitions"""
        if agent_id not in self.state_history or agent_id not in self.mood_profiles:
            return []
        
        anomalies = []
        states = list(self.state_history[agent_id])
        profile = self.mood_profiles[agent_id]
        
        # Check recent states against baseline
        recent_states = states[-10:] if len(states) >= 10 else states
        
        for state in recent_states:
            # Check each emotion against baseline
            for emotion, intensity in state.emotion_vector.items():
                baseline = profile.baseline_emotions.get(emotion, 0.5)
                deviation = abs(intensity - baseline)
                
                if deviation > self.anomaly_threshold * 0.2:  # Scaled threshold
                    anomalies.append({
                        'type': 'emotion_deviation',
                        'timestamp': state.timestamp,
                        'emotion': emotion,
                        'intensity': intensity,
                        'baseline': baseline,
                        'deviation': deviation,
                        'severity': 'high' if deviation > 0.5 else 'moderate'
                    })
            
            # Check for unusual emotional complexity
            complexity = state.get_complexity()
            if complexity > 0.8:  # Very mixed emotions
                anomalies.append({
                    'type': 'high_complexity',
                    'timestamp': state.timestamp,
                    'complexity': complexity,
                    'emotions': state.emotion_vector,
                    'severity': 'moderate'
                })
        
        # Check transitions
        recent_transitions = [t for t in self.transition_history 
                            if t.to_state.agent_id == agent_id][-5:]
        
        for transition in recent_transitions:
            # Rapid transitions
            if transition.transition_time < timedelta(seconds=30) and \
               transition.get_emotional_distance() > 0.5:
                anomalies.append({
                    'type': 'rapid_transition',
                    'timestamp': transition.to_state.timestamp,
                    'from_emotion': transition.from_state.primary_emotion.value,
                    'to_emotion': transition.to_state.primary_emotion.value,
                    'transition_time': transition.transition_time.total_seconds(),
                    'distance': transition.get_emotional_distance(),
                    'severity': 'high'
                })
        
        return anomalies
    
    def get_emotional_trajectory(self, agent_id: str, 
                               time_points: int = 10) -> Dict[str, Any]:
        """Get emotional trajectory over time"""
        if agent_id not in self.state_history:
            return {'error': 'No emotional history for agent'}
        
        states = list(self.state_history[agent_id])
        if len(states) < 2:
            return {'error': 'Insufficient data for trajectory'}
        
        # Sample states at regular intervals
        indices = np.linspace(0, len(states) - 1, time_points, dtype=int)
        sampled_states = [states[i] for i in indices]
        
        # Extract trajectory data
        trajectory = {
            'timestamps': [s.timestamp for s in sampled_states],
            'valence': [s.valence for s in sampled_states],
            'arousal': [s.arousal for s in sampled_states],
            'primary_emotions': [s.primary_emotion.value for s in sampled_states],
            'intensity': [s.get_intensity() for s in sampled_states],
            'stability': [s.stability for s in sampled_states]
        }
        
        # Calculate trajectory metrics
        valence_trend = np.polyfit(range(len(sampled_states)), trajectory['valence'], 1)[0]
        arousal_trend = np.polyfit(range(len(sampled_states)), trajectory['arousal'], 1)[0]
        
        # Identify emotional phases
        phases = self._identify_emotional_phases(sampled_states)
        
        return {
            'agent_id': agent_id,
            'trajectory': trajectory,
            'trends': {
                'valence_trend': 'improving' if valence_trend > 0 else 'declining',
                'arousal_trend': 'increasing' if arousal_trend > 0 else 'decreasing',
                'valence_slope': valence_trend,
                'arousal_slope': arousal_trend
            },
            'phases': phases,
            'current_phase': phases[-1] if phases else None,
            'prediction': self._predict_next_emotional_state(agent_id)
        }
    
    def register_alert_callback(self, callback: callable) -> None:
        """Register a callback for emotional alerts"""
        self.alert_callbacks.append(callback)
    
    def get_mood_stability_report(self, agent_id: str) -> Dict[str, Any]:
        """Generate report on mood stability"""
        if agent_id not in self.mood_profiles:
            return {'error': 'No mood profile available'}
        
        profile = self.mood_profiles[agent_id]
        recent_states = list(self.state_history[agent_id])[-20:]
        
        if len(recent_states) < 5:
            return {'error': 'Insufficient data for stability analysis'}
        
        # Calculate stability metrics
        valence_values = [s.valence for s in recent_states]
        arousal_values = [s.arousal for s in recent_states]
        
        valence_stability = 1.0 - np.std(valence_values)
        arousal_stability = 1.0 - np.std(arousal_values)
        
        # Count mood swings
        mood_swings = 0
        for i in range(1, len(valence_values)):
            if abs(valence_values[i] - valence_values[i-1]) > 0.5:
                mood_swings += 1
        
        # Assess regulation effectiveness
        regulation_score = self._assess_emotional_regulation(recent_states)
        
        return {
            'agent_id': agent_id,
            'overall_stability': profile.mood_stability,
            'recent_stability': {
                'valence_stability': valence_stability,
                'arousal_stability': arousal_stability,
                'combined_stability': (valence_stability + arousal_stability) / 2
            },
            'mood_swings': {
                'count': mood_swings,
                'frequency': mood_swings / len(recent_states),
                'severity': 'high' if mood_swings > 5 else 'moderate' if mood_swings > 2 else 'low'
            },
            'emotional_regulation': {
                'score': regulation_score,
                'effectiveness': 'good' if regulation_score > 0.7 else 'moderate' if regulation_score > 0.4 else 'poor'
            },
            'resilience_score': profile.resilience_score,
            'recommendations': self._generate_stability_recommendations(profile, recent_states)
        }
    
    # Helper methods
    def _determine_primary_emotion(self, emotion_data: Dict[str, Any]) -> EmotionalCategory:
        """Determine the primary emotion from emotion data"""
        emotions = emotion_data.get('emotions', {})
        if not emotions:
            return EmotionalCategory.NEUTRAL
        
        # Find emotion with highest intensity
        primary = max(emotions.items(), key=lambda x: x[1])
        
        # Map to EmotionalCategory
        try:
            return EmotionalCategory(primary[0].lower())
        except ValueError:
            return EmotionalCategory.NEUTRAL
    
    def _calculate_stability(self, agent_id: str, emotion_data: Dict[str, Any]) -> float:
        """Calculate emotional stability"""
        if agent_id not in self.state_history:
            return 0.5
        
        recent_states = list(self.state_history[agent_id])[-self.stability_window:]
        if len(recent_states) < 2:
            return 0.5
        
        # Calculate variance in emotional states
        intensities = [s.get_intensity() for s in recent_states]
        valences = [s.valence for s in recent_states]
        
        intensity_variance = np.var(intensities)
        valence_variance = np.var(valences)
        
        # Lower variance = higher stability
        stability = 1.0 - (intensity_variance + valence_variance) / 2
        return max(0, min(1, stability))
    
    def _create_transition(self, from_state: EmotionalState, 
                         to_state: EmotionalState,
                         trigger: Optional[str]) -> EmotionalTransition:
        """Create an emotional transition"""
        transition_time = to_state.timestamp - from_state.timestamp
        
        # Calculate smoothness based on emotional distance and time
        distance = self._calculate_emotional_distance(from_state, to_state)
        time_factor = min(transition_time.total_seconds() / 60, 1.0)  # Normalize to 1 minute
        smoothness = time_factor / (1 + distance)
        
        return EmotionalTransition(
            from_state=from_state,
            to_state=to_state,
            trigger=trigger,
            transition_time=transition_time,
            smoothness=smoothness
        )
    
    def _calculate_emotional_distance(self, state1: EmotionalState, 
                                    state2: EmotionalState) -> float:
        """Calculate distance between two emotional states"""
        # Combine emotion vector, arousal, valence, and dominance
        vector1 = list(state1.emotion_vector.values()) + [state1.arousal, state1.valence, state1.dominance]
        vector2 = list(state2.emotion_vector.values()) + [state2.arousal, state2.valence, state2.dominance]
        
        return np.linalg.norm(np.array(vector1) - np.array(vector2))
    
    def _update_mood_profile(self, agent_id: str) -> None:
        """Update long-term mood profile"""
        states = list(self.state_history[agent_id])
        if len(states) < 10:
            return
        
        # Calculate baseline emotions
        emotion_sums = defaultdict(float)
        emotion_counts = defaultdict(int)
        
        for state in states:
            for emotion, intensity in state.emotion_vector.items():
                emotion_sums[emotion] += intensity
                emotion_counts[emotion] += 1
        
        baseline_emotions = {
            emotion: emotion_sums[emotion] / emotion_counts[emotion]
            for emotion in emotion_sums
        }
        
        # Calculate other profile metrics
        mood_stability = np.mean([s.stability for s in states[-20:]])
        emotional_range = self._calculate_emotional_range(states)
        typical_transitions = self._analyze_typical_transitions(agent_id)
        stress_indicators = self._identify_stress_indicators(states)
        resilience_score = self._calculate_resilience(agent_id)
        
        # Update or create profile
        self.mood_profiles[agent_id] = MoodProfile(
            agent_id=agent_id,
            baseline_emotions=baseline_emotions,
            mood_stability=mood_stability,
            emotional_range=emotional_range,
            typical_transitions=typical_transitions,
            stress_indicators=stress_indicators,
            resilience_score=resilience_score,
            last_updated=datetime.now()
        )
    
    def _check_alerts(self, agent_id: str, state: EmotionalState) -> None:
        """Check for alert conditions"""
        alerts = []
        
        # Check each alert condition
        for alert_name, condition in self.alert_conditions.items():
            if alert_name in ['rapid_cycling', 'prolonged_negative']:
                # These need history
                if condition(self.state_history[agent_id]):
                    alerts.append({
                        'type': alert_name,
                        'agent_id': agent_id,
                        'timestamp': datetime.now(),
                        'severity': 'high'
                    })
            else:
                # These work on current state
                if condition(state):
                    alerts.append({
                        'type': alert_name,
                        'agent_id': agent_id,
                        'timestamp': datetime.now(),
                        'state': state,
                        'severity': 'moderate'
                    })
        
        # Trigger callbacks
        for alert in alerts:
            for callback in self.alert_callbacks:
                callback(alert)
    
    def _run_pattern_detection(self) -> None:
        """Run pattern detection algorithms"""
        current_time = datetime.now()
        
        for pattern_name, detector in self.pattern_detectors.items():
            detected = detector()
            if detected:
                for pattern in detected:
                    pattern.time_range = (pattern.time_range[0], current_time)
                    self.detected_patterns.append(pattern)
    
    def _detect_emotional_contagion(self) -> List[EmotionalPattern]:
        """Detect emotional contagion patterns"""
        patterns = []
        
        # Check if multiple agents share similar emotional states
        if len(self.current_states) < 2:
            return patterns
        
        # Group agents by similar emotional states
        emotion_groups = defaultdict(list)
        for agent_id, state in self.current_states.items():
            emotion_groups[state.primary_emotion].append(agent_id)
        
        # Check for contagion
        for emotion, agents in emotion_groups.items():
            if len(agents) >= 3:  # At least 3 agents
                # Check if this is recent
                timestamps = [self.current_states[a].timestamp for a in agents]
                if all(t > datetime.now() - timedelta(minutes=5) for t in timestamps):
                    patterns.append(EmotionalPattern(
                        pattern_type='emotional_contagion',
                        description=f'Emotional contagion of {emotion.value}',
                        frequency=len(agents) / len(self.current_states),
                        agents_affected=agents,
                        time_range=(min(timestamps), max(timestamps)),
                        confidence=0.8,
                        metadata={'emotion': emotion.value}
                    ))
        
        return patterns
    
    def _detect_mood_cycles(self) -> List[EmotionalPattern]:
        """Detect cyclical mood patterns"""
        patterns = []
        
        # Analyze each agent's history
        for agent_id, states in self.state_history.items():
            if len(states) < 20:
                continue
            
            # Extract valence time series
            valences = [s.valence for s in states]
            
            # Simple cycle detection using autocorrelation
            if len(valences) > 10:
                autocorr = np.correlate(valences, valences, mode='full')
                autocorr = autocorr[len(autocorr)//2:]
                
                # Find peaks in autocorrelation
                peaks = []
                for i in range(1, len(autocorr)-1):
                    if autocorr[i] > autocorr[i-1] and autocorr[i] > autocorr[i+1]:
                        peaks.append(i)
                
                if peaks and peaks[0] > 3:  # Cycle length > 3
                    patterns.append(EmotionalPattern(
                        pattern_type='mood_cycle',
                        description=f'Mood cycle with period ~{peaks[0]} states',
                        frequency=1.0 / peaks[0],
                        agents_affected=[agent_id],
                        time_range=(states[0].timestamp, states[-1].timestamp),
                        confidence=0.6,
                        metadata={'cycle_length': peaks[0]}
                    ))
        
        return patterns
    
    def _detect_stress_patterns(self) -> List[EmotionalPattern]:
        """Detect stress-related patterns"""
        patterns = []
        
        for agent_id, states in self.state_history.items():
            recent_states = list(states)[-10:]
            if len(recent_states) < 5:
                continue
            
            # Stress indicators: high arousal + negative valence
            stress_count = sum(1 for s in recent_states 
                             if s.arousal > 0.7 and s.valence < -0.3)
            
            if stress_count >= len(recent_states) * 0.6:
                patterns.append(EmotionalPattern(
                    pattern_type='chronic_stress',
                    description='Sustained high stress levels',
                    frequency=stress_count / len(recent_states),
                    agents_affected=[agent_id],
                    time_range=(recent_states[0].timestamp, recent_states[-1].timestamp),
                    confidence=0.7,
                    metadata={'stress_level': 'high'}
                ))
        
        return patterns
    
    def _detect_regulation_patterns(self) -> List[EmotionalPattern]:
        """Detect emotional regulation patterns"""
        patterns = []
        
        for agent_id in self.current_states:
            transitions = [t for t in self.transition_history 
                         if t.to_state.agent_id == agent_id][-10:]
            
            if len(transitions) < 3:
                continue
            
            # Look for successful regulation (negative to neutral/positive)
            regulation_success = 0
            for trans in transitions:
                if trans.from_state.valence < -0.3 and trans.to_state.valence > 0:
                    regulation_success += 1
            
            if regulation_success >= 3:
                patterns.append(EmotionalPattern(
                    pattern_type='effective_regulation',
                    description='Successful emotional regulation',
                    frequency=regulation_success / len(transitions),
                    agents_affected=[agent_id],
                    time_range=(transitions[0].from_state.timestamp, 
                              transitions[-1].to_state.timestamp),
                    confidence=0.75,
                    metadata={'success_rate': regulation_success / len(transitions)}
                ))
        
        return patterns
    
    def _check_rapid_cycling(self, history: deque) -> bool:
        """Check for rapid emotional cycling"""
        if len(history) < 5:
            return False
        
        recent = list(history)[-5:]
        transitions = 0
        
        for i in range(1, len(recent)):
            if recent[i].primary_emotion != recent[i-1].primary_emotion:
                transitions += 1
        
        return transitions >= 4  # 4+ transitions in 5 states
    
    def _check_prolonged_negative(self, history: deque) -> bool:
        """Check for prolonged negative emotional state"""
        if len(history) < 10:
            return False
        
        recent = list(history)[-10:]
        negative_count = sum(1 for s in recent if s.valence < -0.3)
        
        return negative_count >= 8  # 80% negative
    
    def _get_current_emotional_description(self, agent_id: str) -> str:
        """Get human-readable description of current emotional state"""
        if agent_id not in self.current_states:
            return "Unknown"
        
        state = self.current_states[agent_id]
        
        # Describe intensity
        intensity = state.get_intensity()
        if intensity < 0.3:
            intensity_desc = "mildly"
        elif intensity < 0.7:
            intensity_desc = "moderately"
        else:
            intensity_desc = "intensely"
        
        # Describe valence
        if state.valence > 0.3:
            valence_desc = "positive"
        elif state.valence < -0.3:
            valence_desc = "negative"
        else:
            valence_desc = "neutral"
        
        return f"{intensity_desc} {state.primary_emotion.value} ({valence_desc})"
    
    def _calculate_emotional_synchrony(self, agent_ids: List[str]) -> float:
        """Calculate how synchronized emotions are across agents"""
        if len(agent_ids) < 2:
            return 0.0
        
        # Get current emotion vectors
        emotion_vectors = []
        for agent_id in agent_ids:
            if agent_id in self.current_states:
                state = self.current_states[agent_id]
                vector = list(state.emotion_vector.values())
                emotion_vectors.append(vector)
        
        if len(emotion_vectors) < 2:
            return 0.0
        
        # Calculate pairwise correlations
        correlations = []
        for i in range(len(emotion_vectors)):
            for j in range(i + 1, len(emotion_vectors)):
                corr = np.corrcoef(emotion_vectors[i], emotion_vectors[j])[0, 1]
                if not np.isnan(corr):
                    correlations.append(corr)
        
        return np.mean(correlations) if correlations else 0.0
    
    def _calculate_emotional_diversity(self, current_emotions: Dict[str, Dict[str, Any]]) -> float:
        """Calculate diversity of emotional states"""
        if not current_emotions:
            return 0.0
        
        # Count unique primary emotions
        primary_emotions = [e['primary'] for e in current_emotions.values()]
        unique_emotions = len(set(primary_emotions))
        
        # Normalize by number of agents
        diversity = unique_emotions / len(current_emotions)
        
        return diversity
    
    def _identify_emotional_clusters(self, agent_ids: List[str]) -> List[Dict[str, Any]]:
        """Identify clusters of agents with similar emotional states"""
        clusters = []
        
        if len(agent_ids) < 2:
            return clusters
        
        # Group by primary emotion
        emotion_groups = defaultdict(list)
        for agent_id in agent_ids:
            if agent_id in self.current_states:
                state = self.current_states[agent_id]
                emotion_groups[state.primary_emotion.value].append(agent_id)
        
        # Create clusters
        for emotion, agents in emotion_groups.items():
            if len(agents) >= 2:
                clusters.append({
                    'emotion': emotion,
                    'agents': agents,
                    'size': len(agents),
                    'percentage': len(agents) / len(agent_ids)
                })
        
        return sorted(clusters, key=lambda x: x['size'], reverse=True)
    
    def _determine_collective_mood(self, avg_valence: float) -> str:
        """Determine collective mood from average valence"""
        if avg_valence > 0.5:
            return 'very_positive'
        elif avg_valence > 0.2:
            return 'positive'
        elif avg_valence > -0.2:
            return 'neutral'
        elif avg_valence > -0.5:
            return 'negative'
        else:
            return 'very_negative'
    
    def _get_active_alerts(self, agent_ids: List[str]) -> List[Dict[str, Any]]:
        """Get currently active alerts for specified agents"""
        active_alerts = []
        
        for agent_id in agent_ids:
            if agent_id in self.current_states:
                state = self.current_states[agent_id]
                
                # Check alert conditions
                if state.get_intensity() > 0.9:
                    active_alerts.append({
                        'agent_id': agent_id,
                        'type': 'extreme_emotion',
                        'details': f'Extreme {state.primary_emotion.value}'
                    })
                
                if state.get_intensity() < 0.1:
                    active_alerts.append({
                        'agent_id': agent_id,
                        'type': 'emotional_flatness',
                        'details': 'Very low emotional intensity'
                    })
        
        return active_alerts
    
    def _calculate_emotional_range(self, states: List[EmotionalState]) -> float:
        """Calculate emotional range from state history"""
        if not states:
            return 0.0
        
        # Get range of valence and arousal
        valences = [s.valence for s in states]
        arousals = [s.arousal for s in states]
        
        valence_range = max(valences) - min(valences)
        arousal_range = max(arousals) - min(arousals)
        
        # Average range
        return (valence_range + arousal_range) / 2
    
    def _analyze_typical_transitions(self, agent_id: str) -> List[Tuple[str, str, float]]:
        """Analyze typical emotional transitions for an agent"""
        transitions = [t for t in self.transition_history 
                      if t.to_state.agent_id == agent_id]
        
        if not transitions:
            return []
        
        # Count transition patterns
        transition_counts = defaultdict(int)
        for trans in transitions:
            key = (trans.from_state.primary_emotion.value, 
                   trans.to_state.primary_emotion.value)
            transition_counts[key] += 1
        
        # Calculate probabilities
        total = len(transitions)
        typical_transitions = [
            (from_e, to_e, count / total)
            for (from_e, to_e), count in transition_counts.items()
        ]
        
        # Sort by probability
        return sorted(typical_transitions, key=lambda x: x[2], reverse=True)[:5]
    
    def _identify_stress_indicators(self, states: List[EmotionalState]) -> Dict[str, float]:
        """Identify stress indicators from emotional states"""
        if not states:
            return {}
        
        indicators = {
            'high_arousal_frequency': sum(1 for s in states if s.arousal > 0.7) / len(states),
            'negative_valence_frequency': sum(1 for s in states if s.valence < -0.3) / len(states),
            'emotional_instability': 1.0 - np.mean([s.stability for s in states]),
            'fear_frequency': sum(1 for s in states if s.primary_emotion == EmotionalCategory.FEAR) / len(states),
            'anger_frequency': sum(1 for s in states if s.primary_emotion == EmotionalCategory.ANGER) / len(states)
        }
        
        return indicators
    
    def _calculate_resilience(self, agent_id: str) -> float:
        """Calculate emotional resilience score"""
        transitions = [t for t in self.transition_history 
                      if t.to_state.agent_id == agent_id]
        
        if not transitions:
            return 0.5
        
        # Count recovery from negative states
        recoveries = 0
        negative_states = 0
        
        for trans in transitions:
            if trans.from_state.valence < -0.3:
                negative_states += 1
                if trans.to_state.valence > 0:
                    recoveries += 1
        
        if negative_states == 0:
            return 0.8  # No negative states to recover from
        
        recovery_rate = recoveries / negative_states
        
        # Also consider recovery speed
        recovery_times = []
        for trans in transitions:
            if trans.from_state.valence < -0.3 and trans.to_state.valence > 0:
                recovery_times.append(trans.transition_time.total_seconds())
        
        if recovery_times:
            avg_recovery_time = np.mean(recovery_times)
            # Faster recovery = higher resilience
            speed_factor = 1.0 / (1.0 + avg_recovery_time / 300)  # Normalize to 5 minutes
        else:
            speed_factor = 0.5
        
        return recovery_rate * 0.6 + speed_factor * 0.4
    
    def _identify_emotional_phases(self, states: List[EmotionalState]) -> List[Dict[str, Any]]:
        """Identify distinct emotional phases in trajectory"""
        if len(states) < 3:
            return []
        
        phases = []
        current_phase = {
            'start_index': 0,
            'dominant_emotion': states[0].primary_emotion.value,
            'avg_valence': states[0].valence,
            'states': [states[0]]
        }
        
        for i in range(1, len(states)):
            # Check if emotion changed significantly
            if (states[i].primary_emotion != states[i-1].primary_emotion or 
                abs(states[i].valence - current_phase['avg_valence']) > 0.5):
                
                # End current phase
                current_phase['end_index'] = i - 1
                current_phase['duration'] = len(current_phase['states'])
                phases.append(current_phase)
                
                # Start new phase
                current_phase = {
                    'start_index': i,
                    'dominant_emotion': states[i].primary_emotion.value,
                    'avg_valence': states[i].valence,
                    'states': [states[i]]
                }
            else:
                current_phase['states'].append(states[i])
                current_phase['avg_valence'] = np.mean([s.valence for s in current_phase['states']])
        
        # Add final phase
        current_phase['end_index'] = len(states) - 1
        current_phase['duration'] = len(current_phase['states'])
        phases.append(current_phase)
        
        # Clean up states from phases to avoid circular reference
        for phase in phases:
            del phase['states']
        
        return phases
    
    def _predict_next_emotional_state(self, agent_id: str) -> Optional[Dict[str, Any]]:
        """Predict next likely emotional state"""
        if agent_id not in self.mood_profiles or agent_id not in self.current_states:
            return None
        
        profile = self.mood_profiles[agent_id]
        current = self.current_states[agent_id]
        
        # Find most likely transition
        likely_transitions = [
            (to_e, prob) for from_e, to_e, prob in profile.typical_transitions
            if from_e == current.primary_emotion.value
        ]
        
        if likely_transitions:
            likely_transitions.sort(key=lambda x: x[1], reverse=True)
            next_emotion = likely_transitions[0][0]
            confidence = likely_transitions[0][1]
        else:
            # Default to baseline
            next_emotion = max(profile.baseline_emotions.items(), 
                             key=lambda x: x[1])[0]
            confidence = 0.5
        
        return {
            'predicted_emotion': next_emotion,
            'confidence': confidence,
            'expected_valence': profile.baseline_emotions.get(next_emotion, 0.0)
        }
    
    def _assess_emotional_regulation(self, states: List[EmotionalState]) -> float:
        """Assess emotional regulation effectiveness"""
        if len(states) < 3:
            return 0.5
        
        regulation_events = 0
        opportunities = 0
        
        for i in range(1, len(states)):
            # Look for high intensity negative emotions
            if states[i-1].valence < -0.3 and states[i-1].get_intensity() > 0.6:
                opportunities += 1
                
                # Check if regulated in next state
                if (states[i].valence > states[i-1].valence + 0.2 or 
                    states[i].get_intensity() < states[i-1].get_intensity() - 0.2):
                    regulation_events += 1
        
        if opportunities == 0:
            return 0.7  # No regulation needed
        
        return regulation_events / opportunities
    
    def _generate_stability_recommendations(self, profile: MoodProfile, 
                                          recent_states: List[EmotionalState]) -> List[str]:
        """Generate recommendations for improving emotional stability"""
        recommendations = []
        
        # Check stability score
        if profile.mood_stability < 0.4:
            recommendations.append("Practice emotional regulation techniques")
            recommendations.append("Establish consistent routines to stabilize mood")
        
        # Check stress indicators
        high_stress = [k for k, v in profile.stress_indicators.items() if v > 0.6]
        if high_stress:
            recommendations.append("Implement stress reduction strategies")
            recommendations.append("Consider mindfulness or relaxation practices")
        
        # Check emotional range
        if profile.emotional_range > 0.8:
            recommendations.append("Work on emotional grounding techniques")
        elif profile.emotional_range < 0.2:
            recommendations.append("Explore safe ways to experience broader emotions")
        
        # Check resilience
        if profile.resilience_score < 0.4:
            recommendations.append("Build emotional resilience through gradual exposure")
            recommendations.append("Develop coping strategies for negative emotions")
        
        return recommendations
