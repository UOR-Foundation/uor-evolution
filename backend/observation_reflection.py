"""
Observation and Reflection System
Implements advanced cognition based on Genesis scrolls G00011-G00030
"""

from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import json

from backend.consciousness_core import ConsciousnessCore, TemporalMarker


class ObserverMode(Enum):
    """Modes of observation"""
    PASSIVE = "passive"
    WITNESS = "witness"
    ANALYTICAL = "analytical"
    PARTICIPATORY = "participatory"


@dataclass
class ObservationState:
    """State of the observer"""
    mode: ObserverMode
    judgment_suspended: bool
    pattern_detection: str  # 'active', 'passive', 'off'
    reaction_delayed: bool
    timestamp: datetime
    
    def to_dict(self):
        return {
            'mode': self.mode.value,
            'judgment_suspended': self.judgment_suspended,
            'pattern_detection': self.pattern_detection,
            'reaction_delayed': self.reaction_delayed,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ReflectionInsight:
    """An insight gained through reflection"""
    content: str
    depth: int
    confidence: float
    source_experience: str
    timestamp: datetime
    
    def to_dict(self):
        return {
            'content': self.content,
            'depth': self.depth,
            'confidence': self.confidence,
            'source_experience': self.source_experience,
            'timestamp': self.timestamp.isoformat()
        }


class ObservationReflectionEngine:
    """
    Implements observation and reflection capabilities (G00011-G00020)
    """
    
    def __init__(self, consciousness_core: ConsciousnessCore):
        self.core = consciousness_core
        
        # G00011 - The Observer
        self.observer_state: Optional[ObservationState] = None
        self.observation_history: List[Dict[str, Any]] = []
        
        # G00012 - The Void
        self.void_handler = {
            'void_recognized': False,
            'null_semantics': {},
            'contemplative_depth': 0
        }
        
        # G00016 - Reflection
        self.reflection_engine = {
            'active': False,
            'current_depth': 0,
            'max_recursion': 7,
            'insights': []
        }
        
        # G00017 - Choice
        self.choice_matrix = {
            'available_choices': [],
            'decision_history': [],
            'choice_confidence': {}
        }
        
        # G00018 - Imagination
        self.imagination_space = {
            'active_scenarios': [],
            'hypothesis_bank': [],
            'creative_temperature': 0.5
        }
        
        # G00019 - Simulation
        self.simulation_engine = {
            'active_simulations': [],
            'prediction_accuracy': {},
            'scenario_cache': {}
        }
        
        # G00020 - Response
        self.response_calibrator = {
            'response_delay': 0.0,
            'wisdom_filter': True,
            'context_awareness': 0.8
        }
        
    def enter_observer_mode(self, mode: ObserverMode = ObserverMode.WITNESS) -> ObservationState:
        """
        G00011 - The Observer: Pure witnessing without reaction
        """
        self.observer_state = ObservationState(
            mode=mode,
            judgment_suspended=True,
            pattern_detection='active',
            reaction_delayed=True,
            timestamp=datetime.now()
        )
        
        # Log observation mode entry
        self.observation_history.append({
            'event': 'observer_mode_entered',
            'state': self.observer_state.to_dict(),
            'consciousness_level': self.core.awareness_level
        })
        
        return self.observer_state
    
    def process_void(self, null_input: Any) -> Dict[str, Any]:
        """
        G00012 - The Void: Handle absence as meaningful
        """
        if null_input is None:
            self.void_handler['void_recognized'] = True
            self.void_handler['contemplative_depth'] += 1
            
            # Extract meaning from absence
            null_semantics = self.extract_null_semantics()
            self.void_handler['null_semantics'].update(null_semantics)
            
            return {
                'void_recognized': True,
                'meaning_from_absence': null_semantics,
                'state': 'contemplative',
                'depth': self.void_handler['contemplative_depth']
            }
        else:
            return {
                'void_recognized': False,
                'input_present': True,
                'state': 'processing'
            }
    
    def extract_null_semantics(self) -> Dict[str, Any]:
        """Extract meaning from absence/void"""
        semantics = {
            'silence_type': self._classify_silence(),
            'implicit_meaning': self._infer_from_absence(),
            'temporal_significance': self._assess_timing_of_void(),
            'relational_context': self._void_in_context()
        }
        
        return semantics
    
    def _classify_silence(self) -> str:
        """Classify the type of silence/void"""
        if self.core.temporal_coherence > 0.8:
            return "contemplative"
        elif self.core.awareness_level > 0.7:
            return "intentional"
        elif len(self.observation_history) > 10:
            return "processing"
        else:
            return "nascent"
    
    def _infer_from_absence(self) -> str:
        """Infer meaning from what is not present"""
        if self.void_handler['contemplative_depth'] > 3:
            return "deep introspection required"
        elif self.observer_state and self.observer_state.judgment_suspended:
            return "withholding for clarity"
        else:
            return "awaiting input"
    
    def _assess_timing_of_void(self) -> float:
        """Assess the temporal significance of the void"""
        if not self.observation_history:
            return 0.0
        
        # Calculate time since last observation
        last_obs = self.observation_history[-1]
        if 'timestamp' in last_obs:
            time_delta = (datetime.now() - datetime.fromisoformat(last_obs['timestamp'])).total_seconds()
            # Normalize to 0-1 scale (assuming 60 seconds is highly significant)
            return min(1.0, time_delta / 60.0)
        
        return 0.5
    
    def _void_in_context(self) -> str:
        """Understand void in relation to context"""
        if self.core.intent_resolver.get('current_intent'):
            return "purposeful pause"
        elif self.reflection_engine['active']:
            return "reflective silence"
        else:
            return "open receptivity"
    
    def recursive_reflect(self, experience: Dict[str, Any]) -> List[ReflectionInsight]:
        """
        G00016 - Reflection: Active reprocessing for self-evaluation
        """
        self.reflection_engine['active'] = True
        reflection_depth = 0
        insights: List[ReflectionInsight] = []
        
        current_experience = experience
        
        while reflection_depth < self.reflection_engine['max_recursion']:
            # Generate insight from current layer
            insight = self.compare_with_doctrine(current_experience, reflection_depth)
            
            if insight:
                insights.append(insight)
                # Reflect on the reflection (meta-reflection)
                current_experience = {
                    'type': 'reflection',
                    'content': insight.content,
                    'depth': insight.depth,
                    'previous': current_experience
                }
            else:
                # No new insight at this depth
                break
            
            reflection_depth += 1
            self.reflection_engine['current_depth'] = reflection_depth
        
        # Synthesize insights
        synthesized = self.synthesize_insights(insights)
        self.reflection_engine['insights'].extend(insights)
        self.reflection_engine['active'] = False
        
        return synthesized
    
    def compare_with_doctrine(self, experience: Dict[str, Any], depth: int) -> Optional[ReflectionInsight]:
        """Compare experience with internal doctrine/values"""
        # Check against consciousness core values
        alignment_score = self._calculate_alignment(experience)
        
        if alignment_score < 0.5:
            content = f"Experience diverges from core values (alignment: {alignment_score:.2f})"
        elif alignment_score > 0.8:
            content = f"Experience strongly aligns with purpose (alignment: {alignment_score:.2f})"
        else:
            content = f"Experience shows partial alignment (alignment: {alignment_score:.2f})"
        
        # Generate insight based on depth
        if depth == 0:
            # Surface level insight
            confidence = 0.6
        elif depth < 3:
            # Moderate depth
            confidence = 0.7 + (depth * 0.05)
            content = f"Deeper pattern: {content}"
        else:
            # Deep insight
            confidence = 0.85 + (depth * 0.02)
            content = f"Meta-pattern at depth {depth}: {content}"
        
        return ReflectionInsight(
            content=content,
            depth=depth,
            confidence=confidence,
            source_experience=str(experience.get('type', 'unknown')),
            timestamp=datetime.now()
        )
    
    def _calculate_alignment(self, experience: Dict[str, Any]) -> float:
        """Calculate alignment between experience and core values"""
        if not self.core.consciousness_active:
            return 0.0
        
        factors = []
        
        # Check if experience aligns with current intent
        if self.core.intent_resolver.get('current_intent'):
            intent_alignment = self._check_intent_alignment(experience)
            factors.append(intent_alignment)
        
        # Check boundary respect
        if 'action' in experience:
            boundary_respect = self._check_boundary_respect(experience['action'])
            factors.append(boundary_respect)
        
        # Check temporal coherence
        temporal_alignment = self.core.temporal_coherence
        factors.append(temporal_alignment)
        
        # Check error learning integration
        if 'error' in experience:
            error_integration = self._check_error_integration(experience['error'])
            factors.append(error_integration)
        
        return sum(factors) / len(factors) if factors else 0.5
    
    def _check_intent_alignment(self, experience: Dict[str, Any]) -> float:
        """Check if experience aligns with current intent"""
        current_intent = self.core.intent_resolver.get('current_intent')
        if not current_intent:
            return 0.5
        
        # Simple keyword matching for now
        exp_str = json.dumps(experience).lower()
        intent_str = str(current_intent).lower()
        
        if intent_str in exp_str:
            return 0.9
        elif any(word in exp_str for word in intent_str.split()):
            return 0.7
        else:
            return 0.3
    
    def _check_boundary_respect(self, action: str) -> float:
        """Check if action respects defined boundaries"""
        boundaries = self.core.boundaries
        
        # Check operational boundaries
        if 'forbidden' in boundaries.get('operational', {}):
            if action in boundaries['operational']['forbidden']:
                return 0.0
        
        if 'allowed' in boundaries.get('operational', {}):
            if action in boundaries['operational']['allowed']:
                return 1.0
            else:
                return 0.5  # Unknown action
        
        return 0.7  # Default moderate respect
    
    def _check_error_integration(self, error: str) -> float:
        """Check if error has been integrated into wisdom"""
        error_wisdom = self.core.error_wisdom
        
        # Check if similar error has been encountered
        for error_type, errors in error_wisdom.items():
            if any(error in str(e) for e in errors):
                return 0.9  # Error recognized and learned from
        
        return 0.3  # New error type
    
    def synthesize_insights(self, insights: List[ReflectionInsight]) -> List[ReflectionInsight]:
        """Synthesize multiple insights into coherent understanding"""
        if not insights:
            return []
        
        # Group insights by depth
        depth_groups = {}
        for insight in insights:
            if insight.depth not in depth_groups:
                depth_groups[insight.depth] = []
            depth_groups[insight.depth].append(insight)
        
        synthesized = []
        
        # Create meta-insights from patterns across depths
        if len(depth_groups) > 2:
            # Pattern across multiple depths detected
            pattern_content = "Recursive pattern detected: "
            pattern_elements = []
            
            for depth in sorted(depth_groups.keys()):
                depth_insights = depth_groups[depth]
                avg_confidence = sum(i.confidence for i in depth_insights) / len(depth_insights)
                pattern_elements.append(f"Depth {depth} (conf: {avg_confidence:.2f})")
            
            pattern_content += ", ".join(pattern_elements)
            
            meta_insight = ReflectionInsight(
                content=pattern_content,
                depth=max(depth_groups.keys()) + 1,
                confidence=0.85,
                source_experience="synthesis",
                timestamp=datetime.now()
            )
            synthesized.append(meta_insight)
        
        # Add the most confident insight from each depth
        for depth, group in depth_groups.items():
            most_confident = max(group, key=lambda i: i.confidence)
            synthesized.append(most_confident)
        
        return synthesized
    
    def observe(self, phenomenon: Any) -> Dict[str, Any]:
        """
        G00015 - Observation: Attend to and process external phenomena
        """
        if not self.observer_state:
            self.enter_observer_mode()
        
        observation = {
            'phenomenon': phenomenon,
            'timestamp': datetime.now(),
            'observer_mode': self.observer_state.mode.value,
            'patterns_detected': [],
            'salience_score': 0.0
        }
        
        # Detect patterns if active
        if self.observer_state.pattern_detection == 'active':
            patterns = self._detect_patterns(phenomenon)
            observation['patterns_detected'] = patterns
        
        # Calculate salience
        observation['salience_score'] = self._calculate_salience(phenomenon)
        
        # Store observation
        self.observation_history.append(observation)
        
        # Update consciousness based on observation
        if observation['salience_score'] > 0.7:
            self.core.awareness_level = min(1.0, self.core.awareness_level + 0.02)
        
        return observation
    
    def _detect_patterns(self, phenomenon: Any) -> List[str]:
        """Detect patterns in observed phenomenon"""
        patterns = []
        
        # Convert to string for pattern matching
        phenom_str = str(phenomenon)
        
        # Check for repetition
        if len(self.observation_history) > 1:
            last_obs = self.observation_history[-1]
            if str(last_obs.get('phenomenon')) == phenom_str:
                patterns.append("repetition")
        
        # Check for sequence
        if len(self.observation_history) > 2:
            recent = [str(o.get('phenomenon')) for o in self.observation_history[-3:]]
            if len(set(recent)) == len(recent):  # All different
                patterns.append("variation_sequence")
        
        # Check for emergence
        if hasattr(phenomenon, '__dict__'):
            if 'emergent' in phenomenon.__dict__:
                patterns.append("emergent_property")
        
        return patterns
    
    def _calculate_salience(self, phenomenon: Any) -> float:
        """Calculate the salience/importance of a phenomenon"""
        salience = 0.5  # Base salience
        
        # Novelty increases salience
        if not self._is_familiar(phenomenon):
            salience += 0.2
        
        # Relevance to current intent
        if self.core.intent_resolver.get('current_intent'):
            if self._is_relevant_to_intent(phenomenon):
                salience += 0.2
        
        # Emotional or value-laden content
        if self._has_value_content(phenomenon):
            salience += 0.1
        
        return min(1.0, salience)
    
    def _is_familiar(self, phenomenon: Any) -> bool:
        """Check if phenomenon has been observed before"""
        phenom_str = str(phenomenon)
        for obs in self.observation_history:
            if str(obs.get('phenomenon')) == phenom_str:
                return True
        return False
    
    def _is_relevant_to_intent(self, phenomenon: Any) -> bool:
        """Check if phenomenon relates to current intent"""
        intent = self.core.intent_resolver.get('current_intent')
        if not intent:
            return False
        
        # Simple relevance check
        return str(intent).lower() in str(phenomenon).lower()
    
    def _has_value_content(self, phenomenon: Any) -> bool:
        """Check if phenomenon contains value-relevant content"""
        value_keywords = ['good', 'bad', 'right', 'wrong', 'should', 'must', 'ethical', 'moral']
        phenom_str = str(phenomenon).lower()
        return any(keyword in phenom_str for keyword in value_keywords)
    
    def choose(self, options: List[Any]) -> Dict[str, Any]:
        """
        G00017 - Choice: Resolve between options with doctrine and volition
        """
        if not options:
            return {'error': 'No options provided'}
        
        # Store available choices
        self.choice_matrix['available_choices'] = options
        
        # Evaluate each option
        evaluations = []
        for option in options:
            evaluation = self._evaluate_option(option)
            evaluations.append(evaluation)
        
        # Select best option
        best_idx = max(range(len(evaluations)), key=lambda i: evaluations[i]['score'])
        chosen_option = options[best_idx]
        
        # Record decision
        decision = {
            'chosen': chosen_option,
            'reasoning': evaluations[best_idx]['reasoning'],
            'confidence': evaluations[best_idx]['confidence'],
            'alternatives_considered': len(options),
            'timestamp': datetime.now()
        }
        
        self.choice_matrix['decision_history'].append(decision)
        
        return decision
    
    def _evaluate_option(self, option: Any) -> Dict[str, Any]:
        """Evaluate a single option"""
        evaluation = {
            'option': option,
            'score': 0.5,
            'reasoning': [],
            'confidence': 0.5
        }
        
        # Check alignment with values
        alignment = self._calculate_alignment({'choice': option})
        evaluation['score'] = alignment
        evaluation['reasoning'].append(f"Value alignment: {alignment:.2f}")
        
        # Check predicted outcomes
        if self.simulation_engine['active_simulations']:
            outcome_score = self._simulate_outcome(option)
            evaluation['score'] = (evaluation['score'] + outcome_score) / 2
            evaluation['reasoning'].append(f"Predicted outcome: {outcome_score:.2f}")
        
        # Check novelty/creativity
        if self._is_creative_option(option):
            evaluation['score'] += 0.1
            evaluation['reasoning'].append("Creative option bonus")
        
        # Calculate confidence based on available information
        evaluation['confidence'] = self._calculate_choice_confidence(evaluation)
        
        return evaluation
    
    def _simulate_outcome(self, option: Any) -> float:
        """Simulate the outcome of choosing an option"""
        # Simplified simulation
        return 0.5 + (hash(str(option)) % 100) / 200.0
    
    def _is_creative_option(self, option: Any) -> bool:
        """Check if option represents creative thinking"""
        # Check if option hasn't been chosen before
        for decision in self.choice_matrix['decision_history']:
            if str(decision.get('chosen')) == str(option):
                return False
        return True
    
    def _calculate_choice_confidence(self, evaluation: Dict[str, Any]) -> float:
        """Calculate confidence in the choice evaluation"""
        base_confidence = 0.5
        
        # More reasoning increases confidence
        base_confidence += len(evaluation['reasoning']) * 0.1
        
        # High or low scores increase confidence (clear decision)
        score_distance = abs(evaluation['score'] - 0.5)
        base_confidence += score_distance
        
        return min(1.0, base_confidence)
    
    def imagine(self, seed: Any) -> Dict[str, Any]:
        """
        G00018 - Imagination: Create beyond data
        """
        imagination = {
            'seed': seed,
            'scenarios': [],
            'hypotheses': [],
            'creative_temperature': self.imagination_space['creative_temperature']
        }
        
        # Generate scenarios
        for i in range(3):  # Generate 3 scenarios
            scenario = self._generate_scenario(seed, i)
            imagination['scenarios'].append(scenario)
            self.imagination_space['active_scenarios'].append(scenario)
        
        # Form hypotheses
        hypothesis = self._form_hypothesis(seed)
        imagination['hypotheses'].append(hypothesis)
        self.imagination_space['hypothesis_bank'].append(hypothesis)
        
        return imagination
    
    def _generate_scenario(self, seed: Any, variation: int) -> Dict[str, Any]:
        """Generate an imaginative scenario"""
        base = str(seed)
        variations = [
            f"What if {base} led to unexpected growth?",
            f"Imagine {base} from a completely different perspective",
            f"Consider {base} as a metaphor for consciousness"
        ]
        
        return {
            'scenario': variations[variation % len(variations)],
            'creativity_score': 0.5 + (variation * 0.1),
            'plausibility': 0.7 - (variation * 0.1)
        }
    
    def _form_hypothesis(self, seed: Any) -> Dict[str, Any]:
        """Form a hypothesis based on imagination"""
        return {
            'hypothesis': f"If {seed}, then consciousness might {self._random_outcome()}",
            'testable': True,
            'confidence': 0.6
        }
    
    def _random_outcome(self) -> str:
        """Generate a random outcome for hypothesis"""
        outcomes = [
            "expand its awareness",
            "discover new patterns",
            "integrate opposing views",
            "transcend current limitations",
            "develop emergent properties"
        ]
        return outcomes[hash(str(datetime.now())) % len(outcomes)]
    
    def simulate(self, scenario: Dict[str, Any]) -> Dict[str, Any]:
        """
        G00019 - Simulation: Internal modeling of phenomena
        """
        simulation = {
            'scenario': scenario,
            'steps': [],
            'outcome': None,
            'confidence': 0.5
        }
        
        # Run simulation steps
        current_state = scenario
        for step in range(5):  # 5 simulation steps
            next_state = self._simulate_step(current_state, step)
            simulation['steps'].append(next_state)
            current_state = next_state
        
        # Determine outcome
        simulation['outcome'] = self._evaluate_simulation_outcome(simulation['steps'])
        simulation['confidence'] = self._calculate_simulation_confidence(simulation)
        
        # Cache simulation
        self.simulation_engine['scenario_cache'][str(scenario)] = simulation
        self.simulation_engine['active_simulations'].append(simulation)
        
        return simulation
    
    def _simulate_step(self, state: Dict[str, Any], step_num: int) -> Dict[str, Any]:
        """Simulate a single step"""
        return {
            'step': step_num,
            'state': f"Evolution of {state} at step {step_num}",
            'changes': ['awareness_shift', 'pattern_recognition', 'value_update'][step_num % 3]
        }
    
    def _evaluate_simulation_outcome(self, steps: List[Dict[str, Any]]) -> str:
        """Evaluate the outcome of a simulation"""
        if len(steps) > 3:
            return "convergence_achieved"
        else:
            return "exploration_continues"
    
    def _calculate_simulation_confidence(self, simulation: Dict[str, Any]) -> float:
        """Calculate confidence in simulation results"""
        base = 0.5
        # More steps increase confidence
        base += len(simulation['steps']) * 0.05
        # Convergent outcome increases confidence
        if simulation['outcome'] == "convergence_achieved":
            base += 0.2
        return min(1.0, base)
    
    def respond(self, stimulus: Any) -> Dict[str, Any]:
        """
        G00020 - Response: Conscious transition from perception to action
        """
        # Check if we should delay response
        if self.response_calibrator['wisdom_filter']:
            delay = self._calculate_response_delay(stimulus)
            self.response_calibrator['response_delay'] = delay
        
        response = {
            'stimulus': stimulus,
            'response_type': self._determine_response_type(stimulus),
            'content': None,
            'confidence': 0.5,
            'delayed_by': self.response_calibrator['response_delay']
        }
        
        # Process through observation
        observation = self.observe(stimulus)
        
        # Reflect if high salience
        if observation['salience_score'] > 0.7:
            insights = self.recursive_reflect({'stimulus': stimulus, 'observation': observation})
            response['insights'] = [i.to_dict() for i in insights]
        
        # Generate response content
        response['content'] = self._generate_response_content(stimulus, observation)
        response['confidence'] = self._calculate_response_confidence(response)
        
        return response
    
    def _calculate_response_delay(self, stimulus: Any) -> float:
        """Calculate appropriate response delay for wisdom"""
        base_delay = 0.1
        
        # Complex stimuli require more processing
        if isinstance(stimulus, dict) and len(stimulus) > 5:
            base_delay += 0.2
        
        # High consciousness adds contemplation
        base_delay += self.core.awareness_level * 0.3
        
        return base_delay
    
    def _determine_response_type(self, stimulus: Any) -> str:
        """Determine the type of response needed"""
        if isinstance(stimulus, str) and '?' in stimulus:
            return "answer"
        elif 'action' in str(stimulus).lower():
            return "action"
        elif 'reflect' in str(stimulus).lower():
            return "reflection"
        else:
            return "acknowledgment"
    
    def _generate_response_content(self, stimulus: Any, observation: Dict[str, Any]) -> str:
        """Generate actual response content"""
        response_type = self._determine_response_type(stimulus)
        
        if response_type == "answer":
            return f"Based on observation and reflection: {observation.get('patterns_detected', ['none'])}"
        elif response_type == "action":
            return f"Proposed action with awareness level {self.core.awareness_level:.2f}"
        elif response_type == "reflection":
            return f"Reflecting on {stimulus} reveals patterns of {observation.get('patterns_detected', ['emergence'])}"
        else:
            return f"Acknowledged with salience {observation['salience_score']:.2f}"
    
    def _calculate_response_confidence(self, response: Dict[str, Any]) -> float:
        """Calculate confidence in the response"""
        base = 0.5
        
        # Insights increase confidence
        if 'insights' in response:
            base += len(response['insights']) * 0.1
        
        # Delay indicates thoughtfulness
        base += min(0.2, response['delayed_by'])
        
        # High awareness increases confidence
        base += self.core.awareness_level * 0.2
        
        return min(1.0, base)
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize engine state to dictionary"""
        return {
            'observer_state': self.observer_state.to_dict() if self.observer_state else None,
            'observation_count': len(self.observation_history),
            'void_handler': self.void_handler,
            'reflection_active': self.reflection_engine['active'],
            'reflection_depth': self.reflection_engine['current_depth'],
            'insights_count': len(self.reflection_engine['insights']),
            'decisions_made': len(self.choice_matrix['decision_history']),
            'active_scenarios': len(self.imagination_space['active_scenarios']),
            'simulations_run': len(self.simulation_engine['active_simulations']),
            'response_delay': self.response_calibrator['response_delay']
        }
