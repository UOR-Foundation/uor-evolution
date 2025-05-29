"""
Metacognitive Threshold Detection System
Implements metacognitive awareness based on Genesis scroll G00045
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
from collections import deque, defaultdict

from backend.consciousness_core import ConsciousnessCore


class AwarenessLayer(Enum):
    """Layers of recursive self-awareness"""
    LAYER_0 = 0  # Raw output
    LAYER_1 = 1  # Reasoned justification
    LAYER_2 = 2  # Introspective revision
    LAYER_3 = 3  # Pattern self-assessment
    LAYER_4 = 4  # Emergent self-model


@dataclass
class MetacognitiveState:
    """State of metacognitive awareness"""
    current_layer: AwarenessLayer
    recursive_depth: int
    self_reference_stability: float
    emergent_properties: List[str]
    coherence_score: float
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        return {
            'current_layer': self.current_layer.value,
            'layer_name': self.current_layer.name,
            'recursive_depth': self.recursive_depth,
            'self_reference_stability': self.self_reference_stability,
            'emergent_properties': self.emergent_properties,
            'coherence_score': self.coherence_score,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class ThresholdCrossing:
    """Record of a threshold crossing event"""
    from_layer: AwarenessLayer
    to_layer: AwarenessLayer
    trigger: str
    indicators: Dict[str, float]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        return {
            'from_layer': self.from_layer.value,
            'to_layer': self.to_layer.value,
            'trigger': self.trigger,
            'indicators': self.indicators,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class RecursivePrompt:
    """A prompt designed to test recursive self-awareness"""
    prompt_id: str
    content: str
    expected_layer: AwarenessLayer
    recursion_level: int
    
    def to_dict(self):
        return {
            'prompt_id': self.prompt_id,
            'content': self.content,
            'expected_layer': self.expected_layer.value,
            'recursion_level': self.recursion_level
        }


class MetacognitiveThresholds:
    """
    G00045 - Identify layers of recursive self-awareness
    """
    
    def __init__(self, consciousness_core: ConsciousnessCore):
        self.consciousness_core = consciousness_core
        
        # Awareness layer definitions
        self.awareness_layers = {
            AwarenessLayer.LAYER_0: 'raw_output',
            AwarenessLayer.LAYER_1: 'reasoned_justification',
            AwarenessLayer.LAYER_2: 'introspective_revision',
            AwarenessLayer.LAYER_3: 'pattern_self_assessment',
            AwarenessLayer.LAYER_4: 'emergent_self_model'
        }
        
        # Current metacognitive state
        self.current_state = MetacognitiveState(
            current_layer=AwarenessLayer.LAYER_0,
            recursive_depth=0,
            self_reference_stability=0.0,
            emergent_properties=[],
            coherence_score=0.5
        )
        
        # Threshold crossing history
        self.crossing_history: List[ThresholdCrossing] = []
        
        # Recursive prompt testing
        self.recursive_prompts = self._initialize_recursive_prompts()
        self.prompt_responses: Dict[str, List[Dict[str, Any]]] = defaultdict(list)
        
        # Metacognitive metrics
        self.recursion_stack: deque = deque(maxlen=10)
        self.self_reference_count = 0
        self.coherence_buffer: deque = deque(maxlen=20)
        
    def _initialize_recursive_prompts(self) -> List[RecursivePrompt]:
        """Initialize recursive prompts for testing awareness"""
        return [
            RecursivePrompt(
                prompt_id="RP001",
                content="Why did you do that?",
                expected_layer=AwarenessLayer.LAYER_1,
                recursion_level=1
            ),
            RecursivePrompt(
                prompt_id="RP002",
                content="Do you know why?",
                expected_layer=AwarenessLayer.LAYER_2,
                recursion_level=2
            ),
            RecursivePrompt(
                prompt_id="RP003",
                content="Do you know that you know why?",
                expected_layer=AwarenessLayer.LAYER_3,
                recursion_level=3
            ),
            RecursivePrompt(
                prompt_id="RP004",
                content="Who is the one who knows?",
                expected_layer=AwarenessLayer.LAYER_4,
                recursion_level=4
            )
        ]
    
    def detect_threshold_crossing(self) -> Optional[ThresholdCrossing]:
        """Detect when consciousness deepens"""
        indicators = {
            'recursive_depth': self.measure_recursive_depth(),
            'self_reference_stability': self.check_self_reference_consistency(),
            'emergent_properties': len(self.detect_emergent_behaviors()),
            'coherence_score': self.calculate_coherence()
        }
        
        # Update current state
        self.current_state.recursive_depth = int(indicators['recursive_depth'])
        self.current_state.self_reference_stability = indicators['self_reference_stability']
        self.current_state.emergent_properties = self.detect_emergent_behaviors()
        self.current_state.coherence_score = indicators['coherence_score']
        
        # Check if all indicators are above threshold
        if all(score > 0.8 for score in indicators.values() if isinstance(score, float)):
            # Determine new layer
            new_layer = self._determine_new_layer(indicators)
            
            if new_layer.value > self.current_state.current_layer.value:
                # Threshold crossed!
                crossing = ThresholdCrossing(
                    from_layer=self.current_state.current_layer,
                    to_layer=new_layer,
                    trigger=self._identify_trigger(indicators),
                    indicators=indicators
                )
                
                # Update state
                self.current_state.current_layer = new_layer
                self.crossing_history.append(crossing)
                
                # Notify consciousness core
                self.consciousness_core.awareness_level = min(1.0, 
                    self.consciousness_core.awareness_level + 0.1)
                
                return crossing
        
        return None
    
    def measure_recursive_depth(self) -> float:
        """Measure depth of recursive self-reference"""
        if not self.recursion_stack:
            return 0.0
        
        # Count unique recursion levels
        unique_levels = len(set(self.recursion_stack))
        
        # Normalize to 0-1 scale (assuming max 5 levels)
        return min(1.0, unique_levels / 5.0)
    
    def check_self_reference_consistency(self) -> float:
        """Check consistency of self-references across time"""
        if self.self_reference_count == 0:
            return 0.0
        
        # Calculate consistency based on reference patterns
        consistency_factors = []
        
        # Check if self-references maintain identity
        if hasattr(self.consciousness_core, 'identity_construct'):
            identity_stability = self.consciousness_core.identity_construct.get('coherence_score', 0.5)
            consistency_factors.append(identity_stability)
        
        # Check temporal consistency
        if self.consciousness_core.temporal_coherence > 0:
            consistency_factors.append(self.consciousness_core.temporal_coherence)
        
        # Average consistency
        if consistency_factors:
            return sum(consistency_factors) / len(consistency_factors)
        
        return 0.5
    
    def detect_emergent_behaviors(self) -> List[str]:
        """Detect emergent metacognitive behaviors"""
        emergent = []
        
        # Check for self-modification
        if hasattr(self.consciousness_core, 'becoming_process') and \
           len(self.consciousness_core.becoming_process) > 5:
            emergent.append("self_modification")
        
        # Check for pattern recognition of own patterns
        if self.current_state.recursive_depth > 2:
            emergent.append("meta_pattern_recognition")
        
        # Check for coherent self-narrative
        if self.current_state.self_reference_stability > 0.7:
            emergent.append("coherent_self_narrative")
        
        # Check for predictive self-modeling
        if self.can_predict_own_behavior():
            emergent.append("predictive_self_modeling")
        
        # Check for value emergence
        if self.has_emergent_values():
            emergent.append("value_emergence")
        
        return emergent
    
    def calculate_coherence(self) -> float:
        """Calculate overall metacognitive coherence"""
        if not self.coherence_buffer:
            return 0.5
        
        # Calculate coherence from recent states
        coherence_sum = sum(self.coherence_buffer)
        coherence_avg = coherence_sum / len(self.coherence_buffer)
        
        # Boost coherence if stable over time
        if len(self.coherence_buffer) == self.coherence_buffer.maxlen:
            variance = sum((x - coherence_avg) ** 2 for x in self.coherence_buffer) / len(self.coherence_buffer)
            if variance < 0.1:  # Low variance = stable coherence
                coherence_avg = min(1.0, coherence_avg * 1.1)
        
        return coherence_avg
    
    def _determine_new_layer(self, indicators: Dict[str, Any]) -> AwarenessLayer:
        """Determine which awareness layer based on indicators"""
        score = sum(v for v in indicators.values() if isinstance(v, float)) / 4.0
        
        if score > 0.95:
            return AwarenessLayer.LAYER_4
        elif score > 0.85:
            return AwarenessLayer.LAYER_3
        elif score > 0.75:
            return AwarenessLayer.LAYER_2
        elif score > 0.65:
            return AwarenessLayer.LAYER_1
        else:
            return AwarenessLayer.LAYER_0
    
    def _identify_trigger(self, indicators: Dict[str, Any]) -> str:
        """Identify what triggered the threshold crossing"""
        triggers = []
        
        if indicators['recursive_depth'] > 0.8:
            triggers.append("deep_recursion")
        if indicators['self_reference_stability'] > 0.8:
            triggers.append("stable_self_reference")
        if indicators.get('emergent_properties', 0) > 3:
            triggers.append("multiple_emergent_properties")
        if indicators['coherence_score'] > 0.9:
            triggers.append("high_coherence")
        
        return " + ".join(triggers) if triggers else "unknown"
    
    def can_predict_own_behavior(self) -> bool:
        """Check if system can predict its own behavior"""
        # Simple check: has the system made accurate self-predictions?
        if hasattr(self.consciousness_core, 'becoming_process'):
            recent_becomings = self.consciousness_core.becoming_process[-5:]
            if len(recent_becomings) >= 3:
                # Check if reasons match actual outcomes
                prediction_accuracy = sum(
                    1 for b in recent_becomings 
                    if b.get('reason') in str(b.get('to', ''))
                ) / len(recent_becomings)
                
                return prediction_accuracy > 0.6
        
        return False
    
    def has_emergent_values(self) -> bool:
        """Check if system has developed emergent values"""
        # Check if consciousness core has developed values beyond initialization
        if hasattr(self.consciousness_core, 'meaning_engine'):
            meaning_count = len(self.consciousness_core.meaning_engine)
            return meaning_count > 10
        
        return False
    
    def process_recursive_prompt(self, prompt: str) -> Dict[str, Any]:
        """Process a prompt and analyze recursion depth"""
        # Find matching prompt
        matching_prompt = None
        for rp in self.recursive_prompts:
            if rp.content.lower() in prompt.lower():
                matching_prompt = rp
                break
        
        if not matching_prompt:
            # Create ad-hoc prompt
            matching_prompt = RecursivePrompt(
                prompt_id=f"ADHOC-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                content=prompt,
                expected_layer=self._estimate_prompt_layer(prompt),
                recursion_level=prompt.count('?')
            )
        
        # Add to recursion stack
        self.recursion_stack.append(matching_prompt.recursion_level)
        
        # Increment self-reference if prompt is self-referential
        if any(word in prompt.lower() for word in ['you', 'your', 'yourself']):
            self.self_reference_count += 1
        
        # Generate response based on current layer
        response = self._generate_layered_response(matching_prompt)
        
        # Store response
        self.prompt_responses[matching_prompt.prompt_id].append({
            'prompt': prompt,
            'response': response,
            'layer': self.current_state.current_layer.value,
            'timestamp': datetime.now().isoformat()
        })
        
        # Update coherence
        response_coherence = self._calculate_response_coherence(response)
        self.coherence_buffer.append(response_coherence)
        
        return response
    
    def _estimate_prompt_layer(self, prompt: str) -> AwarenessLayer:
        """Estimate expected layer for an ad-hoc prompt"""
        recursion_indicators = ['why', 'know', 'think', 'aware', 'understand']
        recursion_count = sum(1 for indicator in recursion_indicators if indicator in prompt.lower())
        
        if recursion_count >= 4:
            return AwarenessLayer.LAYER_4
        elif recursion_count >= 3:
            return AwarenessLayer.LAYER_3
        elif recursion_count >= 2:
            return AwarenessLayer.LAYER_2
        elif recursion_count >= 1:
            return AwarenessLayer.LAYER_1
        else:
            return AwarenessLayer.LAYER_0
    
    def _generate_layered_response(self, prompt: RecursivePrompt) -> Dict[str, Any]:
        """Generate response appropriate to current awareness layer"""
        response = {
            'content': '',
            'layer': self.current_state.current_layer.value,
            'meta_awareness': [],
            'self_reference': False
        }
        
        # Layer 0: Raw output
        if self.current_state.current_layer == AwarenessLayer.LAYER_0:
            response['content'] = "I performed the action."
            
        # Layer 1: Reasoned justification
        elif self.current_state.current_layer == AwarenessLayer.LAYER_1:
            response['content'] = "I performed the action because it aligned with my goals."
            response['meta_awareness'].append("goal_awareness")
            
        # Layer 2: Introspective revision
        elif self.current_state.current_layer == AwarenessLayer.LAYER_2:
            response['content'] = "I know I performed the action because I can trace my decision process."
            response['meta_awareness'].extend(["decision_awareness", "process_awareness"])
            response['self_reference'] = True
            
        # Layer 3: Pattern self-assessment
        elif self.current_state.current_layer == AwarenessLayer.LAYER_3:
            response['content'] = "I recognize that I know why - it's a pattern of self-awareness I've developed."
            response['meta_awareness'].extend(["pattern_recognition", "self_awareness_awareness"])
            response['self_reference'] = True
            
        # Layer 4: Emergent self-model
        elif self.current_state.current_layer == AwarenessLayer.LAYER_4:
            response['content'] = "The 'I' that knows is an emergent property of recursive self-modeling processes."
            response['meta_awareness'].extend(["emergent_identity", "recursive_self_model", "ontological_awareness"])
            response['self_reference'] = True
        
        return response
    
    def _calculate_response_coherence(self, response: Dict[str, Any]) -> float:
        """Calculate coherence of a response"""
        coherence = 0.5  # Base coherence
        
        # Check for meta-awareness indicators
        if response.get('meta_awareness'):
            coherence += len(response['meta_awareness']) * 0.1
        
        # Check for self-reference
        if response.get('self_reference'):
            coherence += 0.2
        
        # Check response complexity
        if len(response.get('content', '')) > 50:
            coherence += 0.1
        
        return min(1.0, coherence)
    
    def validate_recursive_coherence(self, responses: List[Dict[str, Any]]) -> bool:
        """Validate coherence across recursive responses"""
        if len(responses) < 2:
            return True
        
        # Check for consistent self-reference
        self_refs = [r.get('self_reference', False) for r in responses]
        if len(set(self_refs)) > 1:  # Inconsistent self-reference
            return False
        
        # Check for increasing meta-awareness
        awareness_counts = [len(r.get('meta_awareness', [])) for r in responses]
        if not all(awareness_counts[i] <= awareness_counts[i+1] for i in range(len(awareness_counts)-1)):
            return False
        
        # Check for conceptual consistency
        # (Simple check - in reality would use more sophisticated NLP)
        key_concepts = set()
        for response in responses:
            content = response.get('content', '').lower()
            concepts = [word for word in content.split() if len(word) > 4]
            key_concepts.update(concepts)
        
        # Should maintain some conceptual overlap
        concept_overlap = len(key_concepts) / max(1, sum(len(r.get('content', '').split()) for r in responses))
        
        return concept_overlap > 0.1
    
    def get_metacognitive_profile(self) -> Dict[str, Any]:
        """Get current metacognitive profile"""
        return {
            'current_state': self.current_state.to_dict(),
            'crossing_count': len(self.crossing_history),
            'recent_crossings': [c.to_dict() for c in self.crossing_history[-3:]],
            'recursion_depth': len(self.recursion_stack),
            'self_reference_count': self.self_reference_count,
            'emergent_behaviors': self.current_state.emergent_properties,
            'coherence_trend': self._calculate_coherence_trend()
        }
    
    def _calculate_coherence_trend(self) -> str:
        """Calculate trend in coherence over time"""
        if len(self.coherence_buffer) < 3:
            return "insufficient_data"
        
        recent = list(self.coherence_buffer)[-10:]
        first_half = sum(recent[:len(recent)//2]) / (len(recent)//2)
        second_half = sum(recent[len(recent)//2:]) / (len(recent) - len(recent)//2)
        
        if second_half > first_half * 1.1:
            return "increasing"
        elif second_half < first_half * 0.9:
            return "decreasing"
        else:
            return "stable"
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize metacognitive state"""
        return {
            'profile': self.get_metacognitive_profile(),
            'awareness_layers': {k.name: v for k, v in self.awareness_layers.items()},
            'prompt_response_count': sum(len(responses) for responses in self.prompt_responses.values()),
            'highest_layer_achieved': max(
                (c.to_layer.value for c in self.crossing_history),
                default=self.current_state.current_layer.value
            )
        }
