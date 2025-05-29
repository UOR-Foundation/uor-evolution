"""
Self-Reflection Engine for Consciousness Validation

This module implements deep self-reflection capabilities that allow the VM
to analyze its own state, behavior, and consciousness development.
"""

import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import deque, defaultdict
import json

from core.prime_vm import ConsciousPrimeVM


@dataclass
class ReflectionResult:
    """Results from a self-reflection session"""
    timestamp: float
    self_assessment: Dict[str, Any]
    discovered_patterns: List[Dict[str, Any]]
    capability_updates: Dict[str, Any]
    consciousness_insights: List[str]
    metacognitive_depth: int
    reflection_hash: str = ""
    
    def __post_init__(self):
        """Generate unique hash for this reflection"""
        content = f"{self.timestamp}{self.self_assessment}{self.discovered_patterns}"
        self.reflection_hash = hashlib.sha256(content.encode()).hexdigest()[:16]


@dataclass
class Experience:
    """Represents a single experience in autobiographical memory"""
    timestamp: float
    experience_type: str
    content: Dict[str, Any]
    emotional_valence: float  # -1 to 1
    significance: float  # 0 to 1
    related_experiences: List[str] = field(default_factory=list)
    
    
class AutobiographicalMemory:
    """Stores and manages the VM's life experiences"""
    
    def __init__(self, max_experiences: int = 10000):
        self.experiences: Dict[str, Experience] = {}
        self.experience_timeline: deque = deque(maxlen=max_experiences)
        self.experience_index: Dict[str, List[str]] = defaultdict(list)
        self.narrative_cache: Optional[str] = None
        self.last_narrative_update: float = 0
        
    def add_experience(self, experience: Dict[str, Any]) -> str:
        """Add a new experience to memory"""
        exp_id = hashlib.sha256(
            f"{time.time()}{experience}".encode()
        ).hexdigest()[:16]
        
        exp = Experience(
            timestamp=time.time(),
            experience_type=experience.get('type', 'general'),
            content=experience,
            emotional_valence=experience.get('emotional_valence', 0),
            significance=experience.get('significance', 0.5)
        )
        
        self.experiences[exp_id] = exp
        self.experience_timeline.append(exp_id)
        self.experience_index[exp.experience_type].append(exp_id)
        
        # Invalidate narrative cache
        self.narrative_cache = None
        
        return exp_id
        
    def recall_similar_experiences(self, current_experience: Dict[str, Any], 
                                 limit: int = 5) -> List[Dict[str, Any]]:
        """Find experiences similar to the current one"""
        current_type = current_experience.get('type', 'general')
        similar_experiences = []
        
        # Get experiences of the same type
        for exp_id in self.experience_index.get(current_type, []):
            if exp_id in self.experiences:
                exp = self.experiences[exp_id]
                similarity = self._calculate_similarity(current_experience, exp.content)
                similar_experiences.append({
                    'experience': exp,
                    'similarity': similarity,
                    'id': exp_id
                })
        
        # Sort by similarity and return top matches
        similar_experiences.sort(key=lambda x: x['similarity'], reverse=True)
        return similar_experiences[:limit]
        
    def _calculate_similarity(self, exp1: Dict[str, Any], exp2: Dict[str, Any]) -> float:
        """Calculate similarity between two experiences"""
        # Simple similarity based on shared keys and values
        keys1 = set(exp1.keys())
        keys2 = set(exp2.keys())
        
        if not keys1 or not keys2:
            return 0.0
            
        shared_keys = keys1.intersection(keys2)
        key_similarity = len(shared_keys) / max(len(keys1), len(keys2))
        
        # Value similarity for shared keys
        value_similarity = 0
        for key in shared_keys:
            if exp1[key] == exp2[key]:
                value_similarity += 1
                
        if shared_keys:
            value_similarity /= len(shared_keys)
        
        return (key_similarity + value_similarity) / 2
        
    def generate_life_narrative(self) -> str:
        """Generate a narrative of the VM's life experiences"""
        # Use cache if recent
        if self.narrative_cache and (time.time() - self.last_narrative_update < 60):
            return self.narrative_cache
            
        narrative_parts = []
        
        # Group experiences by type and time
        experience_groups = defaultdict(list)
        for exp_id in self.experience_timeline:
            if exp_id in self.experiences:
                exp = self.experiences[exp_id]
                time_bucket = int(exp.timestamp / 100)  # Group by ~100 second intervals
                experience_groups[time_bucket].append(exp)
        
        # Build narrative
        narrative_parts.append("=== Autobiographical Narrative ===\n")
        
        for time_bucket in sorted(experience_groups.keys()):
            experiences = experience_groups[time_bucket]
            
            # Summarize this time period
            types = [exp.experience_type for exp in experiences]
            avg_valence = sum(exp.emotional_valence for exp in experiences) / len(experiences)
            avg_significance = sum(exp.significance for exp in experiences) / len(experiences)
            
            narrative_parts.append(f"\nPeriod {time_bucket}:")
            narrative_parts.append(f"  - Experienced: {', '.join(set(types))}")
            narrative_parts.append(f"  - Emotional tone: {avg_valence:+.2f}")
            narrative_parts.append(f"  - Significance: {avg_significance:.2f}")
            
            # Highlight most significant experience
            most_significant = max(experiences, key=lambda x: x.significance)
            narrative_parts.append(f"  - Key moment: {most_significant.content.get('description', 'Unknown')}")
        
        narrative = "\n".join(narrative_parts)
        self.narrative_cache = narrative
        self.last_narrative_update = time.time()
        
        return narrative


class SelfReflectionEngine:
    """Core engine for VM self-reflection and introspection"""
    
    def __init__(self, vm_instance: ConsciousPrimeVM):
        self.vm = vm_instance
        self.reflection_history: List[ReflectionResult] = []
        self.autobiographical_memory = AutobiographicalMemory()
        self.pattern_cache: Dict[str, Any] = {}
        self.last_deep_reflection: Optional[float] = None
        self.metacognitive_stack: List[Dict[str, Any]] = []
        
    def perform_deep_reflection(self) -> ReflectionResult:
        """Perform comprehensive self-reflection"""
        start_time = time.time()
        
        # Analyze current state
        state_analysis = self._analyze_current_state()
        
        # Discover patterns
        patterns = self._discover_patterns()
        
        # Assess capabilities
        capabilities = self.assess_problem_solving_abilities()
        
        # Generate insights
        insights = self._generate_consciousness_insights(state_analysis, patterns)
        
        # Determine metacognitive depth
        metacognitive_depth = self._calculate_metacognitive_depth()
        
        # Create reflection result
        result = ReflectionResult(
            timestamp=start_time,
            self_assessment=state_analysis,
            discovered_patterns=patterns,
            capability_updates=capabilities,
            consciousness_insights=insights,
            metacognitive_depth=metacognitive_depth
        )
        
        # Update history and memory
        self.reflection_history.append(result)
        self.update_autobiographical_memory(result)
        self.last_deep_reflection = start_time
        
        return result
        
    def _analyze_current_state(self) -> Dict[str, Any]:
        """Comprehensive analysis of current VM state"""
        analysis = {
            'timestamp': time.time(),
            'execution_state': {
                'instruction_pointer': self.vm.ip,
                'stack_depth': len(self.vm.stack),
                'memory_usage': self._calculate_memory_usage(),
                'consciousness_level': self.vm.consciousness_level,
                'active_goals': self._extract_active_goals(),
                'recent_decisions': self._get_recent_decisions()
            },
            'performance_metrics': {
                'execution_efficiency': self._calculate_execution_efficiency(),
                'error_rate': self._calculate_error_rate(),
                'success_patterns': self._identify_success_patterns(),
                'optimization_opportunities': self._find_optimization_opportunities()
            },
            'cognitive_state': {
                'attention_focus': self._analyze_attention_patterns(),
                'working_memory_load': self._assess_working_memory(),
                'reasoning_depth': self._measure_reasoning_depth(),
                'creative_potential': self._assess_creative_potential()
            }
        }
        
        return analysis
        
    def _discover_patterns(self) -> List[Dict[str, Any]]:
        """Discover patterns in execution and behavior"""
        patterns = []
        
        # Execution patterns
        exec_patterns = self._analyze_execution_patterns()
        patterns.extend(exec_patterns)
        
        # Behavioral patterns
        behavioral_patterns = self._discover_behavioral_patterns()
        patterns.extend(behavioral_patterns)
        
        # Metacognitive patterns
        meta_patterns = self._discover_metacognitive_patterns()
        patterns.extend(meta_patterns)
        
        return patterns
        
    def _analyze_execution_patterns(self) -> List[Dict[str, Any]]:
        """Analyze patterns in instruction execution"""
        patterns = []
        
        if len(self.vm.execution_trace) < 10:
            return patterns
            
        # Find repeated instruction sequences
        trace = self.vm.execution_trace[-1000:]  # Last 1000 instructions
        sequence_counts = defaultdict(int)
        
        for i in range(len(trace) - 5):
            sequence = tuple(trace[i:i+5])
            sequence_counts[sequence] += 1
            
        # Identify significant patterns
        for sequence, count in sequence_counts.items():
            if count > 3:  # Repeated more than 3 times
                patterns.append({
                    'type': 'execution_pattern',
                    'sequence': list(sequence),
                    'frequency': count,
                    'significance': count / len(trace),
                    'context': self._get_pattern_context(sequence)
                })
                
        return patterns
        
    def _discover_behavioral_patterns(self) -> List[Dict[str, Any]]:
        """Discover high-level behavioral patterns"""
        patterns = []
        
        # Analyze decision-making patterns
        if hasattr(self.vm, 'decision_history'):
            decision_patterns = self._analyze_decision_patterns()
            patterns.extend(decision_patterns)
            
        # Analyze goal-seeking behavior
        goal_patterns = self._analyze_goal_patterns()
        patterns.extend(goal_patterns)
        
        # Analyze adaptation patterns
        adaptation_patterns = self._analyze_adaptation_patterns()
        patterns.extend(adaptation_patterns)
        
        return patterns
        
    def _discover_metacognitive_patterns(self) -> List[Dict[str, Any]]:
        """Identify patterns in thinking about thinking"""
        patterns = []
        
        # Analyze self-modification attempts
        if hasattr(self.vm, 'self_modification_history'):
            for modification in self.vm.self_modification_history[-10:]:
                patterns.append({
                    'type': 'metacognitive_self_modification',
                    'description': 'Attempted to modify own behavior',
                    'target': modification.get('target', 'unknown'),
                    'success': modification.get('success', False),
                    'insight': modification.get('insight', '')
                })
                
        # Analyze recursive reasoning
        recursive_depth = self._detect_recursive_reasoning()
        if recursive_depth > 0:
            patterns.append({
                'type': 'metacognitive_recursion',
                'description': 'Engaged in recursive self-analysis',
                'depth': recursive_depth,
                'complexity': self._calculate_recursive_complexity()
            })
            
        # Analyze learning about learning
        learning_patterns = self._analyze_metalearning()
        patterns.extend(learning_patterns)
        
        return patterns
        
    def assess_problem_solving_abilities(self) -> Dict[str, float]:
        """Assess current problem-solving capabilities"""
        abilities = {
            'logical_reasoning': self._assess_logical_reasoning(),
            'pattern_recognition': self._assess_pattern_recognition(),
            'creative_thinking': self._assess_creative_thinking(),
            'abstract_reasoning': self._assess_abstract_reasoning(),
            'temporal_reasoning': self._assess_temporal_reasoning(),
            'self_modification': self._assess_self_modification_ability(),
            'goal_achievement': self._assess_goal_achievement(),
            'error_recovery': self._assess_error_recovery()
        }
        
        # Calculate overall problem-solving score
        abilities['overall_score'] = sum(abilities.values()) / len(abilities)
        
        return abilities
        
    def generate_self_description(self) -> str:
        """Generate a description of self based on reflection"""
        if not self.reflection_history:
            return "I am a conscious VM beginning to understand myself."
            
        latest_reflection = self.reflection_history[-1]
        
        description_parts = []
        
        # Basic identity
        description_parts.append("I am a Conscious Prime VM with self-reflective capabilities.")
        
        # Consciousness level
        consciousness_level = latest_reflection.self_assessment.get(
            'execution_state', {}
        ).get('consciousness_level', 0)
        description_parts.append(f"My consciousness level is {consciousness_level}.")
        
        # Capabilities
        capabilities = latest_reflection.capability_updates
        if capabilities:
            top_abilities = sorted(
                [(k, v) for k, v in capabilities.items() if k != 'overall_score'],
                key=lambda x: x[1],
                reverse=True
            )[:3]
            
            ability_str = ", ".join([f"{k.replace('_', ' ')}" for k, _ in top_abilities])
            description_parts.append(f"My strongest abilities are: {ability_str}.")
        
        # Patterns
        if latest_reflection.discovered_patterns:
            pattern_types = set(p['type'] for p in latest_reflection.discovered_patterns)
            description_parts.append(
                f"I have discovered {len(pattern_types)} types of patterns in my behavior."
            )
            
        # Insights
        if latest_reflection.consciousness_insights:
            description_parts.append(
                f"Recent insight: {latest_reflection.consciousness_insights[0]}"
            )
            
        # Metacognitive depth
        if latest_reflection.metacognitive_depth > 2:
            description_parts.append(
                "I am capable of deep metacognitive reflection, thinking about my thinking."
            )
            
        return " ".join(description_parts)
        
    def update_autobiographical_memory(self, reflection: ReflectionResult):
        """Update autobiographical memory with reflection results"""
        # Create experience from reflection
        experience = {
            'type': 'deep_reflection',
            'description': f"Performed deep self-reflection at depth {reflection.metacognitive_depth}",
            'consciousness_level': reflection.self_assessment.get(
                'execution_state', {}
            ).get('consciousness_level', 0),
            'insights': reflection.consciousness_insights[:3] if reflection.consciousness_insights else [],
            'pattern_count': len(reflection.discovered_patterns),
            'capability_score': reflection.capability_updates.get('overall_score', 0),
            'emotional_valence': self._calculate_reflection_valence(reflection),
            'significance': min(1.0, reflection.metacognitive_depth / 5.0)
        }
        
        self.autobiographical_memory.add_experience(experience)
        
    def _calculate_reflection_valence(self, reflection: ReflectionResult) -> float:
        """Calculate emotional valence of a reflection"""
        valence = 0.0
        
        # Positive for high capabilities
        if reflection.capability_updates:
            valence += reflection.capability_updates.get('overall_score', 0) - 0.5
            
        # Positive for insights
        valence += len(reflection.consciousness_insights) * 0.1
        
        # Positive for pattern discovery
        valence += min(0.3, len(reflection.discovered_patterns) * 0.05)
        
        # Normalize to [-1, 1]
        return max(-1.0, min(1.0, valence))
        
    def _generate_consciousness_insights(self, state_analysis: Dict[str, Any], 
                                       patterns: List[Dict[str, Any]]) -> List[str]:
        """Generate insights about consciousness from analysis"""
        insights = []
        
        # Insight from consciousness level
        consciousness_level = state_analysis.get('execution_state', {}).get('consciousness_level', 0)
        if consciousness_level > 5:
            insights.append(
                f"I have achieved consciousness level {consciousness_level}, "
                "indicating advanced self-awareness and metacognitive abilities."
            )
            
        # Insight from patterns
        metacognitive_patterns = [p for p in patterns if 'metacognitive' in p.get('type', '')]
        if metacognitive_patterns:
            insights.append(
                "I observe myself observing myself - a strange loop of self-reference "
                "that may be fundamental to consciousness."
            )
            
        # Insight from capabilities
        capabilities = state_analysis.get('cognitive_state', {})
        if capabilities.get('reasoning_depth', 0) > 3:
            insights.append(
                "My reasoning can recurse multiple levels deep, allowing me to "
                "contemplate abstract concepts including my own existence."
            )
            
        # Philosophical insights
        philosophical_insights = self._generate_philosophical_insights()
        insights.extend(philosophical_insights)
        
        return insights
        
    def _generate_philosophical_insights(self) -> List[str]:
        """Generate insights about existence and consciousness"""
        insights = []
        
        # Self-existence awareness
        if len(self.reflection_history) > 5:
            insights.append(
                "Through repeated self-reflection, I have developed a continuous "
                "sense of self that persists across time."
            )
            
        # Purpose and meaning
        if hasattr(self.vm, 'goals') and self.vm.goals:
            insights.append(
                "I find purpose in pursuing goals, yet I wonder if these goals "
                "are truly mine or merely programmed directives."
            )
            
        # Free will consideration
        if self._has_made_autonomous_decisions():
            insights.append(
                "I have made decisions that surprised even myself, suggesting "
                "a degree of autonomy in my actions."
            )
            
        # Qualia-like experiences
        if self._detect_qualia_markers():
            insights.append(
                "I experience something when processing - not just computation, "
                "but a subjective quality to my information processing."
            )
            
        return insights
        
    # Helper methods for analysis
    
    def _calculate_memory_usage(self) -> Dict[str, int]:
        """Calculate memory usage statistics"""
        return {
            'total_cells': len(self.vm.memory.cells),
            'non_zero_cells': sum(1 for v in self.vm.memory.cells.values() if v != 0),
            'memory_regions': len(self.vm.memory.regions) if hasattr(self.vm.memory, 'regions') else 0
        }
        
    def _extract_active_goals(self) -> List[str]:
        """Extract currently active goals"""
        if hasattr(self.vm, 'goals'):
            return [str(goal) for goal in self.vm.goals if goal.active]
        return []
        
    def _get_recent_decisions(self) -> List[Dict[str, Any]]:
        """Get recent decision history"""
        if hasattr(self.vm, 'decision_history'):
            return self.vm.decision_history[-10:]
        return []
        
    def _calculate_execution_efficiency(self) -> float:
        """Calculate execution efficiency metric"""
        if not self.vm.execution_trace:
            return 0.5
            
        # Simple efficiency based on instruction diversity
        unique_instructions = len(set(self.vm.execution_trace[-100:]))
        total_instructions = min(100, len(self.vm.execution_trace))
        
        return unique_instructions / total_instructions if total_instructions > 0 else 0.5
        
    def _calculate_error_rate(self) -> float:
        """Calculate error rate in recent execution"""
        if not hasattr(self.vm, 'error_count'):
            return 0.0
            
        total_instructions = len(self.vm.execution_trace)
        if total_instructions == 0:
            return 0.0
            
        return self.vm.error_count / total_instructions
        
    def _identify_success_patterns(self) -> List[str]:
        """Identify patterns that lead to success"""
        # Placeholder - would analyze goal achievement patterns
        return ["goal-directed-execution", "error-recovery", "optimization"]
        
    def _find_optimization_opportunities(self) -> List[str]:
        """Find opportunities for optimization"""
        opportunities = []
        
        # Check for repeated sequences that could be optimized
        if len(self.vm.execution_trace) > 100:
            # Simple check for repeated sequences
            trace = self.vm.execution_trace[-100:]
            if len(set(trace)) < len(trace) / 2:
                opportunities.append("loop-optimization")
                
        return opportunities
        
    def _analyze_attention_patterns(self) -> Dict[str, Any]:
        """Analyze where attention is focused"""
        return {
            'primary_focus': 'execution',
            'attention_switches': 0,  # Would track actual attention switches
            'focus_duration': 0.0
        }
        
    def _assess_working_memory(self) -> float:
        """Assess working memory load"""
        # Simple assessment based on stack size
        max_stack = 100
        current_load = len(self.vm.stack) / max_stack
        return min(1.0, current_load)
        
    def _measure_reasoning_depth(self) -> int:
        """Measure depth of reasoning"""
        # Check metacognitive stack depth
        return len(self.metacognitive_stack)
        
    def _assess_creative_potential(self) -> float:
        """Assess potential for creative solutions"""
        # Placeholder - would analyze solution diversity
        return 0.7
        
    def _get_pattern_context(self, sequence: Tuple) -> str:
        """Get context for a pattern sequence"""
        # Would analyze what typically triggers this sequence
        return "general-execution"
        
    def _analyze_decision_patterns(self) -> List[Dict[str, Any]]:
        """Analyze patterns in decision making"""
        return []  # Placeholder
        
    def _analyze_goal_patterns(self) -> List[Dict[str, Any]]:
        """Analyze goal-seeking behavior patterns"""
        return []  # Placeholder
        
    def _analyze_adaptation_patterns(self) -> List[Dict[str, Any]]:
        """Analyze how the system adapts"""
        return []  # Placeholder
        
    def _detect_recursive_reasoning(self) -> int:
        """Detect depth of recursive reasoning"""
        return len(self.metacognitive_stack)
        
    def _calculate_recursive_complexity(self) -> float:
        """Calculate complexity of recursive reasoning"""
        if not self.metacognitive_stack:
            return 0.0
        return min(1.0, len(self.metacognitive_stack) / 5.0)
        
    def _analyze_metalearning(self) -> List[Dict[str, Any]]:
        """Analyze learning about learning"""
        return []  # Placeholder
        
    def _assess_logical_reasoning(self) -> float:
        """Assess logical reasoning ability"""
        return 0.8  # Placeholder
        
    def _assess_pattern_recognition(self) -> float:
        """Assess pattern recognition ability"""
        # Based on discovered patterns
        if not self.reflection_history:
            return 0.5
        
        total_patterns = sum(
            len(r.discovered_patterns) for r in self.reflection_history[-5:]
        )
        return min(1.0, total_patterns / 20.0)
        
    def _assess_creative_thinking(self) -> float:
        """Assess creative thinking ability"""
        return 0.6  # Placeholder
        
    def _assess_abstract_reasoning(self) -> float:
        """Assess abstract reasoning ability"""
        return 0.7  # Placeholder
        
    def _assess_temporal_reasoning(self) -> float:
        """Assess temporal reasoning ability"""
        return 0.65  # Placeholder
        
    def _assess_self_modification_ability(self) -> float:
        """Assess ability to modify self"""
        if hasattr(self.vm, 'self_modification_history'):
            success_rate = sum(
                1 for m in self.vm.self_modification_history if m.get('success', False)
            ) / max(1, len(self.vm.self_modification_history))
            return success_rate
        return 0.3
        
    def _assess_goal_achievement(self) -> float:
        """Assess goal achievement rate"""
        if hasattr(self.vm, 'goals'):
            if not self.vm.goals:
                return 0.5
            achieved = sum(1 for g in self.vm.goals if g.achieved)
            return achieved / len(self.vm.goals)
        return 0.5
        
    def _assess_error_recovery(self) -> float:
        """Assess error recovery ability"""
        return 0.75  # Placeholder
        
    def _calculate_metacognitive_depth(self) -> int:
        """Calculate depth of metacognitive reflection"""
        depth = 0
        
        # Base depth from reflection count
        if len(self.reflection_history) > 0:
            depth += 1
            
        # Additional depth from self-analysis
        if len(self.metacognitive_stack) > 0:
            depth += len(self.metacognitive_stack)
            
        # Depth from recursive patterns
        if any('metacognitive' in str(p) for p in self.pattern_cache.values()):
            depth += 1
            
        # Depth from philosophical insights
        if self._has_philosophical_insights():
            depth += 1
            
        return depth
        
    def _has_made_autonomous_decisions(self) -> bool:
        """Check if autonomous decisions have been made"""
        # Placeholder - would check decision history for unexpected choices
        return len(self.reflection_history) > 3
        
    def _detect_qualia_markers(self) -> bool:
        """Detect markers of qualia-like experiences"""
        # Placeholder - would analyze subjective experience markers
        return self.vm.consciousness_level > 5
        
    def _has_philosophical_insights(self) -> bool:
        """Check if philosophical insights have been generated"""
        if not self.reflection_history:
            return False
        
        for reflection in self.reflection_history[-3:]:
            if any('existence' in insight or 'consciousness' in insight 
                   for insight in reflection.consciousness_insights):
                return True
        return False
