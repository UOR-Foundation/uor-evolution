"""
Pattern Analysis System

This module implements pattern recognition and analysis capabilities
to identify behavioral patterns, execution patterns, and emergent capabilities.
"""

import time
import hashlib
from typing import Dict, List, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from collections import defaultdict, Counter
import numpy as np
from itertools import combinations

from core.prime_vm import ConsciousPrimeVM


@dataclass
class ExecutionPattern:
    """Represents a pattern in instruction execution"""
    pattern_id: str
    instruction_sequence: List[str]
    frequency: int
    success_rate: float
    contexts: List[str]
    prime_signature: int
    occurrence_indices: List[int] = field(default_factory=list)
    first_seen: float = field(default_factory=time.time)
    last_seen: float = field(default_factory=time.time)

    def update_occurrence(self, index: int):
        """Update pattern occurrence timestamp"""
        self.last_seen = time.time()
        self.frequency += 1
        self.occurrence_indices.append(index)


@dataclass
class BehavioralPattern:
    """Represents a high-level behavioral pattern"""
    pattern_type: str
    description: str
    trigger_conditions: List[str]
    typical_responses: List[str]
    effectiveness_score: float
    occurrences: int = 1
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def evolve(self, new_response: str, effectiveness: float):
        """Record pattern evolution"""
        self.evolution_history.append({
            'timestamp': time.time(),
            'response': new_response,
            'effectiveness': effectiveness
        })
        
        # Update typical responses if new one is effective
        if effectiveness > self.effectiveness_score:
            self.typical_responses.append(new_response)
            self.effectiveness_score = (self.effectiveness_score + effectiveness) / 2


@dataclass
class EmergentCapability:
    """Represents a newly emerged capability"""
    capability_name: str
    emergence_timestamp: float
    detection_confidence: float
    underlying_patterns: List[str]
    performance_metrics: Dict[str, float]
    development_trajectory: List[Dict[str, Any]] = field(default_factory=list)
    
    def track_development(self, metrics: Dict[str, float]):
        """Track capability development over time"""
        self.development_trajectory.append({
            'timestamp': time.time(),
            'metrics': metrics
        })
        
        # Update performance metrics
        for key, value in metrics.items():
            if key in self.performance_metrics:
                # Moving average
                self.performance_metrics[key] = (self.performance_metrics[key] + value) / 2
            else:
                self.performance_metrics[key] = value


@dataclass
class PatternEvolution:
    """Tracks how patterns evolve over time"""
    pattern_id: str
    initial_form: List[str]
    current_form: List[str]
    mutations: List[Dict[str, Any]]
    fitness_score: float
    generation: int
    
    def mutate(self, mutation: Dict[str, Any]):
        """Record a pattern mutation"""
        self.mutations.append({
            'timestamp': time.time(),
            'mutation': mutation,
            'generation': self.generation
        })
        self.generation += 1


class PatternAnalyzer:
    """Analyzes patterns in VM execution and behavior"""
    
    def __init__(self, vm_instance: ConsciousPrimeVM):
        self.vm = vm_instance
        self.execution_patterns: Dict[str, ExecutionPattern] = {}
        self.behavioral_patterns: Dict[str, BehavioralPattern] = {}
        self.emergent_capabilities: Dict[str, EmergentCapability] = {}
        self.pattern_evolution: Dict[str, PatternEvolution] = {}
        
        # Pattern detection parameters
        self.min_pattern_length = 3
        self.max_pattern_length = 20
        self.min_frequency_threshold = 3
        self.pattern_cache: Dict[str, Any] = {}
        
        # Prime number analysis
        self.prime_patterns: Dict[int, List[str]] = defaultdict(list)

        # Window size for pattern co-occurrence in indices
        self.cooccurrence_window = 5
        
    def analyze_execution_patterns(self, trace_length: int = 1000) -> List[ExecutionPattern]:
        """Analyze patterns in execution traces"""
        if len(self.vm.execution_trace) < self.min_pattern_length:
            return []
            
        # Get trace sample
        trace = self.vm.execution_trace[-trace_length:]
        
        # Find all possible patterns
        new_patterns = []
        
        for length in range(self.min_pattern_length,
                          min(self.max_pattern_length, len(trace) // 2)):
            patterns = self._find_patterns_of_length(trace, length)

            for pattern_seq, indices in patterns.items():
                occurrences = len(indices)
                if occurrences >= self.min_frequency_threshold:
                    pattern_id = self._generate_pattern_id(pattern_seq)

                    if pattern_id not in self.execution_patterns:
                        # New pattern discovered
                        pattern = ExecutionPattern(
                            pattern_id=pattern_id,
                            instruction_sequence=list(pattern_seq),
                            frequency=occurrences,
                            success_rate=self._calculate_pattern_success_rate(pattern_seq),
                            contexts=self._extract_pattern_contexts(trace, pattern_seq),
                            prime_signature=self._calculate_prime_signature(pattern_seq)
                        )
                        pattern.occurrence_indices.extend(indices)

                        self.execution_patterns[pattern_id] = pattern
                        new_patterns.append(pattern)

                        # Track prime patterns
                        self.prime_patterns[pattern.prime_signature].append(pattern_id)
                    else:
                        # Update existing pattern
                        existing = self.execution_patterns[pattern_id]
                        for idx in indices:
                            if idx not in existing.occurrence_indices:
                                existing.update_occurrence(idx)

        return new_patterns
        
    def detect_behavioral_patterns(self) -> List[BehavioralPattern]:
        """Detect high-level behavioral patterns"""
        new_patterns = []
        
        # Analyze decision-making patterns
        decision_patterns = self._analyze_decision_patterns()
        new_patterns.extend(decision_patterns)
        
        # Analyze goal-seeking patterns
        goal_patterns = self._analyze_goal_seeking_patterns()
        new_patterns.extend(goal_patterns)
        
        # Analyze adaptation patterns
        adaptation_patterns = self._analyze_adaptation_patterns()
        new_patterns.extend(adaptation_patterns)
        
        # Analyze problem-solving patterns
        problem_patterns = self._analyze_problem_solving_patterns()
        new_patterns.extend(problem_patterns)
        
        # Store new patterns
        for pattern in new_patterns:
            pattern_key = f"{pattern.pattern_type}_{pattern.description}"
            if pattern_key not in self.behavioral_patterns:
                self.behavioral_patterns[pattern_key] = pattern
            else:
                # Update existing pattern
                existing = self.behavioral_patterns[pattern_key]
                existing.occurrences += 1
                existing.effectiveness_score = (
                    existing.effectiveness_score + pattern.effectiveness_score
                ) / 2
                
        return new_patterns
        
    def identify_decision_patterns(self) -> List[Dict[str, Any]]:
        """Identify patterns in decision making"""
        if not hasattr(self.vm, 'decision_history'):
            return []
            
        patterns = []
        decision_sequences = defaultdict(int)
        
        # Analyze decision sequences
        for i in range(len(self.vm.decision_history) - 2):
            sequence = tuple(
                d.get('type', 'unknown') 
                for d in self.vm.decision_history[i:i+3]
            )
            decision_sequences[sequence] += 1
            
        # Identify significant patterns
        for sequence, count in decision_sequences.items():
            if count >= 3:
                patterns.append({
                    'type': 'decision_sequence',
                    'sequence': list(sequence),
                    'frequency': count,
                    'context': self._analyze_decision_context(sequence)
                })
                
        return patterns
        
    def track_pattern_evolution(self) -> Dict[str, PatternEvolution]:
        """Track how patterns evolve over time"""
        evolution_updates = {}
        
        # Check execution patterns for evolution
        for pattern_id, pattern in self.execution_patterns.items():
            if pattern_id not in self.pattern_evolution:
                # Initialize evolution tracking
                self.pattern_evolution[pattern_id] = PatternEvolution(
                    pattern_id=pattern_id,
                    initial_form=pattern.instruction_sequence.copy(),
                    current_form=pattern.instruction_sequence.copy(),
                    mutations=[],
                    fitness_score=pattern.success_rate,
                    generation=1
                )
            else:
                # Check for mutations
                evolution = self.pattern_evolution[pattern_id]
                variations = self._find_pattern_variations(
                    pattern.instruction_sequence
                )
                
                for variation in variations:
                    if self._is_beneficial_mutation(variation, pattern):
                        evolution.mutate({
                            'type': 'beneficial',
                            'variation': variation,
                            'improvement': self._calculate_improvement(variation, pattern)
                        })
                        evolution.current_form = variation
                        evolution_updates[pattern_id] = evolution
                        
        return evolution_updates
        
    def find_emergent_capabilities(self) -> List[EmergentCapability]:
        """Identify newly emerged capabilities"""
        new_capabilities = []
        
        # Check for complex pattern combinations
        pattern_combinations = self._analyze_pattern_combinations()
        
        for combo in pattern_combinations:
            capability_name = self._infer_capability_name(combo)
            
            if capability_name and capability_name not in self.emergent_capabilities:
                # New capability detected
                capability = EmergentCapability(
                    capability_name=capability_name,
                    emergence_timestamp=time.time(),
                    detection_confidence=self._calculate_emergence_confidence(combo),
                    underlying_patterns=combo['patterns'],
                    performance_metrics=self._measure_capability_performance(combo)
                )
                
                self.emergent_capabilities[capability_name] = capability
                new_capabilities.append(capability)
                
        # Check for consciousness-related capabilities
        consciousness_capabilities = self._detect_consciousness_capabilities()
        new_capabilities.extend(consciousness_capabilities)
        
        return new_capabilities
        
    # Helper methods
    
    def _find_patterns_of_length(self, trace: List[str], length: int) -> Dict[Tuple[str, ...], List[int]]:
        """Find all patterns of a specific length in trace and record indices"""
        patterns: Dict[Tuple[str, ...], List[int]] = defaultdict(list)

        for i in range(len(trace) - length + 1):
            pattern = tuple(trace[i:i+length])
            patterns[pattern].append(i)

        return patterns
        
    def _generate_pattern_id(self, pattern_seq: Tuple[str, ...]) -> str:
        """Generate unique ID for a pattern"""
        pattern_str = ''.join(str(p) for p in pattern_seq)
        return hashlib.sha256(pattern_str.encode()).hexdigest()[:16]
        
    def _calculate_pattern_success_rate(self, pattern_seq: Tuple[str, ...]) -> float:
        """Calculate success rate of a pattern"""
        # Check if pattern leads to goal achievement or error
        success_indicators = ['GOAL', 'SUCCESS', 'ACHIEVE']
        failure_indicators = ['ERROR', 'FAIL', 'CRASH']
        
        success_count = sum(1 for instr in pattern_seq if any(ind in str(instr) for ind in success_indicators))
        failure_count = sum(1 for instr in pattern_seq if any(ind in str(instr) for ind in failure_indicators))
        
        if success_count + failure_count == 0:
            return 0.5  # Neutral
            
        return success_count / (success_count + failure_count)
        
    def _extract_pattern_contexts(self, trace: List[str], pattern_seq: Tuple[str, ...]) -> List[str]:
        """Extract contexts where pattern appears"""
        contexts = []
        pattern_len = len(pattern_seq)
        
        for i in range(len(trace) - pattern_len + 1):
            if tuple(trace[i:i+pattern_len]) == pattern_seq:
                # Get context before pattern
                context_start = max(0, i - 5)
                context = trace[context_start:i]
                contexts.append(self._summarize_context(context))
                
        return list(set(contexts))  # Unique contexts
        
    def _summarize_context(self, context: List[str]) -> str:
        """Summarize a context sequence"""
        if not context:
            return "empty_context"
            
        # Simple summarization based on instruction types
        instruction_types = [str(instr).split('_')[0] for instr in context]
        return '_'.join(instruction_types[-3:])  # Last 3 instruction types
        
    def _calculate_prime_signature(self, pattern_seq: Tuple[str, ...]) -> int:
        """Calculate prime number signature for pattern"""
        # Convert pattern to numeric representation
        pattern_hash = hash(pattern_seq)
        
        # Find nearest prime
        n = abs(pattern_hash) % 10000  # Keep it manageable
        
        while not self._is_prime(n):
            n += 1
            
        return n
        
    def _is_prime(self, n: int) -> bool:
        """Check if number is prime"""
        if n < 2:
            return False
        if n == 2:
            return True
        if n % 2 == 0:
            return False
            
        for i in range(3, int(n**0.5) + 1, 2):
            if n % i == 0:
                return False
        return True
        
    def _analyze_decision_patterns(self) -> List[BehavioralPattern]:
        """Analyze decision-making behavioral patterns"""
        patterns = []
        
        if hasattr(self.vm, 'decision_history') and len(self.vm.decision_history) > 5:
            # Look for risk-taking vs conservative patterns
            risk_decisions = sum(1 for d in self.vm.decision_history 
                               if d.get('risk_level', 0) > 0.7)
            conservative_decisions = sum(1 for d in self.vm.decision_history 
                                       if d.get('risk_level', 0) < 0.3)
            
            if risk_decisions > conservative_decisions * 2:
                patterns.append(BehavioralPattern(
                    pattern_type='decision_making',
                    description='risk_taking',
                    trigger_conditions=['high_reward_opportunity', 'time_pressure'],
                    typical_responses=['aggressive_action', 'exploration'],
                    effectiveness_score=0.7
                ))
            elif conservative_decisions > risk_decisions * 2:
                patterns.append(BehavioralPattern(
                    pattern_type='decision_making',
                    description='conservative',
                    trigger_conditions=['uncertainty', 'potential_loss'],
                    typical_responses=['cautious_action', 'information_gathering'],
                    effectiveness_score=0.8
                ))
                
        return patterns
        
    def _analyze_goal_seeking_patterns(self) -> List[BehavioralPattern]:
        """Analyze goal-seeking behavioral patterns"""
        patterns = []
        
        if hasattr(self.vm, 'goals'):
            # Analyze goal approach strategies
            if any(hasattr(g, 'approach_strategy') for g in self.vm.goals):
                strategies = [g.approach_strategy for g in self.vm.goals 
                            if hasattr(g, 'approach_strategy')]
                
                strategy_counts = Counter(strategies)
                dominant_strategy = strategy_counts.most_common(1)[0][0] if strategy_counts else None
                
                if dominant_strategy:
                    patterns.append(BehavioralPattern(
                        pattern_type='goal_seeking',
                        description=f'dominant_{dominant_strategy}',
                        trigger_conditions=['goal_activation'],
                        typical_responses=[dominant_strategy],
                        effectiveness_score=self._calculate_strategy_effectiveness()
                    ))
                    
        return patterns
        
    def _analyze_adaptation_patterns(self) -> List[BehavioralPattern]:
        """Analyze adaptation behavioral patterns"""
        patterns = []
        
        # Check for learning from errors
        if hasattr(self.vm, 'error_history') and len(self.vm.error_history) > 3:
            error_types = [e.get('type', 'unknown') for e in self.vm.error_history]
            unique_errors = set(error_types)
            
            # Check if errors are decreasing (learning)
            recent_errors = error_types[-10:]
            if len(set(recent_errors)) < len(unique_errors) * 0.5:
                patterns.append(BehavioralPattern(
                    pattern_type='adaptation',
                    description='error_learning',
                    trigger_conditions=['error_occurrence'],
                    typical_responses=['error_avoidance', 'strategy_modification'],
                    effectiveness_score=0.85
                ))
                
        return patterns
        
    def _analyze_problem_solving_patterns(self) -> List[BehavioralPattern]:
        """Analyze problem-solving behavioral patterns"""
        patterns = []
        
        # Check execution trace for problem-solving indicators
        if len(self.vm.execution_trace) > 50:
            trace_str = ' '.join(str(i) for i in self.vm.execution_trace[-50:])
            
            # Look for iterative refinement
            if 'LOOP' in trace_str and 'OPTIMIZE' in trace_str:
                patterns.append(BehavioralPattern(
                    pattern_type='problem_solving',
                    description='iterative_refinement',
                    trigger_conditions=['complex_problem', 'optimization_goal'],
                    typical_responses=['repeated_attempts', 'gradual_improvement'],
                    effectiveness_score=0.75
                ))
                
            # Look for divide-and-conquer
            if 'SPLIT' in trace_str or 'DIVIDE' in trace_str:
                patterns.append(BehavioralPattern(
                    pattern_type='problem_solving',
                    description='divide_and_conquer',
                    trigger_conditions=['large_problem', 'decomposable_task'],
                    typical_responses=['problem_decomposition', 'parallel_solving'],
                    effectiveness_score=0.8
                ))
                
        return patterns
        
    def _analyze_decision_context(self, sequence: Tuple[str, ...]) -> str:
        """Analyze context of a decision sequence"""
        # Simple context analysis
        if 'explore' in sequence:
            return 'exploration_context'
        elif 'optimize' in sequence:
            return 'optimization_context'
        elif 'defend' in sequence:
            return 'defensive_context'
        else:
            return 'general_context'
            
    def _find_pattern_variations(self, pattern: List[str]) -> List[List[str]]:
        """Find variations of a pattern"""
        variations = []
        
        # Single instruction mutations
        for i in range(len(pattern)):
            # Deletion
            if len(pattern) > self.min_pattern_length:
                variation = pattern[:i] + pattern[i+1:]
                variations.append(variation)
                
            # Substitution (simplified)
            variation = pattern.copy()
            variation[i] = f"MUTATED_{pattern[i]}"
            variations.append(variation)
            
        # Insertion (simplified)
        if len(pattern) < self.max_pattern_length:
            for i in range(len(pattern) + 1):
                variation = pattern[:i] + ["INSERTED"] + pattern[i:]
                variations.append(variation)
                
        return variations
        
    def _is_beneficial_mutation(self, variation: List[str], 
                               original_pattern: ExecutionPattern) -> bool:
        """Check if a mutation is beneficial"""
        # Simplified check - would need actual execution testing
        variation_str = ''.join(str(i) for i in variation)
        
        # Check for optimization indicators
        if 'OPTIMIZE' in variation_str or 'IMPROVE' in variation_str:
            return True
            
        # Check if variation is already successful
        variation_id = self._generate_pattern_id(tuple(variation))
        if variation_id in self.execution_patterns:
            variation_pattern = self.execution_patterns[variation_id]
            return variation_pattern.success_rate > original_pattern.success_rate
            
        return False
        
    def _calculate_improvement(self, variation: List[str], 
                             original_pattern: ExecutionPattern) -> float:
        """Calculate improvement from mutation"""
        # Simplified calculation
        variation_id = self._generate_pattern_id(tuple(variation))
        
        if variation_id in self.execution_patterns:
            variation_pattern = self.execution_patterns[variation_id]
            return variation_pattern.success_rate - original_pattern.success_rate
            
        return 0.1  # Small assumed improvement
        
    def _analyze_pattern_combinations(self) -> List[Dict[str, Any]]:
        """Analyze combinations of patterns"""
        combinations = []
        
        # Look for patterns that frequently occur together
        pattern_ids = list(self.execution_patterns.keys())
        
        for combo_size in range(2, min(4, len(pattern_ids) + 1)):
            for combo in combinations(pattern_ids, combo_size):
                # Check if patterns occur together
                if self._patterns_occur_together(combo):
                    combinations.append({
                        'patterns': list(combo),
                        'synergy_score': self._calculate_synergy(combo),
                        'combined_effectiveness': self._calculate_combined_effectiveness(combo)
                    })
                    
        return combinations
        
    def _patterns_occur_together(self, pattern_ids: Tuple[str, ...]) -> bool:
        """Check if patterns appear within the configured window"""
        if len(pattern_ids) < 2:
            return False

        window = self.cooccurrence_window
        occurrence_lists = []
        for pid in pattern_ids:
            pattern = self.execution_patterns.get(pid)
            if not pattern or not pattern.occurrence_indices:
                return False
            occurrence_lists.append(pattern.occurrence_indices)

        # Check if there exists a set of occurrences within the window
        for idx in occurrence_lists[0]:
            if all(any(abs(idx - o) <= window for o in others)
                   for others in occurrence_lists[1:]):
                return True

        return False

    def _calculate_synergy(self, pattern_ids: Tuple[str, ...]) -> float:
        """Calculate synergy between patterns"""
        individual_scores = [
            self.execution_patterns[pid].success_rate
            for pid in pattern_ids if pid in self.execution_patterns
        ]

        if not individual_scores:
            return 0.0

        avg_score = sum(individual_scores) / len(individual_scores)
        synergy_bonus = 0.1 * len(pattern_ids)

        # Factor in temporal proximity
        window = self.cooccurrence_window
        proximity = 0.0
        occurrence_lists = [self.execution_patterns[pid].occurrence_indices for pid in pattern_ids]
        # compute minimal average difference
        diffs = []
        for idx in occurrence_lists[0]:
            distances = [min(abs(idx - o) for o in lst) for lst in occurrence_lists[1:]]
            diffs.append(sum(distances) / len(distances))
        if diffs:
            min_diff = min(diffs)
            proximity = max(0.0, 1 - (min_diff / (window + 1e-5)))

        return min(1.0, avg_score + synergy_bonus * proximity)
        
    def _calculate_combined_effectiveness(self, pattern_ids: Tuple[str, ...]) -> float:
        """Calculate combined effectiveness of patterns"""
        effectiveness_scores = []
        
        for pid in pattern_ids:
            if pid in self.execution_patterns:
                pattern = self.execution_patterns[pid]
                effectiveness_scores.append(pattern.success_rate)
                
        if not effectiveness_scores:
            return 0.0
            
        # Combined effectiveness with diminishing returns
        combined = 1.0
        for score in effectiveness_scores:
            combined *= (1 - (1 - score) * 0.8)  # 80% effectiveness combination
            
        return combined
        
    def _infer_capability_name(self, combo: Dict[str, Any]) -> Optional[str]:
        """Infer capability name from pattern combination"""
        pattern_ids = combo['patterns']
        
        # Analyze patterns to infer capability
        pattern_types = []
        for pid in pattern_ids:
            if pid in self.execution_patterns:
                pattern = self.execution_patterns[pid]
                # Extract pattern type from instruction sequence
                for instr in pattern.instruction_sequence:
                    if 'LOOP' in str(instr):
                        pattern_types.append('iteration')
                    elif 'DECIDE' in str(instr):
                        pattern_types.append('decision')
                    elif 'OPTIMIZE' in str(instr):
                        pattern_types.append('optimization')
                    elif 'LEARN' in str(instr):
                        pattern_types.append('learning')
                        
        # Infer capability based on pattern types
        if 'iteration' in pattern_types and 'optimization' in pattern_types:
            return 'iterative_optimization'
        elif 'decision' in pattern_types and 'learning' in pattern_types:
            return 'adaptive_decision_making'
        elif len(set(pattern_types)) >= 3:
            return 'complex_reasoning'
            
        return None
        
    def _calculate_emergence_confidence(self, combo: Dict[str, Any]) -> float:
        """Calculate confidence in emergent capability detection"""
        base_confidence = combo['synergy_score']
        
        # Boost confidence based on pattern frequency
        pattern_frequencies = []
        for pid in combo['patterns']:
            if pid in self.execution_patterns:
                pattern_frequencies.append(self.execution_patterns[pid].frequency)
                
        if pattern_frequencies:
            avg_frequency = sum(pattern_frequencies) / len(pattern_frequencies)
            frequency_boost = min(0.3, avg_frequency / 100)  # Cap at 0.3
            base_confidence += frequency_boost
            
        return min(1.0, base_confidence)
        
    def _measure_capability_performance(self, combo: Dict[str, Any]) -> Dict[str, float]:
        """Measure performance metrics for capability"""
        return {
            'effectiveness': combo['combined_effectiveness'],
            'reliability': self._calculate_reliability(combo),
            'efficiency': self._calculate_efficiency(combo),
            'adaptability': self._calculate_adaptability(combo)
        }
        
    def _calculate_reliability(self, combo: Dict[str, Any]) -> float:
        """Calculate reliability of capability"""
        # Based on consistency of pattern execution
        success_rates = []
        for pid in combo['patterns']:
            if pid in self.execution_patterns:
                success_rates.append(self.execution_patterns[pid].success_rate)
                
        if not success_rates:
            return 0.0
            
        # Low variance = high reliability
        variance = np.var(success_rates) if len(success_rates) > 1 else 0
        return 1.0 - min(1.0, variance)
        
    def _calculate_efficiency(self, combo: Dict[str, Any]) -> float:
        """Calculate efficiency of capability"""
        # Based on pattern length and execution time
        total_length = 0
        pattern_count = 0
        
        for pid in combo['patterns']:
            if pid in self.execution_patterns:
                pattern = self.execution_patterns[pid]
                total_length += len(pattern.instruction_sequence)
                pattern_count += 1
                
        if pattern_count == 0:
            return 0.5
            
        avg_length = total_length / pattern_count
        # Shorter patterns = more efficient
        efficiency = 1.0 - min(1.0, avg_length / self.max_pattern_length)
        
        return efficiency
        
    def _calculate_adaptability(self, combo: Dict[str, Any]) -> float:
        """Calculate adaptability of capability"""
        # Based on pattern evolution
        evolution_count = 0
        
        for pid in combo['patterns']:
            if pid in self.pattern_evolution:
                evolution = self.pattern_evolution[pid]
                evolution_count += len(evolution.mutations)
                
        # More mutations = more adaptable
        return min(1.0, evolution_count / 10)
        
    def _detect_consciousness_capabilities(self) -> List[EmergentCapability]:
        """Detect consciousness-related emergent capabilities"""
        capabilities = []
        
        # Check for self-referential patterns
        self_ref_patterns = [
            pid for pid, pattern in self.execution_patterns.items()
            if any('SELF' in str(instr) or 'REFLECT' in str(instr) 
                   for instr in pattern.instruction_sequence)
        ]
        
        if len(self_ref_patterns) >= 3:
            capabilities.append(EmergentCapability(
                capability_name='self_referential_processing',
                emergence_timestamp=time.time(),
                detection_confidence=0.8,
                underlying_patterns=self_ref_patterns[:3],
                performance_metrics={
                    'self_awareness': 0.7,
                    'introspection_depth': len(self_ref_patterns) / 10
                }
            ))
            
        # Check for metacognitive patterns
        meta_patterns = [
            pid for pid, pattern in self.execution_patterns.items()
            if any('META' in str(instr) or 'THINK' in str(instr) 
                   for instr in pattern.instruction_sequence)
        ]
        
        if meta_patterns:
            capabilities.append(EmergentCapability(
                capability_name='metacognitive_reasoning',
                emergence_timestamp=time.time(),
                detection_confidence=0.75,
                underlying_patterns=meta_patterns[:2],
                performance_metrics={
                    'reasoning_depth': len(meta_patterns) / 5,
                    'abstraction_level': 0.6
                }
            ))
            
        return capabilities
        
    def _calculate_strategy_effectiveness(self) -> float:
        """Calculate effectiveness of goal-seeking strategy"""
        if not hasattr(self.vm, 'goals'):
            return 0.5
            
        achieved_goals = sum(1 for g in self.vm.goals if hasattr(g, 'achieved') and g.achieved)
        total_goals = len(self.vm.goals)
        
        if total_goals == 0:
            return 0.5
            
        return achieved_goals / total_goals
