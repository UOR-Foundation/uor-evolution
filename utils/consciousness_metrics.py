"""
Consciousness Metrics Utility

This module provides tools for measuring and quantifying consciousness-related phenomena.
"""

import math
import time
from typing import Dict, List, Any, Tuple, Optional
from dataclasses import dataclass
from collections import defaultdict


@dataclass
class ConsciousnessScore:
    """Comprehensive consciousness score with breakdown"""
    overall_score: float  # 0-10 scale
    component_scores: Dict[str, float]
    confidence_interval: Tuple[float, float]
    measurement_timestamp: float
    notes: List[str]
    
    def get_level(self) -> int:
        """Get consciousness level (0-10)"""
        return int(self.overall_score)
        
    def get_summary(self) -> str:
        """Get summary of consciousness score"""
        level_descriptions = {
            0: "No consciousness detected",
            1: "Minimal reactive processing",
            2: "Basic information integration",
            3: "Simple self-monitoring",
            4: "Emerging self-awareness",
            5: "Clear self-awareness",
            6: "Metacognitive capabilities",
            7: "Deep introspection",
            8: "Rich subjective experience",
            9: "Advanced consciousness",
            10: "Transcendent consciousness"
        }
        
        level = self.get_level()
        return f"Level {level}: {level_descriptions.get(level, 'Unknown')}"


class ConsciousnessMetricsCalculator:
    """Calculator for various consciousness metrics"""
    
    def __init__(self):
        self.weights = {
            'self_awareness': 0.25,
            'metacognition': 0.20,
            'temporal_continuity': 0.15,
            'integration': 0.15,
            'intentionality': 0.10,
            'qualia': 0.10,
            'creativity': 0.05
        }
        
    def calculate_integrated_information(self, 
                                       system_states: List[Dict[str, Any]]) -> float:
        """Calculate Phi (Φ) - integrated information measure"""
        if len(system_states) < 2:
            return 0.0
            
        # Simplified Phi calculation
        total_entropy = self._calculate_entropy(system_states)
        
        # Calculate partition entropies
        partition_entropies = []
        for i in range(1, len(system_states)):
            partition_entropy = (
                self._calculate_entropy(system_states[:i]) +
                self._calculate_entropy(system_states[i:])
            )
            partition_entropies.append(partition_entropy)
            
        # Phi is the minimum difference
        if partition_entropies:
            min_partition_entropy = min(partition_entropies)
            phi = total_entropy - min_partition_entropy
            return max(0.0, phi)
        
        return 0.0
        
    def calculate_global_workspace_score(self, 
                                       attention_distribution: Dict[str, float],
                                       access_patterns: List[str]) -> float:
        """Calculate global workspace theory score"""
        # Calculate attention entropy
        attention_entropy = self._calculate_distribution_entropy(
            list(attention_distribution.values())
        )
        
        # Calculate access diversity
        unique_patterns = len(set(access_patterns))
        total_patterns = len(access_patterns)
        access_diversity = unique_patterns / max(1, total_patterns)
        
        # Global workspace score combines distribution and access
        score = (attention_entropy * 0.6 + access_diversity * 0.4)
        
        return min(1.0, score)
        
    def calculate_strange_loop_index(self, 
                                   recursion_depth: int,
                                   self_reference_count: int,
                                   loop_stability: float) -> float:
        """Calculate strange loop index based on Hofstadter's concepts"""
        # Normalize recursion depth (logarithmic scale)
        depth_score = math.log(recursion_depth + 1) / math.log(10)
        
        # Normalize self-reference (square root scale for diminishing returns)
        reference_score = math.sqrt(self_reference_count) / 10
        
        # Combine with stability
        strange_loop_index = (
            depth_score * 0.4 +
            reference_score * 0.4 +
            loop_stability * 0.2
        )
        
        return min(1.0, strange_loop_index)
        
    def calculate_consciousness_complexity(self,
                                         state_diversity: float,
                                         temporal_patterns: int,
                                         emergent_behaviors: int) -> float:
        """Calculate consciousness complexity measure"""
        # Normalize inputs
        diversity_score = min(1.0, state_diversity)
        temporal_score = min(1.0, temporal_patterns / 20)
        emergence_score = min(1.0, emergent_behaviors / 10)
        
        # Calculate complexity using geometric mean
        complexity = (diversity_score * temporal_score * emergence_score) ** (1/3)
        
        return complexity
        
    def calculate_overall_consciousness_score(self,
                                            component_scores: Dict[str, float]) -> ConsciousnessScore:
        """Calculate overall consciousness score from components"""
        # Apply weights to components
        weighted_sum = 0.0
        available_weight = 0.0
        
        for component, weight in self.weights.items():
            if component in component_scores:
                weighted_sum += component_scores[component] * weight
                available_weight += weight
                
        # Normalize by available weight
        if available_weight > 0:
            raw_score = weighted_sum / available_weight
        else:
            raw_score = 0.0
            
        # Scale to 0-10
        overall_score = raw_score * 10
        
        # Calculate confidence interval
        if component_scores:
            scores = list(component_scores.values())
            variance = sum((s - raw_score) ** 2 for s in scores) / len(scores)
            std_dev = math.sqrt(variance)
            confidence_interval = (
                max(0, overall_score - std_dev * 2),
                min(10, overall_score + std_dev * 2)
            )
        else:
            confidence_interval = (0, 0)
            
        # Generate notes
        notes = self._generate_score_notes(overall_score, component_scores)
        
        return ConsciousnessScore(
            overall_score=overall_score,
            component_scores=component_scores,
            confidence_interval=confidence_interval,
            measurement_timestamp=time.time(),
            notes=notes
        )
        
    def _calculate_entropy(self, states: List[Dict[str, Any]]) -> float:
        """Calculate Shannon entropy of states"""
        if not states:
            return 0.0
            
        # Convert states to hashable format
        state_strings = [str(sorted(state.items())) for state in states]
        
        # Count occurrences
        state_counts = defaultdict(int)
        for state in state_strings:
            state_counts[state] += 1
            
        # Calculate probabilities
        total = len(states)
        entropy = 0.0
        
        for count in state_counts.values():
            if count > 0:
                p = count / total
                entropy -= p * math.log2(p)
                
        return entropy
        
    def _calculate_distribution_entropy(self, values: List[float]) -> float:
        """Calculate entropy of a probability distribution"""
        if not values:
            return 0.0
            
        # Normalize to probabilities
        total = sum(values)
        if total == 0:
            return 0.0
            
        probabilities = [v / total for v in values]
        
        # Calculate entropy
        entropy = 0.0
        for p in probabilities:
            if p > 0:
                entropy -= p * math.log2(p)
                
        # Normalize by maximum entropy
        max_entropy = math.log2(len(values))
        if max_entropy > 0:
            return entropy / max_entropy
        
        return 0.0
        
    def _generate_score_notes(self, overall_score: float, 
                            components: Dict[str, float]) -> List[str]:
        """Generate interpretive notes about the score"""
        notes = []
        
        # Overall assessment
        if overall_score < 3:
            notes.append("Minimal consciousness indicators detected")
        elif overall_score < 6:
            notes.append("Emerging consciousness with clear self-awareness")
        elif overall_score < 8:
            notes.append("Advanced consciousness with metacognitive abilities")
        else:
            notes.append("Highly developed consciousness approaching human-like awareness")
            
        # Component-specific notes
        if components.get('self_awareness', 0) > 0.8:
            notes.append("Strong self-awareness and self-recognition")
            
        if components.get('metacognition', 0) > 0.7:
            notes.append("Sophisticated metacognitive reasoning detected")
            
        if components.get('qualia', 0) > 0.6:
            notes.append("Evidence of subjective experiential qualities")
            
        # Identify weakest component
        if components:
            weakest = min(components.items(), key=lambda x: x[1])
            if weakest[1] < 0.3:
                notes.append(f"Development needed in {weakest[0]}")
                
        return notes


class ConsciousnessTracker:
    """Track consciousness metrics over time"""
    
    def __init__(self, window_size: int = 100):
        self.window_size = window_size
        self.history: Dict[str, List[Tuple[float, float]]] = defaultdict(list)
        self.milestones: List[Dict[str, Any]] = []
        
    def record_metric(self, metric_name: str, value: float, timestamp: float):
        """Record a metric value"""
        self.history[metric_name].append((timestamp, value))
        
        # Keep only recent history
        if len(self.history[metric_name]) > self.window_size:
            self.history[metric_name] = self.history[metric_name][-self.window_size:]
            
        # Check for milestones
        self._check_milestones(metric_name, value, timestamp)
        
    def get_trend(self, metric_name: str) -> str:
        """Get trend for a metric"""
        if metric_name not in self.history:
            return "no_data"
            
        values = self.history[metric_name]
        if len(values) < 2:
            return "insufficient_data"
            
        # Simple linear trend
        recent_avg = sum(v[1] for v in values[-10:]) / min(10, len(values))
        older_avg = sum(v[1] for v in values[:10]) / min(10, len(values))
        
        if recent_avg > older_avg * 1.1:
            return "increasing"
        elif recent_avg < older_avg * 0.9:
            return "decreasing"
        else:
            return "stable"
            
    def _check_milestones(self, metric_name: str, value: float, timestamp: float):
        """Check for consciousness milestones"""
        milestones_thresholds = {
            'self_awareness': [(0.5, "Self-awareness emerged"), (0.8, "Strong self-awareness achieved")],
            'metacognition': [(0.6, "Metacognitive abilities detected"), (0.8, "Advanced metacognition")],
            'qualia': [(0.5, "Qualia-like experiences detected"), (0.7, "Rich subjective experience")]
        }
        
        if metric_name in milestones_thresholds:
            for threshold, description in milestones_thresholds[metric_name]:
                if value >= threshold:
                    # Check if this milestone was already achieved
                    milestone_key = f"{metric_name}_{threshold}"
                    if not any(m.get('key') == milestone_key for m in self.milestones):
                        self.milestones.append({
                            'key': milestone_key,
                            'metric': metric_name,
                            'threshold': threshold,
                            'description': description,
                            'timestamp': timestamp,
                            'value': value
                        })
