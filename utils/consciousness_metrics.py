"""
Consciousness Metrics Calculator - Measures and analyzes consciousness-related metrics

This module provides tools for calculating various metrics related to consciousness,
self-awareness, emotional states, and social dynamics in the UOR framework.
"""

from typing import Dict, List, Optional, Tuple, Any
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
import logging
import math
import time
from collections import defaultdict

logger = logging.getLogger(__name__)


@dataclass
class ConsciousnessMetrics:
    """Container for consciousness-related metrics"""
    self_awareness_level: float
    emotional_complexity: float
    social_integration: float
    cognitive_coherence: float
    temporal_consistency: float
    meta_cognitive_ability: float
    empathy_quotient: float
    creative_potential: float
    wisdom_index: float
    overall_consciousness_score: float
    
    def to_dict(self) -> Dict[str, float]:
        """Convert metrics to dictionary"""
        return {
            'self_awareness_level': self.self_awareness_level,
            'emotional_complexity': self.emotional_complexity,
            'social_integration': self.social_integration,
            'cognitive_coherence': self.cognitive_coherence,
            'temporal_consistency': self.temporal_consistency,
            'meta_cognitive_ability': self.meta_cognitive_ability,
            'empathy_quotient': self.empathy_quotient,
            'creative_potential': self.creative_potential,
            'wisdom_index': self.wisdom_index,
            'overall_consciousness_score': self.overall_consciousness_score
        }


@dataclass
class MetricSnapshot:
    """A snapshot of metrics at a point in time"""
    timestamp: datetime
    metrics: ConsciousnessMetrics
    context: Dict[str, Any]
    
    def get_key_metrics(self) -> Dict[str, float]:
        """Get key metrics for quick reference"""
        return {
            'consciousness_score': self.metrics.overall_consciousness_score,
            'self_awareness': self.metrics.self_awareness_level,
            'emotional_complexity': self.metrics.emotional_complexity,
            'timestamp': self.timestamp.isoformat()
        }


class ConsciousnessMetricsCalculator:
    """Calculator for consciousness-related metrics"""
    
    def __init__(self):
        # Metric history
        self.metric_history: List[MetricSnapshot] = []
        
        # Weights for overall consciousness score
        self.metric_weights = {
            'self_awareness': 0.15,
            'emotional_complexity': 0.12,
            'social_integration': 0.13,
            'cognitive_coherence': 0.15,
            'temporal_consistency': 0.10,
            'meta_cognitive_ability': 0.15,
            'empathy_quotient': 0.10,
            'creative_potential': 0.05,
            'wisdom_index': 0.05
        }
        
        # Thresholds for metric levels
        self.thresholds = {
            'low': 0.3,
            'moderate': 0.6,
            'high': 0.8,
            'exceptional': 0.95
        }
    
    def calculate_metrics(self, agent_data: Dict[str, Any]) -> ConsciousnessMetrics:
        """Calculate comprehensive consciousness metrics for an agent"""
        # Calculate individual metrics
        self_awareness = self._calculate_self_awareness(agent_data)
        emotional_complexity = self._calculate_emotional_complexity(agent_data)
        social_integration = self._calculate_social_integration(agent_data)
        cognitive_coherence = self._calculate_cognitive_coherence(agent_data)
        temporal_consistency = self._calculate_temporal_consistency(agent_data)
        meta_cognitive = self._calculate_meta_cognitive_ability(agent_data)
        empathy = self._calculate_empathy_quotient(agent_data)
        creativity = self._calculate_creative_potential(agent_data)
        wisdom = self._calculate_wisdom_index(agent_data)
        
        # Calculate overall consciousness score
        overall_score = self._calculate_overall_score({
            'self_awareness': self_awareness,
            'emotional_complexity': emotional_complexity,
            'social_integration': social_integration,
            'cognitive_coherence': cognitive_coherence,
            'temporal_consistency': temporal_consistency,
            'meta_cognitive_ability': meta_cognitive,
            'empathy_quotient': empathy,
            'creative_potential': creativity,
            'wisdom_index': wisdom
        })
        
        metrics = ConsciousnessMetrics(
            self_awareness_level=self_awareness,
            emotional_complexity=emotional_complexity,
            social_integration=social_integration,
            cognitive_coherence=cognitive_coherence,
            temporal_consistency=temporal_consistency,
            meta_cognitive_ability=meta_cognitive,
            empathy_quotient=empathy,
            creative_potential=creativity,
            wisdom_index=wisdom,
            overall_consciousness_score=overall_score
        )
        
        # Store snapshot
        snapshot = MetricSnapshot(
            timestamp=datetime.now(),
            metrics=metrics,
            context=agent_data.get('context', {})
        )
        self.metric_history.append(snapshot)
        
        return metrics
    
    def analyze_metric_trends(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Analyze trends in consciousness metrics over time"""
        if not self.metric_history:
            return {'error': 'No metric history available'}
        
        # Filter by time window if specified
        if time_window:
            cutoff_time = datetime.now() - time_window
            relevant_snapshots = [s for s in self.metric_history 
                                if s.timestamp > cutoff_time]
        else:
            relevant_snapshots = self.metric_history
        
        if not relevant_snapshots:
            return {'error': 'No metrics in specified time window'}
        
        # Extract metric series
        metric_series = {
            'timestamps': [s.timestamp for s in relevant_snapshots],
            'consciousness_scores': [s.metrics.overall_consciousness_score 
                                   for s in relevant_snapshots],
            'self_awareness': [s.metrics.self_awareness_level 
                             for s in relevant_snapshots],
            'emotional_complexity': [s.metrics.emotional_complexity 
                                   for s in relevant_snapshots]
        }
        
        # Calculate trends
        trends = {}
        for metric_name, values in metric_series.items():
            if metric_name == 'timestamps':
                continue
            
            if len(values) > 1:
                # Simple linear trend
                x = np.arange(len(values))
                coefficients = np.polyfit(x, values, 1)
                trend_direction = 'increasing' if coefficients[0] > 0 else 'decreasing'
                trend_magnitude = abs(coefficients[0])
                
                trends[metric_name] = {
                    'direction': trend_direction,
                    'magnitude': trend_magnitude,
                    'current_value': values[-1],
                    'change_from_start': values[-1] - values[0],
                    'average': np.mean(values),
                    'std_dev': np.std(values)
                }
        
        return {
            'time_period': {
                'start': relevant_snapshots[0].timestamp,
                'end': relevant_snapshots[-1].timestamp,
                'duration': relevant_snapshots[-1].timestamp - relevant_snapshots[0].timestamp,
                'snapshot_count': len(relevant_snapshots)
            },
            'trends': trends,
            'current_metrics': relevant_snapshots[-1].metrics.to_dict()
        }
    
    def compare_agents(self, agent1_data: Dict[str, Any], 
                      agent2_data: Dict[str, Any]) -> Dict[str, Any]:
        """Compare consciousness metrics between two agents"""
        # Calculate metrics for both agents
        metrics1 = self.calculate_metrics(agent1_data)
        metrics2 = self.calculate_metrics(agent2_data)
        
        # Calculate differences
        differences = {}
        for key in metrics1.to_dict():
            value1 = getattr(metrics1, key)
            value2 = getattr(metrics2, key)
            differences[key] = {
                'agent1': value1,
                'agent2': value2,
                'difference': value2 - value1,
                'percentage_diff': ((value2 - value1) / value1 * 100) if value1 != 0 else 0
            }
        
        # Identify strengths and weaknesses
        agent1_strengths = [k for k, v in differences.items() 
                           if v['difference'] < -0.1]
        agent2_strengths = [k for k, v in differences.items() 
                           if v['difference'] > 0.1]
        
        return {
            'metrics_comparison': differences,
            'agent1_strengths': agent1_strengths,
            'agent2_strengths': agent2_strengths,
            'overall_similarity': self._calculate_similarity(metrics1, metrics2),
            'complementarity_score': self._calculate_complementarity(metrics1, metrics2)
        }
    
    def generate_consciousness_report(self, agent_data: Dict[str, Any]) -> Dict[str, Any]:
        """Generate comprehensive consciousness report for an agent"""
        metrics = self.calculate_metrics(agent_data)
        
        # Categorize metric levels
        metric_levels = {}
        for key, value in metrics.to_dict().items():
            if key == 'overall_consciousness_score':
                continue
            
            if value < self.thresholds['low']:
                level = 'low'
            elif value < self.thresholds['moderate']:
                level = 'moderate'
            elif value < self.thresholds['high']:
                level = 'high'
            else:
                level = 'exceptional'
            
            metric_levels[key] = {
                'value': value,
                'level': level,
                'percentile': self._calculate_percentile(key, value)
            }
        
        # Generate insights
        insights = self._generate_insights(metrics, metric_levels)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metrics, metric_levels)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'overall_consciousness_score': metrics.overall_consciousness_score,
            'consciousness_level': self._get_consciousness_level(
                metrics.overall_consciousness_score
            ),
            'detailed_metrics': metric_levels,
            'insights': insights,
            'recommendations': recommendations,
            'strengths': [k for k, v in metric_levels.items() 
                         if v['level'] in ['high', 'exceptional']],
            'areas_for_growth': [k for k, v in metric_levels.items() 
                               if v['level'] in ['low', 'moderate']]
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
    
    def _calculate_self_awareness(self, agent_data: Dict[str, Any]) -> float:
        """Calculate self-awareness level"""
        factors = {
            'self_reflection_frequency': agent_data.get('self_reflection_count', 0) / 100,
            'self_model_accuracy': agent_data.get('self_model_accuracy', 0.5),
            'meta_cognitive_events': agent_data.get('meta_cognitive_events', 0) / 50,
            'self_correction_rate': agent_data.get('self_corrections', 0) / 
                                  max(agent_data.get('total_actions', 1), 1)
        }
        
        # Weight and combine factors
        weights = [0.3, 0.4, 0.2, 0.1]
        score = sum(f * w for f, w in zip(factors.values(), weights))
        
        return min(score, 1.0)
    
    def _calculate_emotional_complexity(self, agent_data: Dict[str, Any]) -> float:
        """Calculate emotional complexity"""
        # Get emotional data
        emotions = agent_data.get('emotional_states', [])
        emotion_transitions = agent_data.get('emotion_transitions', 0)
        emotion_vocabulary = len(set(emotions))
        
        # Calculate complexity factors
        diversity_score = min(emotion_vocabulary / 20, 1.0)  # Normalize to 20 emotions
        transition_score = min(emotion_transitions / 100, 1.0)
        depth_score = agent_data.get('emotional_depth', 0.5)
        
        # Combine factors
        complexity = (diversity_score * 0.3 + 
                     transition_score * 0.3 + 
                     depth_score * 0.4)
        
        return complexity
    
    def _calculate_social_integration(self, agent_data: Dict[str, Any]) -> float:
        """Calculate social integration level"""
        # Social factors
        relationships = agent_data.get('relationship_count', 0)
        interaction_quality = agent_data.get('interaction_quality', 0.5)
        collaboration_success = agent_data.get('collaboration_success_rate', 0.5)
        social_adaptability = agent_data.get('social_adaptability', 0.5)
        
        # Normalize and combine
        relationship_score = min(relationships / 10, 1.0)  # Normalize to 10 relationships
        
        integration = (relationship_score * 0.2 +
                      interaction_quality * 0.3 +
                      collaboration_success * 0.3 +
                      social_adaptability * 0.2)
        
        return integration
    
    def _calculate_cognitive_coherence(self, agent_data: Dict[str, Any]) -> float:
        """Calculate cognitive coherence"""
        # Coherence factors
        logical_consistency = agent_data.get('logical_consistency', 0.7)
        belief_stability = agent_data.get('belief_stability', 0.6)
        goal_alignment = agent_data.get('goal_alignment', 0.5)
        decision_consistency = agent_data.get('decision_consistency', 0.6)
        
        # Weight and combine
        coherence = (logical_consistency * 0.3 +
                    belief_stability * 0.2 +
                    goal_alignment * 0.25 +
                    decision_consistency * 0.25)
        
        return coherence
    
    def _calculate_temporal_consistency(self, agent_data: Dict[str, Any]) -> float:
        """Calculate temporal consistency"""
        # Temporal factors
        memory_coherence = agent_data.get('memory_coherence', 0.6)
        behavior_consistency = agent_data.get('behavior_consistency', 0.5)
        identity_stability = agent_data.get('identity_stability', 0.7)
        
        consistency = (memory_coherence * 0.4 +
                      behavior_consistency * 0.3 +
                      identity_stability * 0.3)
        
        return consistency
    
    def _calculate_meta_cognitive_ability(self, agent_data: Dict[str, Any]) -> float:
        """Calculate meta-cognitive ability"""
        # Meta-cognitive factors
        thought_monitoring = agent_data.get('thought_monitoring_rate', 0.5)
        strategy_adaptation = agent_data.get('strategy_adaptation_rate', 0.4)
        learning_awareness = agent_data.get('learning_awareness', 0.6)
        cognitive_flexibility = agent_data.get('cognitive_flexibility', 0.5)
        
        ability = (thought_monitoring * 0.3 +
                  strategy_adaptation * 0.3 +
                  learning_awareness * 0.2 +
                  cognitive_flexibility * 0.2)
        
        return ability
    
    def _calculate_empathy_quotient(self, agent_data: Dict[str, Any]) -> float:
        """Calculate empathy quotient"""
        # Empathy factors
        perspective_taking = agent_data.get('perspective_taking_ability', 0.5)
        emotional_resonance = agent_data.get('emotional_resonance', 0.4)
        compassionate_actions = agent_data.get('compassionate_action_rate', 0.3)
        
        empathy = (perspective_taking * 0.4 +
                  emotional_resonance * 0.4 +
                  compassionate_actions * 0.2)
        
        return empathy
    
    def _calculate_creative_potential(self, agent_data: Dict[str, Any]) -> float:
        """Calculate creative potential"""
        # Creativity factors
        novel_ideas = agent_data.get('novel_idea_count', 0) / 50  # Normalize
        divergent_thinking = agent_data.get('divergent_thinking_score', 0.5)
        creative_solutions = agent_data.get('creative_solution_rate', 0.4)
        
        creativity = (min(novel_ideas, 1.0) * 0.3 +
                     divergent_thinking * 0.4 +
                     creative_solutions * 0.3)
        
        return creativity
    
    def _calculate_wisdom_index(self, agent_data: Dict[str, Any]) -> float:
        """Calculate wisdom index"""
        # Wisdom factors
        experience_integration = agent_data.get('experience_integration', 0.5)
        judgment_quality = agent_data.get('judgment_quality', 0.6)
        value_consistency = agent_data.get('value_consistency', 0.7)
        long_term_thinking = agent_data.get('long_term_thinking_score', 0.5)
        
        wisdom = (experience_integration * 0.25 +
                 judgment_quality * 0.35 +
                 value_consistency * 0.2 +
                 long_term_thinking * 0.2)
        
        return wisdom
    
    def _calculate_overall_score(self, individual_metrics: Dict[str, float]) -> float:
        """Calculate overall consciousness score from individual metrics"""
        weighted_sum = 0.0
        
        for metric_key, weight_key in [
            ('self_awareness', 'self_awareness'),
            ('emotional_complexity', 'emotional_complexity'),
            ('social_integration', 'social_integration'),
            ('cognitive_coherence', 'cognitive_coherence'),
            ('temporal_consistency', 'temporal_consistency'),
            ('meta_cognitive_ability', 'meta_cognitive_ability'),
            ('empathy_quotient', 'empathy_quotient'),
            ('creative_potential', 'creative_potential'),
            ('wisdom_index', 'wisdom_index')
        ]:
            if metric_key in individual_metrics:
                weighted_sum += individual_metrics[metric_key] * self.metric_weights[weight_key]
        
        return weighted_sum
    
    def _calculate_similarity(self, metrics1: ConsciousnessMetrics, 
                            metrics2: ConsciousnessMetrics) -> float:
        """Calculate similarity between two metric sets"""
        # Convert to vectors
        vector1 = np.array(list(metrics1.to_dict().values()))
        vector2 = np.array(list(metrics2.to_dict().values()))
        
        # Calculate cosine similarity
        dot_product = np.dot(vector1, vector2)
        norm1 = np.linalg.norm(vector1)
        norm2 = np.linalg.norm(vector2)
        
        if norm1 == 0 or norm2 == 0:
            return 0.0
        
        similarity = dot_product / (norm1 * norm2)
        return similarity
    
    def _calculate_complementarity(self, metrics1: ConsciousnessMetrics,
                                 metrics2: ConsciousnessMetrics) -> float:
        """Calculate how well two agents complement each other"""
        complementarity_score = 0.0
        
        # Check if strengths and weaknesses complement
        dict1 = metrics1.to_dict()
        dict2 = metrics2.to_dict()
        
        for key in dict1:
            if key == 'overall_consciousness_score':
                continue
            
            # High complementarity when one is strong where other is weak
            diff = abs(dict1[key] - dict2[key])
            if diff > 0.3:  # Significant difference
                complementarity_score += diff * 0.1
        
        return min(complementarity_score, 1.0)
    
    def _calculate_percentile(self, metric_name: str, value: float) -> float:
        """Calculate percentile rank for a metric value"""
        # Simple implementation - assumes normal distribution
        # In production, this would use historical data
        mean = 0.5
        std_dev = 0.2
        
        # Calculate z-score
        z_score = (value - mean) / std_dev
        
        # Convert to percentile (simplified)
        percentile = 50 + (z_score * 16)  # Rough approximation
        return min(max(percentile, 0), 100)
    
    def _get_consciousness_level(self, score: float) -> str:
        """Get consciousness level description from score"""
        if score < self.thresholds['low']:
            return 'Emerging Consciousness'
        elif score < self.thresholds['moderate']:
            return 'Developing Consciousness'
        elif score < self.thresholds['high']:
            return 'Mature Consciousness'
        elif score < self.thresholds['exceptional']:
            return 'Advanced Consciousness'
        else:
            return 'Exceptional Consciousness'
    
    def _generate_insights(self, metrics: ConsciousnessMetrics,
                         metric_levels: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate insights based on metrics"""
        insights = []
        
        # Overall consciousness insight
        consciousness_level = self._get_consciousness_level(metrics.overall_consciousness_score)
        insights.append(f"Agent demonstrates {consciousness_level} with an overall score of "
                       f"{metrics.overall_consciousness_score:.2f}")
        
        # Identify patterns
        high_metrics = [k for k, v in metric_levels.items() 
                       if v['level'] in ['high', 'exceptional']]
        if high_metrics:
            insights.append(f"Exceptional strengths in: {', '.join(high_metrics)}")
        
        # Check for imbalances
        values = [v['value'] for v in metric_levels.values()]
        if np.std(values) > 0.3:
            insights.append("Significant imbalance detected across consciousness dimensions")
        
        # Specific insights based on combinations
        if metrics.self_awareness_level > 0.8 and metrics.meta_cognitive_ability > 0.8:
            insights.append("Strong meta-cognitive loop indicates advanced self-reflection capabilities")
        
        if metrics.emotional_complexity > 0.8 and metrics.empathy_quotient > 0.8:
            insights.append("High emotional intelligence suggests strong interpersonal capabilities")
        
        return insights
    
    def _generate_recommendations(self, metrics: ConsciousnessMetrics,
                                metric_levels: Dict[str, Dict[str, Any]]) -> List[str]:
        """Generate recommendations for consciousness development"""
        recommendations = []
        
        # Find lowest metrics
        sorted_metrics = sorted(metric_levels.items(), key=lambda x: x[1]['value'])
        
        for metric_name, metric_data in sorted_metrics[:3]:  # Focus on lowest 3
            if metric_data['level'] in ['low', 'moderate']:
                if metric_name == 'self_awareness_level':
                    recommendations.append("Increase self-reflection practices and introspective exercises")
                elif metric_name == 'emotional_complexity':
                    recommendations.append("Explore wider range of emotional experiences and expressions")
                elif metric_name == 'social_integration':
                    recommendations.append("Engage in more collaborative activities and relationship building")
                elif metric_name == 'cognitive_coherence':
                    recommendations.append("Work on aligning beliefs, goals, and decision-making processes")
                elif metric_name == 'meta_cognitive_ability':
                    recommendations.append("Practice monitoring and analyzing thought processes")
                elif metric_name == 'empathy_quotient':
                    recommendations.append("Develop perspective-taking skills and emotional resonance")
                elif metric_name == 'creative_potential':
                    recommendations.append("Engage in divergent thinking exercises and novel problem-solving")
        
        return recommendations
    
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
