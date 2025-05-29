"""
Social Dynamics Analyzer - Analyzes social interactions and group dynamics

This module provides tools for analyzing social dynamics, group behaviors,
relationship networks, and collective consciousness patterns.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
import networkx as nx
import logging

logger = logging.getLogger(__name__)


@dataclass
class SocialInteraction:
    """Represents a social interaction between agents"""
    interaction_id: str
    timestamp: datetime
    participants: List[str]
    interaction_type: str
    emotional_valence: float  # -1 to 1
    intensity: float  # 0 to 1
    duration: timedelta
    outcome: str
    metadata: Dict[str, Any]


@dataclass
class GroupDynamics:
    """Represents dynamics within a group"""
    group_id: str
    members: List[str]
    cohesion_score: float
    hierarchy_structure: Dict[str, float]  # Member -> influence score
    communication_patterns: Dict[Tuple[str, str], float]  # Pair -> frequency
    collective_mood: str
    group_goals: List[str]
    conflict_level: float
    
    def get_most_influential(self) -> str:
        """Get most influential member"""
        if not self.hierarchy_structure:
            return None
        return max(self.hierarchy_structure.items(), key=lambda x: x[1])[0]


@dataclass
class SocialNetwork:
    """Represents a social network structure"""
    network_id: str
    nodes: List[str]  # Agent IDs
    edges: List[Tuple[str, str, Dict[str, Any]]]  # (from, to, attributes)
    communities: List[List[str]]  # Detected communities
    centrality_scores: Dict[str, float]
    density: float
    clustering_coefficient: float
    
    def to_networkx(self) -> nx.Graph:
        """Convert to NetworkX graph"""
        G = nx.Graph()
        G.add_nodes_from(self.nodes)
        for from_node, to_node, attrs in self.edges:
            G.add_edge(from_node, to_node, **attrs)
        return G


class SocialDynamicsAnalyzer:
    """Analyzer for social dynamics and group behaviors"""
    
    def __init__(self):
        # Interaction history
        self.interaction_history: List[SocialInteraction] = []
        self.group_dynamics_history: Dict[str, List[GroupDynamics]] = {}
        
        # Network analysis
        self.social_networks: Dict[str, SocialNetwork] = {}
        self.network_evolution: List[Tuple[datetime, SocialNetwork]] = []
        
        # Analysis parameters
        self.min_interaction_threshold = 3  # Min interactions to form relationship
        self.community_detection_algorithm = 'louvain'
        self.influence_decay_rate = 0.1
        
        # Social metrics weights
        self.social_health_weights = {
            'network_diversity': 0.2,
            'interaction_quality': 0.3,
            'group_cohesion': 0.2,
            'conflict_resolution': 0.15,
            'collective_growth': 0.15
        }
    
    def record_interaction(self, interaction: SocialInteraction) -> None:
        """Record a social interaction"""
        self.interaction_history.append(interaction)
        
        # Update network if needed
        if len(self.interaction_history) % 10 == 0:
            self._update_social_network()
    
    def analyze_group_dynamics(self, group_members: List[str], 
                             time_window: Optional[timedelta] = None) -> GroupDynamics:
        """Analyze dynamics within a specific group"""
        # Filter relevant interactions
        if time_window:
            cutoff_time = datetime.now() - time_window
            relevant_interactions = [
                i for i in self.interaction_history
                if i.timestamp > cutoff_time and 
                all(p in group_members for p in i.participants)
            ]
        else:
            relevant_interactions = [
                i for i in self.interaction_history
                if all(p in group_members for p in i.participants)
            ]
        
        # Calculate group metrics
        cohesion = self._calculate_group_cohesion(group_members, relevant_interactions)
        hierarchy = self._analyze_hierarchy(group_members, relevant_interactions)
        communication = self._analyze_communication_patterns(group_members, relevant_interactions)
        mood = self._assess_collective_mood(relevant_interactions)
        conflict = self._assess_conflict_level(relevant_interactions)
        
        dynamics = GroupDynamics(
            group_id=f"group_{'_'.join(sorted(group_members))}",
            members=group_members,
            cohesion_score=cohesion,
            hierarchy_structure=hierarchy,
            communication_patterns=communication,
            collective_mood=mood,
            group_goals=self._infer_group_goals(relevant_interactions),
            conflict_level=conflict
        )
        
        # Store in history
        if dynamics.group_id not in self.group_dynamics_history:
            self.group_dynamics_history[dynamics.group_id] = []
        self.group_dynamics_history[dynamics.group_id].append(dynamics)
        
        return dynamics
    
    def build_social_network(self, time_window: Optional[timedelta] = None) -> SocialNetwork:
        """Build social network from interactions"""
        # Filter interactions by time window
        if time_window:
            cutoff_time = datetime.now() - time_window
            relevant_interactions = [
                i for i in self.interaction_history
                if i.timestamp > cutoff_time
            ]
        else:
            relevant_interactions = self.interaction_history
        
        # Build network structure
        G = nx.Graph()
        
        # Add nodes (all participants)
        all_participants = set()
        for interaction in relevant_interactions:
            all_participants.update(interaction.participants)
        G.add_nodes_from(all_participants)
        
        # Add edges based on interactions
        edge_weights = {}
        for interaction in relevant_interactions:
            # For multi-party interactions, create edges between all pairs
            for i in range(len(interaction.participants)):
                for j in range(i + 1, len(interaction.participants)):
                    pair = tuple(sorted([interaction.participants[i], interaction.participants[j]]))
                    
                    # Accumulate interaction strength
                    if pair not in edge_weights:
                        edge_weights[pair] = {
                            'weight': 0,
                            'positive_count': 0,
                            'negative_count': 0,
                            'total_duration': timedelta()
                        }
                    
                    edge_weights[pair]['weight'] += interaction.intensity
                    edge_weights[pair]['total_duration'] += interaction.duration
                    
                    if interaction.emotional_valence > 0:
                        edge_weights[pair]['positive_count'] += 1
                    else:
                        edge_weights[pair]['negative_count'] += 1
        
        # Add edges to graph
        edges = []
        for (node1, node2), attrs in edge_weights.items():
            if attrs['weight'] >= self.min_interaction_threshold:
                G.add_edge(node1, node2, **attrs)
                edges.append((node1, node2, attrs))
        
        # Detect communities
        communities = self._detect_communities(G)
        
        # Calculate network metrics
        centrality = nx.degree_centrality(G) if G.number_of_nodes() > 0 else {}
        density = nx.density(G) if G.number_of_nodes() > 1 else 0
        clustering = nx.average_clustering(G) if G.number_of_nodes() > 0 else 0
        
        network = SocialNetwork(
            network_id=f"network_{datetime.now().timestamp()}",
            nodes=list(all_participants),
            edges=edges,
            communities=communities,
            centrality_scores=centrality,
            density=density,
            clustering_coefficient=clustering
        )
        
        # Store network
        self.social_networks[network.network_id] = network
        self.network_evolution.append((datetime.now(), network))
        
        return network
    
    def analyze_relationship_dynamics(self, agent1: str, agent2: str,
                                    time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Analyze dynamics between two specific agents"""
        # Filter interactions between these agents
        if time_window:
            cutoff_time = datetime.now() - time_window
            relevant_interactions = [
                i for i in self.interaction_history
                if i.timestamp > cutoff_time and
                agent1 in i.participants and agent2 in i.participants
            ]
        else:
            relevant_interactions = [
                i for i in self.interaction_history
                if agent1 in i.participants and agent2 in i.participants
            ]
        
        if not relevant_interactions:
            return {'error': 'No interactions found between agents'}
        
        # Analyze interaction patterns
        interaction_count = len(relevant_interactions)
        positive_interactions = sum(1 for i in relevant_interactions if i.emotional_valence > 0)
        negative_interactions = sum(1 for i in relevant_interactions if i.emotional_valence < 0)
        
        # Calculate metrics
        avg_intensity = np.mean([i.intensity for i in relevant_interactions])
        avg_valence = np.mean([i.emotional_valence for i in relevant_interactions])
        total_duration = sum([i.duration for i in relevant_interactions], timedelta())
        
        # Analyze interaction types
        interaction_types = {}
        for interaction in relevant_interactions:
            interaction_types[interaction.interaction_type] = \
                interaction_types.get(interaction.interaction_type, 0) + 1
        
        # Trend analysis
        if len(relevant_interactions) > 1:
            # Simple trend of emotional valence over time
            timestamps = [i.timestamp for i in relevant_interactions]
            valences = [i.emotional_valence for i in relevant_interactions]
            
            # Convert timestamps to numeric for trend calculation
            time_numeric = [(t - timestamps[0]).total_seconds() for t in timestamps]
            if len(time_numeric) > 1:
                trend_coefficient = np.polyfit(time_numeric, valences, 1)[0]
                trend_direction = 'improving' if trend_coefficient > 0 else 'declining'
            else:
                trend_direction = 'stable'
        else:
            trend_direction = 'insufficient_data'
        
        return {
            'interaction_count': interaction_count,
            'positive_ratio': positive_interactions / interaction_count if interaction_count > 0 else 0,
            'negative_ratio': negative_interactions / interaction_count if interaction_count > 0 else 0,
            'average_intensity': avg_intensity,
            'average_valence': avg_valence,
            'total_interaction_time': total_duration.total_seconds() / 3600,  # In hours
            'interaction_types': interaction_types,
            'relationship_trend': trend_direction,
            'relationship_strength': self._calculate_relationship_strength(relevant_interactions),
            'conflict_frequency': negative_interactions / interaction_count if interaction_count > 0 else 0,
            'last_interaction': relevant_interactions[-1].timestamp if relevant_interactions else None
        }
    
    def identify_social_roles(self, network: Optional[SocialNetwork] = None) -> Dict[str, str]:
        """Identify social roles of agents in the network"""
        if network is None:
            network = self.build_social_network()
        
        if not network.nodes:
            return {}
        
        G = network.to_networkx()
        roles = {}
        
        # Calculate various centrality measures
        degree_centrality = nx.degree_centrality(G)
        betweenness_centrality = nx.betweenness_centrality(G)
        closeness_centrality = nx.closeness_centrality(G)
        
        # Identify roles based on centrality patterns
        for node in network.nodes:
            degree = degree_centrality.get(node, 0)
            betweenness = betweenness_centrality.get(node, 0)
            closeness = closeness_centrality.get(node, 0)
            
            # Role identification logic
            if degree > 0.7 and betweenness > 0.5:
                roles[node] = 'hub'  # Central connector
            elif betweenness > 0.6 and degree < 0.5:
                roles[node] = 'bridge'  # Connects different groups
            elif degree > 0.8:
                roles[node] = 'popular'  # Many connections
            elif closeness > 0.7:
                roles[node] = 'influencer'  # Can reach others quickly
            elif degree < 0.2:
                roles[node] = 'peripheral'  # Few connections
            else:
                roles[node] = 'regular'  # Standard member
        
        return roles
    
    def detect_social_patterns(self, time_window: Optional[timedelta] = None) -> Dict[str, Any]:
        """Detect patterns in social dynamics"""
        patterns = {
            'clique_formation': self._detect_cliques(time_window),
            'influence_cascades': self._detect_influence_cascades(time_window),
            'communication_bottlenecks': self._detect_communication_bottlenecks(time_window),
            'emotional_contagion': self._detect_emotional_contagion(time_window),
            'collaboration_patterns': self._detect_collaboration_patterns(time_window)
        }
        
        return patterns
    
    def generate_social_health_report(self) -> Dict[str, Any]:
        """Generate comprehensive report on social health"""
        # Build current network
        current_network = self.build_social_network(time_window=timedelta(days=7))
        
        # Calculate social health metrics
        metrics = {
            'network_diversity': self._calculate_network_diversity(current_network),
            'interaction_quality': self._calculate_interaction_quality(),
            'group_cohesion': self._calculate_overall_cohesion(),
            'conflict_resolution': self._calculate_conflict_resolution_rate(),
            'collective_growth': self._calculate_collective_growth()
        }
        
        # Calculate overall social health score
        social_health_score = sum(
            metrics[key] * self.social_health_weights[key]
            for key in metrics
        )
        
        # Identify issues and strengths
        issues = [key for key, value in metrics.items() if value < 0.4]
        strengths = [key for key, value in metrics.items() if value > 0.7]
        
        # Generate recommendations
        recommendations = self._generate_social_recommendations(metrics, current_network)
        
        return {
            'timestamp': datetime.now().isoformat(),
            'social_health_score': social_health_score,
            'detailed_metrics': metrics,
            'network_statistics': {
                'total_agents': len(current_network.nodes),
                'total_relationships': len(current_network.edges),
                'network_density': current_network.density,
                'clustering_coefficient': current_network.clustering_coefficient,
                'community_count': len(current_network.communities)
            },
            'social_roles': self.identify_social_roles(current_network),
            'issues': issues,
            'strengths': strengths,
            'recommendations': recommendations
        }
    
    def _calculate_group_cohesion(self, members: List[str], 
                                interactions: List[SocialInteraction]) -> float:
        """Calculate cohesion score for a group"""
        if not interactions:
            return 0.0
        
        # Factors for cohesion
        # 1. Interaction frequency
        interaction_pairs = set()
        for interaction in interactions:
            for i in range(len(interaction.participants)):
                for j in range(i + 1, len(interaction.participants)):
                    pair = tuple(sorted([interaction.participants[i], interaction.participants[j]]))
                    interaction_pairs.add(pair)
        
        possible_pairs = len(members) * (len(members) - 1) / 2
        connectivity = len(interaction_pairs) / possible_pairs if possible_pairs > 0 else 0
        
        # 2. Positive interaction ratio
        positive_ratio = sum(1 for i in interactions if i.emotional_valence > 0) / len(interactions)
        
        # 3. Shared activities
        activity_diversity = len(set(i.interaction_type for i in interactions))
        activity_score = min(activity_diversity / 10, 1.0)  # Normalize to 10 activity types
        
        # Combine factors
        cohesion = connectivity * 0.4 + positive_ratio * 0.4 + activity_score * 0.2
        
        return cohesion
    
    def _analyze_hierarchy(self, members: List[str],
                         interactions: List[SocialInteraction]) -> Dict[str, float]:
        """Analyze hierarchy structure in group"""
        influence_scores = {member: 0.5 for member in members}  # Start with equal influence
        
        # Analyze who initiates interactions, who others defer to, etc.
        for interaction in interactions:
            # Simple heuristic: first participant has slightly more influence
            if interaction.participants:
                initiator = interaction.participants[0]
                if initiator in influence_scores:
                    influence_scores[initiator] += 0.01
                
                # Positive outcomes increase influence
                if interaction.outcome == 'positive' and initiator in influence_scores:
                    influence_scores[initiator] += 0.02
        
        # Normalize scores
        total = sum(influence_scores.values())
        if total > 0:
            influence_scores = {k: v/total for k, v in influence_scores.items()}
        
        return influence_scores
    
    def _analyze_communication_patterns(self, members: List[str],
                                      interactions: List[SocialInteraction]) -> Dict[Tuple[str, str], float]:
        """Analyze communication patterns between group members"""
        communication_freq = {}
        
        for interaction in interactions:
            # Count interactions between each pair
            for i in range(len(interaction.participants)):
                for j in range(i + 1, len(interaction.participants)):
                    if (interaction.participants[i] in members and 
                        interaction.participants[j] in members):
                        pair = tuple(sorted([interaction.participants[i], interaction.participants[j]]))
                        communication_freq[pair] = communication_freq.get(pair, 0) + 1
        
        # Normalize by total interactions
        total = sum(communication_freq.values())
        if total > 0:
            communication_freq = {k: v/total for k, v in communication_freq.items()}
        
        return communication_freq
    
    def _assess_collective_mood(self, interactions: List[SocialInteraction]) -> str:
        """Assess the collective mood from interactions"""
        if not interactions:
            return 'neutral'
        
        avg_valence = np.mean([i.emotional_valence for i in interactions])
        
        if avg_valence > 0.5:
            return 'positive'
        elif avg_valence > 0.2:
            return 'optimistic'
        elif avg_valence > -0.2:
            return 'neutral'
        elif avg_valence > -0.5:
            return 'tense'
        else:
            return 'negative'
    
    def _assess_conflict_level(self, interactions: List[SocialInteraction]) -> float:
        """Assess conflict level in interactions"""
        if not interactions:
            return 0.0
        
        conflict_indicators = 0
        for interaction in interactions:
            if interaction.emotional_valence < -0.3:
                conflict_indicators += 1
            if interaction.outcome in ['conflict', 'disagreement', 'tension']:
                conflict_indicators += 1
        
        conflict_level = conflict_indicators / (len(interactions) * 2)  # Normalize
        return min(conflict_level, 1.0)
    
    def _infer_group_goals(self, interactions: List[SocialInteraction]) -> List[str]:
        """Infer group goals from interaction patterns"""
        # Simple implementation - could be enhanced with NLP
        goals = []
        
        # Count interaction types
        type_counts = {}
        for interaction in interactions:
            type_counts[interaction.interaction_type] = type_counts.get(interaction.interaction_type, 0) + 1
        
        # Infer goals from frequent interaction types
        for itype, count in type_counts.items():
            if count > len(interactions) * 0.2:  # More than 20% of interactions
                if 'collaborat' in itype.lower():
                    goals.append('collaboration')
                elif 'learn' in itype.lower():
                    goals.append('learning')
                elif 'social' in itype.lower():
                    goals.append('social bonding')
        
        return goals
    
    def _update_social_network(self) -> None:
        """Update the social network with recent interactions"""
        # Build network for last 24 hours
        self.build_social_network(time_window=timedelta(hours=24))
    
    def _detect_communities(self, G: nx.Graph) -> List[List[str]]:
        """Detect communities in the network"""
        if G.number_of_nodes() == 0:
            return []
        
        # Use Louvain method for community detection
        try:
            import community
            partition = community.best_partition(G)
            
            # Convert partition to list of communities
            communities = {}
            for node, comm_id in partition.items():
                if comm_id not in communities:
                    communities[comm_id] = []
                communities[comm_id].append(node)
            
            return list(communities.values())
        except ImportError:
            # Fallback to connected components
            return [list(c) for c in nx.connected_components(G)]
    
    def _calculate_relationship_strength(self, interactions: List[SocialInteraction]) -> float:
        """Calculate strength of a relationship from interactions"""
        if not interactions:
            return 0.0
        
        # Factors: frequency, duration, emotional positivity, consistency
        frequency_score = min(len(interactions) / 50, 1.0)  # Normalize to 50 interactions
        
        avg_duration = np.mean([i.duration.total_seconds() for i in interactions])
        duration_score = min(avg_duration / 3600, 1.0)  # Normalize to 1 hour
        
        positivity_score = sum(1 for i in interactions if i.emotional_valence > 0) / len(interactions)
        
        # Consistency - low variance in interaction intervals
        if len(interactions) > 1:
            intervals = []
            for i in range(1, len(interactions)):
                interval = (interactions[i].timestamp - interactions[i-1].timestamp).total_seconds()
                intervals.append(interval)
            
            if intervals:
                consistency_score = 1.0 / (1.0 + np.std(intervals) / np.mean(intervals))
            else:
                consistency_score = 0.5
        else:
            consistency_score = 0.3
        
        # Weighted combination
        strength = (frequency_score * 0.3 + 
                   duration_score * 0.2 + 
                   positivity_score * 0.3 + 
                   consistency_score * 0.2)
        
        return strength
    
    def _detect_cliques(self, time_window: Optional[timedelta]) -> List[List[str]]:
        """Detect cliques (tightly connected groups)"""
        network = self.build_social_network(time_window)
        G = network.to_networkx()
        
        # Find maximal cliques
        cliques = list(nx.find_cliques(G))
        
        # Filter to meaningful cliques (3+ members)
        meaningful_cliques = [c for c in cliques if len(c) >= 3]
        
        return meaningful_cliques
    
    def _detect_influence_cascades(self, time_window: Optional[timedelta]) -> List[Dict[str, Any]]:
        """Detect influence cascades in the network"""
        # Simplified implementation - tracks behavior/mood spreading
        cascades = []
        
        # This would require tracking behavior changes over time
        # For now, return empty list
        return cascades
    
    def _detect_communication_bottlenecks(self, time_window: Optional[timedelta]) -> List[str]:
        """Detect agents who are communication bottlenecks"""
        network = self.build_social_network(time_window)
        G = network.to_networkx()
        
        if G.number_of_nodes() == 0:
            return []
        
        # Calculate betweenness centrality
        betweenness = nx.betweenness_centrality(G)
        
        # Identify bottlenecks (high betweenness)
        bottlenecks = [node for node, centrality in betweenness.items() if centrality > 0.5]
        
        return bottlenecks
    
    def _detect_emotional_contagion(self, time_window: Optional[timedelta]) -> Dict[str, Any]:
        """Detect emotional contagion patterns"""
        # Simplified implementation
        return {
            'contagion_detected': False,
            'primary_emotion': 'neutral',
            'affected_agents': []
        }
    
    def _detect_collaboration_patterns(self, time_window: Optional[timedelta]) -> Dict[str, Any]:
        """Detect collaboration patterns"""
        if time_window:
            cutoff_time = datetime.now() - time_window
            relevant_interactions = [
                i for i in self.interaction_history
                if i.timestamp > cutoff_time and 
                'collaborat' in i.interaction_type.lower()
            ]
        else:
            relevant_interactions = [
                i for i in self.interaction_history
                if 'collaborat' in i.interaction_type.lower()
            ]
        
        # Analyze collaboration patterns
        frequent_collaborators = {}
        successful_collaborations = 0
        
        for interaction in relevant_interactions:
            if len(interaction.participants) >= 2:
                pair = tuple(sorted(interaction.participants[:2]))
                frequent_collaborators[pair] = frequent_collaborators.get(pair, 0) + 1
                
                if interaction.outcome == 'success':
                    successful_collaborations += 1
        
        success_rate = successful_collaborations / len(relevant_interactions) if relevant_interactions else 0
        
        return {
            'total_collaborations': len(relevant_interactions),
            'success_rate': success_rate,
            'frequent_pairs': sorted(frequent_collaborators.items(), key=lambda x: x[1], reverse=True)[:5]
        }
    
    def _calculate_network_diversity(self, network: SocialNetwork) -> float:
        """Calculate diversity of the social network"""
        if not network.nodes:
            return 0.0
        
        # Factors: community diversity, connection diversity
        community_count = len(network.communities)
        community_diversity = min(community_count / 5, 1.0)  # Normalize to 5 communities
        
        # Connection diversity - how evenly distributed are connections
        if network.centrality_scores:
            centrality_values = list(network.centrality_scores.values())
            connection_diversity = 1.0 - np.std(centrality_values) / (np.mean(centrality_values) + 0.01)
        else:
            connection_diversity = 0.5
        
        diversity = community_diversity * 0.5 + connection_diversity * 0.5
        return max(0, min(diversity, 1.0))
    
    def _calculate_interaction_quality(self) -> float:
        """Calculate overall interaction quality"""
        recent_interactions = [
            i for i in self.interaction_history
            if i.timestamp > datetime.now() - timedelta(days=7)
        ]
        
        if not recent_interactions:
            return 0.5
        
        # Quality factors
        positive_ratio = sum(1 for i in recent_interactions if i.emotional_valence > 0) / len(recent_interactions)
        avg_intensity = np.mean([i.intensity for i in recent_interactions])
        successful_outcomes = sum(1 for i in recent_interactions if i.outcome in ['positive', 'success']) / len(recent_interactions)
        
        quality = positive_ratio * 0.4 + avg_intensity * 0.3 + successful_outcomes * 0.3
        return quality
    
    def _calculate_overall_cohesion(self) -> float:
        """Calculate overall cohesion across all groups"""
        if not self.group_dynamics_history:
            return 0.5
        
        recent_cohesions = []
        for group_history in self.group_dynamics_history.values():
            if group_history:
                recent_cohesions.append(group_history[-1].cohesion_score)
        
        if recent_cohesions:
            return np.mean(recent_cohesions)
        return 0.5
    
    def _calculate_conflict_resolution_rate(self) -> float:
        """Calculate rate of successful conflict resolution"""
        conflicts = [
            i for i in self.interaction_history
            if i.emotional_valence < -0.3 or 'conflict' in i.interaction_type.lower()
        ]
        
        if not conflicts:
            return 1.0  # No conflicts is good
        
        # Look for resolution patterns
        resolved = 0
        for i, conflict in enumerate(conflicts):
            # Check if followed by positive interaction with same participants
            for j in range(i + 1, min(i + 10, len(self.interaction_history))):
                next_interaction = self.interaction_history[j]
                if (set(conflict.participants) == set(next_interaction.participants) and
                    next_interaction.emotional_valence > 0.3):
                    resolved += 1
                    break
        
        resolution_rate = resolved / len(conflicts)
        return resolution_rate
    
    def _calculate_collective_growth(self) -> float:
        """Calculate collective growth metric"""
        # Simplified - would track learning, skill development, etc.
        # For now, use interaction diversity and quality improvement
        
        if len(self.interaction_history) < 20:
            return 0.5
        
        # Compare early and recent interactions
        early_interactions = self.interaction_history[:10]
        recent_interactions = self.interaction_history[-10:]
        
        early_quality = np.mean([i.emotional_valence for i in early_interactions])
        recent_quality = np.mean([i.emotional_valence for i in recent_interactions])
        
        quality_improvement = (recent_quality - early_quality + 1) / 2  # Normalize to 0-1
        
        # Diversity of interaction types
        early_types = len(set(i.interaction_type for i in early_interactions))
        recent_types = len(set(i.interaction_type for i in recent_interactions))
        
        diversity_growth = min((recent_types - early_types) / 5 + 0.5, 1.0)
        
        growth = quality_improvement * 0.6 + diversity_growth * 0.4
        return max(0, min(growth, 1.0))
    
    def _generate_social_recommendations(self, metrics: Dict[str, float],
                                       network: SocialNetwork) -> List[str]:
        """Generate recommendations for improving social dynamics"""
        recommendations = []
        
        # Based on metrics
        if metrics['network_diversity'] < 0.4:
            recommendations.append("Encourage cross-group interactions to increase network diversity")
        
        if metrics['interaction_quality'] < 0.5:
            recommendations.append("Focus on improving quality of interactions through active listening and empathy")
        
        if metrics['group_cohesion'] < 0.4:
            recommendations.append("Organize team-building activities to strengthen group cohesion")
        
        if metrics['conflict_resolution'] < 0.5:
            recommendations.append("Implement conflict resolution training and mediation processes")
        
        if metrics['collective_growth'] < 0.3:
            recommendations.append("Create opportunities for collaborative learning and skill sharing")
        
        # Based on network structure
        if network.density < 0.2:
            recommendations.append("Network is too sparse - facilitate more interactions between agents")
        elif network.density > 0.8:
            recommendations.append("Network may be too dense - encourage some independent activities")
        
        # Check for isolated nodes
        isolated_nodes = [node for node, centrality in network.centrality_scores.items() 
                         if centrality < 0.1]
        if isolated_nodes:
            recommendations.append(f"Include isolated members: {', '.join(isolated_nodes[:3])}")
        
        # Check for communication bottlenecks
        bottlenecks = self._detect_communication_bottlenecks(None)
        if bottlenecks:
            recommendations.append(f"Address communication bottlenecks through: {', '.join(bottlenecks[:2])}")
        
        return recommendations
