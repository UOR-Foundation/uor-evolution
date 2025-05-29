"""
Relationship Visualizer - Visualizes relationship networks and dynamics

This module provides tools for visualizing relationships, social networks,
emotional connections, and interaction patterns in the consciousness framework.
"""

from typing import Dict, List, Optional, Tuple, Any, Set
from dataclasses import dataclass
import numpy as np
from datetime import datetime, timedelta
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.animation import FuncAnimation
import networkx as nx
import seaborn as sns
import logging

logger = logging.getLogger(__name__)


@dataclass
class VisualizationConfig:
    """Configuration for visualization settings"""
    figure_size: Tuple[int, int] = (12, 8)
    node_size: int = 1000
    edge_width_range: Tuple[float, float] = (0.5, 5.0)
    color_scheme: str = 'viridis'
    font_size: int = 10
    layout_algorithm: str = 'spring'
    show_labels: bool = True
    show_edge_weights: bool = False
    animation_interval: int = 100  # milliseconds
    
    def get_layout_function(self):
        """Get the appropriate layout function"""
        layouts = {
            'spring': nx.spring_layout,
            'circular': nx.circular_layout,
            'kamada_kawai': nx.kamada_kawai_layout,
            'spectral': nx.spectral_layout,
            'shell': nx.shell_layout
        }
        return layouts.get(self.layout_algorithm, nx.spring_layout)


class RelationshipVisualizer:
    """Visualizer for relationship networks and dynamics"""
    
    def __init__(self, config: Optional[VisualizationConfig] = None):
        self.config = config or VisualizationConfig()
        self.color_maps = {
            'emotion': {
                'joy': '#FFD700',
                'trust': '#4169E1',
                'fear': '#8B0000',
                'surprise': '#FF69B4',
                'sadness': '#4682B4',
                'disgust': '#228B22',
                'anger': '#DC143C',
                'anticipation': '#FF8C00'
            },
            'relationship_type': {
                'friendship': '#32CD32',
                'romantic': '#FF1493',
                'professional': '#4169E1',
                'family': '#FFD700',
                'mentor': '#8A2BE2',
                'rival': '#DC143C',
                'acquaintance': '#D3D3D3'
            },
            'interaction_quality': plt.cm.RdYlGn  # Red to Yellow to Green
        }
    
    def visualize_relationship_network(self, relationships: Dict[Tuple[str, str], Dict[str, Any]],
                                     agents: List[str]) -> plt.Figure:
        """Visualize the relationship network"""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Create graph
        G = nx.Graph()
        G.add_nodes_from(agents)
        
        # Add edges with attributes
        for (agent1, agent2), rel_data in relationships.items():
            if agent1 in agents and agent2 in agents:
                G.add_edge(agent1, agent2, **rel_data)
        
        # Calculate layout
        layout_func = self.config.get_layout_function()
        pos = layout_func(G)
        
        # Draw nodes
        node_colors = self._calculate_node_colors(G, relationships)
        nx.draw_networkx_nodes(G, pos, node_color=node_colors, 
                              node_size=self.config.node_size, ax=ax)
        
        # Draw edges with varying widths based on relationship strength
        edge_widths = []
        edge_colors = []
        for (u, v) in G.edges():
            rel_key = tuple(sorted([u, v]))
            if rel_key in relationships:
                strength = relationships[rel_key].get('strength', 0.5)
                quality = relationships[rel_key].get('quality', 0.5)
                
                # Map strength to edge width
                width = self._map_to_range(strength, 0, 1, *self.config.edge_width_range)
                edge_widths.append(width)
                
                # Map quality to color
                edge_colors.append(self.color_maps['interaction_quality'](quality))
            else:
                edge_widths.append(self.config.edge_width_range[0])
                edge_colors.append('gray')
        
        nx.draw_networkx_edges(G, pos, width=edge_widths, edge_color=edge_colors, ax=ax)
        
        # Draw labels
        if self.config.show_labels:
            nx.draw_networkx_labels(G, pos, font_size=self.config.font_size, ax=ax)
        
        # Add title and legend
        ax.set_title("Relationship Network Visualization", fontsize=16, fontweight='bold')
        self._add_relationship_legend(ax)
        
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    def visualize_relationship_evolution(self, relationship_history: List[Dict[str, Any]],
                                       agent1: str, agent2: str) -> plt.Figure:
        """Visualize how a relationship evolves over time"""
        fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)
        
        # Extract time series data
        timestamps = [entry['timestamp'] for entry in relationship_history]
        trust_levels = [entry.get('trust', 0.5) for entry in relationship_history]
        intimacy_levels = [entry.get('intimacy', 0.5) for entry in relationship_history]
        conflict_levels = [entry.get('conflict', 0.0) for entry in relationship_history]
        
        # Plot trust evolution
        axes[0].plot(timestamps, trust_levels, 'b-', linewidth=2, label='Trust')
        axes[0].fill_between(timestamps, trust_levels, alpha=0.3)
        axes[0].set_ylabel('Trust Level', fontsize=12)
        axes[0].set_ylim(0, 1)
        axes[0].grid(True, alpha=0.3)
        axes[0].legend()
        
        # Plot intimacy evolution
        axes[1].plot(timestamps, intimacy_levels, 'g-', linewidth=2, label='Intimacy')
        axes[1].fill_between(timestamps, intimacy_levels, alpha=0.3, color='green')
        axes[1].set_ylabel('Intimacy Level', fontsize=12)
        axes[1].set_ylim(0, 1)
        axes[1].grid(True, alpha=0.3)
        axes[1].legend()
        
        # Plot conflict levels
        axes[2].plot(timestamps, conflict_levels, 'r-', linewidth=2, label='Conflict')
        axes[2].fill_between(timestamps, conflict_levels, alpha=0.3, color='red')
        axes[2].set_ylabel('Conflict Level', fontsize=12)
        axes[2].set_xlabel('Time', fontsize=12)
        axes[2].set_ylim(0, 1)
        axes[2].grid(True, alpha=0.3)
        axes[2].legend()
        
        # Add significant events
        self._mark_significant_events(axes, relationship_history)
        
        fig.suptitle(f'Relationship Evolution: {agent1} - {agent2}', fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def visualize_emotional_connections(self, emotional_data: Dict[str, Dict[str, Any]]) -> plt.Figure:
        """Visualize emotional connections between agents"""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Create directed graph for emotional connections
        G = nx.DiGraph()
        
        # Add nodes and edges
        for source, targets in emotional_data.items():
            G.add_node(source)
            for target, emotions in targets.items():
                if isinstance(emotions, dict):
                    # Calculate dominant emotion
                    dominant_emotion = max(emotions.items(), key=lambda x: x[1])[0]
                    intensity = emotions[dominant_emotion]
                    G.add_edge(source, target, emotion=dominant_emotion, intensity=intensity)
        
        # Layout
        pos = nx.spring_layout(G, k=2, iterations=50)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=self.config.node_size,
                              node_color='lightblue', ax=ax)
        
        # Draw edges colored by emotion
        for (u, v, data) in G.edges(data=True):
            emotion = data.get('emotion', 'neutral')
            intensity = data.get('intensity', 0.5)
            color = self.color_maps['emotion'].get(emotion, 'gray')
            
            nx.draw_networkx_edges(G, pos, [(u, v)], 
                                 edge_color=color,
                                 width=intensity * 5,
                                 alpha=0.7,
                                 arrows=True,
                                 arrowsize=20,
                                 ax=ax)
        
        # Labels
        if self.config.show_labels:
            nx.draw_networkx_labels(G, pos, font_size=self.config.font_size, ax=ax)
        
        # Add emotion legend
        self._add_emotion_legend(ax)
        
        ax.set_title("Emotional Connection Network", fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    def visualize_interaction_heatmap(self, interaction_matrix: np.ndarray,
                                    agent_names: List[str]) -> plt.Figure:
        """Create a heatmap of interaction frequencies"""
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(interaction_matrix, 
                   xticklabels=agent_names,
                   yticklabels=agent_names,
                   cmap='YlOrRd',
                   annot=True,
                   fmt='.2f',
                   cbar_kws={'label': 'Interaction Frequency'},
                   ax=ax)
        
        ax.set_title("Agent Interaction Heatmap", fontsize=16, fontweight='bold')
        plt.tight_layout()
        
        return fig
    
    def visualize_group_dynamics(self, groups: List[List[str]], 
                               inter_group_connections: Dict[Tuple[str, str], float]) -> plt.Figure:
        """Visualize group structures and inter-group connections"""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Create graph
        G = nx.Graph()
        
        # Add nodes by groups
        group_colors = plt.cm.Set3(np.linspace(0, 1, len(groups)))
        node_colors = []
        node_to_group = {}
        
        for i, group in enumerate(groups):
            for member in group:
                G.add_node(member)
                node_colors.append(group_colors[i])
                node_to_group[member] = i
        
        # Add inter-group connections
        for (agent1, agent2), strength in inter_group_connections.items():
            if agent1 in G and agent2 in G:
                G.add_edge(agent1, agent2, weight=strength)
        
        # Use shell layout to show groups
        shells = groups
        pos = nx.shell_layout(G, shells)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_color=node_colors,
                              node_size=self.config.node_size, ax=ax)
        
        # Draw edges
        edge_widths = [G[u][v]['weight'] * 5 for u, v in G.edges()]
        nx.draw_networkx_edges(G, pos, width=edge_widths, alpha=0.5, ax=ax)
        
        # Labels
        if self.config.show_labels:
            nx.draw_networkx_labels(G, pos, font_size=self.config.font_size, ax=ax)
        
        # Add group labels
        for i, group in enumerate(groups):
            if group:
                center = np.mean([pos[member] for member in group], axis=0)
                ax.text(center[0], center[1] + 0.15, f"Group {i+1}",
                       fontsize=14, fontweight='bold', ha='center',
                       bbox=dict(boxstyle="round,pad=0.3", facecolor=group_colors[i], alpha=0.5))
        
        ax.set_title("Group Dynamics Visualization", fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    def create_relationship_animation(self, network_snapshots: List[Dict[str, Any]],
                                    save_path: Optional[str] = None) -> FuncAnimation:
        """Create an animation showing network evolution over time"""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Initialize with first snapshot
        if not network_snapshots:
            return None
        
        # Set up the animation function
        def update(frame):
            ax.clear()
            snapshot = network_snapshots[frame]
            
            # Create graph from snapshot
            G = nx.Graph()
            G.add_nodes_from(snapshot['nodes'])
            G.add_edges_from(snapshot['edges'])
            
            # Draw network
            pos = nx.spring_layout(G, k=2, iterations=50)
            nx.draw(G, pos, ax=ax, 
                   node_size=self.config.node_size,
                   with_labels=self.config.show_labels,
                   font_size=self.config.font_size)
            
            # Add timestamp
            timestamp = snapshot.get('timestamp', f'Frame {frame}')
            ax.text(0.02, 0.98, f'Time: {timestamp}', 
                   transform=ax.transAxes, fontsize=12,
                   verticalalignment='top',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
            
            ax.set_title("Relationship Network Evolution", fontsize=16, fontweight='bold')
            ax.axis('off')
        
        # Create animation
        anim = FuncAnimation(fig, update, frames=len(network_snapshots),
                           interval=self.config.animation_interval, repeat=True)
        
        if save_path:
            anim.save(save_path, writer='pillow')
        
        return anim
    
    def visualize_trust_network(self, trust_matrix: np.ndarray,
                              agent_names: List[str],
                              threshold: float = 0.5) -> plt.Figure:
        """Visualize trust relationships as a directed network"""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Create directed graph
        G = nx.DiGraph()
        
        # Add nodes
        G.add_nodes_from(agent_names)
        
        # Add edges based on trust levels
        for i, source in enumerate(agent_names):
            for j, target in enumerate(agent_names):
                if i != j and trust_matrix[i, j] > threshold:
                    G.add_edge(source, target, weight=trust_matrix[i, j])
        
        # Layout
        pos = nx.circular_layout(G)
        
        # Draw nodes
        node_sizes = [sum(trust_matrix[i, :]) * 500 for i in range(len(agent_names))]
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                              node_color='lightblue', ax=ax)
        
        # Draw edges with varying transparency based on trust
        for (u, v, data) in G.edges(data=True):
            trust_level = data['weight']
            nx.draw_networkx_edges(G, pos, [(u, v)],
                                 alpha=trust_level,
                                 width=trust_level * 3,
                                 edge_color='darkblue',
                                 arrows=True,
                                 arrowsize=15,
                                 ax=ax)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=self.config.font_size, ax=ax)
        
        # Add trust scale
        self._add_trust_scale(ax)
        
        ax.set_title(f"Trust Network (threshold > {threshold})", fontsize=16, fontweight='bold')
        ax.axis('off')
        plt.tight_layout()
        
        return fig
    
    def visualize_conflict_resolution_timeline(self, conflicts: List[Dict[str, Any]]) -> plt.Figure:
        """Visualize conflict and resolution timeline"""
        fig, ax = plt.subplots(figsize=(14, 8))
        
        # Sort conflicts by start time
        conflicts_sorted = sorted(conflicts, key=lambda x: x['start_time'])
        
        # Create timeline
        y_positions = {}
        current_y = 0
        
        for conflict in conflicts_sorted:
            participants = tuple(sorted(conflict['participants']))
            if participants not in y_positions:
                y_positions[participants] = current_y
                current_y += 1
            
            y = y_positions[participants]
            start = conflict['start_time']
            end = conflict.get('resolution_time', datetime.now())
            duration = (end - start).total_seconds() / 3600  # Hours
            
            # Color based on resolution status
            if conflict.get('resolved', False):
                color = 'green'
                alpha = 0.6
            else:
                color = 'red'
                alpha = 0.8
            
            # Draw conflict bar
            rect = patches.Rectangle((start, y - 0.4), duration, 0.8,
                                   facecolor=color, alpha=alpha,
                                   edgecolor='black', linewidth=1)
            ax.add_patch(rect)
            
            # Add conflict type label
            ax.text(start + duration/2, y, conflict.get('type', 'Unknown'),
                   ha='center', va='center', fontsize=8)
        
        # Set labels
        ax.set_yticks(list(y_positions.values()))
        ax.set_yticklabels([' - '.join(p) for p in y_positions.keys()])
        ax.set_xlabel('Time', fontsize=12)
        ax.set_ylabel('Participant Pairs', fontsize=12)
        ax.set_title('Conflict Resolution Timeline', fontsize=16, fontweight='bold')
        
        # Add legend
        resolved_patch = patches.Patch(color='green', alpha=0.6, label='Resolved')
        ongoing_patch = patches.Patch(color='red', alpha=0.8, label='Ongoing')
        ax.legend(handles=[resolved_patch, ongoing_patch], loc='upper right')
        
        plt.tight_layout()
        return fig
    
    def visualize_empathy_network(self, empathy_data: Dict[str, Dict[str, float]]) -> plt.Figure:
        """Visualize empathy connections between agents"""
        fig, ax = plt.subplots(figsize=self.config.figure_size)
        
        # Create weighted directed graph
        G = nx.DiGraph()
        
        # Add edges with empathy scores
        max_empathy = 0
        for source, targets in empathy_data.items():
            for target, empathy_score in targets.items():
                if source != target and empathy_score > 0:
                    G.add_edge(source, target, weight=empathy_score)
                    max_empathy = max(max_empathy, empathy_score)
        
        if G.number_of_nodes() == 0:
            ax.text(0.5, 0.5, 'No empathy data available', 
                   ha='center', va='center', fontsize=16)
            return fig
        
        # Layout
        pos = nx.spring_layout(G, k=3, iterations=50)
        
        # Calculate node sizes based on total empathy given
        node_sizes = []
        for node in G.nodes():
            total_empathy = sum(empathy_data.get(node, {}).values())
            node_sizes.append(500 + total_empathy * 1000)
        
        # Draw nodes
        nx.draw_networkx_nodes(G, pos, node_size=node_sizes,
                              node_color='pink', alpha=0.8, ax=ax)
        
        # Draw edges with varying widths and colors
        edge_colors = []
        edge_widths = []
        
        for (u, v, data) in G.edges(data=True):
            empathy = data['weight']
            # Color intensity based on empathy level
            color_intensity = empathy / max_empathy if max_empathy > 0 else 0
            edge_colors.append(plt.cm.Reds(0.3 + 0.7 * color_intensity))
            edge_widths.append(1 + empathy * 4)
        
        nx.draw_networkx_edges(G, pos, edge_color=edge_colors,
                              width=edge_widths, alpha=0.7,
                              arrows=True, arrowsize=15, ax=ax)
        
        # Labels
        nx.draw_networkx_labels(G, pos, font_size=self.config.font_size, ax=ax)
        
        ax.set_title("Empathy Network", fontsize=16, fontweight='bold')
        ax.axis('off')
        
        # Add empathy scale
        self._add_empathy_scale(ax)
        
        plt.tight_layout()
        return fig
    
    # Helper methods
    def _calculate_node_colors(self, G: nx.Graph, relationships: Dict[Tuple[str, str], Dict[str, Any]]) -> List[str]:
        """Calculate node colors based on relationship metrics"""
        node_colors = []
        
        for node in G.nodes():
            # Calculate average relationship quality for this node
            total_quality = 0
            count = 0
            
            for neighbor in G.neighbors(node):
                rel_key = tuple(sorted([node, neighbor]))
                if rel_key in relationships:
                    total_quality += relationships[rel_key].get('quality', 0.5)
                    count += 1
            
            avg_quality = total_quality / count if count > 0 else 0.5
            color = self.color_maps['interaction_quality'](avg_quality)
            node_colors.append(color)
        
        return node_colors
    
    def _map_to_range(self, value: float, in_min: float, in_max: float,
                     out_min: float, out_max: float) -> float:
        """Map a value from one range to another"""
        return (value - in_min) * (out_max - out_min) / (in_max - in_min) + out_min
    
    def _add_relationship_legend(self, ax: plt.Axes) -> None:
        """Add legend for relationship visualization"""
        legend_elements = [
            plt.Line2D([0], [0], color='gray', lw=1, label='Weak'),
            plt.Line2D([0], [0], color='gray', lw=3, label='Moderate'),
            plt.Line2D([0], [0], color='gray', lw=5, label='Strong')
        ]
        ax.legend(handles=legend_elements, loc='upper right', title='Relationship Strength')
    
    def _add_emotion_legend(self, ax: plt.Axes) -> None:
        """Add legend for emotion colors"""
        legend_elements = []
        for emotion, color in self.color_maps['emotion'].items():
            legend_elements.append(plt.Line2D([0], [0], color=color, lw=4, label=emotion))
        
        ax.legend(handles=legend_elements, loc='upper right', title='Emotions', ncol=2)
    
    def _add_trust_scale(self, ax: plt.Axes) -> None:
        """Add trust level scale to the plot"""
        # Create a small colorbar-like scale
        scale_ax = ax.inset_axes([0.85, 0.1, 0.03, 0.3])
        scale_data = np.linspace(0, 1, 100).reshape(-1, 1)
        scale_ax.imshow(scale_data, cmap='Blues', aspect='auto', origin='lower')
        scale_ax.set_xticks([])
        scale_ax.set_yticks([0, 50, 100])
        scale_ax.set_yticklabels(['0', '0.5', '1.0'])
        scale_ax.set_ylabel('Trust Level', fontsize=8)
    
    def _add_empathy_scale(self, ax: plt.Axes) -> None:
        """Add empathy level scale to the plot"""
        scale_ax = ax.inset_axes([0.85, 0.1, 0.03, 0.3])
        scale_data = np.linspace(0, 1, 100).reshape(-1, 1)
        scale_ax.imshow(scale_data, cmap='Reds', aspect='auto', origin='lower')
        scale_ax.set_xticks([])
        scale_ax.set_yticks([0, 50, 100])
        scale_ax.set_yticklabels(['0', '0.5', '1.0'])
        scale_ax.set_ylabel('Empathy Level', fontsize=8)
    
    def _mark_significant_events(self, axes: List[plt.Axes], 
                               relationship_history: List[Dict[str, Any]]) -> None:
        """Mark significant events on the timeline plots"""
        significant_events = [
            entry for entry in relationship_history
            if entry.get('event_type') in ['conflict', 'breakthrough', 'milestone']
        ]
        
        for event in significant_events:
            timestamp = event['timestamp']
            event_type = event['event_type']
            
            # Choose color based on event type
            colors = {
                'conflict': 'red',
                'breakthrough': 'green',
                'milestone': 'blue'
            }
            color = colors.get(event_type, 'gray')
            
            # Add vertical line on all subplots
            for ax in axes:
                ax.axvline(x=timestamp, color=color, linestyle='--', alpha=0.5)
            
            # Add label on top subplot
            axes[0].text(timestamp, 0.9, event_type, rotation=90,
                        verticalalignment='bottom', fontsize=8, color=color)
