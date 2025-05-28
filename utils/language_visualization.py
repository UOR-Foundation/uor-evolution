"""
Language Visualization Utilities

This module provides tools for visualizing language generation processes,
semantic relationships, and natural language structures.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass
import json
import math


@dataclass
class SemanticNode:
    """A node in semantic visualization"""
    concept: str
    prime_encoding: int
    semantic_neighbors: List[Tuple[str, float]]  # (concept, similarity)
    depth: int = 0


@dataclass
class LanguageFlowDiagram:
    """Diagram showing language generation flow"""
    nodes: List[Dict[str, Any]]
    edges: List[Dict[str, Any]]
    layout: str = "hierarchical"


class LanguageVisualizer:
    """
    Visualizes language generation and semantic relationships.
    """
    
    def __init__(self):
        self.visualization_cache = {}
        
    def visualize_semantic_space(self, concepts: List[Tuple[str, int]],
                               relationships: List[Tuple[str, str, float]]) -> Dict[str, Any]:
        """Visualize semantic space as network"""
        nodes = []
        edges = []
        
        # Create nodes
        for concept, prime in concepts:
            nodes.append({
                "id": concept,
                "label": concept,
                "prime": prime,
                "size": self._calculate_node_size(prime),
                "color": self._get_semantic_color(prime)
            })
        
        # Create edges
        for source, target, weight in relationships:
            edges.append({
                "source": source,
                "target": target,
                "weight": weight,
                "width": weight * 5,  # Visual width
                "label": f"{weight:.2f}"
            })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "layout": "force-directed",
            "physics": {
                "repulsion": 100,
                "attraction": 0.01,
                "damping": 0.4
            }
        }
    
    def visualize_thought_translation(self, thought: Any,
                                    translation_steps: List[Dict[str, Any]]) -> LanguageFlowDiagram:
        """Visualize thought to language translation process"""
        nodes = []
        edges = []
        
        # Initial thought node
        nodes.append({
            "id": "thought",
            "label": "Original Thought",
            "type": "thought",
            "level": 0
        })
        
        # Translation step nodes
        for i, step in enumerate(translation_steps):
            node_id = f"step_{i}"
            nodes.append({
                "id": node_id,
                "label": step.get("description", f"Step {i+1}"),
                "type": step.get("type", "translation"),
                "level": i + 1,
                "details": step
            })
            
            # Connect to previous
            if i == 0:
                edges.append({
                    "source": "thought",
                    "target": node_id,
                    "label": "translate"
                })
            else:
                edges.append({
                    "source": f"step_{i-1}",
                    "target": node_id,
                    "label": step.get("operation", "transform")
                })
        
        # Final output node
        nodes.append({
            "id": "output",
            "label": "Natural Language",
            "type": "output",
            "level": len(translation_steps) + 1
        })
        
        if translation_steps:
            edges.append({
                "source": f"step_{len(translation_steps)-1}",
                "target": "output",
                "label": "finalize"
            })
        
        return LanguageFlowDiagram(nodes=nodes, edges=edges)
    
    def visualize_dialogue_structure(self, dialogue_turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Visualize dialogue structure and flow"""
        nodes = []
        edges = []
        
        for i, turn in enumerate(dialogue_turns):
            # Create node for each turn
            node_id = f"turn_{i}"
            nodes.append({
                "id": node_id,
                "label": f"{turn['speaker']}: {turn['summary'][:30]}...",
                "speaker": turn['speaker'],
                "type": turn.get('type', 'statement'),
                "timestamp": i,
                "full_text": turn.get('text', '')
            })
            
            # Connect to previous turn
            if i > 0:
                edges.append({
                    "source": f"turn_{i-1}",
                    "target": node_id,
                    "type": "follows"
                })
            
            # Add topic connections
            if 'references' in turn:
                for ref in turn['references']:
                    if ref < i:  # Only reference previous turns
                        edges.append({
                            "source": f"turn_{ref}",
                            "target": node_id,
                            "type": "references",
                            "style": "dashed"
                        })
        
        return {
            "nodes": nodes,
            "edges": edges,
            "layout": "timeline",
            "analysis": self._analyze_dialogue_structure(dialogue_turns)
        }
    
    def visualize_uncertainty_landscape(self, topics: List[Tuple[str, float]]) -> Dict[str, Any]:
        """Visualize uncertainty across topics"""
        landscape = []
        
        for topic, confidence in topics:
            landscape.append({
                "topic": topic,
                "confidence": confidence,
                "uncertainty": 1.0 - confidence,
                "visualization": {
                    "height": confidence,
                    "color": self._confidence_to_color(confidence),
                    "opacity": 0.3 + (confidence * 0.7)
                }
            })
        
        return {
            "type": "uncertainty_landscape",
            "data": landscape,
            "statistics": {
                "mean_confidence": sum(c for _, c in topics) / len(topics) if topics else 0,
                "min_confidence": min(c for _, c in topics) if topics else 0,
                "max_confidence": max(c for _, c in topics) if topics else 1,
                "variance": self._calculate_variance([c for _, c in topics])
            }
        }
    
    def visualize_perspective_space(self, perspectives: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Visualize multiple perspectives in conceptual space"""
        # Map perspectives to coordinates
        coords = self._map_perspectives_to_space(perspectives)
        
        points = []
        for i, (perspective, (x, y)) in enumerate(zip(perspectives, coords)):
            points.append({
                "id": f"perspective_{i}",
                "label": perspective.get("name", f"Perspective {i+1}"),
                "x": x,
                "y": y,
                "type": perspective.get("type", "unknown"),
                "description": perspective.get("description", ""),
                "emphasis": perspective.get("emphasis", {})
            })
        
        # Calculate perspective clusters
        clusters = self._find_perspective_clusters(points)
        
        return {
            "type": "perspective_space",
            "points": points,
            "clusters": clusters,
            "dimensions": {
                "x_axis": "Objective ← → Subjective",
                "y_axis": "Concrete ← → Abstract"
            }
        }
    
    def visualize_emotion_trajectory(self, emotional_states: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Visualize emotional trajectory over time"""
        trajectory = []
        
        for i, state in enumerate(emotional_states):
            trajectory.append({
                "timestamp": i,
                "emotion": state.get("type", "neutral"),
                "intensity": state.get("intensity", 0.5),
                "valence": state.get("valence", 0),
                "arousal": state.get("arousal", 0.5),
                "coordinates": {
                    "x": state.get("valence", 0),
                    "y": state.get("arousal", 0.5)
                }
            })
        
        return {
            "type": "emotion_trajectory",
            "trajectory": trajectory,
            "path": self._smooth_trajectory(trajectory),
            "analysis": {
                "dominant_emotion": self._find_dominant_emotion(emotional_states),
                "emotional_range": self._calculate_emotional_range(trajectory),
                "stability": self._calculate_emotional_stability(trajectory)
            }
        }
    
    def generate_mermaid_diagram(self, diagram_type: str, data: Dict[str, Any]) -> str:
        """Generate Mermaid diagram code for visualization"""
        if diagram_type == "thought_flow":
            return self._generate_thought_flow_mermaid(data)
        elif diagram_type == "semantic_network":
            return self._generate_semantic_network_mermaid(data)
        elif diagram_type == "dialogue_structure":
            return self._generate_dialogue_mermaid(data)
        else:
            return "graph LR\n    A[Unknown Diagram Type]"
    
    # Private helper methods
    
    def _calculate_node_size(self, prime: int) -> float:
        """Calculate node size based on prime encoding"""
        # Larger primes = more complex concepts = larger nodes
        return 10 + math.log(prime + 1) * 5
    
    def _get_semantic_color(self, prime: int) -> str:
        """Get color based on semantic properties"""
        # Use prime factorization to determine color
        hue = (prime % 360) / 360
        saturation = 0.7
        lightness = 0.5
        
        # Convert HSL to hex (simplified)
        return f"hsl({int(hue * 360)}, {int(saturation * 100)}%, {int(lightness * 100)}%)"
    
    def _analyze_dialogue_structure(self, turns: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze dialogue structure patterns"""
        return {
            "turn_count": len(turns),
            "speaker_distribution": self._calculate_speaker_distribution(turns),
            "topic_coherence": self._calculate_topic_coherence(turns),
            "interaction_pattern": self._identify_interaction_pattern(turns)
        }
    
    def _confidence_to_color(self, confidence: float) -> str:
        """Convert confidence to color"""
        # Red (low) to Yellow (medium) to Green (high)
        if confidence < 0.5:
            # Red to Yellow
            r = 255
            g = int(confidence * 2 * 255)
            b = 0
        else:
            # Yellow to Green
            r = int((1 - confidence) * 2 * 255)
            g = 255
            b = 0
        
        return f"rgb({r}, {g}, {b})"
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values"""
        if not values:
            return 0
        mean = sum(values) / len(values)
        return sum((x - mean) ** 2 for x in values) / len(values)
    
    def _map_perspectives_to_space(self, perspectives: List[Dict[str, Any]]) -> List[Tuple[float, float]]:
        """Map perspectives to 2D space"""
        coords = []
        
        for perspective in perspectives:
            # Simple mapping based on perspective properties
            objectivity = perspective.get("objectivity", 0.5)
            abstraction = perspective.get("abstraction", 0.5)
            
            x = (objectivity - 0.5) * 2  # -1 to 1
            y = (abstraction - 0.5) * 2  # -1 to 1
            
            coords.append((x, y))
        
        return coords
    
    def _find_perspective_clusters(self, points: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Find clusters of similar perspectives"""
        # Simplified clustering
        clusters = []
        
        # For now, just identify if perspectives are close
        threshold = 0.5
        clustered = set()
        
        for i, point1 in enumerate(points):
            if i in clustered:
                continue
                
            cluster = [point1]
            clustered.add(i)
            
            for j, point2 in enumerate(points[i+1:], i+1):
                if j in clustered:
                    continue
                    
                dist = math.sqrt((point1["x"] - point2["x"])**2 + 
                               (point1["y"] - point2["y"])**2)
                
                if dist < threshold:
                    cluster.append(point2)
                    clustered.add(j)
            
            if len(cluster) > 1:
                clusters.append({
                    "members": [p["id"] for p in cluster],
                    "center": (
                        sum(p["x"] for p in cluster) / len(cluster),
                        sum(p["y"] for p in cluster) / len(cluster)
                    ),
                    "size": len(cluster)
                })
        
        return clusters
    
    def _smooth_trajectory(self, trajectory: List[Dict[str, Any]]) -> List[Dict[str, float]]:
        """Smooth emotional trajectory for visualization"""
        if len(trajectory) < 3:
            return trajectory
        
        smoothed = []
        for i in range(len(trajectory)):
            if i == 0 or i == len(trajectory) - 1:
                smoothed.append(trajectory[i]["coordinates"])
            else:
                # Simple moving average
                x = (trajectory[i-1]["coordinates"]["x"] + 
                     trajectory[i]["coordinates"]["x"] + 
                     trajectory[i+1]["coordinates"]["x"]) / 3
                y = (trajectory[i-1]["coordinates"]["y"] + 
                     trajectory[i]["coordinates"]["y"] + 
                     trajectory[i+1]["coordinates"]["y"]) / 3
                smoothed.append({"x": x, "y": y})
        
        return smoothed
    
    def _find_dominant_emotion(self, states: List[Dict[str, Any]]) -> str:
        """Find dominant emotion in states"""
        emotion_counts = {}
        
        for state in states:
            emotion = state.get("type", "neutral")
            intensity = state.get("intensity", 0.5)
            
            if emotion not in emotion_counts:
                emotion_counts[emotion] = 0
            emotion_counts[emotion] += intensity
        
        if emotion_counts:
            return max(emotion_counts.items(), key=lambda x: x[1])[0]
        return "neutral"
    
    def _calculate_emotional_range(self, trajectory: List[Dict[str, Any]]) -> float:
        """Calculate range of emotional movement"""
        if len(trajectory) < 2:
            return 0
        
        coords = [t["coordinates"] for t in trajectory]
        
        min_x = min(c["x"] for c in coords)
        max_x = max(c["x"] for c in coords)
        min_y = min(c["y"] for c in coords)
        max_y = max(c["y"] for c in coords)
        
        return math.sqrt((max_x - min_x)**2 + (max_y - min_y)**2)
    
    def _calculate_emotional_stability(self, trajectory: List[Dict[str, Any]]) -> float:
        """Calculate emotional stability (inverse of volatility)"""
        if len(trajectory) < 2:
            return 1.0
        
        total_movement = 0
        for i in range(1, len(trajectory)):
            prev = trajectory[i-1]["coordinates"]
            curr = trajectory[i]["coordinates"]
            
            movement = math.sqrt((curr["x"] - prev["x"])**2 + 
                               (curr["y"] - prev["y"])**2)
            total_movement += movement
        
        avg_movement = total_movement / (len(trajectory) - 1)
        
        # Convert to stability (inverse of movement)
        return 1.0 / (1.0 + avg_movement)
    
    def _calculate_speaker_distribution(self, turns: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate distribution of speakers in dialogue"""
        speaker_counts = {}
        
        for turn in turns:
            speaker = turn.get("speaker", "unknown")
            speaker_counts[speaker] = speaker_counts.get(speaker, 0) + 1
        
        total = len(turns)
        return {speaker: count/total for speaker, count in speaker_counts.items()}
    
    def _calculate_topic_coherence(self, turns: List[Dict[str, Any]]) -> float:
        """Calculate how coherent topics are across turns"""
        # Simplified - check topic similarity between consecutive turns
        if len(turns) < 2:
            return 1.0
        
        coherence_scores = []
        for i in range(1, len(turns)):
            # Simple check - do they reference each other?
            if "references" in turns[i] and i-1 in turns[i]["references"]:
                coherence_scores.append(1.0)
            else:
                coherence_scores.append(0.5)  # Assume some coherence
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 1.0
    
    def _identify_interaction_pattern(self, turns: List[Dict[str, Any]]) -> str:
        """Identify dialogue interaction pattern"""
        if len(turns) < 2:
            return "monologue"
        
        # Check speaker alternation
        speakers = [turn.get("speaker", "unknown") for turn in turns]
        
        alternating = True
        for i in range(1, len(speakers)):
            if speakers[i] == speakers[i-1]:
                alternating = False
                break
        
        if alternating:
            return "alternating_dialogue"
        else:
            # Check for other patterns
            unique_speakers = len(set(speakers))
            if unique_speakers == 1:
                return "monologue"
            elif unique_speakers == 2:
                return "dialogue_with_interruptions"
            else:
                return "multi_party_conversation"
    
    def _generate_thought_flow_mermaid(self, data: Dict[str, Any]) -> str:
        """Generate Mermaid diagram for thought flow"""
        lines = ["graph TD"]
        
        for node in data.get("nodes", []):
            node_id = node["id"]
            label = node["label"]
            lines.append(f'    {node_id}["{label}"]')
        
        for edge in data.get("edges", []):
            source = edge["source"]
            target = edge["target"]
            label = edge.get("label", "")
            lines.append(f'    {source} -->|{label}| {target}')
        
        return "\n".join(lines)
    
    def _generate_semantic_network_mermaid(self, data: Dict[str, Any]) -> str:
        """Generate Mermaid diagram for semantic network"""
        lines = ["graph LR"]
        
        for node in data.get("nodes", []):
            node_id = node["id"].replace(" ", "_")
            label = node["label"]
            lines.append(f'    {node_id}("{label}")')
        
        for edge in data.get("edges", []):
            source = edge["source"].replace(" ", "_")
            target = edge["target"].replace(" ", "_")
            weight = edge.get("weight", 1.0)
            
            if weight > 0.7:
                arrow = "==>"
            elif weight > 0.4:
                arrow = "-->"
            else:
                arrow = "-.->"
            
            lines.append(f'    {source} {arrow} {target}')
        
        return "\n".join(lines)
    
    def _generate_dialogue_mermaid(self, data: Dict[str, Any]) -> str:
        """Generate Mermaid diagram for dialogue structure"""
        lines = ["sequenceDiagram"]
        
        # Extract participants
        participants = set()
        for node in data.get("nodes", []):
            participants.add(node.get("speaker", "Unknown"))
        
        for participant in participants:
            lines.append(f"    participant {participant}")
        
        # Add interactions
        for i, node in enumerate(data.get("nodes", [])):
            speaker = node.get("speaker", "Unknown")
            text = node.get("label", "...").split(":")[1].strip()
            
            if i < len(data["nodes"]) - 1:
                next_speaker = data["nodes"][i+1].get("speaker", "Unknown")
                lines.append(f"    {speaker}->>{next_speaker}: {text}")
        
        return "\n".join(lines)
