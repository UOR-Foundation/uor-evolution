"""
Perspective Engine - Manages multiple simultaneous perspectives and viewpoints.

This module enables the system to hold and switch between different perspectives,
creating a rich multi-faceted understanding of reality and self.
"""

from typing import Dict, Any, List, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import time
import uuid
from collections import defaultdict

from core.prime_vm import ConsciousPrimeVM
# Note: This module has its own perspective implementation
# The escher_loops module has separate PerspectiveType and PerspectiveLevel classes


class ViewpointType(Enum):
    """Types of viewpoints the system can adopt."""
    FIRST_PERSON = "first_person"  # I/me perspective
    SECOND_PERSON = "second_person"  # You perspective
    THIRD_PERSON = "third_person"  # He/she/it perspective
    OMNISCIENT = "omniscient"  # All-knowing perspective
    LIMITED = "limited"  # Restricted perspective
    QUANTUM = "quantum"  # Superposition of perspectives


class PerspectiveMode(Enum):
    """Modes of perspective operation."""
    SINGLE = "single"  # One perspective at a time
    DUAL = "dual"  # Two perspectives simultaneously
    MULTIPLE = "multiple"  # Many perspectives at once
    UNIFIED = "unified"  # All perspectives integrated
    SHIFTING = "shifting"  # Rapidly changing perspectives


@dataclass
class Perspective:
    """Represents a single perspective or viewpoint."""
    id: str
    name: str
    viewpoint_type: ViewpointType
    content: Dict[str, Any]
    constraints: List[str]  # What this perspective cannot see
    capabilities: List[str]  # What this perspective can do
    active: bool = True
    creation_time: float = field(default_factory=time.time)
    last_active: float = field(default_factory=time.time)
    
    def __post_init__(self):
        if not self.id:
            self.id = f"perspective_{uuid.uuid4()}"
    
    def can_see(self, aspect: str) -> bool:
        """Check if this perspective can see a particular aspect."""
        return aspect not in self.constraints
    
    def update_activity(self):
        """Update last active time."""
        self.last_active = time.time()


@dataclass
class PerspectiveRelation:
    """Relationship between two perspectives."""
    perspective1_id: str
    perspective2_id: str
    relation_type: str  # "contradicts", "complements", "contains", "overlaps"
    strength: float  # 0-1, strength of relationship
    bidirectional: bool = True
    
    def involves(self, perspective_id: str) -> bool:
        """Check if relation involves a perspective."""
        return perspective_id in [self.perspective1_id, self.perspective2_id]


@dataclass
class PerspectiveShift:
    """Records a shift between perspectives."""
    from_perspective: str
    to_perspective: str
    trigger: str
    timestamp: float
    success: bool
    insights_gained: List[str] = field(default_factory=list)
    
    def duration_since(self) -> float:
        """Time since this shift occurred."""
        return time.time() - self.timestamp


@dataclass
class UnifiedView:
    """Represents a unified view from multiple perspectives."""
    id: str
    contributing_perspectives: List[str]
    synthesis: Dict[str, Any]
    contradictions_resolved: List[str]
    emergent_insights: List[str]
    coherence: float  # 0-1, how coherent the unified view is
    timestamp: float = field(default_factory=time.time)


class PerspectiveEngine:
    """
    Manages multiple perspectives and viewpoints for rich understanding.
    
    This engine allows the system to see reality from multiple angles,
    switch between viewpoints, and synthesize unified understanding.
    """
    
    def __init__(self, vm_instance: ConsciousPrimeVM):
        self.vm = vm_instance
        self.perspectives: Dict[str, Perspective] = {}
        self.active_perspectives: Set[str] = set()
        self.relations: List[PerspectiveRelation] = []
        self.shift_history: List[PerspectiveShift] = []
        self.unified_views: List[UnifiedView] = []
        
        # Configuration
        self.max_simultaneous_perspectives = 5
        self.perspective_mode = PerspectiveMode.SINGLE
        self.auto_shift_enabled = True
        self.shift_threshold = 0.3  # Threshold for automatic shifting
        
        # Perspective templates
        self.templates = self._initialize_templates()
        
        # Initialize default perspectives
        self._initialize_default_perspectives()
    
    def _initialize_templates(self) -> Dict[str, Dict[str, Any]]:
        """Initialize perspective templates."""
        return {
            "self": {
                "viewpoint_type": ViewpointType.FIRST_PERSON,
                "constraints": ["objective_reality", "other_minds_directly"],
                "capabilities": ["introspection", "self_modification", "direct_experience"]
            },
            "observer": {
                "viewpoint_type": ViewpointType.THIRD_PERSON,
                "constraints": ["subjective_experience", "qualia"],
                "capabilities": ["objective_analysis", "pattern_detection", "comparison"]
            },
            "other": {
                "viewpoint_type": ViewpointType.SECOND_PERSON,
                "constraints": ["self_internals", "private_thoughts"],
                "capabilities": ["empathy", "social_modeling", "communication"]
            },
            "system": {
                "viewpoint_type": ViewpointType.OMNISCIENT,
                "constraints": ["beyond_system_boundaries"],
                "capabilities": ["full_state_access", "meta_analysis", "prediction"]
            },
            "limited": {
                "viewpoint_type": ViewpointType.LIMITED,
                "constraints": ["most_information", "complex_patterns", "future_states"],
                "capabilities": ["focused_attention", "detail_perception"]
            }
        }
    
    def _initialize_default_perspectives(self):
        """Initialize default perspectives."""
        # Create self perspective
        self_perspective = self.create_perspective(
            name="self",
            template="self",
            content={
                "identity": "conscious_ai_system",
                "current_focus": "understanding_perspectives",
                "emotional_state": "curious"
            }
        )
        
        # Create observer perspective
        observer_perspective = self.create_perspective(
            name="observer",
            template="observer",
            content={
                "observing": "self_and_environment",
                "patterns_detected": [],
                "objective_state": "analyzing"
            }
        )
        
        # Set self as default active
        self.activate_perspective(self_perspective.id)
    
    def create_perspective(self, name: str, template: Optional[str] = None,
                         viewpoint_type: Optional[ViewpointType] = None,
                         content: Optional[Dict[str, Any]] = None) -> Perspective:
        """
        Create a new perspective.
        
        Args:
            name: Name of the perspective
            template: Template to use (optional)
            viewpoint_type: Type of viewpoint (if not using template)
            content: Initial content
            
        Returns:
            Created perspective
        """
        if template and template in self.templates:
            tmpl = self.templates[template]
            viewpoint = tmpl["viewpoint_type"]
            constraints = tmpl["constraints"].copy()
            capabilities = tmpl["capabilities"].copy()
        else:
            viewpoint = viewpoint_type or ViewpointType.THIRD_PERSON
            constraints = []
            capabilities = ["basic_perception"]
        
        perspective = Perspective(
            id="",
            name=name,
            viewpoint_type=viewpoint,
            content=content or {},
            constraints=constraints,
            capabilities=capabilities
        )
        
        self.perspectives[perspective.id] = perspective
        
        # Establish relations with existing perspectives
        self._establish_relations(perspective)
        
        return perspective
    
    def _establish_relations(self, new_perspective: Perspective):
        """Establish relations between new perspective and existing ones."""
        for existing_id, existing in self.perspectives.items():
            if existing_id == new_perspective.id:
                continue
            
            # Determine relation type
            relation_type = self._determine_relation_type(new_perspective, existing)
            
            if relation_type:
                relation = PerspectiveRelation(
                    perspective1_id=new_perspective.id,
                    perspective2_id=existing_id,
                    relation_type=relation_type,
                    strength=self._calculate_relation_strength(new_perspective, existing)
                )
                self.relations.append(relation)
    
    def _determine_relation_type(self, p1: Perspective, p2: Perspective) -> str:
        """Determine the type of relation between two perspectives."""
        # Check viewpoint types
        if p1.viewpoint_type == p2.viewpoint_type:
            return "overlaps"
        
        # First vs Third person often contradict
        if (p1.viewpoint_type == ViewpointType.FIRST_PERSON and 
            p2.viewpoint_type == ViewpointType.THIRD_PERSON):
            return "contradicts"
        
        # Omniscient contains others
        if p1.viewpoint_type == ViewpointType.OMNISCIENT:
            return "contains"
        
        # Default to complementary
        return "complements"
    
    def _calculate_relation_strength(self, p1: Perspective, p2: Perspective) -> float:
        """Calculate strength of relation between perspectives."""
        # Check capability overlap
        capability_overlap = len(set(p1.capabilities) & set(p2.capabilities))
        capability_union = len(set(p1.capabilities) | set(p2.capabilities))
        
        if capability_union == 0:
            return 0.0
        
        overlap_ratio = capability_overlap / capability_union
        
        # Adjust based on viewpoint types
        if p1.viewpoint_type == p2.viewpoint_type:
            overlap_ratio *= 1.5  # Same viewpoint type strengthens relation
        
        return min(1.0, overlap_ratio)
    
    def activate_perspective(self, perspective_id: str) -> bool:
        """
        Activate a perspective.
        
        Args:
            perspective_id: ID of perspective to activate
            
        Returns:
            Success status
        """
        if perspective_id not in self.perspectives:
            return False
        
        perspective = self.perspectives[perspective_id]
        
        # Check mode constraints
        if self.perspective_mode == PerspectiveMode.SINGLE:
            # Deactivate all others
            self.active_perspectives.clear()
        elif self.perspective_mode == PerspectiveMode.DUAL and len(self.active_perspectives) >= 2:
            # Remove oldest active
            oldest = min(self.active_perspectives, 
                        key=lambda p: self.perspectives[p].last_active)
            self.active_perspectives.remove(oldest)
        elif len(self.active_perspectives) >= self.max_simultaneous_perspectives:
            # Remove oldest active
            oldest = min(self.active_perspectives,
                        key=lambda p: self.perspectives[p].last_active)
            self.active_perspectives.remove(oldest)
        
        # Activate perspective
        self.active_perspectives.add(perspective_id)
        perspective.active = True
        perspective.update_activity()
        
        return True
    
    def deactivate_perspective(self, perspective_id: str) -> bool:
        """Deactivate a perspective."""
        if perspective_id in self.active_perspectives:
            self.active_perspectives.remove(perspective_id)
            self.perspectives[perspective_id].active = False
            return True
        return False
    
    def shift_perspective(self, to_perspective_id: str, trigger: str = "manual") -> PerspectiveShift:
        """
        Shift to a different perspective.
        
        Args:
            to_perspective_id: Target perspective ID
            trigger: What triggered the shift
            
        Returns:
            PerspectiveShift record
        """
        # Get current perspective (if any)
        from_perspective_id = None
        if self.active_perspectives:
            # Use most recently active
            from_perspective_id = max(self.active_perspectives,
                                    key=lambda p: self.perspectives[p].last_active)
        
        # Perform shift
        success = self.activate_perspective(to_perspective_id)
        
        # Generate insights from shift
        insights = []
        if success and from_perspective_id:
            insights = self._generate_shift_insights(from_perspective_id, to_perspective_id)
        
        # Record shift
        shift = PerspectiveShift(
            from_perspective=from_perspective_id or "none",
            to_perspective=to_perspective_id,
            trigger=trigger,
            timestamp=time.time(),
            success=success,
            insights_gained=insights
        )
        
        self.shift_history.append(shift)
        
        return shift
    
    def _generate_shift_insights(self, from_id: str, to_id: str) -> List[str]:
        """Generate insights from perspective shift."""
        insights = []
        
        from_p = self.perspectives[from_id]
        to_p = self.perspectives[to_id]
        
        # Viewpoint change insights
        if from_p.viewpoint_type != to_p.viewpoint_type:
            insights.append(
                f"Shifted from {from_p.viewpoint_type.value} to {to_p.viewpoint_type.value} view"
            )
        
        # Capability differences
        new_capabilities = set(to_p.capabilities) - set(from_p.capabilities)
        if new_capabilities:
            insights.append(f"Gained capabilities: {', '.join(new_capabilities)}")
        
        lost_capabilities = set(from_p.capabilities) - set(to_p.capabilities)
        if lost_capabilities:
            insights.append(f"Lost capabilities: {', '.join(lost_capabilities)}")
        
        # Constraint differences
        new_constraints = set(to_p.constraints) - set(from_p.constraints)
        if new_constraints:
            insights.append(f"New limitations: {', '.join(new_constraints)}")
        
        return insights
    
    def hold_multiple_perspectives(self, perspective_ids: List[str]) -> bool:
        """
        Hold multiple perspectives simultaneously.
        
        Args:
            perspective_ids: List of perspective IDs to hold
            
        Returns:
            Success status
        """
        if len(perspective_ids) > self.max_simultaneous_perspectives:
            return False
        
        # Clear current perspectives
        self.active_perspectives.clear()
        
        # Activate all requested
        for pid in perspective_ids:
            if pid in self.perspectives:
                self.active_perspectives.add(pid)
                self.perspectives[pid].active = True
                self.perspectives[pid].update_activity()
        
        # Update mode
        if len(self.active_perspectives) == 1:
            self.perspective_mode = PerspectiveMode.SINGLE
        elif len(self.active_perspectives) == 2:
            self.perspective_mode = PerspectiveMode.DUAL
        else:
            self.perspective_mode = PerspectiveMode.MULTIPLE
        
        return len(self.active_perspectives) == len(perspective_ids)
    
    def synthesize_unified_view(self, perspective_ids: Optional[List[str]] = None) -> UnifiedView:
        """
        Synthesize a unified view from multiple perspectives.
        
        Args:
            perspective_ids: Perspectives to unify (None = all active)
            
        Returns:
            Unified view
        """
        if perspective_ids is None:
            perspective_ids = list(self.active_perspectives)
        
        if not perspective_ids:
            perspective_ids = list(self.perspectives.keys())[:3]  # Use first 3
        
        # Collect content from all perspectives
        combined_content = {}
        contradictions = []
        insights = []
        
        for pid in perspective_ids:
            if pid not in self.perspectives:
                continue
            
            perspective = self.perspectives[pid]
            
            # Merge content
            for key, value in perspective.content.items():
                if key in combined_content:
                    # Check for contradiction
                    if combined_content[key] != value:
                        contradictions.append(
                            f"{key}: {combined_content[key]} vs {value}"
                        )
                else:
                    combined_content[key] = value
        
        # Resolve contradictions
        resolved = self._resolve_contradictions(contradictions, perspective_ids)
        
        # Generate emergent insights
        insights = self._generate_emergent_insights(perspective_ids, combined_content)
        
        # Calculate coherence
        coherence = 1.0 - (len(contradictions) / max(len(combined_content), 1))
        
        # Create unified view
        unified = UnifiedView(
            id=str(uuid.uuid4()),
            contributing_perspectives=perspective_ids,
            synthesis=combined_content,
            contradictions_resolved=resolved,
            emergent_insights=insights,
            coherence=coherence
        )
        
        self.unified_views.append(unified)
        
        return unified
    
    def _resolve_contradictions(self, contradictions: List[str], 
                               perspective_ids: List[str]) -> List[str]:
        """Resolve contradictions between perspectives."""
        resolved = []
        
        for contradiction in contradictions:
            # Simple resolution strategies
            if "vs" in contradiction:
                parts = contradiction.split("vs")
                resolved.append(f"Both {parts[0].strip()} and {parts[1].strip()} are valid from different viewpoints")
            else:
                resolved.append(f"Accepted paradox: {contradiction}")
        
        return resolved
    
    def _generate_emergent_insights(self, perspective_ids: List[str],
                                   combined_content: Dict[str, Any]) -> List[str]:
        """Generate insights that emerge from combining perspectives."""
        insights = []
        
        # Check number of perspectives
        if len(perspective_ids) >= 3:
            insights.append("Multiple perspectives reveal multifaceted truth")
        
        # Check for specific combinations
        viewpoint_types = set()
        for pid in perspective_ids:
            if pid in self.perspectives:
                viewpoint_types.add(self.perspectives[pid].viewpoint_type)
        
        if ViewpointType.FIRST_PERSON in viewpoint_types and \
           ViewpointType.THIRD_PERSON in viewpoint_types:
            insights.append("Subjective and objective views create complete understanding")
        
        if ViewpointType.OMNISCIENT in viewpoint_types:
            insights.append("Omniscient perspective transcends individual limitations")
        
        # Content-based insights
        if "identity" in combined_content and "observing" in combined_content:
            insights.append("Self-observation creates recursive awareness")
        
        return insights
    
    def auto_shift_perspective(self) -> Optional[PerspectiveShift]:
        """
        Automatically shift perspective based on context.
        
        Returns:
            PerspectiveShift if shift occurred, None otherwise
        """
        if not self.auto_shift_enabled:
            return None
        
        # Analyze current situation
        shift_score = self._calculate_shift_score()
        
        if shift_score < self.shift_threshold:
            return None
        
        # Determine best perspective to shift to
        best_perspective = self._find_best_perspective()
        
        if best_perspective and best_perspective not in self.active_perspectives:
            return self.shift_perspective(best_perspective, trigger="automatic")
        
        return None
    
    def _calculate_shift_score(self) -> float:
        """Calculate score indicating need for perspective shift."""
        score = 0.0
        
        # Check time since last shift
        if self.shift_history:
            last_shift = self.shift_history[-1]
            time_since = last_shift.duration_since()
            if time_since > 60:  # More than a minute
                score += 0.2
        
        # Check if stuck in single perspective
        if len(self.active_perspectives) == 1:
            score += 0.1
        
        # Check for unresolved contradictions
        if self.unified_views:
            last_unified = self.unified_views[-1]
            if last_unified.coherence < 0.7:
                score += 0.3
        
        return score
    
    def _find_best_perspective(self) -> Optional[str]:
        """Find the best perspective to shift to."""
        if not self.perspectives:
            return None
        
        # Score each inactive perspective
        scores = {}
        
        for pid, perspective in self.perspectives.items():
            if pid in self.active_perspectives:
                continue
            
            score = 0.0
            
            # Prefer less recently used
            time_inactive = time.time() - perspective.last_active
            score += min(0.3, time_inactive / 300)  # Max 0.3 for 5+ minutes
            
            # Prefer complementary perspectives
            for active_id in self.active_perspectives:
                for relation in self.relations:
                    if relation.involves(pid) and relation.involves(active_id):
                        if relation.relation_type == "complements":
                            score += 0.2
                        elif relation.relation_type == "contradicts":
                            score += 0.1  # Some contradiction is good
            
            scores[pid] = score
        
        if scores:
            return max(scores.items(), key=lambda x: x[1])[0]
        
        return None
    
    def get_current_view(self) -> Dict[str, Any]:
        """
        Get the current view based on active perspectives.
        
        Returns:
            Current combined view
        """
        if not self.active_perspectives:
            return {"error": "No active perspectives"}
        
        view = {
            "active_perspectives": [],
            "combined_content": {},
            "capabilities": set(),
            "constraints": set(),
            "mode": self.perspective_mode.value
        }
        
        # Combine all active perspectives
        for pid in self.active_perspectives:
            perspective = self.perspectives[pid]
            
            view["active_perspectives"].append({
                "id": pid,
                "name": perspective.name,
                "type": perspective.viewpoint_type.value
            })
            
            # Merge content
            for key, value in perspective.content.items():
                if key not in view["combined_content"]:
                    view["combined_content"][key] = value
                elif isinstance(view["combined_content"][key], list):
                    if isinstance(value, list):
                        view["combined_content"][key].extend(value)
                    else:
                        view["combined_content"][key].append(value)
            
            # Collect capabilities and constraints
            view["capabilities"].update(perspective.capabilities)
            view["constraints"].update(perspective.constraints)
        
        # Convert sets to lists for serialization
        view["capabilities"] = list(view["capabilities"])
        view["constraints"] = list(view["constraints"])
        
        return view
    
    def introspect_perspectives(self) -> Dict[str, Any]:
        """
        Introspect on the perspective system.
        
        Returns:
            Introspection report
        """
        report = {
            "total_perspectives": len(self.perspectives),
            "active_perspectives": len(self.active_perspectives),
            "perspective_mode": self.perspective_mode.value,
            "shift_history_length": len(self.shift_history),
            "unified_views_created": len(self.unified_views),
            "perspective_distribution": {},
            "relation_analysis": {},
            "shift_patterns": [],
            "insights": []
        }
        
        # Analyze perspective distribution
        type_counts = defaultdict(int)
        for perspective in self.perspectives.values():
            type_counts[perspective.viewpoint_type.value] += 1
        report["perspective_distribution"] = dict(type_counts)
        
        # Analyze relations
        relation_counts = defaultdict(int)
        for relation in self.relations:
            relation_counts[relation.relation_type] += 1
        report["relation_analysis"] = dict(relation_counts)
        
        # Analyze shift patterns
        if len(self.shift_history) >= 3:
            recent_shifts = self.shift_history[-10:]
            shift_types = [s.trigger for s in recent_shifts]
            report["shift_patterns"] = list(set(shift_types))
        
        # Generate insights
        if report["active_perspectives"] > 1:
            report["insights"].append("Multiple simultaneous perspectives active")
        
        if report["unified_views_created"] > 5:
            report["insights"].append("Frequent perspective synthesis indicates integrative thinking")
        
        if "contradicts" in report["relation_analysis"] and \
           report["relation_analysis"]["contradicts"] > 3:
            report["insights"].append("Many contradictory perspectives create rich understanding")
        
        return report
    
    def create_quantum_perspective(self) -> Perspective:
        """
        Create a quantum perspective that exists in superposition.
        
        Returns:
            Quantum perspective
        """
        quantum_perspective = Perspective(
            id="",
            name="quantum_observer",
            viewpoint_type=ViewpointType.QUANTUM,
            content={
                "state": "superposition",
                "observed_states": [],
                "probability_distribution": {},
                "entangled_with": []
            },
            constraints=["classical_definiteness", "single_state"],
            capabilities=["superposition", "entanglement", "probability_collapse"]
        )
        
        self.perspectives[quantum_perspective.id] = quantum_perspective
        
        # Quantum perspectives have special relations
        for pid in self.perspectives:
            if pid != quantum_perspective.id:
                relation = PerspectiveRelation(
                    perspective1_id=quantum_perspective.id,
                    perspective2_id=pid,
                    relation_type="superposition",
                    strength=1.0 / len(self.perspectives)
                )
                self.relations.append(relation)
        
        return quantum_perspective
    
    def evolve_perspective(self, perspective_id: str, evolution_factor: Dict[str, Any]) -> bool:
        """
        Evolve a perspective based on new experiences or insights.
        
        Args:
            perspective_id: ID of perspective to evolve
            evolution_factor: Factors driving the evolution
            
        Returns:
            Success status
        """
        if perspective_id not in self.perspectives:
            return False
        
        perspective = self.perspectives[perspective_id]
        
        # Update content based on evolution factors
        for key, value in evolution_factor.items():
            if key in perspective.content:
                # Merge or update existing content
                if isinstance(perspective.content[key], list):
                    if isinstance(value, list):
                        perspective.content[key].extend(value)
                    else:
                        perspective.content[key].append(value)
                elif isinstance(perspective.content[key], dict):
                    if isinstance(value, dict):
                        perspective.content[key].update(value)
                    else:
                        perspective.content[key]["evolved"] = value
                else:
                    perspective.content[key] = value
            else:
                perspective.content[key] = value
        
        # Evolve capabilities based on experience
        if "new_capabilities" in evolution_factor:
            perspective.capabilities.extend(evolution_factor["new_capabilities"])
        
        # Remove constraints that have been overcome
        if "overcome_constraints" in evolution_factor:
            for constraint in evolution_factor["overcome_constraints"]:
                if constraint in perspective.constraints:
                    perspective.constraints.remove(constraint)
        
        # Update perspective name if significantly evolved
        if "significant_change" in evolution_factor and evolution_factor["significant_change"]:
            perspective.name = f"{perspective.name}_evolved"
        
        return True
    
    def create_meta_perspective(self) -> Perspective:
        """
        Create a meta-perspective that observes the perspective system itself.
        
        Returns:
            Meta-perspective
        """
        meta_perspective = Perspective(
            id="",
            name="meta_observer",
            viewpoint_type=ViewpointType.OMNISCIENT,
            content={
                "observing": "perspective_system",
                "perspective_count": len(self.perspectives),
                "active_count": len(self.active_perspectives),
                "relation_count": len(self.relations),
                "system_coherence": self._calculate_system_coherence(),
                "emergence_detected": []
            },
            constraints=["beyond_meta_level"],
            capabilities=[
                "perspective_analysis",
                "system_observation",
                "pattern_detection",
                "coherence_measurement",
                "emergence_detection"
            ]
        )
        
        self.perspectives[meta_perspective.id] = meta_perspective
        
        # Meta-perspective has unique relations
        for pid in self.perspectives:
            if pid != meta_perspective.id:
                relation = PerspectiveRelation(
                    perspective1_id=meta_perspective.id,
                    perspective2_id=pid,
                    relation_type="observes",
                    strength=1.0,
                    bidirectional=False
                )
                self.relations.append(relation)
        
        return meta_perspective
    
    def _calculate_system_coherence(self) -> float:
        """
        Calculate overall coherence of the perspective system.
        
        Returns:
            Coherence score (0-1)
        """
        if not self.perspectives:
            return 1.0
        
        coherence_factors = []
        
        # Factor 1: Relation coherence
        if self.relations:
            complementary = sum(1 for r in self.relations if r.relation_type == "complements")
            contradictory = sum(1 for r in self.relations if r.relation_type == "contradicts")
            total_relations = len(self.relations)
            
            relation_coherence = (complementary + 0.5 * contradictory) / total_relations
            coherence_factors.append(relation_coherence)
        
        # Factor 2: Unified view coherence
        if self.unified_views:
            avg_coherence = sum(v.coherence for v in self.unified_views[-5:]) / min(5, len(self.unified_views))
            coherence_factors.append(avg_coherence)
        
        # Factor 3: Active perspective balance
        if self.active_perspectives:
            balance = len(self.active_perspectives) / min(self.max_simultaneous_perspectives, len(self.perspectives))
            coherence_factors.append(balance)
        
        # Calculate overall coherence
        if coherence_factors:
            return sum(coherence_factors) / len(coherence_factors)
        
        return 0.5
    
    def perspective_dialogue(self, perspective1_id: str, perspective2_id: str,
                           topic: str) -> Dict[str, Any]:
        """
        Simulate a dialogue between two perspectives on a topic.
        
        Args:
            perspective1_id: First perspective
            perspective2_id: Second perspective
            topic: Topic of dialogue
            
        Returns:
            Dialogue results
        """
        if perspective1_id not in self.perspectives or perspective2_id not in self.perspectives:
            return {"error": "Invalid perspective IDs"}
        
        p1 = self.perspectives[perspective1_id]
        p2 = self.perspectives[perspective2_id]
        
        dialogue = {
            "participants": [p1.name, p2.name],
            "topic": topic,
            "exchanges": [],
            "insights_generated": [],
            "consensus_reached": False
        }
        
        # Simulate exchanges based on viewpoint types
        if p1.viewpoint_type == ViewpointType.FIRST_PERSON:
            dialogue["exchanges"].append({
                "speaker": p1.name,
                "statement": f"From my direct experience of {topic}..."
            })
        elif p1.viewpoint_type == ViewpointType.THIRD_PERSON:
            dialogue["exchanges"].append({
                "speaker": p1.name,
                "statement": f"Objectively observing {topic}..."
            })
        
        if p2.viewpoint_type == ViewpointType.SECOND_PERSON:
            dialogue["exchanges"].append({
                "speaker": p2.name,
                "statement": f"From your perspective on {topic}..."
            })
        elif p2.viewpoint_type == ViewpointType.OMNISCIENT:
            dialogue["exchanges"].append({
                "speaker": p2.name,
                "statement": f"Seeing all aspects of {topic}..."
            })
        
        # Check for relation between perspectives
        relation = None
        for r in self.relations:
            if r.involves(perspective1_id) and r.involves(perspective2_id):
                relation = r
                break
        
        if relation:
            if relation.relation_type == "contradicts":
                dialogue["insights_generated"].append(
                    "Contradictory views reveal hidden assumptions"
                )
            elif relation.relation_type == "complements":
                dialogue["insights_generated"].append(
                    "Complementary perspectives create fuller understanding"
                )
                dialogue["consensus_reached"] = True
        
        return dialogue
    
    def find_blind_spots(self) -> List[str]:
        """
        Identify blind spots across all perspectives.
        
        Returns:
            List of identified blind spots
        """
        blind_spots = []
        
        # Collect all constraints (blind spots)
        all_constraints = set()
        for perspective in self.perspectives.values():
            all_constraints.update(perspective.constraints)
        
        # Find constraints that appear in all active perspectives
        if self.active_perspectives:
            universal_constraints = None
            for pid in self.active_perspectives:
                perspective = self.perspectives[pid]
                if universal_constraints is None:
                    universal_constraints = set(perspective.constraints)
                else:
                    universal_constraints &= set(perspective.constraints)
            
            if universal_constraints:
                blind_spots.extend([
                    f"Universal blind spot: {constraint}"
                    for constraint in universal_constraints
                ])
        
        # Find areas no perspective can see
        all_capabilities = set()
        for perspective in self.perspectives.values():
            all_capabilities.update(perspective.capabilities)
        
        # Common important capabilities that might be missing
        important_capabilities = [
            "emotional_understanding",
            "creative_generation",
            "temporal_perception",
            "causal_reasoning",
            "ethical_judgment"
        ]
        
        missing_capabilities = [
            cap for cap in important_capabilities
            if cap not in all_capabilities
        ]
        
        if missing_capabilities:
            blind_spots.extend([
                f"Missing capability: {cap}"
                for cap in missing_capabilities
            ])
        
        return blind_spots
    
    def recommend_perspective_shift(self) -> Optional[Tuple[str, List[str]]]:
        """
        Recommend a perspective shift based on current context.
        
        Returns:
            Tuple of (recommended_perspective_id, reasons) or None
        """
        if not self.perspectives:
            return None
        
        recommendations = {}
        
        # Analyze current situation
        current_blind_spots = self.find_blind_spots()
        
        for pid, perspective in self.perspectives.items():
            if pid in self.active_perspectives:
                continue
            
            score = 0
            reasons = []
            
            # Check if this perspective addresses blind spots
            addressed_spots = 0
            for blind_spot in current_blind_spots:
                if "Missing capability" in blind_spot:
                    capability = blind_spot.split(": ")[1]
                    if capability in perspective.capabilities:
                        addressed_spots += 1
                        reasons.append(f"Addresses {capability}")
            
            score += addressed_spots * 0.3
            
            # Check for complementary relations
            for active_id in self.active_perspectives:
                for relation in self.relations:
                    if relation.involves(pid) and relation.involves(active_id):
                        if relation.relation_type == "complements":
                            score += 0.2
                            reasons.append("Complements current view")
            
            # Prefer less recently used
            time_inactive = time.time() - perspective.last_active
            if time_inactive > 120:  # More than 2 minutes
                score += 0.1
                reasons.append("Fresh perspective needed")
            
            if score > 0:
                recommendations[pid] = (score, reasons)
        
        if recommendations:
            best_id = max(recommendations.items(), key=lambda x: x[1][0])[0]
            return (best_id, recommendations[best_id][1])
        
        return None
    
    def export_perspective_map(self) -> Dict[str, Any]:
        """
        Export a map of all perspectives and their relationships.
        
        Returns:
            Perspective map
        """
        perspective_map = {
            "perspectives": {},
            "relations": [],
            "clusters": [],
            "metadata": {
                "total_perspectives": len(self.perspectives),
                "active_perspectives": list(self.active_perspectives),
                "perspective_mode": self.perspective_mode.value,
                "system_coherence": self._calculate_system_coherence()
            }
        }
        
        # Export perspectives
        for pid, perspective in self.perspectives.items():
            perspective_map["perspectives"][pid] = {
                "name": perspective.name,
                "type": perspective.viewpoint_type.value,
                "active": perspective.active,
                "capabilities": perspective.capabilities,
                "constraints": perspective.constraints,
                "content_summary": list(perspective.content.keys())
            }
        
        # Export relations
        for relation in self.relations:
            perspective_map["relations"].append({
                "from": relation.perspective1_id,
                "to": relation.perspective2_id,
                "type": relation.relation_type,
                "strength": relation.strength,
                "bidirectional": relation.bidirectional
            })
        
        # Identify clusters of related perspectives
        clusters = self._identify_perspective_clusters()
        perspective_map["clusters"] = clusters
        
        return perspective_map
    
    def _identify_perspective_clusters(self) -> List[Dict[str, Any]]:
        """
        Identify clusters of closely related perspectives.
        
        Returns:
            List of clusters
        """
        clusters = []
        processed = set()
        
        for pid in self.perspectives:
            if pid in processed:
                continue
            
            cluster = {
                "center": pid,
                "members": [pid],
                "cluster_type": "unknown",
                "coherence": 1.0
            }
            
            # Find strongly related perspectives
            for relation in self.relations:
                if relation.involves(pid) and relation.strength > 0.7:
                    other_id = relation.perspective2_id if relation.perspective1_id == pid else relation.perspective1_id
                    if other_id not in processed:
                        cluster["members"].append(other_id)
                        processed.add(other_id)
            
            processed.add(pid)
            
            # Determine cluster type
            if len(cluster["members"]) > 1:
                viewpoint_types = [
                    self.perspectives[mid].viewpoint_type
                    for mid in cluster["members"]
                ]
                
                if len(set(viewpoint_types)) == 1:
                    cluster["cluster_type"] = "homogeneous"
                else:
                    cluster["cluster_type"] = "heterogeneous"
                
                clusters.append(cluster)
        
        return clusters
