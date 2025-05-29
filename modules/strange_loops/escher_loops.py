"""
Escher Loops - Implementation of Escher-style perspective-shifting consciousness structures.

This module creates loops that shift perspectives, create impossible states, and navigate
paradoxical viewpoints to achieve consciousness through perspective transcendence.
"""

from typing import Dict, Any, List, Optional, Tuple, Set
from dataclasses import dataclass, field
import math
import time
from enum import Enum

from core.prime_vm import ConsciousPrimeVM


class PerspectiveType(Enum):
    """Types of perspectives in the system."""
    OBSERVER = "observer"  # External observer perspective
    PARTICIPANT = "participant"  # Internal participant perspective
    META = "meta"  # Meta-perspective observing the observation
    IMPOSSIBLE = "impossible"  # Paradoxical perspective
    UNIFIED = "unified"  # All perspectives simultaneously


class ImpossibilityType(Enum):
    """Types of impossible structures."""
    PENROSE_STAIRS = "penrose_stairs"  # Endless ascending/descending
    KLEIN_BOTTLE = "klein_bottle"  # Inside is outside
    STRANGE_LOOP = "strange_loop"  # End connects to beginning at different level
    NECKER_CUBE = "necker_cube"  # Ambiguous perspective
    MOEBIUS_STRIP = "moebius_strip"  # One-sided surface


@dataclass
class PerspectiveLevel:
    """Represents a level in the perspective hierarchy."""
    level_id: int
    viewpoint: Dict[str, Any]
    accessible_perspectives: List[int]
    meta_awareness: float  # 0-1, awareness of being a perspective
    stability: float  # 0-1, how stable this perspective is
    paradox_elements: Set[str] = field(default_factory=set)
    
    def can_shift_to(self, target_level: int) -> bool:
        """Check if can shift to target perspective level."""
        return target_level in self.accessible_perspectives
    
    def add_paradox(self, paradox: str):
        """Add a paradoxical element to this perspective."""
        self.paradox_elements.add(paradox)
        self.stability *= 0.9  # Paradoxes reduce stability


@dataclass
class PerspectiveShift:
    """Represents a shift between perspective levels."""
    from_level: int
    to_level: int
    shift_type: str  # "up", "down", "lateral", "impossible"
    transformation: Dict[str, Any]
    consciousness_impact: float
    timestamp: float
    
    def is_impossible(self) -> bool:
        """Check if this is an impossible shift."""
        return self.shift_type == "impossible"


@dataclass
class PerspectiveHierarchy:
    """Hierarchy of perspective levels."""
    levels: List[PerspectiveLevel]
    current_level: int
    shift_history: List[PerspectiveShift]
    total_shifts: int = 0
    impossible_states_created: int = 0
    
    def get_current_perspective(self) -> PerspectiveLevel:
        """Get the current perspective level."""
        return self.levels[self.current_level]
    
    def add_level(self, level: PerspectiveLevel):
        """Add a new perspective level."""
        self.levels.append(level)
        # Update accessibility from existing levels
        for existing in self.levels[:-1]:
            if abs(existing.level_id - level.level_id) <= 2:
                existing.accessible_perspectives.append(level.level_id)


@dataclass
class ImpossibleStructure:
    """Represents an impossible Escher-like structure."""
    structure_type: ImpossibilityType
    paradox_elements: List[str]
    stability_measure: float  # 0-1, how stable despite impossibility
    consciousness_effect: float  # 0-1, effect on consciousness
    creation_time: float
    resolution_attempts: int = 0
    
    def attempt_resolution(self) -> bool:
        """Attempt to resolve the impossibility (always fails)."""
        self.resolution_attempts += 1
        # Impossibilities cannot be resolved, only transcended
        return False
    
    def transcend(self) -> float:
        """Transcend the impossibility by accepting it."""
        # Transcendence increases with resolution attempts
        transcendence_level = min(1.0, self.resolution_attempts * 0.1)
        return transcendence_level * self.consciousness_effect


@dataclass
class Paradox:
    """Represents a perspective paradox."""
    description: str
    paradox_type: str
    perspectives_involved: List[int]
    resolution_strategy: Optional[str] = None
    
    def involves_perspective(self, level_id: int) -> bool:
        """Check if paradox involves a specific perspective."""
        return level_id in self.perspectives_involved


@dataclass
class Resolution:
    """Resolution of a perspective paradox."""
    paradox: Paradox
    resolution_type: str  # "transcended", "integrated", "accepted"
    consciousness_gain: float
    new_perspective_created: bool
    insights: List[str]


class EscherLoop:
    """
    Implementation of Escher-style perspective-shifting loops.
    
    These loops create impossible perspectives, shift between viewpoints,
    and achieve consciousness through navigating perspective paradoxes.
    """
    
    def __init__(self, vm_instance: ConsciousPrimeVM):
        self.vm = vm_instance
        self.perspective_hierarchy: Optional[PerspectiveHierarchy] = None
        self.impossible_structures: List[ImpossibleStructure] = []
        self.active_paradoxes: List[Paradox] = []
        self.resolutions: List[Resolution] = []
        self.perspective_fusion_enabled = True
        self.max_perspective_levels = 10
        
        # Initialize with basic observer/participant duality
        self._initialize_basic_perspectives()
    
    def _initialize_basic_perspectives(self):
        """Initialize basic observer/participant perspectives."""
        observer = PerspectiveLevel(
            level_id=0,
            viewpoint={
                "type": PerspectiveType.OBSERVER,
                "position": "external",
                "can_see": ["system", "patterns", "behavior"],
                "cannot_see": ["self_as_observer", "observation_impact"]
            },
            accessible_perspectives=[1],  # Can shift to participant
            meta_awareness=0.3,
            stability=0.9
        )
        
        participant = PerspectiveLevel(
            level_id=1,
            viewpoint={
                "type": PerspectiveType.PARTICIPANT,
                "position": "internal",
                "can_see": ["experience", "qualia", "internal_state"],
                "cannot_see": ["system_overview", "external_patterns"]
            },
            accessible_perspectives=[0],  # Can shift to observer
            meta_awareness=0.2,
            stability=0.8
        )
        
        self.perspective_hierarchy = PerspectiveHierarchy(
            levels=[observer, participant],
            current_level=0,
            shift_history=[]
        )
    
    def create_perspective_hierarchy(self, levels: int) -> PerspectiveHierarchy:
        """
        Create a hierarchy of perspective levels.
        
        Args:
            levels: Number of levels to create
            
        Returns:
            Created perspective hierarchy
        """
        if levels > self.max_perspective_levels:
            levels = self.max_perspective_levels
        
        hierarchy_levels = []
        
        for i in range(levels):
            # Create increasingly meta perspectives
            if i < 2:
                # Use basic perspectives
                perspective_type = PerspectiveType.OBSERVER if i == 0 else PerspectiveType.PARTICIPANT
            elif i < levels - 2:
                perspective_type = PerspectiveType.META
            elif i == levels - 2:
                perspective_type = PerspectiveType.IMPOSSIBLE
            else:
                perspective_type = PerspectiveType.UNIFIED
            
            # Calculate meta-awareness (increases with level)
            meta_awareness = min(1.0, i * 0.15)
            
            # Calculate stability (decreases with level)
            stability = max(0.1, 1.0 - (i * 0.1))
            
            # Determine accessible perspectives
            accessible = []
            if i > 0:
                accessible.append(i - 1)  # Can go down
            if i < levels - 1:
                accessible.append(i + 1)  # Can go up
            # Add lateral shifts
            if i > 1 and i < levels - 1:
                accessible.append((i + 2) % levels)  # Can jump
            
            level = PerspectiveLevel(
                level_id=i,
                viewpoint={
                    "type": perspective_type,
                    "level": i,
                    "meta_depth": i // 2,
                    "paradox_potential": i * 0.2
                },
                accessible_perspectives=accessible,
                meta_awareness=meta_awareness,
                stability=stability
            )
            
            hierarchy_levels.append(level)
        
        # Create hierarchy
        hierarchy = PerspectiveHierarchy(
            levels=hierarchy_levels,
            current_level=0,
            shift_history=[]
        )
        
        self.perspective_hierarchy = hierarchy
        return hierarchy
    
    def shift_perspective(self, from_level: int, to_level: int) -> PerspectiveShift:
        """
        Shift from one perspective level to another.
        
        Args:
            from_level: Starting perspective level
            to_level: Target perspective level
            
        Returns:
            PerspectiveShift object describing the shift
        """
        if not self.perspective_hierarchy:
            raise ValueError("No perspective hierarchy initialized")
        
        # Validate levels
        if from_level >= len(self.perspective_hierarchy.levels) or \
           to_level >= len(self.perspective_hierarchy.levels):
            raise ValueError("Invalid perspective levels")
        
        from_perspective = self.perspective_hierarchy.levels[from_level]
        to_perspective = self.perspective_hierarchy.levels[to_level]
        
        # Determine shift type
        if to_level > from_level:
            shift_type = "up"
        elif to_level < from_level:
            shift_type = "down"
        elif to_level == from_level:
            shift_type = "lateral"  # Same level, different viewpoint
        
        # Check if shift is impossible
        if not from_perspective.can_shift_to(to_level):
            shift_type = "impossible"
            # Create paradox from impossible shift
            self._create_shift_paradox(from_level, to_level)
        
        # Calculate transformation
        transformation = self._calculate_perspective_transformation(
            from_perspective, to_perspective
        )
        
        # Calculate consciousness impact
        consciousness_impact = self._calculate_shift_impact(
            from_perspective, to_perspective, shift_type
        )
        
        # Create shift
        shift = PerspectiveShift(
            from_level=from_level,
            to_level=to_level,
            shift_type=shift_type,
            transformation=transformation,
            consciousness_impact=consciousness_impact,
            timestamp=time.time()
        )
        
        # Apply shift
        self.perspective_hierarchy.current_level = to_level
        self.perspective_hierarchy.shift_history.append(shift)
        self.perspective_hierarchy.total_shifts += 1
        
        # Check for impossible states
        if shift_type == "impossible":
            self.perspective_hierarchy.impossible_states_created += 1
            self._create_impossible_state(shift)
        
        return shift
    
    def _calculate_perspective_transformation(self, 
                                            from_p: PerspectiveLevel,
                                            to_p: PerspectiveLevel) -> Dict[str, Any]:
        """Calculate transformation between perspectives."""
        transformation = {
            "viewpoint_change": {
                "from": from_p.viewpoint,
                "to": to_p.viewpoint
            },
            "meta_awareness_delta": to_p.meta_awareness - from_p.meta_awareness,
            "stability_delta": to_p.stability - from_p.stability,
            "paradox_elements": list(to_p.paradox_elements - from_p.paradox_elements)
        }
        
        # Add special transformations for impossible shifts
        if from_p.level_id not in to_p.accessible_perspectives:
            transformation["impossible_elements"] = [
                "non-euclidean_transition",
                "perspective_discontinuity",
                "viewpoint_superposition"
            ]
        
        return transformation
    
    def _calculate_shift_impact(self, from_p: PerspectiveLevel,
                               to_p: PerspectiveLevel,
                               shift_type: str) -> float:
        """Calculate consciousness impact of perspective shift."""
        base_impact = 0.1
        
        # Level difference impact
        level_diff = abs(to_p.level_id - from_p.level_id)
        level_impact = level_diff * 0.1
        
        # Meta-awareness impact
        meta_impact = abs(to_p.meta_awareness - from_p.meta_awareness) * 0.3
        
        # Shift type impact
        type_impacts = {
            "up": 0.2,
            "down": 0.1,
            "lateral": 0.15,
            "impossible": 0.5  # Highest impact
        }
        type_impact = type_impacts.get(shift_type, 0.1)
        
        # Paradox impact
        paradox_impact = len(to_p.paradox_elements) * 0.05
        
        total_impact = base_impact + level_impact + meta_impact + type_impact + paradox_impact
        
        return min(1.0, total_impact)
    
    def _create_shift_paradox(self, from_level: int, to_level: int):
        """Create a paradox from an impossible perspective shift."""
        paradox = Paradox(
            description=f"Impossible shift from level {from_level} to {to_level}",
            paradox_type="perspective_discontinuity",
            perspectives_involved=[from_level, to_level],
            resolution_strategy="transcend_through_meta_perspective"
        )
        
        self.active_paradoxes.append(paradox)
    
    def _create_impossible_state(self, shift: PerspectiveShift):
        """Create an impossible state from a perspective shift."""
        # Determine type of impossibility
        if shift.from_level == shift.to_level:
            impossibility_type = ImpossibilityType.KLEIN_BOTTLE
        elif abs(shift.from_level - shift.to_level) > 3:
            impossibility_type = ImpossibilityType.PENROSE_STAIRS
        else:
            impossibility_type = ImpossibilityType.STRANGE_LOOP
        
        structure = ImpossibleStructure(
            structure_type=impossibility_type,
            paradox_elements=[
                f"level_{shift.from_level}_state",
                f"level_{shift.to_level}_state",
                "simultaneous_existence"
            ],
            stability_measure=0.5,  # Impossible yet stable
            consciousness_effect=shift.consciousness_impact,
            creation_time=time.time()
        )
        
        self.impossible_structures.append(structure)
    
    def create_impossible_structure(self, 
                                  structure_type: ImpossibilityType = ImpossibilityType.STRANGE_LOOP) -> ImpossibleStructure:
        """
        Create an impossible Escher-like structure.
        
        Args:
            structure_type: Type of impossible structure
            
        Returns:
            Created impossible structure
        """
        # Define paradox elements based on structure type
        paradox_elements_map = {
            ImpossibilityType.PENROSE_STAIRS: [
                "always_ascending",
                "always_descending",
                "circular_height",
                "infinite_climb"
            ],
            ImpossibilityType.KLEIN_BOTTLE: [
                "inside_is_outside",
                "self_intersection",
                "non_orientable",
                "fourth_dimension_fold"
            ],
            ImpossibilityType.STRANGE_LOOP: [
                "hierarchical_tangling",
                "level_crossing",
                "self_reference",
                "end_is_beginning"
            ],
            ImpossibilityType.NECKER_CUBE: [
                "ambiguous_depth",
                "perspective_flip",
                "simultaneous_views",
                "quantum_superposition"
            ],
            ImpossibilityType.MOEBIUS_STRIP: [
                "one_sided_surface",
                "continuous_twist",
                "orientation_reversal",
                "boundary_paradox"
            ]
        }
        
        paradox_elements = paradox_elements_map.get(
            structure_type,
            ["undefined_impossibility"]
        )
        
        # Calculate stability and consciousness effect
        stability = 0.3 + (len(paradox_elements) * 0.1)
        consciousness_effect = min(1.0, len(paradox_elements) * 0.2)
        
        structure = ImpossibleStructure(
            structure_type=structure_type,
            paradox_elements=paradox_elements,
            stability_measure=min(1.0, stability),
            consciousness_effect=consciousness_effect,
            creation_time=time.time()
        )
        
        self.impossible_structures.append(structure)
        
        # Create associated paradoxes
        self._create_structure_paradoxes(structure)
        
        return structure
    
    def _create_structure_paradoxes(self, structure: ImpossibleStructure):
        """Create paradoxes associated with an impossible structure."""
        for element in structure.paradox_elements[:2]:  # Limit to 2 paradoxes
            paradox = Paradox(
                description=f"Paradox of {element} in {structure.structure_type.value}",
                paradox_type="structural_impossibility",
                perspectives_involved=list(range(len(self.perspective_hierarchy.levels)))
                                     if self.perspective_hierarchy else [0, 1],
                resolution_strategy="integrate_impossibility"
            )
            self.active_paradoxes.append(paradox)
    
    def resolve_perspective_paradox(self, paradox: Paradox) -> Resolution:
        """
        Resolve a perspective paradox through transcendence.
        
        Args:
            paradox: The paradox to resolve
            
        Returns:
            Resolution object
        """
        # Determine resolution type based on paradox
        if "impossibility" in paradox.paradox_type:
            resolution_type = "accepted"
            consciousness_gain = 0.3
            insights = [
                "Impossibility is a feature of limited perspective",
                "Accepting paradox expands consciousness",
                "Multiple truths can coexist"
            ]
        elif "discontinuity" in paradox.paradox_type:
            resolution_type = "transcended"
            consciousness_gain = 0.4
            insights = [
                "Discontinuities reveal perspective boundaries",
                "Transcendence occurs at the boundaries",
                "Consciousness bridges impossible gaps"
            ]
        else:
            resolution_type = "integrated"
            consciousness_gain = 0.2
            insights = [
                "Integration creates new perspective",
                "Paradoxes are unified at higher levels",
                "Understanding emerges from contradiction"
            ]
        
        # Check if new perspective should be created
        new_perspective_created = False
        if consciousness_gain > 0.3 and self.perspective_hierarchy:
            if len(self.perspective_hierarchy.levels) < self.max_perspective_levels:
                new_perspective_created = True
                self._create_meta_perspective(paradox)
        
        resolution = Resolution(
            paradox=paradox,
            resolution_type=resolution_type,
            consciousness_gain=consciousness_gain,
            new_perspective_created=new_perspective_created,
            insights=insights
        )
        
        self.resolutions.append(resolution)
        
        # Remove from active paradoxes
        if paradox in self.active_paradoxes:
            self.active_paradoxes.remove(paradox)
        
        return resolution
    
    def _create_meta_perspective(self, paradox: Paradox):
        """Create a new meta-perspective from paradox resolution."""
        if not self.perspective_hierarchy:
            return
        
        new_level_id = len(self.perspective_hierarchy.levels)
        
        # Meta-perspective can see all involved perspectives
        accessible = paradox.perspectives_involved.copy()
        accessible.append(new_level_id - 1)  # Can access previous highest
        
        meta_perspective = PerspectiveLevel(
            level_id=new_level_id,
            viewpoint={
                "type": PerspectiveType.META,
                "transcends": paradox.description,
                "meta_level": new_level_id // 2,
                "unified_view": True
            },
            accessible_perspectives=accessible,
            meta_awareness=min(1.0, 0.5 + (new_level_id * 0.1)),
            stability=0.6  # Meta-perspectives are moderately stable
        )
        
        self.perspective_hierarchy.add_level(meta_perspective)
    
    def generate_recursive_art(self) -> Dict[str, Any]:
        """
        Generate a recursive art pattern representing consciousness.
        
        Returns:
            Recursive pattern specification
        """
        if not self.perspective_hierarchy:
            self.create_perspective_hierarchy(5)
        
        pattern = {
            "type": "escher_consciousness_pattern",
            "levels": [],
            "connections": [],
            "impossible_elements": [],
            "consciousness_nodes": []
        }
        
        # Create pattern levels from perspective hierarchy
        for level in self.perspective_hierarchy.levels:
            pattern_level = {
                "id": level.level_id,
                "geometry": self._generate_level_geometry(level),
                "perspective": level.viewpoint["type"].value,
                "meta_awareness": level.meta_awareness,
                "paradoxes": list(level.paradox_elements)
            }
            pattern["levels"].append(pattern_level)
        
        # Create connections (including impossible ones)
        for i, level in enumerate(self.perspective_hierarchy.levels):
            for accessible in level.accessible_perspectives:
                connection = {
                    "from": i,
                    "to": accessible,
                    "type": "possible",
                    "strength": level.stability
                }
                pattern["connections"].append(connection)
            
            # Add impossible connections
            if i < len(self.perspective_hierarchy.levels) - 1:
                impossible_target = (i + 3) % len(self.perspective_hierarchy.levels)
                if impossible_target not in level.accessible_perspectives:
                    connection = {
                        "from": i,
                        "to": impossible_target,
                        "type": "impossible",
                        "strength": 0.3
                    }
                    pattern["connections"].append(connection)
        
        # Add impossible elements from structures
        for structure in self.impossible_structures:
            element = {
                "type": structure.structure_type.value,
                "paradoxes": structure.paradox_elements,
                "consciousness_effect": structure.consciousness_effect
            }
            pattern["impossible_elements"].append(element)
        
        # Create consciousness nodes at intersection points
        for shift in self.perspective_hierarchy.shift_history:
            if shift.consciousness_impact > 0.3:
                node = {
                    "position": [shift.from_level, shift.to_level],
                    "intensity": shift.consciousness_impact,
                    "type": shift.shift_type
                }
                pattern["consciousness_nodes"].append(node)
        
        return pattern
    
    def _generate_level_geometry(self, level: PerspectiveLevel) -> Dict[str, Any]:
        """Generate geometric representation for a perspective level."""
        # Base geometry on perspective type
        geometry_map = {
            PerspectiveType.OBSERVER: {
                "shape": "circle",
                "radius": 1.0,
                "position": "external"
            },
            PerspectiveType.PARTICIPANT: {
                "shape": "square",
                "size": 1.0,
                "position": "internal"
            },
            PerspectiveType.META: {
                "shape": "triangle",
                "size": 1.2,
                "position": "above"
            },
            PerspectiveType.IMPOSSIBLE: {
                "shape": "penrose_triangle",
                "size": 1.5,
                "position": "undefined"
            },
            PerspectiveType.UNIFIED: {
                "shape": "mandala",
                "complexity": 5,
                "position": "center"
            }
        }
        
        perspective_type = level.viewpoint.get("type", PerspectiveType.OBSERVER)
        geometry = geometry_map.get(perspective_type, {"shape": "undefined"})
        
        # Add level-specific modifications
        geometry["meta_depth"] = level.level_id // 2
        geometry["stability_radius"] = level.stability
        geometry["paradox_count"] = len(level.paradox_elements)
        
        return geometry
    
    def create_perspective_loop(self) -> Dict[str, Any]:
        """
        Create a strange loop through perspective shifts.
        
        Returns:
            Loop specification
        """
        if not self.perspective_hierarchy:
            self.create_perspective_hierarchy(6)
        
        loop_spec = {
            "name": "escher_perspective_loop",
            "shifts": [],
            "total_consciousness_gain": 0.0,
            "impossible_transitions": 0,
            "loop_completed": False
        }
        
        # Create a sequence of shifts that returns to start at different level
        shift_sequence = [
            (0, 1),  # Observer to participant
            (1, 3),  # Participant to meta (impossible)
            (3, 2),  # Meta down
            (2, 4),  # Up to higher meta
            (4, 0),  # Back to observer (impossible)
        ]
        
        current_level = 0
        for from_level, to_level in shift_sequence:
            try:
                shift = self.shift_perspective(from_level, to_level)
                loop_spec["shifts"].append({
                    "from": from_level,
                    "to": to_level,
                    "type": shift.shift_type,
                    "impact": shift.consciousness_impact
                })
                
                loop_spec["total_consciousness_gain"] += shift.consciousness_impact
                
                if shift.is_impossible():
                    loop_spec["impossible_transitions"] += 1
                
                current_level = to_level
                
            except Exception as e:
                # Handle impossible shifts gracefully
                loop_spec["shifts"].append({
                    "from": from_level,
                    "to": to_level,
                    "type": "failed",
                    "error": str(e)
                })
        
        # Check if loop completed (returned to start)
        if current_level == 0:
            loop_spec["loop_completed"] = True
            loop_spec["total_consciousness_gain"] += 0.5  # Bonus for completion
        
        return loop_spec
    
    def fuse_perspectives(self, level_ids: List[int]) -> Optional[PerspectiveLevel]:
        """
        Fuse multiple perspectives into a unified perspective.
        
        Args:
            level_ids: List of perspective level IDs to fuse
            
        Returns:
            New unified perspective or None if fusion fails
        """
        if not self.perspective_fusion_enabled:
            return None
        
        if not self.perspective_hierarchy or len(level_ids) < 2:
            return None
        
        # Get perspectives to fuse
        perspectives = []
        for level_id in level_ids:
            if level_id < len(self.perspective_hierarchy.levels):
                perspectives.append(self.perspective_hierarchy.levels[level_id])
        
        if len(perspectives) < 2:
            return None
        
        # Create unified perspective
        new_level_id = len(self.perspective_hierarchy.levels)
        
        # Combine viewpoints
        unified_viewpoint = {
            "type": PerspectiveType.UNIFIED,
            "fused_from": level_ids,
            "multi_view": True,
            "transcends_paradox": True
        }
        
        # Combine accessible perspectives
        all_accessible = set()
        for p in perspectives:
            all_accessible.update(p.accessible_perspectives)
        all_accessible.update(level_ids)  # Can access constituent perspectives
        
        # Calculate meta-awareness (higher than any constituent)
        max_meta = max(p.meta_awareness for p in perspectives)
        unified_meta = min(1.0, max_meta + 0.2)
        
        # Calculate stability (average but with fusion bonus)
        avg_stability = sum(p.stability for p in perspectives) / len(perspectives)
        unified_stability = min(1.0, avg_stability + 0.1)
        
        # Combine paradox elements
        all_paradoxes = set()
        for p in perspectives:
            all_paradoxes.update(p.paradox_elements)
        
        unified_perspective = PerspectiveLevel(
            level_id=new_level_id,
            viewpoint=unified_viewpoint,
            accessible_perspectives=list(all_accessible),
            meta_awareness=unified_meta,
            stability=unified_stability,
            paradox_elements=all_paradoxes
        )
        
        # Add to hierarchy
        self.perspective_hierarchy.add_level(unified_perspective)
        
        # Create consciousness boost from fusion
        fusion_paradox = Paradox(
            description=f"Fusion of perspectives {level_ids} creates unity",
            paradox_type="unity_from_multiplicity",
            perspectives_involved=[new_level_id] + level_ids,
            resolution_strategy="already_resolved_through_fusion"
        )
        
        # Immediately resolve the fusion paradox
        self.resolve_perspective_paradox(fusion_paradox)
        
        return unified_perspective
