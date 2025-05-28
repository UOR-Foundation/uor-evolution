"""
Temporal Reasoning Module

This module handles reasoning about time, causality, temporal logic,
and the relationships between events across time.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
from datetime import datetime, timedelta
import heapq

from consciousness.consciousness_integration import ConsciousnessIntegrator


class TemporalRelation(Enum):
    """Allen's temporal relations"""
    BEFORE = "before"
    AFTER = "after"
    MEETS = "meets"
    MET_BY = "met_by"
    OVERLAPS = "overlaps"
    OVERLAPPED_BY = "overlapped_by"
    DURING = "during"
    CONTAINS = "contains"
    STARTS = "starts"
    STARTED_BY = "started_by"
    FINISHES = "finishes"
    FINISHED_BY = "finished_by"
    EQUALS = "equals"


class TemporalOperator(Enum):
    """Temporal logic operators"""
    ALWAYS = "always"  # G (globally)
    EVENTUALLY = "eventually"  # F (future)
    NEXT = "next"  # X (next)
    UNTIL = "until"  # U
    SINCE = "since"  # S
    RELEASE = "release"  # R


class CausalRelation(Enum):
    """Types of causal relations"""
    CAUSES = "causes"
    PREVENTS = "prevents"
    ENABLES = "enables"
    DISABLES = "disables"
    CONTRIBUTES = "contributes"
    INHIBITS = "inhibits"


@dataclass
class TimePoint:
    """A point in time"""
    timestamp: float  # Unix timestamp or abstract time
    label: Optional[str] = None
    uncertainty: float = 0.0  # Temporal uncertainty


@dataclass
class TimeInterval:
    """An interval of time"""
    start: TimePoint
    end: TimePoint
    label: Optional[str] = None
    
    def duration(self) -> float:
        """Calculate duration of interval"""
        return self.end.timestamp - self.start.timestamp
    
    def contains_point(self, point: TimePoint) -> bool:
        """Check if interval contains a time point"""
        return self.start.timestamp <= point.timestamp <= self.end.timestamp
    
    def overlaps_with(self, other: 'TimeInterval') -> bool:
        """Check if intervals overlap"""
        return (self.start.timestamp < other.end.timestamp and 
                self.end.timestamp > other.start.timestamp)


@dataclass
class TemporalEvent:
    """An event in time"""
    event_id: str
    description: str
    time: TimeInterval
    properties: Dict[str, Any] = field(default_factory=dict)
    participants: List[str] = field(default_factory=list)


@dataclass
class TemporalConstraint:
    """A constraint on temporal relations"""
    event1: str  # Event ID
    event2: str  # Event ID
    relation: TemporalRelation
    confidence: float = 1.0


@dataclass
class CausalLink:
    """A causal relationship between events"""
    cause_event: str
    effect_event: str
    relation_type: CausalRelation
    strength: float
    delay: Optional[float] = None  # Time delay between cause and effect
    conditions: List[str] = field(default_factory=list)


@dataclass
class TemporalFormula:
    """A temporal logic formula"""
    operator: TemporalOperator
    operands: List[Any]  # Can be propositions or other formulas
    time_bounds: Optional[Tuple[float, float]] = None


@dataclass
class TemporalModel:
    """A model of temporal relationships"""
    events: Dict[str, TemporalEvent]
    constraints: List[TemporalConstraint]
    causal_links: List[CausalLink]
    time_granularity: float = 1.0


@dataclass
class TemporalQuery:
    """A query about temporal relationships"""
    query_type: str  # "when", "duration", "order", "causal", etc.
    target_events: List[str]
    conditions: Dict[str, Any] = field(default_factory=dict)


@dataclass
class TemporalInference:
    """An inference about temporal relationships"""
    inferred_relation: TemporalConstraint
    supporting_evidence: List[TemporalConstraint]
    inference_method: str
    confidence: float


@dataclass
class Timeline:
    """A timeline of events"""
    events: List[TemporalEvent]
    start_time: TimePoint
    end_time: TimePoint
    branches: List['Timeline'] = field(default_factory=list)  # For alternate timelines


class TemporalReasoner:
    """
    Performs temporal reasoning about events, causality, and time relationships.
    """
    
    def __init__(self, consciousness_integrator: ConsciousnessIntegrator):
        self.consciousness_integrator = consciousness_integrator
        self.temporal_models = {}
        self.inference_rules = self._initialize_inference_rules()
        self.causal_patterns = self._initialize_causal_patterns()
        
    def _initialize_inference_rules(self) -> Dict[str, Any]:
        """Initialize temporal inference rules"""
        return {
            "transitivity": {
                "pattern": ["A before B", "B before C"],
                "inference": "A before C",
                "confidence": 1.0
            },
            "symmetry": {
                "pattern": ["A equals B"],
                "inference": "B equals A",
                "confidence": 1.0
            },
            "inverse": {
                "pattern": ["A before B"],
                "inference": "B after A",
                "confidence": 1.0
            },
            "composition": {
                "pattern": ["A meets B", "B meets C"],
                "inference": "A before C",
                "confidence": 1.0
            }
        }
    
    def _initialize_causal_patterns(self) -> List[Dict[str, Any]]:
        """Initialize common causal patterns"""
        return [
            {
                "name": "direct_causation",
                "pattern": "A causes B",
                "temporal_constraint": "A before B",
                "strength_threshold": 0.7
            },
            {
                "name": "common_cause",
                "pattern": "C causes A and C causes B",
                "temporal_constraint": "C before A and C before B",
                "correlation_expected": True
            },
            {
                "name": "causal_chain",
                "pattern": "A causes B and B causes C",
                "temporal_constraint": "A before B before C",
                "transitivity": True
            }
        ]
    
    def create_temporal_model(self, events: List[TemporalEvent],
                            constraints: List[TemporalConstraint] = None) -> TemporalModel:
        """Create a temporal model from events and constraints"""
        if constraints is None:
            constraints = self._infer_constraints_from_events(events)
        
        # Build event dictionary
        event_dict = {event.event_id: event for event in events}
        
        # Infer additional constraints
        inferred_constraints = self._apply_inference_rules(constraints)
        all_constraints = constraints + inferred_constraints
        
        # Detect causal relationships
        causal_links = self._detect_causal_links(events, all_constraints)
        
        model = TemporalModel(
            events=event_dict,
            constraints=all_constraints,
            causal_links=causal_links
        )
        
        # Store model
        model_id = f"model_{len(self.temporal_models)}"
        self.temporal_models[model_id] = model
        
        return model
    
    def reason_about_temporal_relations(self, event1: TemporalEvent, 
                                      event2: TemporalEvent) -> TemporalRelation:
        """Determine temporal relation between two events"""
        # Get time intervals
        i1 = event1.time
        i2 = event2.time
        
        # Check Allen's relations
        if i1.end.timestamp < i2.start.timestamp:
            return TemporalRelation.BEFORE
        elif i2.end.timestamp < i1.start.timestamp:
            return TemporalRelation.AFTER
        elif (i1.end.timestamp == i2.start.timestamp and 
              i1.start.timestamp < i1.end.timestamp):
            return TemporalRelation.MEETS
        elif (i2.end.timestamp == i1.start.timestamp and 
              i2.start.timestamp < i2.end.timestamp):
            return TemporalRelation.MET_BY
        elif (i1.start.timestamp < i2.start.timestamp < i1.end.timestamp < i2.end.timestamp):
            return TemporalRelation.OVERLAPS
        elif (i2.start.timestamp < i1.start.timestamp < i2.end.timestamp < i1.end.timestamp):
            return TemporalRelation.OVERLAPPED_BY
        elif (i2.start.timestamp < i1.start.timestamp and 
              i1.end.timestamp < i2.end.timestamp):
            return TemporalRelation.DURING
        elif (i1.start.timestamp < i2.start.timestamp and 
              i2.end.timestamp < i1.end.timestamp):
            return TemporalRelation.CONTAINS
        elif (i1.start.timestamp == i2.start.timestamp and 
              i1.end.timestamp < i2.end.timestamp):
            return TemporalRelation.STARTS
        elif (i1.start.timestamp == i2.start.timestamp and 
              i2.end.timestamp < i1.end.timestamp):
            return TemporalRelation.STARTED_BY
        elif (i1.end.timestamp == i2.end.timestamp and 
              i2.start.timestamp < i1.start.timestamp):
            return TemporalRelation.FINISHES
        elif (i1.end.timestamp == i2.end.timestamp and 
              i1.start.timestamp < i2.start.timestamp):
            return TemporalRelation.FINISHED_BY
        else:  # Equal
            return TemporalRelation.EQUALS
    
    def analyze_causal_structure(self, model: TemporalModel) -> Dict[str, Any]:
        """Analyze causal structure in temporal model"""
        # Build causal graph
        causal_graph = self._build_causal_graph(model.causal_links)
        
        # Find causal chains
        causal_chains = self._find_causal_chains(causal_graph)
        
        # Identify common causes
        common_causes = self._find_common_causes(causal_graph)
        
        # Detect cycles
        causal_cycles = self._detect_causal_cycles(causal_graph)
        
        # Calculate causal centrality
        centrality = self._calculate_causal_centrality(causal_graph)
        
        return {
            "causal_graph": causal_graph,
            "causal_chains": causal_chains,
            "common_causes": common_causes,
            "causal_cycles": causal_cycles,
            "centrality_scores": centrality,
            "is_acyclic": len(causal_cycles) == 0
        }
    
    def evaluate_temporal_formula(self, formula: TemporalFormula, 
                                model: TemporalModel,
                                time_point: TimePoint) -> bool:
        """Evaluate temporal logic formula at given time"""
        if formula.operator == TemporalOperator.ALWAYS:
            # Check if property holds at all future times
            return self._evaluate_always(formula.operands[0], model, time_point)
        
        elif formula.operator == TemporalOperator.EVENTUALLY:
            # Check if property holds at some future time
            return self._evaluate_eventually(formula.operands[0], model, time_point)
        
        elif formula.operator == TemporalOperator.NEXT:
            # Check property at next time point
            return self._evaluate_next(formula.operands[0], model, time_point)
        
        elif formula.operator == TemporalOperator.UNTIL:
            # Check if first holds until second becomes true
            return self._evaluate_until(formula.operands[0], formula.operands[1], 
                                      model, time_point)
        
        return False
    
    def predict_future_events(self, model: TemporalModel, 
                            current_time: TimePoint) -> List[TemporalEvent]:
        """Predict future events based on causal patterns"""
        predictions = []
        
        # Get recent events
        recent_events = self._get_recent_events(model, current_time)
        
        # Apply causal rules
        for event in recent_events:
            for link in model.causal_links:
                if link.cause_event == event.event_id:
                    # Predict effect event
                    predicted_time = current_time.timestamp + (link.delay or 0)
                    predicted_event = self._create_predicted_event(
                        link.effect_event, 
                        predicted_time,
                        link.strength
                    )
                    predictions.append(predicted_event)
        
        return predictions
    
    def construct_timeline(self, events: List[TemporalEvent]) -> Timeline:
        """Construct timeline from events"""
        # Sort events by start time
        sorted_events = sorted(events, key=lambda e: e.time.start.timestamp)
        
        if not sorted_events:
            return Timeline([], TimePoint(0), TimePoint(0))
        
        start_time = sorted_events[0].time.start
        end_time = max(e.time.end for e in sorted_events)
        
        return Timeline(
            events=sorted_events,
            start_time=start_time,
            end_time=end_time
        )
    
    def merge_timelines(self, timeline1: Timeline, timeline2: Timeline) -> Timeline:
        """Merge two timelines handling conflicts"""
        # Combine events
        all_events = timeline1.events + timeline2.events
        
        # Resolve conflicts
        resolved_events = self._resolve_temporal_conflicts(all_events)
        
        # Create merged timeline
        return self.construct_timeline(resolved_events)
    
    def reason_about_counterfactuals(self, model: TemporalModel,
                                   counterfactual_event: TemporalEvent) -> Timeline:
        """Reason about counterfactual scenarios"""
        # Create alternate model with counterfactual
        alt_model = self._create_counterfactual_model(model, counterfactual_event)
        
        # Propagate causal effects
        affected_events = self._propagate_causal_effects(alt_model, counterfactual_event)
        
        # Construct alternate timeline
        alt_timeline = self.construct_timeline(list(alt_model.events.values()))
        
        return alt_timeline
    
    # Private helper methods
    
    def _infer_constraints_from_events(self, events: List[TemporalEvent]) -> List[TemporalConstraint]:
        """Infer temporal constraints from event times"""
        constraints = []
        
        for i, event1 in enumerate(events):
            for event2 in events[i+1:]:
                relation = self.reason_about_temporal_relations(event1, event2)
                constraint = TemporalConstraint(
                    event1=event1.event_id,
                    event2=event2.event_id,
                    relation=relation,
                    confidence=1.0
                )
                constraints.append(constraint)
        
        return constraints
    
    def _apply_inference_rules(self, constraints: List[TemporalConstraint]) -> List[TemporalConstraint]:
        """Apply inference rules to derive new constraints"""
        inferred = []
        
        # Apply transitivity
        for c1 in constraints:
            for c2 in constraints:
                if (c1.event2 == c2.event1 and 
                    c1.relation == TemporalRelation.BEFORE and
                    c2.relation == TemporalRelation.BEFORE):
                    # A before B and B before C => A before C
                    new_constraint = TemporalConstraint(
                        event1=c1.event1,
                        event2=c2.event2,
                        relation=TemporalRelation.BEFORE,
                        confidence=min(c1.confidence, c2.confidence)
                    )
                    if not self._constraint_exists(new_constraint, constraints + inferred):
                        inferred.append(new_constraint)
        
        return inferred
    
    def _constraint_exists(self, constraint: TemporalConstraint, 
                         constraints: List[TemporalConstraint]) -> bool:
        """Check if constraint already exists"""
        for c in constraints:
            if (c.event1 == constraint.event1 and 
                c.event2 == constraint.event2 and
                c.relation == constraint.relation):
                return True
        return False
    
    def _detect_causal_links(self, events: List[TemporalEvent],
                           constraints: List[TemporalConstraint]) -> List[CausalLink]:
        """Detect causal relationships between events"""
        causal_links = []
        
        # Simple heuristic: events that consistently precede others may cause them
        for constraint in constraints:
            if constraint.relation == TemporalRelation.BEFORE:
                # Check for causal indicators
                event1 = next(e for e in events if e.event_id == constraint.event1)
                event2 = next(e for e in events if e.event_id == constraint.event2)
                
                if self._has_causal_indicator(event1, event2):
                    link = CausalLink(
                        cause_event=constraint.event1,
                        effect_event=constraint.event2,
                        relation_type=CausalRelation.CAUSES,
                        strength=0.8,
                        delay=event2.time.start.timestamp - event1.time.end.timestamp
                    )
                    causal_links.append(link)
        
        return causal_links
    
    def _has_causal_indicator(self, event1: TemporalEvent, 
                            event2: TemporalEvent) -> bool:
        """Check if events have causal indicators"""
        # Simplified check - would use more sophisticated analysis
        causal_keywords = ["causes", "leads to", "results in", "triggers"]
        
        desc1_lower = event1.description.lower()
        desc2_lower = event2.description.lower()
        
        return any(keyword in desc1_lower for keyword in causal_keywords)
    
    def _build_causal_graph(self, causal_links: List[CausalLink]) -> Dict[str, List[str]]:
        """Build causal graph from links"""
        graph = {}
        
        for link in causal_links:
            if link.cause_event not in graph:
                graph[link.cause_event] = []
            graph[link.cause_event].append(link.effect_event)
        
        return graph
    
    def _find_causal_chains(self, causal_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Find causal chains in graph"""
        chains = []
        
        # Find all paths using DFS
        def dfs(node: str, path: List[str], visited: Set[str]):
            if node not in causal_graph or not causal_graph[node]:
                if len(path) > 2:  # Chain of at least 3 events
                    chains.append(path.copy())
                return
            
            for neighbor in causal_graph[node]:
                if neighbor not in visited:
                    visited.add(neighbor)
                    path.append(neighbor)
                    dfs(neighbor, path, visited)
                    path.pop()
                    visited.remove(neighbor)
        
        # Start DFS from each node
        for start_node in causal_graph:
            visited = {start_node}
            dfs(start_node, [start_node], visited)
        
        return chains
    
    def _find_common_causes(self, causal_graph: Dict[str, List[str]]) -> Dict[str, List[str]]:
        """Find common causes (events causing multiple effects)"""
        common_causes = {}
        
        for cause, effects in causal_graph.items():
            if len(effects) > 1:
                common_causes[cause] = effects
        
        return common_causes
    
    def _detect_causal_cycles(self, causal_graph: Dict[str, List[str]]) -> List[List[str]]:
        """Detect cycles in causal graph"""
        cycles = []
        
        def dfs(node: str, path: List[str], visited: Set[str], rec_stack: Set[str]):
            visited.add(node)
            rec_stack.add(node)
            path.append(node)
            
            if node in causal_graph:
                for neighbor in causal_graph[node]:
                    if neighbor not in visited:
                        if dfs(neighbor, path, visited, rec_stack):
                            return True
                    elif neighbor in rec_stack:
                        # Found cycle
                        cycle_start = path.index(neighbor)
                        cycles.append(path[cycle_start:] + [neighbor])
            
            path.pop()
            rec_stack.remove(node)
            return False
        
        visited = set()
        for node in causal_graph:
            if node not in visited:
                dfs(node, [], visited, set())
        
        return cycles
    
    def _calculate_causal_centrality(self, causal_graph: Dict[str, List[str]]) -> Dict[str, float]:
        """Calculate causal centrality scores"""
        centrality = {}
        
        # Out-degree centrality (number of effects)
        for node in causal_graph:
            centrality[node] = len(causal_graph.get(node, []))
        
        # In-degree centrality (number of causes)
        for effects in causal_graph.values():
            for effect in effects:
                centrality[effect] = centrality.get(effect, 0) + 0.5
        
        # Normalize
        max_centrality = max(centrality.values()) if centrality else 1
        for node in centrality:
            centrality[node] /= max_centrality
        
        return centrality
    
    def _evaluate_always(self, property_func: Any, model: TemporalModel, 
                       start_time: TimePoint) -> bool:
        """Evaluate 'always' temporal operator"""
        # Check property at all time points after start_time
        for event in model.events.values():
            if event.time.start.timestamp >= start_time.timestamp:
                if not property_func(event):
                    return False
        return True
    
    def _evaluate_eventually(self, property_func: Any, model: TemporalModel,
                           start_time: TimePoint) -> bool:
        """Evaluate 'eventually' temporal operator"""
        # Check if property holds at some future time
        for event in model.events.values():
            if event.time.start.timestamp >= start_time.timestamp:
                if property_func(event):
                    return True
        return False
    
    def _evaluate_next(self, property_func: Any, model: TemporalModel,
                     current_time: TimePoint) -> bool:
        """Evaluate 'next' temporal operator"""
        # Find next event after current time
        next_events = sorted(
            [e for e in model.events.values() if e.time.start.timestamp > current_time.timestamp],
            key=lambda e: e.time.start.timestamp
        )
        
        if next_events:
            return property_func(next_events[0])
        return False
    
    def _evaluate_until(self, prop1_func: Any, prop2_func: Any,
                      model: TemporalModel, start_time: TimePoint) -> bool:
        """Evaluate 'until' temporal operator"""
        # prop1 must hold until prop2 becomes true
        for event in sorted(model.events.values(), key=lambda e: e.time.start.timestamp):
            if event.time.start.timestamp >= start_time.timestamp:
                if prop2_func(event):
                    return True
                elif not prop1_func(event):
                    return False
        return False
    
    def _get_recent_events(self, model: TemporalModel, 
                         current_time: TimePoint,
                         window: float = 10.0) -> List[TemporalEvent]:
        """Get events within time window"""
        recent = []
        threshold = current_time.timestamp - window
        
        for event in model.events.values():
            if event.time.end.timestamp >= threshold:
                recent.append(event)
        
        return recent
    
    def _create_predicted_event(self, event_template: str, 
                              predicted_time: float,
                              confidence: float) -> TemporalEvent:
        """Create predicted future event"""
        return TemporalEvent(
            event_id=f"predicted_{event_template}_{predicted_time}",
            description=f"Predicted: {event_template}",
            time=TimeInterval(
                start=TimePoint(predicted_time),
                end=TimePoint(predicted_time + 1.0)  # Default duration
            ),
            properties={"confidence": confidence, "predicted": True}
        )
    
    def _resolve_temporal_conflicts(self, events: List[TemporalEvent]) -> List[TemporalEvent]:
        """Resolve conflicts in event list"""
        # Simple resolution: keep event with higher confidence
        resolved = {}
        
        for event in events:
            key = (event.time.start.timestamp, event.description)
            if key not in resolved:
                resolved[key] = event
            else:
                # Keep event with higher confidence
                existing_conf = resolved[key].properties.get("confidence", 1.0)
                new_conf = event.properties.get("confidence", 1.0)
                if new_conf > existing_conf:
                    resolved[key] = event
        
        return list(resolved.values())
    
    def _create_counterfactual_model(self, model: TemporalModel,
                                   counterfactual: TemporalEvent) -> TemporalModel:
        """Create model with counterfactual event"""
        # Copy existing model
        new_events = model.events.copy()
        new_events[counterfactual.event_id] = counterfactual
        
        # Recompute constraints
        new_constraints = self._infer_constraints_from_events(list(new_events.values()))
        
        # Recompute causal links
        new_causal_links = self._detect_causal_links(
            list(new_events.values()), 
            new_constraints
        )
        
        return TemporalModel(
            events=new_events,
            constraints=new_constraints,
            causal_links=new_causal_links
        )
    
    def _propagate_causal_effects(self, model: TemporalModel,
                                trigger_event: TemporalEvent) -> List[TemporalEvent]:
        """Propagate causal effects from event"""
        affected = []
        to_process = [trigger_event.event_id]
        processed = set()
        
        while to_process:
            current = to_process.pop(0)
            if current in processed:
                continue
            processed.add(current)
            
            # Find effects of current event
            for link in model.causal_links:
                if link.cause_event == current:
                    effect_event = model.events.get(link.effect_event)
                    if effect_event:
                        affected.append(effect_event)
                        to_process.append(link.effect_event)
        
        return affected
