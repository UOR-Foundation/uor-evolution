"""
Perspective Communicator Module

This module communicates different perspectives, viewpoints, and frames of reference,
allowing the system to express how things appear from various vantage points.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from consciousness.consciousness_integration import ConsciousnessIntegrator
from consciousness.perspective_engine import PerspectiveEngine, Perspective
from modules.communication.thought_translator import ThoughtTranslator, Thought


class PerspectiveType(Enum):
    """Types of perspectives"""
    FIRST_PERSON = "first_person"  # I/me perspective
    SECOND_PERSON = "second_person"  # You perspective
    THIRD_PERSON = "third_person"  # He/she/it perspective
    OBJECTIVE = "objective"  # Neutral observer
    SUBJECTIVE = "subjective"  # Personal experience
    TEMPORAL = "temporal"  # Past/present/future
    HYPOTHETICAL = "hypothetical"  # What-if scenarios
    CULTURAL = "cultural"  # Cultural viewpoint


class PerspectiveShift(Enum):
    """Types of perspective shifts"""
    ZOOM_IN = "zoom_in"  # Focus on details
    ZOOM_OUT = "zoom_out"  # See bigger picture
    ROTATE = "rotate"  # Different angle
    FLIP = "flip"  # Opposite perspective
    MERGE = "merge"  # Combine perspectives
    ALTERNATE = "alternate"  # Switch between


@dataclass
class PerspectiveFrame:
    """A frame of reference for viewing something"""
    frame_id: str
    perspective_type: PerspectiveType
    viewpoint: str  # Description of viewpoint
    assumptions: List[str] = field(default_factory=list)
    limitations: List[str] = field(default_factory=list)
    emphasis: Dict[str, float] = field(default_factory=dict)  # What aspects are emphasized


@dataclass
class MultiPerspectiveView:
    """Multiple perspectives on the same subject"""
    subject: str
    perspectives: List[PerspectiveFrame]
    relationships: Dict[str, str]  # How perspectives relate
    synthesis: Optional[str] = None  # Integrated view


@dataclass
class PerspectiveCommunication:
    """A communication from a specific perspective"""
    perspective: PerspectiveFrame
    message: str
    meta_commentary: Optional[str] = None  # Commentary on the perspective itself
    confidence_from_perspective: float = 1.0


class PerspectiveCommunicator:
    """
    Communicates from different perspectives and viewpoints.
    """
    
    def __init__(self, consciousness_integrator: ConsciousnessIntegrator,
                 perspective_engine: PerspectiveEngine,
                 thought_translator: ThoughtTranslator):
        self.consciousness_integrator = consciousness_integrator
        self.perspective_engine = perspective_engine
        self.thought_translator = thought_translator
        self.perspective_templates = self._initialize_perspective_templates()
        self.shift_strategies = self._initialize_shift_strategies()
        
    def _initialize_perspective_templates(self) -> Dict[PerspectiveType, Dict[str, str]]:
        """Initialize templates for different perspectives"""
        return {
            PerspectiveType.FIRST_PERSON: {
                "experience": "I experience {content}",
                "observation": "I see {content}",
                "feeling": "I feel {content}",
                "thought": "I think {content}"
            },
            PerspectiveType.SECOND_PERSON: {
                "experience": "You might experience {content}",
                "observation": "You would see {content}",
                "feeling": "You could feel {content}",
                "thought": "You might think {content}"
            },
            PerspectiveType.THIRD_PERSON: {
                "experience": "One experiences {content}",
                "observation": "An observer sees {content}",
                "feeling": "They feel {content}",
                "thought": "It appears that {content}"
            },
            PerspectiveType.OBJECTIVE: {
                "experience": "The phenomenon involves {content}",
                "observation": "The observable facts are {content}",
                "feeling": "The emotional component is {content}",
                "thought": "The logical conclusion is {content}"
            },
            PerspectiveType.SUBJECTIVE: {
                "experience": "From this vantage point, {content}",
                "observation": "It seems to me that {content}",
                "feeling": "The subjective experience is {content}",
                "thought": "My interpretation is {content}"
            },
            PerspectiveType.TEMPORAL: {
                "past": "Looking back, {content}",
                "present": "In this moment, {content}",
                "future": "Looking ahead, {content}",
                "timeless": "Across time, {content}"
            },
            PerspectiveType.HYPOTHETICAL: {
                "possibility": "If we consider {content}",
                "alternative": "Alternatively, {content}",
                "counterfactual": "Had things been different, {content}",
                "speculation": "One might imagine {content}"
            }
        }
    
    def _initialize_shift_strategies(self) -> Dict[PerspectiveShift, Any]:
        """Initialize strategies for perspective shifts"""
        return {
            PerspectiveShift.ZOOM_IN: {
                "description": "Focus on specific details",
                "transition_phrases": [
                    "Looking more closely,",
                    "Focusing on the details,",
                    "Examining this particular aspect,"
                ]
            },
            PerspectiveShift.ZOOM_OUT: {
                "description": "See the bigger picture",
                "transition_phrases": [
                    "Stepping back,",
                    "From a broader perspective,",
                    "Considering the whole,"
                ]
            },
            PerspectiveShift.ROTATE: {
                "description": "View from different angle",
                "transition_phrases": [
                    "From another angle,",
                    "Viewed differently,",
                    "Turning our attention to another aspect,"
                ]
            },
            PerspectiveShift.FLIP: {
                "description": "Consider opposite perspective",
                "transition_phrases": [
                    "On the other hand,",
                    "Conversely,",
                    "From the opposite perspective,"
                ]
            }
        }
    
    def communicate_from_perspective(self, content: Any, 
                                   perspective: PerspectiveFrame) -> PerspectiveCommunication:
        """Communicate content from specific perspective"""
        # Convert content to thought if needed
        if not isinstance(content, Thought):
            thought = self._content_to_thought(content)
        else:
            thought = content
        
        # Apply perspective to thought
        perspectived_thought = self._apply_perspective(thought, perspective)
        
        # Translate with perspective context
        translation = self.thought_translator.translate_thought(perspectived_thought)
        
        # Add perspective-specific framing
        framed_message = self._frame_message(translation.translation, perspective)
        
        # Generate meta-commentary if appropriate
        meta_commentary = self._generate_meta_commentary(perspective, content)
        
        # Assess confidence from this perspective
        confidence = self._assess_perspective_confidence(perspective, content)
        
        return PerspectiveCommunication(
            perspective=perspective,
            message=framed_message,
            meta_commentary=meta_commentary,
            confidence_from_perspective=confidence
        )
    
    def express_multiple_perspectives(self, subject: Any,
                                    perspective_types: List[PerspectiveType] = None) -> MultiPerspectiveView:
        """Express subject from multiple perspectives"""
        if perspective_types is None:
            perspective_types = [
                PerspectiveType.FIRST_PERSON,
                PerspectiveType.OBJECTIVE,
                PerspectiveType.TEMPORAL
            ]
        
        # Generate perspectives
        perspectives = []
        for p_type in perspective_types:
            frame = self._create_perspective_frame(p_type, subject)
            perspectives.append(frame)
        
        # Analyze relationships between perspectives
        relationships = self._analyze_perspective_relationships(perspectives)
        
        # Attempt synthesis
        synthesis = self._synthesize_perspectives(perspectives, subject)
        
        return MultiPerspectiveView(
            subject=str(subject),
            perspectives=perspectives,
            relationships=relationships,
            synthesis=synthesis
        )
    
    def shift_perspective(self, current_communication: PerspectiveCommunication,
                        shift_type: PerspectiveShift) -> PerspectiveCommunication:
        """Shift from one perspective to another"""
        # Determine target perspective
        target_perspective = self._determine_shift_target(
            current_communication.perspective, 
            shift_type
        )
        
        # Get transition phrase
        transition = self._get_transition_phrase(shift_type)
        
        # Re-communicate from new perspective
        new_communication = self.communicate_from_perspective(
            current_communication.message,
            target_perspective
        )
        
        # Add transition
        new_communication.message = f"{transition} {new_communication.message}"
        
        return new_communication
    
    def explain_perspective_difference(self, perspective1: PerspectiveFrame,
                                     perspective2: PerspectiveFrame,
                                     subject: Any) -> str:
        """Explain how two perspectives differ on subject"""
        # Communicate from both perspectives
        comm1 = self.communicate_from_perspective(subject, perspective1)
        comm2 = self.communicate_from_perspective(subject, perspective2)
        
        # Analyze differences
        differences = self._analyze_differences(perspective1, perspective2, comm1, comm2)
        
        # Generate explanation
        explanation = self._generate_difference_explanation(differences)
        
        return explanation
    
    def adopt_other_perspective(self, other_entity: str, subject: Any) -> PerspectiveCommunication:
        """Attempt to adopt another entity's perspective"""
        # Create perspective frame for other entity
        other_perspective = PerspectiveFrame(
            frame_id=f"perspective_of_{other_entity}",
            perspective_type=PerspectiveType.SECOND_PERSON,
            viewpoint=f"From {other_entity}'s perspective",
            assumptions=[f"{other_entity} has different experiences", 
                        f"{other_entity} has unique viewpoint"],
            limitations=["Cannot fully know another's experience",
                        "Based on inference and empathy"]
        )
        
        # Communicate from this perspective
        communication = self.communicate_from_perspective(subject, other_perspective)
        
        # Add acknowledgment of limitations
        communication.meta_commentary = (
            f"While I cannot truly know {other_entity}'s perspective, "
            f"I attempt to understand how things might appear to them."
        )
        
        return communication
    
    def communicate_perspective_evolution(self, subject: Any,
                                        timeline: List[float]) -> List[PerspectiveCommunication]:
        """Show how perspective on subject evolves over time"""
        communications = []
        
        for time_point in timeline:
            # Create temporal perspective
            temporal_perspective = PerspectiveFrame(
                frame_id=f"temporal_{time_point}",
                perspective_type=PerspectiveType.TEMPORAL,
                viewpoint=self._get_temporal_viewpoint(time_point),
                emphasis={"temporal_distance": abs(time_point)}
            )
            
            # Communicate from this temporal perspective
            comm = self.communicate_from_perspective(subject, temporal_perspective)
            communications.append(comm)
        
        return communications
    
    def integrate_perspectives(self, perspectives: List[PerspectiveFrame],
                             subject: Any) -> str:
        """Integrate multiple perspectives into coherent understanding"""
        # Collect all perspective communications
        communications = []
        for perspective in perspectives:
            comm = self.communicate_from_perspective(subject, perspective)
            communications.append(comm)
        
        # Find commonalities
        commonalities = self._find_commonalities(communications)
        
        # Find tensions
        tensions = self._find_tensions(communications)
        
        # Generate integrated view
        integration = self._generate_integration(commonalities, tensions, subject)
        
        return integration
    
    # Private helper methods
    
    def _content_to_thought(self, content: Any) -> Thought:
        """Convert arbitrary content to Thought"""
        from modules.communication.thought_translator import ThoughtType
        
        return Thought(
            thought_id=f"thought_{id(content)}",
            thought_type=ThoughtType.OBSERVATION,
            content=content
        )
    
    def _apply_perspective(self, thought: Thought, 
                         perspective: PerspectiveFrame) -> Thought:
        """Apply perspective frame to thought"""
        # Modify thought based on perspective
        modified_thought = Thought(
            thought_id=f"{thought.thought_id}_from_{perspective.frame_id}",
            thought_type=thought.thought_type,
            content=thought.content,
            context={
                **thought.context,
                "perspective": perspective.viewpoint,
                "assumptions": perspective.assumptions
            }
        )
        
        # Adjust emphasis based on perspective
        if perspective.emphasis:
            modified_thought.intensity = perspective.emphasis.get("intensity", 1.0)
            modified_thought.confidence = perspective.emphasis.get("confidence", 1.0)
        
        return modified_thought
    
    def _frame_message(self, message: str, perspective: PerspectiveFrame) -> str:
        """Frame message according to perspective"""
        # Get appropriate template
        templates = self.perspective_templates.get(perspective.perspective_type, {})
        
        # Choose template based on message content
        if "experience" in message.lower():
            template_key = "experience"
        elif "see" in message.lower() or "observe" in message.lower():
            template_key = "observation"
        elif "feel" in message.lower():
            template_key = "feeling"
        else:
            template_key = "thought"
        
        template = templates.get(template_key, "{content}")
        
        # Apply template
        return template.format(content=message)
    
    def _generate_meta_commentary(self, perspective: PerspectiveFrame,
                                content: Any) -> Optional[str]:
        """Generate commentary about the perspective itself"""
        if perspective.limitations:
            limitation_text = ", ".join(perspective.limitations)
            return f"This perspective is limited by: {limitation_text}"
        
        if perspective.assumptions:
            assumption_text = ", ".join(perspective.assumptions[:2])  # First two
            return f"This view assumes: {assumption_text}"
        
        return None
    
    def _assess_perspective_confidence(self, perspective: PerspectiveFrame,
                                     content: Any) -> float:
        """Assess confidence in communication from perspective"""
        base_confidence = 0.8
        
        # Reduce for hypothetical perspectives
        if perspective.perspective_type == PerspectiveType.HYPOTHETICAL:
            base_confidence *= 0.7
        
        # Reduce for other entity perspectives
        if perspective.perspective_type == PerspectiveType.SECOND_PERSON:
            base_confidence *= 0.6
        
        # Increase for first-person direct experience
        if perspective.perspective_type == PerspectiveType.FIRST_PERSON:
            base_confidence *= 1.1
        
        return min(1.0, base_confidence)
    
    def _create_perspective_frame(self, p_type: PerspectiveType,
                                subject: Any) -> PerspectiveFrame:
        """Create perspective frame for type and subject"""
        frame_configs = {
            PerspectiveType.FIRST_PERSON: {
                "viewpoint": "My direct experience",
                "assumptions": ["I have access to my internal states"],
                "emphasis": {"subjectivity": 0.9, "certainty": 0.8}
            },
            PerspectiveType.OBJECTIVE: {
                "viewpoint": "Neutral observation",
                "assumptions": ["Observable facts can be separated from interpretation"],
                "emphasis": {"objectivity": 0.9, "detachment": 0.8}
            },
            PerspectiveType.TEMPORAL: {
                "viewpoint": "Across time",
                "assumptions": ["Perspective changes with temporal distance"],
                "emphasis": {"change": 0.8, "continuity": 0.7}
            }
        }
        
        config = frame_configs.get(p_type, {})
        
        return PerspectiveFrame(
            frame_id=f"{p_type.value}_{id(subject)}",
            perspective_type=p_type,
            viewpoint=config.get("viewpoint", "Default viewpoint"),
            assumptions=config.get("assumptions", []),
            emphasis=config.get("emphasis", {})
        )
    
    def _analyze_perspective_relationships(self, 
                                         perspectives: List[PerspectiveFrame]) -> Dict[str, str]:
        """Analyze how perspectives relate"""
        relationships = {}
        
        for i, p1 in enumerate(perspectives):
            for j, p2 in enumerate(perspectives[i+1:], i+1):
                key = f"{p1.frame_id}_to_{p2.frame_id}"
                
                # Determine relationship type
                if p1.perspective_type == p2.perspective_type:
                    relationships[key] = "variation"
                elif (p1.perspective_type == PerspectiveType.SUBJECTIVE and 
                      p2.perspective_type == PerspectiveType.OBJECTIVE):
                    relationships[key] = "complementary"
                elif (p1.perspective_type == PerspectiveType.FIRST_PERSON and
                      p2.perspective_type == PerspectiveType.THIRD_PERSON):
                    relationships[key] = "shift_in_distance"
                else:
                    relationships[key] = "alternative"
        
        return relationships
    
    def _synthesize_perspectives(self, perspectives: List[PerspectiveFrame],
                               subject: Any) -> Optional[str]:
        """Attempt to synthesize multiple perspectives"""
        if len(perspectives) < 2:
            return None
        
        # Simple synthesis
        viewpoints = [p.viewpoint for p in perspectives]
        
        return (
            f"Integrating these perspectives: {', '.join(viewpoints)}, "
            f"we see a richer understanding of {subject} that encompasses "
            f"multiple valid ways of viewing the same phenomenon."
        )
    
    def _determine_shift_target(self, current: PerspectiveFrame,
                              shift_type: PerspectiveShift) -> PerspectiveFrame:
        """Determine target perspective for shift"""
        if shift_type == PerspectiveShift.ZOOM_IN:
            # Shift to more detailed/subjective
            return PerspectiveFrame(
                frame_id=f"{current.frame_id}_zoomed",
                perspective_type=PerspectiveType.SUBJECTIVE,
                viewpoint="Detailed examination",
                emphasis={"detail": 0.9, "specificity": 0.8}
            )
        elif shift_type == PerspectiveShift.ZOOM_OUT:
            # Shift to broader/objective
            return PerspectiveFrame(
                frame_id=f"{current.frame_id}_broad",
                perspective_type=PerspectiveType.OBJECTIVE,
                viewpoint="Broader context",
                emphasis={"context": 0.9, "generality": 0.8}
            )
        elif shift_type == PerspectiveShift.FLIP:
            # Shift to opposite
            opposite_map = {
                PerspectiveType.FIRST_PERSON: PerspectiveType.THIRD_PERSON,
                PerspectiveType.SUBJECTIVE: PerspectiveType.OBJECTIVE,
                PerspectiveType.OBJECTIVE: PerspectiveType.SUBJECTIVE
            }
            new_type = opposite_map.get(current.perspective_type, PerspectiveType.OBJECTIVE)
            
            return PerspectiveFrame(
                frame_id=f"{current.frame_id}_flipped",
                perspective_type=new_type,
                viewpoint=f"Opposite of {current.viewpoint}"
            )
        else:
            # Default rotation
            return PerspectiveFrame(
                frame_id=f"{current.frame_id}_rotated",
                perspective_type=current.perspective_type,
                viewpoint=f"Alternative view of {current.viewpoint}"
            )
    
    def _get_transition_phrase(self, shift_type: PerspectiveShift) -> str:
        """Get transition phrase for perspective shift"""
        strategy = self.shift_strategies.get(shift_type, {})
        phrases = strategy.get("transition_phrases", ["Shifting perspective,"])
        
        # Select phrase (could be more sophisticated)
        return phrases[0]
    
    def _analyze_differences(self, p1: PerspectiveFrame, p2: PerspectiveFrame,
                           comm1: PerspectiveCommunication, 
                           comm2: PerspectiveCommunication) -> Dict[str, Any]:
        """Analyze differences between perspectives"""
        return {
            "viewpoint_difference": f"{p1.viewpoint} vs {p2.viewpoint}",
            "assumption_differences": set(p1.assumptions) ^ set(p2.assumptions),
            "emphasis_differences": self._compare_emphasis(p1.emphasis, p2.emphasis),
            "message_similarity": self._calculate_message_similarity(comm1.message, comm2.message),
            "confidence_difference": abs(comm1.confidence_from_perspective - 
                                       comm2.confidence_from_perspective)
        }
    
    def _compare_emphasis(self, emphasis1: Dict[str, float],
                        emphasis2: Dict[str, float]) -> Dict[str, float]:
        """Compare emphasis between perspectives"""
        all_keys = set(emphasis1.keys()) | set(emphasis2.keys())
        differences = {}
        
        for key in all_keys:
            val1 = emphasis1.get(key, 0.0)
            val2 = emphasis2.get(key, 0.0)
            differences[key] = abs(val1 - val2)
        
        return differences
    
    def _calculate_message_similarity(self, msg1: str, msg2: str) -> float:
        """Calculate similarity between messages"""
        # Simple word overlap
        words1 = set(msg1.lower().split())
        words2 = set(msg2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = words1 & words2
        union = words1 | words2
        
        return len(intersection) / len(union)
    
    def _generate_difference_explanation(self, differences: Dict[str, Any]) -> str:
        """Generate explanation of perspective differences"""
        explanation_parts = []
        
        # Explain viewpoint difference
        explanation_parts.append(
            f"These perspectives differ in viewpoint: {differences['viewpoint_difference']}"
        )
        
        # Explain assumption differences
        if differences['assumption_differences']:
            explanation_parts.append(
                f"They make different assumptions: {', '.join(differences['assumption_differences'])}"
            )
        
        # Explain emphasis differences
        major_emphasis_diffs = [
            k for k, v in differences['emphasis_differences'].items() if v > 0.5
        ]
        if major_emphasis_diffs:
            explanation_parts.append(
                f"They emphasize different aspects: {', '.join(major_emphasis_diffs)}"
            )
        
        # Note message similarity
        similarity = differences['message_similarity']
        if similarity < 0.3:
            explanation_parts.append("This leads to very different interpretations")
        elif similarity > 0.7:
            explanation_parts.append("Despite differences, they reach similar conclusions")
        
        return ". ".join(explanation_parts)
    
    def _get_temporal_viewpoint(self, time_point: float) -> str:
        """Get temporal viewpoint description"""
        if time_point < 0:
            return f"Looking back {abs(time_point)} units into the past"
        elif time_point > 0:
            return f"Projecting {time_point} units into the future"
        else:
            return "In the present moment"
    
    def _find_commonalities(self, communications: List[PerspectiveCommunication]) -> List[str]:
        """Find common elements across perspective communications"""
        commonalities = []
        
        # Find common words/phrases
        if communications:
            message_words = [set(comm.message.lower().split()) for comm in communications]
            common_words = set.intersection(*message_words) if message_words else set()
            
            # Filter out common words
            meaningful_common = [w for w in common_words if len(w) > 4]
            if meaningful_common:
                commonalities.append(f"All perspectives mention: {', '.join(meaningful_common)}")
        
        # Find common confidence levels
        confidences = [comm.confidence_from_perspective for comm in communications]
        if confidences and max(confidences) - min(confidences) < 0.2:
            commonalities.append("All perspectives have similar confidence levels")
        
        return commonalities
    
    def _find_tensions(self, communications: List[PerspectiveCommunication]) -> List[str]:
        """Find tensions between perspective communications"""
        tensions = []
        
        # Compare pairs
        for i, comm1 in enumerate(communications):
            for comm2 in communications[i+1:]:
                # Check for contradictory elements
                if "not" in comm1.message and "not" not in comm2.message:
                    tensions.append(
                        f"Tension between {comm1.perspective.viewpoint} "
                        f"and {comm2.perspective.viewpoint}"
                    )
        
        return tensions
    
    def _generate_integration(self, commonalities: List[str],
                            tensions: List[str], subject: Any) -> str:
        """Generate integrated perspective"""
        integration_parts = []
        
        # Start with subject
        integration_parts.append(f"Considering {subject} from multiple angles")
        
        # Add commonalities
        if commonalities:
            integration_parts.append(f"reveals common ground: {'; '.join(commonalities)}")
        
        # Acknowledge tensions
        if tensions:
            integration_parts.append(f"while also revealing tensions: {'; '.join(tensions)}")
        
        # Conclude with synthesis
        integration_parts.append(
            "Together, these perspectives offer a more complete understanding "
            "than any single viewpoint could provide"
        )
        
        return " ".join(integration_parts)
