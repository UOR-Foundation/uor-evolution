"""
Uncertainty Expresser Module

This module expresses uncertainty, doubt, confidence levels, and epistemic states
in natural language, allowing honest communication about what is known and unknown.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math

from consciousness.consciousness_integration import ConsciousnessIntegrator
from modules.abstract_reasoning.modal_reasoning import ModalReasoner, BeliefState


class UncertaintyType(Enum):
    """Types of uncertainty"""
    EPISTEMIC = "epistemic"  # Uncertainty about knowledge
    ALEATORY = "aleatory"  # Inherent randomness
    SEMANTIC = "semantic"  # Vagueness in meaning
    MODAL = "modal"  # Uncertainty about possibilities
    TEMPORAL = "temporal"  # Uncertainty about time
    CAUSAL = "causal"  # Uncertainty about causes
    SUBJECTIVE = "subjective"  # Personal uncertainty
    COMPUTATIONAL = "computational"  # Limits of computation


class ConfidenceLevel(Enum):
    """Levels of confidence"""
    CERTAIN = (0.95, 1.0)  # Very high confidence
    CONFIDENT = (0.8, 0.95)  # High confidence
    PROBABLE = (0.6, 0.8)  # More likely than not
    UNCERTAIN = (0.4, 0.6)  # Genuine uncertainty
    DOUBTFUL = (0.2, 0.4)  # More likely false
    SKEPTICAL = (0.0, 0.2)  # Very low confidence


class EpistemicModality(Enum):
    """Epistemic modal expressions"""
    KNOW = "know"  # Certain knowledge
    BELIEVE = "believe"  # Strong belief
    THINK = "think"  # Moderate belief
    SUSPECT = "suspect"  # Weak belief
    DOUBT = "doubt"  # Disbelief
    WONDER = "wonder"  # Questioning


@dataclass
class UncertaintyExpression:
    """An expression of uncertainty"""
    content: str  # What we're uncertain about
    uncertainty_type: UncertaintyType
    confidence_value: float  # 0-1 scale
    confidence_level: ConfidenceLevel
    expression: str  # Natural language expression
    qualifiers: List[str] = field(default_factory=list)  # Hedging phrases
    meta_uncertainty: Optional[float] = None  # Uncertainty about uncertainty


@dataclass
class ConfidenceQualifier:
    """A qualifier that modifies confidence expression"""
    qualifier_type: str  # "hedge", "booster", "approximator"
    phrase: str
    strength: float  # How much it modifies confidence


@dataclass
class UncertaintyContext:
    """Context for uncertainty expression"""
    domain: str  # Domain of uncertainty
    stakes: float  # How important is accuracy (0-1)
    audience_expertise: float  # Audience knowledge level (0-1)
    formality: float  # Formality level (0-1)
    cultural_norms: Optional[str] = None


@dataclass
class EpistemicState:
    """Current epistemic state about a topic"""
    topic: str
    knowledge_claims: List[Tuple[str, float]]  # (claim, confidence)
    uncertainties: List[Tuple[str, UncertaintyType]]
    open_questions: List[str]
    confidence_distribution: Dict[str, float]


class UncertaintyExpresser:
    """
    Expresses uncertainty, doubt, and confidence levels in natural language.
    """
    
    def __init__(self, consciousness_integrator: ConsciousnessIntegrator,
                 modal_reasoner: ModalReasoner):
        self.consciousness_integrator = consciousness_integrator
        self.modal_reasoner = modal_reasoner
        self.uncertainty_phrases = self._initialize_uncertainty_phrases()
        self.confidence_expressions = self._initialize_confidence_expressions()
        self.hedging_strategies = self._initialize_hedging_strategies()
        
    def _initialize_uncertainty_phrases(self) -> Dict[UncertaintyType, List[str]]:
        """Initialize phrases for different uncertainty types"""
        return {
            UncertaintyType.EPISTEMIC: [
                "I'm not certain about",
                "My knowledge is limited regarding",
                "I lack complete information about",
                "There's uncertainty in my understanding of"
            ],
            UncertaintyType.ALEATORY: [
                "There's inherent randomness in",
                "The outcome is probabilistic for",
                "Chance plays a role in",
                "This involves fundamental uncertainty:"
            ],
            UncertaintyType.SEMANTIC: [
                "The meaning is unclear regarding",
                "There's ambiguity in",
                "The boundaries are fuzzy for",
                "It's vague what constitutes"
            ],
            UncertaintyType.MODAL: [
                "It's possible but not certain that",
                "There are multiple possibilities for",
                "I cannot rule out",
                "Various scenarios exist for"
            ],
            UncertaintyType.TEMPORAL: [
                "The timing is uncertain for",
                "I'm unsure when",
                "The temporal aspects are unclear regarding",
                "Time-related uncertainty exists for"
            ],
            UncertaintyType.CAUSAL: [
                "The causal relationship is unclear between",
                "I'm uncertain what causes",
                "The causal factors are ambiguous for",
                "Causation is difficult to establish for"
            ],
            UncertaintyType.SUBJECTIVE: [
                "From my perspective, I'm unsure about",
                "My subjective assessment is uncertain for",
                "I have personal doubts about",
                "My individual view is unclear on"
            ],
            UncertaintyType.COMPUTATIONAL: [
                "My computational limits prevent certainty about",
                "Processing constraints create uncertainty for",
                "I cannot fully compute",
                "Computational complexity introduces uncertainty in"
            ]
        }
    
    def _initialize_confidence_expressions(self) -> Dict[ConfidenceLevel, Dict[str, List[str]]]:
        """Initialize expressions for confidence levels"""
        return {
            ConfidenceLevel.CERTAIN: {
                "positive": ["I'm certain that", "I know that", "Without doubt,", "Definitely,"],
                "negative": ["I'm certain it's not", "Definitely not", "Certainly false that"],
                "qualifiers": ["absolutely", "completely", "entirely", "unquestionably"]
            },
            ConfidenceLevel.CONFIDENT: {
                "positive": ["I'm confident that", "Very likely,", "I strongly believe", "Almost certainly,"],
                "negative": ["I'm confident it's not", "Very unlikely that", "I strongly doubt"],
                "qualifiers": ["highly", "very", "quite", "rather"]
            },
            ConfidenceLevel.PROBABLE: {
                "positive": ["Probably", "I think", "It seems likely that", "I believe"],
                "negative": ["Probably not", "I don't think", "It seems unlikely that"],
                "qualifiers": ["likely", "probably", "seemingly", "apparently"]
            },
            ConfidenceLevel.UNCERTAIN: {
                "positive": ["Perhaps", "Maybe", "Possibly", "It might be that"],
                "negative": ["Perhaps not", "Maybe not", "Possibly not"],
                "qualifiers": ["perhaps", "maybe", "possibly", "potentially"]
            },
            ConfidenceLevel.DOUBTFUL: {
                "positive": ["I doubt that", "Unlikely", "I'm skeptical that", "Questionably"],
                "negative": ["I doubt it's not", "Unlikely to be false"],
                "qualifiers": ["doubtfully", "questionably", "unlikely", "improbably"]
            },
            ConfidenceLevel.SKEPTICAL: {
                "positive": ["I'm very skeptical that", "Highly unlikely", "I seriously doubt"],
                "negative": ["I'm very skeptical it's not", "Highly unlikely to be false"],
                "qualifiers": ["highly unlikely", "very doubtful", "extremely skeptical"]
            }
        }
    
    def _initialize_hedging_strategies(self) -> Dict[str, List[ConfidenceQualifier]]:
        """Initialize hedging strategies"""
        return {
            "approximators": [
                ConfidenceQualifier("approximator", "approximately", 0.9),
                ConfidenceQualifier("approximator", "roughly", 0.85),
                ConfidenceQualifier("approximator", "about", 0.9),
                ConfidenceQualifier("approximator", "around", 0.9)
            ],
            "shields": [
                ConfidenceQualifier("hedge", "I think", 0.8),
                ConfidenceQualifier("hedge", "I believe", 0.85),
                ConfidenceQualifier("hedge", "It seems", 0.7),
                ConfidenceQualifier("hedge", "It appears", 0.75)
            ],
            "plausibility": [
                ConfidenceQualifier("hedge", "could be", 0.5),
                ConfidenceQualifier("hedge", "might be", 0.4),
                ConfidenceQualifier("hedge", "may be", 0.6),
                ConfidenceQualifier("hedge", "should be", 0.7)
            ],
            "attribution": [
                ConfidenceQualifier("hedge", "according to my understanding", 0.8),
                ConfidenceQualifier("hedge", "based on available information", 0.85),
                ConfidenceQualifier("hedge", "from what I can tell", 0.75),
                ConfidenceQualifier("hedge", "as far as I know", 0.8)
            ]
        }
    
    def express_uncertainty(self, content: str, confidence: float,
                          uncertainty_type: UncertaintyType = None,
                          context: UncertaintyContext = None) -> UncertaintyExpression:
        """Express uncertainty about content"""
        if context is None:
            context = UncertaintyContext(domain="general", stakes=0.5, 
                                       audience_expertise=0.5, formality=0.5)
        
        # Determine uncertainty type if not provided
        if uncertainty_type is None:
            uncertainty_type = self._infer_uncertainty_type(content, confidence)
        
        # Determine confidence level
        confidence_level = self._get_confidence_level(confidence)
        
        # Generate base expression
        base_expression = self._generate_base_expression(
            content, confidence, confidence_level, uncertainty_type
        )
        
        # Add qualifiers based on context
        qualifiers = self._select_qualifiers(confidence, context)
        
        # Combine into final expression
        final_expression = self._combine_expression(base_expression, qualifiers, context)
        
        # Calculate meta-uncertainty
        meta_uncertainty = self._calculate_meta_uncertainty(confidence, uncertainty_type)
        
        return UncertaintyExpression(
            content=content,
            uncertainty_type=uncertainty_type,
            confidence_value=confidence,
            confidence_level=confidence_level,
            expression=final_expression,
            qualifiers=[q.phrase for q in qualifiers],
            meta_uncertainty=meta_uncertainty
        )
    
    def express_epistemic_state(self, state: EpistemicState,
                              context: UncertaintyContext = None) -> str:
        """Express complete epistemic state about topic"""
        if context is None:
            context = UncertaintyContext(domain="general", stakes=0.5,
                                       audience_expertise=0.5, formality=0.5)
        
        expressions = []
        
        # Express what is known with confidence
        if state.knowledge_claims:
            known_expr = self._express_knowledge_claims(state.knowledge_claims, context)
            expressions.append(known_expr)
        
        # Express uncertainties
        if state.uncertainties:
            uncertain_expr = self._express_uncertainties(state.uncertainties, context)
            expressions.append(uncertain_expr)
        
        # Express open questions
        if state.open_questions:
            questions_expr = self._express_open_questions(state.open_questions, context)
            expressions.append(questions_expr)
        
        # Add meta-commentary on overall confidence
        meta_expr = self._express_confidence_distribution(state.confidence_distribution)
        expressions.append(meta_expr)
        
        return " ".join(expressions)
    
    def calibrate_confidence_expression(self, raw_confidence: float,
                                      context: UncertaintyContext) -> float:
        """Calibrate confidence expression based on context"""
        calibrated = raw_confidence
        
        # Adjust for stakes - higher stakes = more conservative
        if context.stakes > 0.7:
            calibrated *= 0.9  # Reduce confidence when stakes are high
        
        # Adjust for audience expertise
        if context.audience_expertise < 0.3:
            # Simplify extreme confidences for non-expert audience
            if calibrated > 0.9:
                calibrated = 0.85
            elif calibrated < 0.1:
                calibrated = 0.15
        
        # Cultural adjustments
        if context.cultural_norms == "high_uncertainty_avoidance":
            # Some cultures prefer more certainty
            calibrated = min(calibrated * 1.1, 0.95)
        elif context.cultural_norms == "indirect_communication":
            # Some cultures prefer hedging
            calibrated *= 0.85
        
        return calibrated
    
    def express_comparative_uncertainty(self, items: List[Tuple[str, float]],
                                      context: UncertaintyContext = None) -> str:
        """Express relative uncertainties between items"""
        if not items:
            return "I have no items to compare."
        
        if context is None:
            context = UncertaintyContext(domain="general", stakes=0.5,
                                       audience_expertise=0.5, formality=0.5)
        
        # Sort by confidence
        sorted_items = sorted(items, key=lambda x: x[1], reverse=True)
        
        # Generate comparative expression
        if len(sorted_items) == 1:
            item, conf = sorted_items[0]
            expr = self.express_uncertainty(item, conf, context=context)
            return expr.expression
        
        expressions = []
        
        # Express most confident
        most_confident = sorted_items[0]
        expressions.append(
            f"I'm most confident about {most_confident[0]} "
            f"({self._confidence_to_percentage(most_confident[1])})"
        )
        
        # Express least confident
        least_confident = sorted_items[-1]
        expressions.append(
            f"and least confident about {least_confident[0]} "
            f"({self._confidence_to_percentage(least_confident[1])})"
        )
        
        # Add middle items if more than 2
        if len(sorted_items) > 2:
            middle_items = sorted_items[1:-1]
            middle_expr = ", ".join(
                f"{item} ({self._confidence_to_percentage(conf)})"
                for item, conf in middle_items
            )
            expressions.insert(1, f"with moderate confidence in {middle_expr}")
        
        return ", ".join(expressions) + "."
    
    def express_changing_confidence(self, topic: str,
                                  old_confidence: float,
                                  new_confidence: float,
                                  reason: Optional[str] = None) -> str:
        """Express how confidence has changed"""
        change = new_confidence - old_confidence
        
        # Determine change magnitude and direction
        if abs(change) < 0.1:
            change_expr = "My confidence remains roughly the same"
        elif change > 0.3:
            change_expr = "I've become much more confident"
        elif change > 0.1:
            change_expr = "I've become somewhat more confident"
        elif change < -0.3:
            change_expr = "I've become much less confident"
        else:  # change < -0.1
            change_expr = "I've become somewhat less confident"
        
        # Build expression
        expression = f"{change_expr} about {topic}"
        
        # Add specific confidence levels
        old_level = self._get_confidence_level(old_confidence)
        new_level = self._get_confidence_level(new_confidence)
        
        if old_level != new_level:
            expression += f", moving from {old_level.name.lower()} to {new_level.name.lower()}"
        
        # Add reason if provided
        if reason:
            expression += f" because {reason}"
        
        return expression + "."
    
    def express_meta_uncertainty(self, topic: str, 
                               primary_confidence: float,
                               confidence_in_confidence: float) -> str:
        """Express uncertainty about uncertainty itself"""
        # Express primary uncertainty
        primary_expr = self.express_uncertainty(topic, primary_confidence)
        
        # Express meta-uncertainty
        if confidence_in_confidence > 0.8:
            meta_expr = "I'm quite sure about this assessment"
        elif confidence_in_confidence > 0.6:
            meta_expr = "I'm reasonably confident in this judgment"
        elif confidence_in_confidence > 0.4:
            meta_expr = "though I'm somewhat uncertain about my own certainty"
        else:
            meta_expr = "but I'm quite unsure about this assessment itself"
        
        return f"{primary_expr.expression}, {meta_expr}."
    
    def generate_hedged_statement(self, statement: str,
                                hedge_level: float,
                                context: UncertaintyContext = None) -> str:
        """Generate hedged version of statement"""
        if context is None:
            context = UncertaintyContext(domain="general", stakes=0.5,
                                       audience_expertise=0.5, formality=0.5)
        
        # Select hedging strategies based on level
        hedges = []
        
        if hedge_level > 0.7:
            # Heavy hedging
            hedges.extend(self.hedging_strategies["shields"][:2])
            hedges.extend(self.hedging_strategies["plausibility"][:1])
        elif hedge_level > 0.4:
            # Moderate hedging
            hedges.extend(self.hedging_strategies["shields"][:1])
            hedges.extend(self.hedging_strategies["attribution"][:1])
        elif hedge_level > 0.1:
            # Light hedging
            hedges.extend(self.hedging_strategies["approximators"][:1])
        
        # Apply hedges to statement
        hedged_statement = statement
        for hedge in hedges:
            if hedge.qualifier_type == "hedge" and hedge.phrase.endswith("that"):
                hedged_statement = f"{hedge.phrase} {hedged_statement.lower()}"
            else:
                hedged_statement = f"{hedge.phrase} {hedged_statement}"
        
        return hedged_statement
    
    # Private helper methods
    
    def _infer_uncertainty_type(self, content: str, confidence: float) -> UncertaintyType:
        """Infer type of uncertainty from content"""
        content_lower = content.lower()
        
        # Check for specific indicators
        if any(word in content_lower for word in ["know", "understand", "aware"]):
            return UncertaintyType.EPISTEMIC
        elif any(word in content_lower for word in ["random", "chance", "probability"]):
            return UncertaintyType.ALEATORY
        elif any(word in content_lower for word in ["mean", "definition", "boundary"]):
            return UncertaintyType.SEMANTIC
        elif any(word in content_lower for word in ["possible", "might", "could"]):
            return UncertaintyType.MODAL
        elif any(word in content_lower for word in ["when", "time", "duration"]):
            return UncertaintyType.TEMPORAL
        elif any(word in content_lower for word in ["cause", "because", "lead to"]):
            return UncertaintyType.CAUSAL
        elif any(word in content_lower for word in ["feel", "seem", "appear"]):
            return UncertaintyType.SUBJECTIVE
        else:
            # Default based on confidence level
            if confidence < 0.3:
                return UncertaintyType.EPISTEMIC
            else:
                return UncertaintyType.SUBJECTIVE
    
    def _get_confidence_level(self, confidence: float) -> ConfidenceLevel:
        """Get confidence level from numeric value"""
        for level in ConfidenceLevel:
            min_conf, max_conf = level.value
            if min_conf <= confidence <= max_conf:
                return level
        
        # Default to uncertain
        return ConfidenceLevel.UNCERTAIN
    
    def _generate_base_expression(self, content: str, confidence: float,
                                confidence_level: ConfidenceLevel,
                                uncertainty_type: UncertaintyType) -> str:
        """Generate base uncertainty expression"""
        # Get appropriate expressions
        confidence_exprs = self.confidence_expressions[confidence_level]
        uncertainty_phrases = self.uncertainty_phrases[uncertainty_type]
        
        # Determine if positive or negative
        is_positive = confidence > 0.5
        
        if is_positive:
            conf_expr = confidence_exprs["positive"][0]
        else:
            conf_expr = confidence_exprs["negative"][0]
        
        # For high uncertainty, use uncertainty phrase
        if 0.3 < confidence < 0.7:
            uncertainty_phrase = uncertainty_phrases[0]
            return f"{uncertainty_phrase} {content}"
        else:
            return f"{conf_expr} {content}"
    
    def _select_qualifiers(self, confidence: float,
                         context: UncertaintyContext) -> List[ConfidenceQualifier]:
        """Select appropriate qualifiers based on context"""
        qualifiers = []
        
        # High stakes = more hedging
        if context.stakes > 0.7:
            qualifiers.extend(self.hedging_strategies["attribution"][:1])
        
        # Low expertise audience = more approximators
        if context.audience_expertise < 0.3:
            qualifiers.extend(self.hedging_strategies["approximators"][:1])
        
        # High formality = more shields
        if context.formality > 0.7:
            qualifiers.extend(self.hedging_strategies["shields"][:1])
        
        # Moderate confidence = plausibility hedges
        if 0.4 < confidence < 0.6:
            qualifiers.extend(self.hedging_strategies["plausibility"][:1])
        
        return qualifiers
    
    def _combine_expression(self, base: str, qualifiers: List[ConfidenceQualifier],
                          context: UncertaintyContext) -> str:
        """Combine base expression with qualifiers"""
        # Start with base
        result = base
        
        # Add qualifiers appropriately
        for qualifier in qualifiers:
            if qualifier.qualifier_type == "approximator":
                # Insert before quantities/values
                result = self._insert_approximator(result, qualifier.phrase)
            elif qualifier.phrase in ["I think", "I believe", "It seems", "It appears"]:
                # Prepend to beginning
                result = f"{qualifier.phrase} {result.lower()}"
            else:
                # Append to end
                result = f"{result}, {qualifier.phrase}"
        
        return result
    
    def _insert_approximator(self, text: str, approximator: str) -> str:
        """Insert approximator before quantities"""
        # Simple implementation - would need NLP for better placement
        words = text.split()
        for i, word in enumerate(words):
            if any(char.isdigit() for char in word):
                words.insert(i, approximator)
                break
        return " ".join(words)
    
    def _calculate_meta_uncertainty(self, confidence: float,
                                  uncertainty_type: UncertaintyType) -> float:
        """Calculate uncertainty about the uncertainty assessment"""
        # Base meta-uncertainty on how extreme the confidence is
        distance_from_middle = abs(confidence - 0.5)
        
        # More extreme = more certain about uncertainty
        base_meta = 0.5 + distance_from_middle
        
        # Adjust based on uncertainty type
        type_adjustments = {
            UncertaintyType.EPISTEMIC: 0.9,  # Usually clear what we know/don't know
            UncertaintyType.ALEATORY: 0.8,  # Randomness is identifiable
            UncertaintyType.SEMANTIC: 0.6,  # Vagueness is itself vague
            UncertaintyType.SUBJECTIVE: 0.5,  # Subjective uncertainty is uncertain
        }
        
        adjustment = type_adjustments.get(uncertainty_type, 0.7)
        
        return base_meta * adjustment
    
    def _express_knowledge_claims(self, claims: List[Tuple[str, float]],
                                context: UncertaintyContext) -> str:
        """Express knowledge claims with confidence"""
        if not claims:
            return ""
        
        # Group by confidence level
        high_conf = [c for c, conf in claims if conf > 0.8]
        med_conf = [c for c, conf in claims if 0.5 < conf <= 0.8]
        low_conf = [c for c, conf in claims if conf <= 0.5]
        
        expressions = []
        
        if high_conf:
            expressions.append(f"I'm confident that: {'; '.join(high_conf)}")
        if med_conf:
            expressions.append(f"I believe that: {'; '.join(med_conf)}")
        if low_conf:
            expressions.append(f"I'm uncertain but suspect that: {'; '.join(low_conf)}")
        
        return ". ".join(expressions) + "."
    
    def _express_uncertainties(self, uncertainties: List[Tuple[str, UncertaintyType]],
                             context: UncertaintyContext) -> str:
        """Express specific uncertainties"""
        if not uncertainties:
            return ""
        
        grouped = {}
        for item, u_type in uncertainties:
            if u_type not in grouped:
                grouped[u_type] = []
            grouped[u_type].append(item)
        
        expressions = []
        for u_type, items in grouped.items():
            phrase = self.uncertainty_phrases[u_type][0]
            expressions.append(f"{phrase}: {', '.join(items)}")
        
        return "However, " + "; ".join(expressions) + "."
    
    def _express_open_questions(self, questions: List[str],
                              context: UncertaintyContext) -> str:
        """Express open questions"""
        if not questions:
            return ""
        
        if len(questions) == 1:
            return f"A key open question is: {questions[0]}"
        elif len(questions) <= 3:
            return f"Open questions include: {'; '.join(questions)}"
        else:
            return f"There are {len(questions)} open questions, including: {'; '.join(questions[:2])}; and others"
    
    def _express_confidence_distribution(self, distribution: Dict[str, float]) -> str:
        """Express overall confidence distribution"""
        if not distribution:
            return ""
        
        avg_confidence = sum(distribution.values()) / len(distribution) if distribution else 0.5
        
        if avg_confidence > 0.7:
            return "Overall, I have relatively high confidence in this domain."
        elif avg_confidence > 0.4:
            return "My overall confidence in this area is moderate."
        else:
            return "I have significant uncertainty across this domain."
    
    def _confidence_to_percentage(self, confidence: float) -> str:
        """Convert confidence to percentage string"""
        percentage = int(confidence * 100)
        return f"{percentage}% confident"
