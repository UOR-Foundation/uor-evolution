"""
Dialogue Quality Metrics Utilities

This module provides tools for measuring and analyzing the quality of
dialogue interactions, including coherence, engagement, and philosophical depth.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum
import math
from collections import defaultdict


class DialogueMetricType(Enum):
    """Types of dialogue quality metrics"""
    COHERENCE = "coherence"
    RELEVANCE = "relevance"
    ENGAGEMENT = "engagement"
    DEPTH = "depth"
    CLARITY = "clarity"
    INFORMATIVENESS = "informativeness"
    PHILOSOPHICAL_RIGOR = "philosophical_rigor"
    EMOTIONAL_RESONANCE = "emotional_resonance"


class TurnType(Enum):
    """Types of dialogue turns"""
    QUESTION = "question"
    ANSWER = "answer"
    STATEMENT = "statement"
    CLARIFICATION = "clarification"
    ELABORATION = "elaboration"
    CHALLENGE = "challenge"
    AGREEMENT = "agreement"
    REFLECTION = "reflection"


@dataclass
class DialogueTurn:
    """A single turn in dialogue"""
    turn_id: str
    speaker: str
    content: str
    turn_type: TurnType
    timestamp: float
    semantic_content: List[str] = field(default_factory=list)
    references_turns: List[str] = field(default_factory=list)
    emotional_tone: float = 0.5  # 0 = negative, 1 = positive
    confidence: float = 1.0


@dataclass
class DialogueSession:
    """A complete dialogue session"""
    session_id: str
    participants: List[str]
    turns: List[DialogueTurn]
    topic: str
    context: Dict[str, Any] = field(default_factory=dict)
    start_time: float = 0.0
    end_time: float = 0.0


@dataclass
class DialogueQualityReport:
    """Comprehensive dialogue quality report"""
    session_id: str
    overall_quality: float
    metric_scores: Dict[DialogueMetricType, float]
    strengths: List[str]
    weaknesses: List[str]
    recommendations: List[str]
    detailed_analysis: Dict[str, Any]


@dataclass
class TurnQualityScore:
    """Quality score for individual turn"""
    turn_id: str
    relevance: float
    clarity: float
    informativeness: float
    coherence_with_context: float
    overall: float


class DialogueQualityAnalyzer:
    """
    Analyzes dialogue quality across multiple dimensions.
    """
    
    def __init__(self):
        self.metric_weights = self._initialize_metric_weights()
        self.coherence_patterns = self._initialize_coherence_patterns()
        self.quality_thresholds = self._initialize_quality_thresholds()
        
    def _initialize_metric_weights(self) -> Dict[DialogueMetricType, float]:
        """Initialize weights for different metrics"""
        return {
            DialogueMetricType.COHERENCE: 0.20,
            DialogueMetricType.RELEVANCE: 0.15,
            DialogueMetricType.ENGAGEMENT: 0.15,
            DialogueMetricType.DEPTH: 0.15,
            DialogueMetricType.CLARITY: 0.15,
            DialogueMetricType.INFORMATIVENESS: 0.10,
            DialogueMetricType.PHILOSOPHICAL_RIGOR: 0.05,
            DialogueMetricType.EMOTIONAL_RESONANCE: 0.05
        }
    
    def _initialize_coherence_patterns(self) -> Dict[str, List[str]]:
        """Initialize patterns for coherence detection"""
        return {
            "reference": ["as mentioned", "referring to", "regarding", "about"],
            "continuation": ["furthermore", "additionally", "moreover", "also"],
            "contrast": ["however", "but", "on the other hand", "conversely"],
            "causation": ["because", "therefore", "thus", "consequently"],
            "elaboration": ["specifically", "in particular", "for example", "namely"],
            "summary": ["in summary", "to summarize", "overall", "in conclusion"]
        }
    
    def _initialize_quality_thresholds(self) -> Dict[str, float]:
        """Initialize quality thresholds"""
        return {
            "excellent": 0.85,
            "good": 0.70,
            "satisfactory": 0.55,
            "poor": 0.40
        }
    
    def analyze_dialogue_quality(self, session: DialogueSession) -> DialogueQualityReport:
        """Analyze overall dialogue quality"""
        # Calculate individual metrics
        metric_scores = {}
        
        metric_scores[DialogueMetricType.COHERENCE] = self._measure_coherence(session)
        metric_scores[DialogueMetricType.RELEVANCE] = self._measure_relevance(session)
        metric_scores[DialogueMetricType.ENGAGEMENT] = self._measure_engagement(session)
        metric_scores[DialogueMetricType.DEPTH] = self._measure_depth(session)
        metric_scores[DialogueMetricType.CLARITY] = self._measure_clarity(session)
        metric_scores[DialogueMetricType.INFORMATIVENESS] = self._measure_informativeness(session)
        metric_scores[DialogueMetricType.PHILOSOPHICAL_RIGOR] = self._measure_philosophical_rigor(session)
        metric_scores[DialogueMetricType.EMOTIONAL_RESONANCE] = self._measure_emotional_resonance(session)
        
        # Calculate overall quality
        overall_quality = self._calculate_overall_quality(metric_scores)
        
        # Identify strengths and weaknesses
        strengths = self._identify_strengths(metric_scores)
        weaknesses = self._identify_weaknesses(metric_scores)
        
        # Generate recommendations
        recommendations = self._generate_recommendations(metric_scores, session)
        
        # Detailed analysis
        detailed_analysis = self._perform_detailed_analysis(session, metric_scores)
        
        return DialogueQualityReport(
            session_id=session.session_id,
            overall_quality=overall_quality,
            metric_scores=metric_scores,
            strengths=strengths,
            weaknesses=weaknesses,
            recommendations=recommendations,
            detailed_analysis=detailed_analysis
        )
    
    def analyze_turn_quality(self, turn: DialogueTurn, 
                           context: List[DialogueTurn]) -> TurnQualityScore:
        """Analyze quality of individual turn"""
        # Measure turn-specific qualities
        relevance = self._measure_turn_relevance(turn, context)
        clarity = self._measure_turn_clarity(turn)
        informativeness = self._measure_turn_informativeness(turn)
        coherence = self._measure_turn_coherence(turn, context)
        
        # Calculate overall turn quality
        overall = (relevance + clarity + informativeness + coherence) / 4
        
        return TurnQualityScore(
            turn_id=turn.turn_id,
            relevance=relevance,
            clarity=clarity,
            informativeness=informativeness,
            coherence_with_context=coherence,
            overall=overall
        )
    
    def measure_dialogue_flow(self, session: DialogueSession) -> Dict[str, Any]:
        """Measure dialogue flow characteristics"""
        flow_analysis = {
            "turn_distribution": self._analyze_turn_distribution(session),
            "topic_progression": self._analyze_topic_progression(session),
            "interaction_patterns": self._identify_interaction_patterns(session),
            "momentum": self._calculate_dialogue_momentum(session),
            "balance": self._calculate_dialogue_balance(session)
        }
        
        return flow_analysis
    
    def identify_dialogue_patterns(self, session: DialogueSession) -> List[Dict[str, Any]]:
        """Identify patterns in dialogue"""
        patterns = []
        
        # Question-answer patterns
        qa_patterns = self._find_qa_patterns(session)
        patterns.extend(qa_patterns)
        
        # Elaboration patterns
        elab_patterns = self._find_elaboration_patterns(session)
        patterns.extend(elab_patterns)
        
        # Argument patterns
        arg_patterns = self._find_argument_patterns(session)
        patterns.extend(arg_patterns)
        
        # Emotional patterns
        emo_patterns = self._find_emotional_patterns(session)
        patterns.extend(emo_patterns)
        
        return patterns
    
    def compare_dialogue_sessions(self, session1: DialogueSession,
                                session2: DialogueSession) -> Dict[str, Any]:
        """Compare quality between two dialogue sessions"""
        # Analyze both sessions
        report1 = self.analyze_dialogue_quality(session1)
        report2 = self.analyze_dialogue_quality(session2)
        
        comparison = {
            "overall_difference": report2.overall_quality - report1.overall_quality,
            "metric_differences": {},
            "improved_aspects": [],
            "declined_aspects": [],
            "similarity_score": 0.0
        }
        
        # Compare individual metrics
        for metric in DialogueMetricType:
            diff = report2.metric_scores[metric] - report1.metric_scores[metric]
            comparison["metric_differences"][metric.value] = diff
            
            if diff > 0.1:
                comparison["improved_aspects"].append(metric.value)
            elif diff < -0.1:
                comparison["declined_aspects"].append(metric.value)
        
        # Calculate similarity
        comparison["similarity_score"] = self._calculate_session_similarity(
            session1, session2
        )
        
        return comparison
    
    def generate_improvement_suggestions(self, session: DialogueSession,
                                       report: DialogueQualityReport) -> List[Dict[str, Any]]:
        """Generate specific suggestions for improvement"""
        suggestions = []
        
        # Based on weakest metrics
        weakest_metrics = sorted(
            report.metric_scores.items(),
            key=lambda x: x[1]
        )[:3]
        
        for metric, score in weakest_metrics:
            suggestion = self._generate_metric_suggestion(metric, score, session)
            if suggestion:
                suggestions.append(suggestion)
        
        # Based on patterns
        patterns = self.identify_dialogue_patterns(session)
        pattern_suggestions = self._generate_pattern_suggestions(patterns)
        suggestions.extend(pattern_suggestions)
        
        return suggestions
    
    # Private measurement methods
    
    def _measure_coherence(self, session: DialogueSession) -> float:
        """Measure dialogue coherence"""
        if len(session.turns) < 2:
            return 1.0
        
        coherence_scores = []
        
        # Check sequential coherence
        for i in range(1, len(session.turns)):
            prev_turn = session.turns[i-1]
            curr_turn = session.turns[i]
            
            # Semantic coherence
            semantic_coherence = self._calculate_semantic_coherence(
                prev_turn.semantic_content,
                curr_turn.semantic_content
            )
            
            # Reference coherence
            reference_coherence = 1.0 if prev_turn.turn_id in curr_turn.references_turns else 0.5
            
            # Pattern coherence
            pattern_coherence = self._check_coherence_patterns(
                prev_turn.content,
                curr_turn.content
            )
            
            turn_coherence = (semantic_coherence + reference_coherence + pattern_coherence) / 3
            coherence_scores.append(turn_coherence)
        
        return sum(coherence_scores) / len(coherence_scores) if coherence_scores else 0
    
    def _measure_relevance(self, session: DialogueSession) -> float:
        """Measure topic relevance"""
        topic_keywords = session.topic.lower().split()
        relevance_scores = []
        
        for turn in session.turns:
            # Check keyword presence
            content_lower = turn.content.lower()
            keyword_matches = sum(1 for kw in topic_keywords if kw in content_lower)
            keyword_relevance = min(1.0, keyword_matches / len(topic_keywords))
            
            # Check semantic relevance
            semantic_relevance = self._calculate_semantic_relevance(
                turn.semantic_content,
                topic_keywords
            )
            
            turn_relevance = (keyword_relevance + semantic_relevance) / 2
            relevance_scores.append(turn_relevance)
        
        return sum(relevance_scores) / len(relevance_scores) if relevance_scores else 0
    
    def _measure_engagement(self, session: DialogueSession) -> float:
        """Measure dialogue engagement"""
        engagement_factors = []
        
        # Turn frequency (responses per minute)
        duration = session.end_time - session.start_time if session.end_time > session.start_time else 1
        turn_frequency = len(session.turns) / (duration / 60)
        engagement_factors.append(min(1.0, turn_frequency / 10))  # Normalize to 10 turns/min
        
        # Question ratio
        questions = sum(1 for t in session.turns if t.turn_type == TurnType.QUESTION)
        question_ratio = questions / len(session.turns) if session.turns else 0
        engagement_factors.append(min(1.0, question_ratio * 3))  # Questions show engagement
        
        # Elaboration ratio
        elaborations = sum(1 for t in session.turns if t.turn_type == TurnType.ELABORATION)
        elaboration_ratio = elaborations / len(session.turns) if session.turns else 0
        engagement_factors.append(elaboration_ratio)
        
        # Emotional variation
        emotional_variance = self._calculate_emotional_variance(session)
        engagement_factors.append(emotional_variance)
        
        return sum(engagement_factors) / len(engagement_factors)
    
    def _measure_depth(self, session: DialogueSession) -> float:
        """Measure philosophical/intellectual depth"""
        depth_indicators = []
        
        # Average turn length (longer = deeper)
        avg_length = sum(len(t.content.split()) for t in session.turns) / len(session.turns)
        normalized_length = min(1.0, avg_length / 100)  # Normalize to 100 words
        depth_indicators.append(normalized_length)
        
        # Conceptual complexity
        unique_concepts = set()
        for turn in session.turns:
            unique_concepts.update(turn.semantic_content)
        concept_diversity = min(1.0, len(unique_concepts) / (len(session.turns) * 3))
        depth_indicators.append(concept_diversity)
        
        # Abstract language ratio
        abstract_ratio = self._calculate_abstract_language_ratio(session)
        depth_indicators.append(abstract_ratio)
        
        # Reasoning indicators
        reasoning_ratio = self._calculate_reasoning_ratio(session)
        depth_indicators.append(reasoning_ratio)
        
        return sum(depth_indicators) / len(depth_indicators)
    
    def _measure_clarity(self, session: DialogueSession) -> float:
        """Measure dialogue clarity"""
        clarity_scores = []
        
        for turn in session.turns:
            # Sentence complexity (inverse)
            avg_sentence_length = self._calculate_avg_sentence_length(turn.content)
            sentence_clarity = 1.0 - min(1.0, avg_sentence_length / 50)
            
            # Ambiguity indicators (inverse)
            ambiguity = self._detect_ambiguity(turn.content)
            ambiguity_clarity = 1.0 - ambiguity
            
            # Structure clarity
            structure_clarity = self._assess_structure_clarity(turn.content)
            
            turn_clarity = (sentence_clarity + ambiguity_clarity + structure_clarity) / 3
            clarity_scores.append(turn_clarity)
        
        return sum(clarity_scores) / len(clarity_scores) if clarity_scores else 0
    
    def _measure_informativeness(self, session: DialogueSession) -> float:
        """Measure information content"""
        info_scores = []
        
        accumulated_info = set()
        
        for turn in session.turns:
            # New information ratio
            turn_info = set(turn.semantic_content)
            new_info = turn_info - accumulated_info
            new_info_ratio = len(new_info) / len(turn_info) if turn_info else 0
            
            accumulated_info.update(turn_info)
            
            # Factual content
            factual_ratio = self._calculate_factual_ratio(turn.content)
            
            # Specificity
            specificity = self._calculate_specificity(turn.content)
            
            turn_informativeness = (new_info_ratio + factual_ratio + specificity) / 3
            info_scores.append(turn_informativeness)
        
        return sum(info_scores) / len(info_scores) if info_scores else 0
    
    def _measure_philosophical_rigor(self, session: DialogueSession) -> float:
        """Measure philosophical rigor"""
        rigor_indicators = []
        
        # Argument structure
        argument_quality = self._assess_argument_quality(session)
        rigor_indicators.append(argument_quality)
        
        # Logical consistency
        logical_consistency = self._assess_logical_consistency(session)
        rigor_indicators.append(logical_consistency)
        
        # Conceptual precision
        conceptual_precision = self._assess_conceptual_precision(session)
        rigor_indicators.append(conceptual_precision)
        
        # Critical thinking indicators
        critical_thinking = self._assess_critical_thinking(session)
        rigor_indicators.append(critical_thinking)
        
        return sum(rigor_indicators) / len(rigor_indicators) if rigor_indicators else 0
    
    def _measure_emotional_resonance(self, session: DialogueSession) -> float:
        """Measure emotional resonance and connection"""
        resonance_factors = []
        
        # Emotional alignment between participants
        emotional_alignment = self._calculate_emotional_alignment(session)
        resonance_factors.append(emotional_alignment)
        
        # Empathy indicators
        empathy_score = self._detect_empathy_indicators(session)
        resonance_factors.append(empathy_score)
        
        # Emotional depth
        emotional_depth = self._calculate_emotional_depth(session)
        resonance_factors.append(emotional_depth)
        
        return sum(resonance_factors) / len(resonance_factors) if resonance_factors else 0
    
    # Helper calculation methods
    
    def _calculate_semantic_coherence(self, content1: List[str], 
                                    content2: List[str]) -> float:
        """Calculate semantic coherence between contents"""
        if not content1 or not content2:
            return 0.5
        
        set1 = set(content1)
        set2 = set(content2)
        
        overlap = len(set1 & set2)
        total = len(set1 | set2)
        
        return overlap / total if total > 0 else 0
    
    def _check_coherence_patterns(self, prev_content: str, curr_content: str) -> float:
        """Check for coherence patterns"""
        curr_lower = curr_content.lower()
        
        pattern_score = 0
        pattern_count = 0
        
        for pattern_type, patterns in self.coherence_patterns.items():
            for pattern in patterns:
                if pattern in curr_lower:
                    pattern_score += 1
                    pattern_count += 1
        
        return min(1.0, pattern_score / 3) if pattern_count > 0 else 0.5
    
    def _calculate_semantic_relevance(self, semantic_content: List[str],
                                    topic_keywords: List[str]) -> float:
        """Calculate semantic relevance to topic"""
        if not semantic_content:
            return 0
        
        relevant_count = sum(1 for concept in semantic_content 
                           if any(kw in concept.lower() for kw in topic_keywords))
        
        return relevant_count / len(semantic_content)
    
    def _calculate_emotional_variance(self, session: DialogueSession) -> float:
        """Calculate emotional variance in dialogue"""
        if len(session.turns) < 2:
            return 0
        
        emotions = [turn.emotional_tone for turn in session.turns]
        
        # Calculate variance
        mean = sum(emotions) / len(emotions)
        variance = sum((e - mean) ** 2 for e in emotions) / len(emotions)
        
        # Normalize (0-0.25 range expected)
        return min(1.0, variance * 4)
    
    def _calculate_abstract_language_ratio(self, session: DialogueSession) -> float:
        """Calculate ratio of abstract language"""
        abstract_words = {
            "concept", "idea", "theory", "principle", "abstract",
            "essence", "nature", "meaning", "significance", "implication",
            "relationship", "pattern", "structure", "system", "framework"
        }
        
        total_words = 0
        abstract_count = 0
        
        for turn in session.turns:
            words = turn.content.lower().split()
            total_words += len(words)
            abstract_count += sum(1 for w in words if w in abstract_words)
        
        return abstract_count / total_words if total_words > 0 else 0
    
    def _calculate_reasoning_ratio(self, session: DialogueSession) -> float:
        """Calculate ratio of reasoning indicators"""
        reasoning_indicators = {
            "because", "therefore", "thus", "hence", "consequently",
            "implies", "suggests", "indicates", "demonstrates", "proves",
            "follows", "entails", "necessitates", "requires", "depends"
        }
        
        total_turns = len(session.turns)
        reasoning_turns = 0
        
        for turn in session.turns:
            if any(indicator in turn.content.lower() for indicator in reasoning_indicators):
                reasoning_turns += 1
        
        return reasoning_turns / total_turns if total_turns > 0 else 0
    
    def _calculate_avg_sentence_length(self, content: str) -> float:
        """Calculate average sentence length"""
        sentences = content.split('.')
        sentences = [s.strip() for s in sentences if s.strip()]
        
        if not sentences:
            return 0
        
        total_words = sum(len(s.split()) for s in sentences)
        
        return total_words / len(sentences)
    
    def _detect_ambiguity(self, content: str) -> float:
        """Detect ambiguity in content"""
        ambiguous_terms = {
            "thing", "stuff", "it", "this", "that", "some", "many",
            "various", "certain", "particular", "specific"
        }
        
        words = content.lower().split()
        ambiguous_count = sum(1 for w in words if w in ambiguous_terms)
        
        return min(1.0, ambiguous_count / len(words)) if words else 0
    
    def _assess_structure_clarity(self, content: str) -> float:
        """Assess structural clarity of content"""
        # Check for clear structure markers
        structure_markers = {
            "first", "second", "third", "finally",
            "initially", "then", "next", "lastly",
            "on one hand", "on the other hand",
            "in conclusion", "to summarize"
        }
        
        content_lower = content.lower()
        marker_count = sum(1 for marker in structure_markers if marker in content_lower)
        
        # More markers = clearer structure
        return min(1.0, marker_count / 3)
    
    def _calculate_factual_ratio(self, content: str) -> float:
        """Calculate ratio of factual content"""
        # Simplified - look for factual indicators
        factual_indicators = {
            "fact", "data", "evidence", "study", "research",
            "statistic", "number", "percent", "measurement", "observation"
        }
        
        words = content.lower().split()
        factual_count = sum(1 for w in words if w in factual_indicators)
        
        return min(1.0, factual_count / 10)  # Normalize to 10 factual words
    
    def _calculate_specificity(self, content: str) -> float:
        """Calculate content specificity"""
        # Check for specific vs general terms
        specific_indicators = {
            "specifically", "precisely", "exactly", "particular",
            "concrete", "detailed", "explicit", "definite"
        }
        
        general_indicators = {
            "generally", "usually", "typically", "often",
            "sometimes", "perhaps", "maybe", "possibly"
        }
        
        content_lower = content.lower()
        specific_count = sum(1 for ind in specific_indicators if ind in content_lower)
        general_count = sum(1 for ind in general_indicators if ind in content_lower)
        
        if specific_count + general_count == 0:
            return 0.5
        
        return specific_count / (specific_count + general_count)
    
    def _calculate_overall_quality(self, metric_scores: Dict[DialogueMetricType, float]) -> float:
        """Calculate weighted overall quality"""
        total_score = 0
        total_weight = 0
        
        for metric, score in metric_scores.items():
            weight = self.metric_weights.get(metric, 0.1)
            total_score += score * weight
            total_weight += weight
        
        return total_score / total_weight if total_weight > 0 else 0
    
    def _identify_strengths(self, metric_scores: Dict[DialogueMetricType, float]) -> List[str]:
        """Identify dialogue strengths"""
        strengths = []
        
        for metric, score in metric_scores.items():
            if score >= self.quality_thresholds["good"]:
                strengths.append(f"Strong {metric.value}: {score:.2f}")
        
        return strengths
    
    def _identify_weaknesses(self, metric_scores: Dict[DialogueMetricType, float]) -> List[str]:
        """Identify dialogue weaknesses"""
        weaknesses = []
        
        for metric, score in metric_scores.items():
            if score < self.quality_thresholds["satisfactory"]:
                weaknesses.append(f"Weak {metric.value}: {score:.2f}")
        
        return weaknesses
    
    def _generate_recommendations(self, metric_scores: Dict[DialogueMetricType, float],
                                session: DialogueSession) -> List[str]:
        """Generate improvement recommendations"""
        recommendations = []
        
        # Low coherence
        if metric_scores[DialogueMetricType.COHERENCE] < 0.6:
            recommendations.append(
                "Improve coherence by better connecting responses to previous turns"
            )
        
        # Low relevance
        if metric_scores[DialogueMetricType.RELEVANCE] < 0.6:
            recommendations.append(
                "Maintain focus on the main topic throughout the dialogue"
            )
        
        # Low engagement
        if metric_scores[DialogueMetricType.ENGAGEMENT] < 0.6:
            recommendations.append(
                "Increase engagement through more questions and elaborations"
            )
        
        # Low depth
        if metric_scores[DialogueMetricType.DEPTH] < 0.6:
            recommendations.append(
                "Explore topics more deeply with detailed analysis"
            )
        
        return recommendations
    
    def _perform_detailed_analysis(self, session: DialogueSession,
                                 metric_scores: Dict[DialogueMetricType, float]) -> Dict[str, Any]:
        """Perform detailed dialogue analysis"""
        return {
            "turn_quality_distribution": self._analyze_turn_quality_distribution(session),
            "topic_coverage": self._analyze_topic_coverage(session),
            "interaction_dynamics": self._analyze_interaction_dynamics(session),
            "linguistic_features": self._analyze_linguistic_features(session),
            "temporal_patterns": self._analyze_temporal_patterns(session)
        }
    
    def _analyze_turn_quality_distribution(self, session: DialogueSession) -> Dict[str, Any]:
        """Analyze distribution of turn quality"""
        turn_scores = []
        
        for i, turn in enumerate(session.turns):
            context = session.turns[max(0, i-3):i]  # Previous 3 turns
            score = self.analyze_turn_quality(turn, context)
            turn_scores.append(score.overall)
        
        return {
            "mean": sum(turn_scores) / len(turn_scores) if turn_scores else 0,
            "min": min(turn_scores) if turn_scores else 0,
            "max": max(turn_scores) if turn_scores else 0,
            "variance": self._calculate_variance(turn_scores)
        }
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values"""
        if not values:
            return 0
        
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)
    
    def _analyze_topic_coverage(self, session: DialogueSession) -> Dict[str, Any]:
        """Analyze how well topics are covered"""
        topic_mentions = defaultdict(int)
        
        for turn in session.turns:
            for concept in turn.semantic_content:
                topic_mentions[concept] += 1
        
        return {
            "unique_topics": len(topic_mentions),
            "topic_distribution": dict(topic_mentions),
            "coverage_breadth": len(topic_mentions) / len(session.turns) if session.turns else 0
        }
    
    def _analyze_interaction_dynamics(self, session: DialogueSession) -> Dict[str, Any]:
        """Analyze interaction dynamics"""
        return {
            "turn_taking_balance": self._calculate_turn_taking_balance(session),
            "response_patterns": self._analyze_response_patterns(session),
            "initiative_distribution": self._analyze_initiative_distribution(session)
        }
    
    def _analyze_linguistic_features(self, session: DialogueSession) -> Dict[str, Any]:
        """Analyze linguistic features"""
        total_words = sum(len(turn.content.split()) for turn in session.turns)
        unique_words = set()
        
        for turn in session.turns:
            unique_words.update(turn.content.lower().split())
        
        return {
            "total_words": total_words,
            "unique_words": len(unique_words),
            "lexical_diversity": len(unique_words) / total_words if total_words > 0 else 0,
            "average_turn_length": total_words / len(session.turns) if session.turns else 0
        }
    
    def _analyze_temporal_patterns(self, session: DialogueSession) -> Dict[str, Any]:
        """Analyze temporal patterns in dialogue"""
        if len(session.turns) < 2:
            return {"pattern": "insufficient_data"}
        
        # Calculate inter-turn times
        inter_turn_times = []
        for i in range(1, len(session.turns)):
            time_diff = session.turns[i].timestamp - session.turns[i-1].timestamp
            inter_turn_times.append(time_diff)
        
        return {
            "average_response_time": sum(inter_turn_times) / len(inter_turn_times),
            "response_time_variance": self._calculate_variance(inter_turn_times),
            "tempo": "fast" if sum(inter_turn_times) / len(inter_turn_times) < 5 else "moderate"
        }
    
    # Additional helper methods for specific analyses
    
    def _measure_turn_relevance(self, turn: DialogueTurn, context: List[DialogueTurn]) -> float:
        """Measure relevance of turn to context"""
        if not context:
            return 0.5  # Neutral if no context
        
        # Semantic relevance to previous turns
        context_concepts = set()
        for ctx_turn in context:
            context_concepts.update(ctx_turn.semantic_content)
        
        turn_concepts = set(turn.semantic_content)
        
        if not turn_concepts or not context_concepts:
            return 0.5
        
        overlap = len(turn_concepts & context_concepts)
        return overlap / len(turn_concepts)
    
    def _measure_turn_clarity(self, turn: DialogueTurn) -> float:
        """Measure clarity of individual turn"""
        # Similar to session clarity but for single turn
        avg_sentence_length = self._calculate_avg_sentence_length(turn.content)
        sentence_clarity = 1.0 - min(1.0, avg_sentence_length / 50)
        
        ambiguity = self._detect_ambiguity(turn.content)
        ambiguity_clarity = 1.0 - ambiguity
        
        structure_clarity = self._assess_structure_clarity(turn.content)
        
        return (sentence_clarity + ambiguity_clarity + structure_clarity) / 3
    
    def _measure_turn_informativeness(self, turn: DialogueTurn) -> float:
        """Measure informativeness of turn"""
        # Content density
        words = turn.content.split()
        unique_words = set(words)
        lexical_diversity = len(unique_words) / len(words) if words else 0
        
        # Semantic content
        semantic_density = len(turn.semantic_content) / max(len(words), 1)
        
        # Factual content
        factual_ratio = self._calculate_factual_ratio(turn.content)
        
        return (lexical_diversity + semantic_density + factual_ratio) / 3
    
    def _measure_turn_coherence(self, turn: DialogueTurn, context: List[DialogueTurn]) -> float:
        """Measure coherence of turn with context"""
        if not context:
            return 1.0  # First turn is coherent by default
        
        # Check references
        reference_score = 0
        if turn.references_turns:
            valid_refs = sum(1 for ref in turn.references_turns 
                           if any(ctx.turn_id == ref for ctx in context))
            reference_score = valid_refs / len(turn.references_turns)
        
        # Check semantic continuity
        last_turn = context[-1]
        semantic_continuity = self._calculate_semantic_coherence(
            last_turn.semantic_content,
            turn.semantic_content
        )
        
        # Check pattern coherence
        pattern_coherence = self._check_coherence_patterns(
            last_turn.content,
            turn.content
        )
        
        scores = [semantic_continuity, pattern_coherence]
        if turn.references_turns:
            scores.append(reference_score)
        
        return sum(scores) / len(scores)
    
    def _analyze_turn_distribution(self, session: DialogueSession) -> Dict[str, Any]:
        """Analyze distribution of turn types"""
        type_counts = defaultdict(int)
        
        for turn in session.turns:
            type_counts[turn.turn_type.value] += 1
        
        total = len(session.turns)
        
        return {
            "distribution": {k: v/total for k, v in type_counts.items()},
            "dominant_type": max(type_counts.items(), key=lambda x: x[1])[0] if type_counts else None
        }
    
    def _analyze_topic_progression(self, session: DialogueSession) -> Dict[str, Any]:
        """Analyze how topics progress through dialogue"""
        topic_timeline = []
        
        for i, turn in enumerate(session.turns):
            topic_timeline.append({
                "turn": i,
                "topics": turn.semantic_content,
                "new_topics": set(turn.semantic_content) - set(sum([t["topics"] for t in topic_timeline[:i]], []))
            })
        
        return {
            "timeline": topic_timeline,
            "topic_introduction_rate": sum(len(t["new_topics"]) for t in topic_timeline) / len(session.turns)
        }
    
    def _identify_interaction_patterns(self, session: DialogueSession) -> List[str]:
        """Identify interaction patterns in dialogue"""
        patterns = []
        
        # Check for Q&A pattern
        qa_count = 0
        for i in range(len(session.turns) - 1):
            if (session.turns[i].turn_type == TurnType.QUESTION and
                session.turns[i+1].turn_type == TurnType.ANSWER):
                qa_count += 1
        
        if qa_count > len(session.turns) / 4:
            patterns.append("question_answer_dominant")
        
        # Check for elaboration pattern
        elab_count = sum(1 for t in session.turns if t.turn_type == TurnType.ELABORATION)
        if elab_count > len(session.turns) / 3:
            patterns.append("elaboration_heavy")
        
        # Check for debate pattern
        challenge_count = sum(1 for t in session.turns if t.turn_type == TurnType.CHALLENGE)
        if challenge_count > len(session.turns) / 5:
            patterns.append("debate_oriented")
        
        return patterns
    
    def _calculate_dialogue_momentum(self, session: DialogueSession) -> float:
        """Calculate dialogue momentum (engagement over time)"""
        if len(session.turns) < 3:
            return 0.5
        
        # Divide into thirds
        third = len(session.turns) // 3
        
        # Calculate average turn length for each third
        first_third_avg = sum(len(t.content.split()) for t in session.turns[:third]) / third
        last_third_avg = sum(len(t.content.split()) for t in session.turns[-third:]) / third
        
        # Positive momentum if increasing, negative if decreasing
        momentum = (last_third_avg - first_third_avg) / first_third_avg if first_third_avg > 0 else 0
        
        # Normalize to 0-1 scale
        return min(1.0, max(0.0, 0.5 + momentum))
    
    def _calculate_dialogue_balance(self, session: DialogueSession) -> float:
        """Calculate balance between participants"""
        participant_turns = defaultdict(int)
        participant_words = defaultdict(int)
        
        for turn in session.turns:
            participant_turns[turn.speaker] += 1
            participant_words[turn.speaker] += len(turn.content.split())
        
        if len(participant_turns) < 2:
            return 0  # No balance in monologue
        
        # Calculate turn balance
        turn_counts = list(participant_turns.values())
        turn_balance = 1.0 - (max(turn_counts) - min(turn_counts)) / sum(turn_counts)
        
        # Calculate word balance
        word_counts = list(participant_words.values())
        word_balance = 1.0 - (max(word_counts) - min(word_counts)) / sum(word_counts)
        
        return (turn_balance + word_balance) / 2
    
    def _find_qa_patterns(self, session: DialogueSession) -> List[Dict[str, Any]]:
        """Find question-answer patterns"""
        patterns = []
        
        i = 0
        while i < len(session.turns) - 1:
            if session.turns[i].turn_type == TurnType.QUESTION:
                # Find answer
                j = i + 1
                while j < len(session.turns) and session.turns[j].turn_type != TurnType.ANSWER:
                    j += 1
                
                if j < len(session.turns):
                    patterns.append({
                        "type": "question_answer",
                        "question_turn": i,
                        "answer_turn": j,
                        "delay": j - i - 1
                    })
                    i = j
            i += 1
        
        return patterns
    
    def _find_elaboration_patterns(self, session: DialogueSession) -> List[Dict[str, Any]]:
        """Find elaboration patterns"""
        patterns = []
        
        for i, turn in enumerate(session.turns):
            if turn.turn_type == TurnType.ELABORATION:
                # Find what it elaborates on
                if turn.references_turns:
                    patterns.append({
                        "type": "elaboration",
                        "elaboration_turn": i,
                        "elaborates_on": turn.references_turns[0],
                        "depth": len(turn.content.split())
                    })
        
        return patterns
    
    def _find_argument_patterns(self, session: DialogueSession) -> List[Dict[str, Any]]:
        """Find argument patterns"""
        patterns = []
        
        # Look for claim-challenge-response sequences
        for i in range(len(session.turns) - 2):
            if (session.turns[i].turn_type == TurnType.STATEMENT and
                session.turns[i+1].turn_type == TurnType.CHALLENGE):
                
                patterns.append({
                    "type": "argument",
                    "claim_turn": i,
                    "challenge_turn": i + 1,
                    "resolved": session.turns[i+2].turn_type == TurnType.AGREEMENT if i+2 < len(session.turns) else False
                })
        
        return patterns
    
    def _find_emotional_patterns(self, session: DialogueSession) -> List[Dict[str, Any]]:
        """Find emotional patterns"""
        patterns = []
        
        # Find emotional shifts
        for i in range(1, len(session.turns)):
            prev_emotion = session.turns[i-1].emotional_tone
            curr_emotion = session.turns[i].emotional_tone
            
            shift = abs(curr_emotion - prev_emotion)
            
            if shift > 0.3:  # Significant shift
                patterns.append({
                    "type": "emotional_shift",
                    "from_turn": i - 1,
                    "to_turn": i,
                    "shift_magnitude": shift,
                    "direction": "positive" if curr_emotion > prev_emotion else "negative"
                })
        
        return patterns
    
    def _calculate_session_similarity(self, session1: DialogueSession,
                                    session2: DialogueSession) -> float:
        """Calculate similarity between two sessions"""
        # Topic similarity
        topics1 = set()
        topics2 = set()
        
        for turn in session1.turns:
            topics1.update(turn.semantic_content)
        for turn in session2.turns:
            topics2.update(turn.semantic_content)
        
        topic_similarity = len(topics1 & topics2) / len(topics1 | topics2) if topics1 | topics2 else 0
        
        # Pattern similarity
        patterns1 = set(self._identify_interaction_patterns(session1))
        patterns2 = set(self._identify_interaction_patterns(session2))
        
        pattern_similarity = len(patterns1 & patterns2) / len(patterns1 | patterns2) if patterns1 | patterns2 else 0
        
        # Length similarity
        length_ratio = min(len(session1.turns), len(session2.turns)) / max(len(session1.turns), len(session2.turns))
        
        return (topic_similarity + pattern_similarity + length_ratio) / 3
    
    def _generate_metric_suggestion(self, metric: DialogueMetricType,
                                  score: float,
                                  session: DialogueSession) -> Optional[Dict[str, Any]]:
        """Generate suggestion for specific metric"""
        suggestions = {
            DialogueMetricType.COHERENCE: {
                "issue": "Low coherence between turns",
                "suggestion": "Use more transitional phrases and reference previous points",
                "priority": "high"
            },
            DialogueMetricType.RELEVANCE: {
                "issue": "Drifting from main topic",
                "suggestion": "Regularly connect back to the central theme",
                "priority": "high"
            },
            DialogueMetricType.ENGAGEMENT: {
                "issue": "Low participant engagement",
                "suggestion": "Ask more open-ended questions and encourage elaboration",
                "priority": "medium"
            },
            DialogueMetricType.DEPTH: {
                "issue": "Superficial treatment of topics",
                "suggestion": "Explore underlying principles and implications",
                "priority": "medium"
            },
            DialogueMetricType.CLARITY: {
                "issue": "Unclear or ambiguous communication",
                "suggestion": "Use specific examples and define key terms",
                "priority": "high"
            }
        }
        
        if metric in suggestions and score < 0.6:
            return suggestions[metric]
        
        return None
    
    def _generate_pattern_suggestions(self, patterns: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """Generate suggestions based on patterns"""
        suggestions = []
        
        # Count pattern types
        pattern_counts = defaultdict(int)
        for pattern in patterns:
            pattern_counts[pattern["type"]] += 1
        
        # Too many unanswered questions
        qa_patterns = [p for p in patterns if p["type"] == "question_answer"]
        if qa_patterns:
            avg_delay = sum(p["delay"] for p in qa_patterns) / len(qa_patterns)
            if avg_delay > 2:
                suggestions.append({
                    "issue": "Questions often go unanswered or have delayed responses",
                    "suggestion": "Address questions more directly and promptly",
                    "priority": "high"
                })
        
        # Unresolved arguments
        arg_patterns = [p for p in patterns if p["type"] == "argument"]
        if arg_patterns:
            unresolved = sum(1 for p in arg_patterns if not p["resolved"])
            if unresolved > len(arg_patterns) / 2:
                suggestions.append({
                    "issue": "Many arguments remain unresolved",
                    "suggestion": "Work towards finding common ground or acknowledging valid points",
                    "priority": "medium"
                })
        
        return suggestions
    
    def _assess_argument_quality(self, session: DialogueSession) -> float:
        """Assess quality of arguments in dialogue"""
        argument_indicators = [
            "because", "therefore", "evidence", "proves",
            "demonstrates", "shows", "indicates", "suggests"
        ]
        
        argument_turns = 0
        for turn in session.turns:
            if any(ind in turn.content.lower() for ind in argument_indicators):
                argument_turns += 1
        
        return argument_turns / len(session.turns) if session.turns else 0
    
    def _assess_logical_consistency(self, session: DialogueSession) -> float:
        """Assess logical consistency across dialogue"""
        # Track stated positions for each speaker and look for contradictions
        positions: Dict[str, List[str]] = defaultdict(list)
        contradictions = 0
        total_positions = 0

        for turn in session.turns:
            content_lower = turn.content.lower()

            # Identify position statements in a very simple way
            if "believe" in content_lower or "think" in content_lower:
                total_positions += 1

                # Compare against previous positions from the same speaker
                for prev in positions[turn.speaker]:
                    if self._are_contradictory(prev, content_lower):
                        contradictions += 1
                        break

                positions[turn.speaker].append(content_lower)

        # Store tracking information in session context for inspection
        session.context["positions"] = positions
        session.context["contradictions"] = contradictions

        if total_positions == 0:
            return 1.0

        # Consistency decreases with more contradictions
        return max(0.0, 1.0 - contradictions / total_positions)

    def _are_contradictory(self, statement1: str, statement2: str) -> bool:
        """Simple check for contradictory statements"""
        s1 = statement1.lower()
        s2 = statement2.lower()

        # Explicit negation pattern
        if " not " in s1 and s1.replace(" not ", " ").strip() == s2.strip():
            return True
        if " not " in s2 and s2.replace(" not ", " ").strip() == s1.strip():
            return True

        opposites = [
            ("true", "false"),
            ("exist", "not exist"),
            ("possible", "impossible"),
            ("necessary", "contingent"),
            ("physical", "non-physical"),
            ("material", "immaterial"),
        ]

        for a, b in opposites:
            if a in s1 and b in s2:
                return True
            if b in s1 and a in s2:
                return True

        return False
    
    def _assess_conceptual_precision(self, session: DialogueSession) -> float:
        """Assess precision of concept usage"""
        # Check for definition and clarification
        precision_indicators = [
            "define", "specifically", "precisely", "exactly",
            "in other words", "that is", "namely"
        ]
        
        precision_turns = 0
        for turn in session.turns:
            if any(ind in turn.content.lower() for ind in precision_indicators):
                precision_turns += 1
        
        return min(1.0, precision_turns / (len(session.turns) / 5))  # Expect precision every 5 turns
    
    def _assess_critical_thinking(self, session: DialogueSession) -> float:
        """Assess critical thinking indicators"""
        critical_indicators = [
            "however", "although", "on the other hand",
            "consider", "examine", "analyze", "evaluate",
            "question", "challenge", "alternative"
        ]
        
        critical_turns = 0
        for turn in session.turns:
            indicator_count = sum(1 for ind in critical_indicators if ind in turn.content.lower())
            if indicator_count > 0:
                critical_turns += 1
        
        return critical_turns / len(session.turns) if session.turns else 0
    
    def _calculate_emotional_alignment(self, session: DialogueSession) -> float:
        """Calculate emotional alignment between participants"""
        if len(session.participants) < 2:
            return 1.0
        
        # Group turns by speaker
        speaker_emotions = defaultdict(list)
        for turn in session.turns:
            speaker_emotions[turn.speaker].append(turn.emotional_tone)
        
        # Calculate correlation between speakers
        speakers = list(speaker_emotions.keys())
        if len(speakers) >= 2:
            emotions1 = speaker_emotions[speakers[0]]
            emotions2 = speaker_emotions[speakers[1]]
            
            # Simple correlation
            min_len = min(len(emotions1), len(emotions2))
            if min_len > 0:
                diffs = [abs(emotions1[i] - emotions2[i]) for i in range(min_len)]
                avg_diff = sum(diffs) / len(diffs)
                return 1.0 - avg_diff
        
        return 0.5
    
    def _detect_empathy_indicators(self, session: DialogueSession) -> float:
        """Detect empathy indicators in dialogue"""
        empathy_phrases = [
            "i understand", "i see", "that must be",
            "i can imagine", "i appreciate", "thank you for sharing",
            "i hear you", "that makes sense", "i relate"
        ]
        
        empathy_count = 0
        for turn in session.turns:
            content_lower = turn.content.lower()
            if any(phrase in content_lower for phrase in empathy_phrases):
                empathy_count += 1
        
        return min(1.0, empathy_count / (len(session.turns) / 4))  # Expect empathy every 4 turns
    
    def _calculate_emotional_depth(self, session: DialogueSession) -> float:
        """Calculate emotional depth of dialogue"""
        emotion_words = [
            "feel", "felt", "feeling", "emotion", "emotional",
            "happy", "sad", "angry", "afraid", "excited",
            "worried", "anxious", "grateful", "disappointed"
        ]
        
        emotional_turns = 0
        total_emotion_words = 0
        
        for turn in session.turns:
            words = turn.content.lower().split()
            emotion_count = sum(1 for word in words if word in emotion_words)
            if emotion_count > 0:
                emotional_turns += 1
                total_emotion_words += emotion_count
        
        # Combine presence and intensity
        presence = emotional_turns / len(session.turns) if session.turns else 0
        intensity = min(1.0, total_emotion_words / (len(session.turns) * 2))  # 2 emotion words per turn = max
        
        return (presence + intensity) / 2
    
    def _calculate_turn_taking_balance(self, session: DialogueSession) -> float:
        """Calculate balance in turn-taking"""
        if len(session.participants) < 2:
            return 0
        
        # Count consecutive turns by same speaker
        consecutive_counts = []
        current_speaker = session.turns[0].speaker
        current_count = 1
        
        for i in range(1, len(session.turns)):
            if session.turns[i].speaker == current_speaker:
                current_count += 1
            else:
                consecutive_counts.append(current_count)
                current_speaker = session.turns[i].speaker
                current_count = 1
        
        consecutive_counts.append(current_count)
        
        # Good balance = mostly 1-2 consecutive turns
        good_counts = sum(1 for c in consecutive_counts if c <= 2)
        
        return good_counts / len(consecutive_counts)
    
    def _analyze_response_patterns(self, session: DialogueSession) -> Dict[str, Any]:
        """Analyze patterns in responses"""
        response_times = []
        response_lengths = []
        
        for i in range(1, len(session.turns)):
            # Time to respond
            time_diff = session.turns[i].timestamp - session.turns[i-1].timestamp
            response_times.append(time_diff)
            
            # Length of response
            response_lengths.append(len(session.turns[i].content.split()))
        
        return {
            "avg_response_time": sum(response_times) / len(response_times) if response_times else 0,
            "avg_response_length": sum(response_lengths) / len(response_lengths) if response_lengths else 0,
            "response_variability": self._calculate_variance(response_lengths)
        }
    
    def _analyze_initiative_distribution(self, session: DialogueSession) -> Dict[str, Any]:
        """Analyze who takes initiative in dialogue"""
        initiative_counts = defaultdict(int)
        
        for turn in session.turns:
            if turn.turn_type in [TurnType.QUESTION, TurnType.CHALLENGE, TurnType.STATEMENT]:
                initiative_counts[turn.speaker] += 1
        
        total_initiatives = sum(initiative_counts.values())
        
        return {
            "distribution": {k: v/total_initiatives for k, v in initiative_counts.items()} if total_initiatives > 0 else {},
            "balance": self._calculate_balance(list(initiative_counts.values()))
        }
    
    def _calculate_balance(self, values: List[int]) -> float:
        """Calculate balance in distribution"""
        if not values or len(values) < 2:
            return 0
        
        total = sum(values)
        if total == 0:
            return 0
        
        # Perfect balance = equal distribution
        expected = total / len(values)
        deviations = [abs(v - expected) for v in values]
        avg_deviation = sum(deviations) / len(deviations)
        
        # Normalize
        max_deviation = expected  # Maximum possible deviation
        
        return 1.0 - (avg_deviation / max_deviation) if max_deviation > 0 else 0
