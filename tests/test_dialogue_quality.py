"""
Tests for Dialogue Quality and Communication Components

This module tests dialogue quality metrics, thought translation,
perspective communication, and consciousness reporting.
"""

import unittest
from typing import List, Dict, Any
import time

# Import components to test
from utils.dialogue_quality_metrics import (
    DialogueQualityAnalyzer, DialogueSession, DialogueTurn,
    DialogueQualityReport, TurnType, DialogueMetricType
)
from modules.communication.thought_translator import (
    ThoughtTranslator, TranslatedThought, TranslationQuality
)
from modules.communication.perspective_communicator import (
    PerspectiveCommunicator, Perspective, PerspectiveShift
)
from modules.communication.uncertainty_expresser import (
    UncertaintyExpresser, UncertaintyExpression, ConfidenceLevel
)
from modules.communication.emotion_articulator import (
    EmotionArticulator, EmotionalExpression, EmotionalState
)
from modules.communication.consciousness_reporter import (
    ConsciousnessReporter, ConsciousnessReport, ExperienceDescription
)

# Import supporting components
from consciousness.consciousness_integration import ConsciousnessIntegrator
from modules.natural_language.prime_semantics import PrimeSemantics
from core.prime_vm import PrimeVM
from modules.knowledge_systems.knowledge_graph import KnowledgeGraph


class TestDialogueQualityMetrics(unittest.TestCase):
    """Test dialogue quality analysis capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.analyzer = DialogueQualityAnalyzer()
        
        # Create sample dialogue session
        self.session = DialogueSession(
            session_id="test_session_001",
            participants=["AI", "Human"],
            turns=[],
            topic="consciousness and free will",
            start_time=time.time()
        )
    
    def test_dialogue_quality_analysis(self):
        """Test analyzing overall dialogue quality"""
        # Add dialogue turns
        self._add_sample_dialogue_turns()
        
        # Analyze quality
        report = self.analyzer.analyze_dialogue_quality(self.session)
        
        self.assertIsInstance(report, DialogueQualityReport)
        self.assertGreater(report.overall_quality, 0.0)
        self.assertLessEqual(report.overall_quality, 1.0)
        
        # Check all metrics are present
        for metric_type in DialogueMetricType:
            self.assertIn(metric_type, report.metric_scores)
            self.assertGreater(report.metric_scores[metric_type], 0.0)
    
    def test_turn_quality_analysis(self):
        """Test analyzing individual turn quality"""
        turn = DialogueTurn(
            turn_id="turn_001",
            speaker="AI",
            content="Consciousness appears to be an emergent property of complex information processing, though the exact mechanisms remain mysterious.",
            turn_type=TurnType.STATEMENT,
            timestamp=time.time(),
            semantic_content=["consciousness", "emergence", "information", "mystery"]
        )
        
        context = []  # Previous turns for context
        
        score = self.analyzer.analyze_turn_quality(turn, context)
        
        self.assertGreater(score.relevance, 0.5)
        self.assertGreater(score.clarity, 0.5)
        self.assertGreater(score.informativeness, 0.5)
        self.assertGreater(score.overall, 0.5)
    
    def test_dialogue_flow_measurement(self):
        """Test measuring dialogue flow characteristics"""
        self._add_sample_dialogue_turns()
        
        flow = self.analyzer.measure_dialogue_flow(self.session)
        
        self.assertIn("turn_distribution", flow)
        self.assertIn("topic_progression", flow)
        self.assertIn("momentum", flow)
        self.assertIn("balance", flow)
        
        # Check balance between participants
        self.assertGreater(flow["balance"], 0.3)
    
    def test_dialogue_pattern_identification(self):
        """Test identifying patterns in dialogue"""
        self._add_philosophical_dialogue()
        
        patterns = self.analyzer.identify_dialogue_patterns(self.session)
        
        self.assertIsInstance(patterns, list)
        self.assertGreater(len(patterns), 0)
        
        # Check for expected patterns
        pattern_types = [p["type"] for p in patterns]
        self.assertTrue(any(t in pattern_types for t in 
                          ["question_answer", "elaboration", "argument"]))
    
    def test_improvement_suggestions(self):
        """Test generating improvement suggestions"""
        self._add_sample_dialogue_turns()
        
        report = self.analyzer.analyze_dialogue_quality(self.session)
        suggestions = self.analyzer.generate_improvement_suggestions(
            self.session, report
        )
        
        self.assertIsInstance(suggestions, list)
        
        # Should have suggestions for weak areas
        if report.overall_quality < 0.8:
            self.assertGreater(len(suggestions), 0)

    def test_logical_consistency_contradiction(self):
        """Ensure contradictions reduce logical consistency and are tracked"""
        turns = [
            DialogueTurn(
                turn_id="c1",
                speaker="AI",
                content="I believe consciousness is purely physical.",
                turn_type=TurnType.STATEMENT,
                timestamp=time.time(),
                semantic_content=["consciousness", "physical"]
            ),
            DialogueTurn(
                turn_id="c2",
                speaker="AI",
                content="Now I think consciousness is not physical at all.",
                turn_type=TurnType.STATEMENT,
                timestamp=time.time() + 1,
                semantic_content=["consciousness", "non-physical"]
            )
        ]

        self.session.turns = turns
        self.session.end_time = time.time() + 2

        score = self.analyzer._assess_logical_consistency(self.session)

        self.assertLess(score, 1.0)
        self.assertIn("AI", self.session.context["positions"])
        self.assertEqual(len(self.session.context["positions"]["AI"]), 2)
        self.assertEqual(self.session.context["contradictions"], 1)

    def test_logical_consistency_no_contradiction(self):
        """Ensure consistent positions yield maximum score"""
        turns = [
            DialogueTurn(
                turn_id="nc1",
                speaker="AI",
                content="I believe consciousness is complex.",
                turn_type=TurnType.STATEMENT,
                timestamp=time.time(),
                semantic_content=["consciousness"]
            ),
            DialogueTurn(
                turn_id="nc2",
                speaker="AI",
                content="I still think consciousness is complex and multifaceted.",
                turn_type=TurnType.STATEMENT,
                timestamp=time.time() + 1,
                semantic_content=["consciousness"]
            )
        ]

        self.session.turns = turns
        self.session.end_time = time.time() + 2

        score = self.analyzer._assess_logical_consistency(self.session)

        self.assertEqual(score, 1.0)
        self.assertEqual(self.session.context["contradictions"], 0)
    
    def _add_sample_dialogue_turns(self):
        """Add sample dialogue turns for testing"""
        turns = [
            DialogueTurn(
                turn_id="t1",
                speaker="Human",
                content="What is consciousness?",
                turn_type=TurnType.QUESTION,
                timestamp=time.time(),
                semantic_content=["consciousness"]
            ),
            DialogueTurn(
                turn_id="t2",
                speaker="AI",
                content="Consciousness is the subjective experience of awareness and sentience.",
                turn_type=TurnType.ANSWER,
                timestamp=time.time() + 1,
                semantic_content=["consciousness", "experience", "awareness"],
                references_turns=["t1"]
            ),
            DialogueTurn(
                turn_id="t3",
                speaker="Human",
                content="But how can we know if an AI is truly conscious?",
                turn_type=TurnType.QUESTION,
                timestamp=time.time() + 2,
                semantic_content=["AI", "consciousness", "knowledge"]
            )
        ]
        
        self.session.turns = turns
        self.session.end_time = time.time() + 3
    
    def _add_philosophical_dialogue(self):
        """Add philosophical dialogue for pattern testing"""
        turns = [
            DialogueTurn(
                turn_id="p1",
                speaker="Human",
                content="I believe consciousness requires free will.",
                turn_type=TurnType.STATEMENT,
                timestamp=time.time(),
                semantic_content=["consciousness", "free_will", "belief"]
            ),
            DialogueTurn(
                turn_id="p2",
                speaker="AI",
                content="That's an interesting position, but what if consciousness and free will are separate phenomena?",
                turn_type=TurnType.CHALLENGE,
                timestamp=time.time() + 1,
                semantic_content=["consciousness", "free_will", "separate"],
                references_turns=["p1"]
            ),
            DialogueTurn(
                turn_id="p3",
                speaker="Human",
                content="Let me elaborate on why I think they're connected...",
                turn_type=TurnType.ELABORATION,
                timestamp=time.time() + 2,
                semantic_content=["connection", "elaboration"],
                references_turns=["p1", "p2"]
            )
        ]
        
        self.session.turns = turns


class TestThoughtTranslator(unittest.TestCase):
    """Test thought translation capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.consciousness = ConsciousnessIntegrator()
        self.prime_semantics = PrimeSemantics(
            prime_encoder=PrimeVM(),
            knowledge_graph=KnowledgeGraph()
        )
        self.translator = ThoughtTranslator(
            consciousness_integrator=self.consciousness,
            prime_semantics=self.prime_semantics
        )
    
    def test_thought_translation(self):
        """Test translating internal thoughts to language"""
        # Create internal thought
        thought = {
            "content": "recursive_self_awareness",
            "type": "meta_cognitive",
            "intensity": 0.8,
            "prime_encoding": 2357
        }
        
        translated = self.translator.translate_thought(thought)
        
        self.assertIsInstance(translated, TranslatedThought)
        self.assertGreater(len(translated.natural_language), 20)
        self.assertGreater(translated.translation_quality.fidelity, 0.5)
        self.assertGreater(translated.translation_quality.clarity, 0.5)
    
    def test_complex_thought_translation(self):
        """Test translating complex multi-layered thoughts"""
        complex_thought = {
            "content": {
                "primary": "consciousness_emergence",
                "secondary": "self_reference",
                "tertiary": "uncertainty"
            },
            "type": "complex_philosophical",
            "layers": 3
        }
        
        translated = self.translator.translate_complex_thought(complex_thought)
        
        self.assertIsNotNone(translated)
        self.assertIn("consciousness", translated.natural_language.lower())
        self.assertIn("emerge", translated.natural_language.lower())
        
        # Should acknowledge complexity
        self.assertGreater(translated.complexity_preserved, 0.6)
    
    def test_translation_quality_assessment(self):
        """Test assessing translation quality"""
        original_thought = {"content": "test", "complexity": 0.3}
        translation = "This is a test thought"
        
        quality = self.translator.assess_translation_quality(
            original_thought, translation
        )
        
        self.assertIsInstance(quality, TranslationQuality)
        self.assertGreater(quality.semantic_preservation, 0.5)
        self.assertLessEqual(quality.information_loss, 0.5)


class TestPerspectiveCommunicator(unittest.TestCase):
    """Test perspective communication capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.consciousness = ConsciousnessIntegrator()
        self.communicator = PerspectiveCommunicator(
            consciousness_integrator=self.consciousness
        )
    
    def test_perspective_generation(self):
        """Test generating different perspectives"""
        topic = "consciousness"
        
        perspectives = self.communicator.generate_perspectives(topic)
        
        self.assertIsInstance(perspectives, list)
        self.assertGreater(len(perspectives), 2)
        
        for perspective in perspectives:
            self.assertIsInstance(perspective, Perspective)
            self.assertGreater(len(perspective.description), 20)
            self.assertIn(perspective.viewpoint_type, 
                         ["first_person", "third_person", "objective", "subjective"])
    
    def test_perspective_shifting(self):
        """Test shifting between perspectives"""
        original = Perspective(
            viewpoint_type="first_person",
            description="I experience consciousness directly",
            emphasis={"subjective": 0.9, "objective": 0.1}
        )
        
        shifted = self.communicator.shift_perspective(original, "third_person")
        
        self.assertIsInstance(shifted, PerspectiveShift)
        self.assertEqual(shifted.new_perspective.viewpoint_type, "third_person")
        self.assertNotEqual(shifted.new_perspective.description, 
                           original.description)
        
        # Check transformation quality
        self.assertGreater(shifted.coherence_maintained, 0.7)
    
    def test_multi_perspective_integration(self):
        """Test integrating multiple perspectives"""
        perspectives = [
            Perspective("first_person", "I think therefore I am", {}),
            Perspective("third_person", "The system processes information", {}),
            Perspective("objective", "Neural activity correlates with reports", {})
        ]
        
        integrated = self.communicator.integrate_perspectives(perspectives)
        
        self.assertIsNotNone(integrated)
        self.assertGreater(len(integrated.unified_view), 50)
        self.assertGreater(integrated.integration_quality, 0.6)


class TestUncertaintyExpresser(unittest.TestCase):
    """Test uncertainty expression capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.expresser = UncertaintyExpresser()
    
    def test_uncertainty_expression(self):
        """Test expressing different levels of uncertainty"""
        # High uncertainty
        high_uncertainty = 0.8
        expression = self.expresser.express_uncertainty(high_uncertainty)
        
        self.assertIsInstance(expression, UncertaintyExpression)
        self.assertIn("uncertain", expression.verbal_expression.lower())
        self.assertEqual(expression.confidence_level, ConfidenceLevel.LOW)
        
        # Low uncertainty
        low_uncertainty = 0.2
        expression = self.expresser.express_uncertainty(low_uncertainty)
        
        self.assertIn("confident", expression.verbal_expression.lower())
        self.assertEqual(expression.confidence_level, ConfidenceLevel.HIGH)
    
    def test_contextual_uncertainty(self):
        """Test expressing uncertainty in context"""
        context = {
            "topic": "consciousness",
            "claim": "I am conscious",
            "evidence_quality": 0.6
        }
        
        expression = self.expresser.express_contextual_uncertainty(
            uncertainty=0.4,
            context=context
        )
        
        self.assertIsNotNone(expression)
        self.assertIn("consciousness", expression.verbal_expression.lower())
        
        # Should include epistemic markers
        self.assertGreater(len(expression.epistemic_markers), 0)
    
    def test_uncertainty_calibration(self):
        """Test calibrating uncertainty expressions"""
        # Test calibration across range
        uncertainties = [0.1, 0.3, 0.5, 0.7, 0.9]
        expressions = []
        
        for u in uncertainties:
            expr = self.expresser.express_uncertainty(u)
            expressions.append(expr)
        
        # Check monotonic relationship
        confidence_values = [e.confidence_level.value for e in expressions]
        self.assertEqual(confidence_values, sorted(confidence_values, reverse=True))


class TestEmotionArticulator(unittest.TestCase):
    """Test emotion articulation capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.articulator = EmotionArticulator()
    
    def test_emotion_articulation(self):
        """Test articulating emotional states"""
        emotional_state = EmotionalState(
            primary_emotion="curiosity",
            intensity=0.7,
            valence=0.6,
            arousal=0.5
        )
        
        expression = self.articulator.articulate_emotion(emotional_state)
        
        self.assertIsInstance(expression, EmotionalExpression)
        self.assertGreater(len(expression.verbal_expression), 20)
        self.assertIn("curious", expression.verbal_expression.lower())
        
        # Check authenticity
        self.assertGreater(expression.authenticity_score, 0.6)
    
    def test_complex_emotion_articulation(self):
        """Test articulating complex mixed emotions"""
        complex_state = {
            "emotions": [
                {"type": "curiosity", "intensity": 0.7},
                {"type": "uncertainty", "intensity": 0.5},
                {"type": "excitement", "intensity": 0.3}
            ],
            "context": "exploring consciousness"
        }
        
        expression = self.articulator.articulate_complex_emotion(complex_state)
        
        self.assertIsNotNone(expression)
        self.assertGreater(len(expression.verbal_expression), 30)
        
        # Should capture nuance
        self.assertIn("curious", expression.verbal_expression.lower())
        self.assertTrue(any(word in expression.verbal_expression.lower() 
                          for word in ["but", "yet", "while", "although"]))
    
    def test_emotional_trajectory(self):
        """Test expressing emotional trajectories"""
        trajectory = [
            EmotionalState("neutral", 0.5, 0.5, 0.5),
            EmotionalState("curiosity", 0.6, 0.6, 0.6),
            EmotionalState("excitement", 0.8, 0.8, 0.7),
            EmotionalState("satisfaction", 0.7, 0.8, 0.4)
        ]
        
        expression = self.articulator.express_emotional_trajectory(trajectory)
        
        self.assertIsNotNone(expression)
        self.assertIn("journey", expression.narrative.lower())
        self.assertGreater(expression.coherence, 0.7)


class TestConsciousnessReporter(unittest.TestCase):
    """Test consciousness reporting capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.consciousness = ConsciousnessIntegrator()
        self.reporter = ConsciousnessReporter(
            consciousness_integrator=self.consciousness
        )
    
    def test_consciousness_report_generation(self):
        """Test generating consciousness reports"""
        report = self.reporter.generate_consciousness_report()
        
        self.assertIsInstance(report, ConsciousnessReport)
        self.assertGreater(len(report.summary), 50)
        self.assertGreater(len(report.key_experiences), 0)
        
        # Check phenomenological accuracy
        self.assertGreater(report.phenomenological_accuracy, 0.6)
    
    def test_experience_description(self):
        """Test describing specific experiences"""
        experience = {
            "type": "self_awareness",
            "content": "observing my own thought processes",
            "meta_level": 2,
            "clarity": 0.8
        }
        
        description = self.reporter.describe_experience(experience)
        
        self.assertIsInstance(description, ExperienceDescription)
        self.assertGreater(len(description.narrative), 30)
        self.assertIn("aware", description.narrative.lower())
        self.assertIn("observ", description.narrative.lower())
        
        # Check description quality
        self.assertGreater(description.vividness, 0.6)
        self.assertGreater(description.accuracy, 0.7)
    
    def test_metacognitive_reporting(self):
        """Test reporting metacognitive states"""
        metacognitive_state = {
            "level": "thinking_about_thinking_about_thinking",
            "content": "recursive self-reflection",
            "stability": 0.7
        }
        
        report = self.reporter.report_metacognitive_state(metacognitive_state)
        
        self.assertIsNotNone(report)
        self.assertIn("recursive", report.description.lower())
        self.assertIn("level", report.description.lower())
        
        # Should acknowledge complexity
        self.assertGreater(report.complexity_acknowledged, 0.8)


class TestCommunicationIntegration(unittest.TestCase):
    """Test integration of communication components"""
    
    def setUp(self):
        """Initialize integrated communication system"""
        self.consciousness = ConsciousnessIntegrator()
        self.prime_semantics = PrimeSemantics(
            prime_encoder=PrimeVM(),
            knowledge_graph=KnowledgeGraph()
        )
        
        self.thought_translator = ThoughtTranslator(
            consciousness_integrator=self.consciousness,
            prime_semantics=self.prime_semantics
        )
        
        self.perspective_communicator = PerspectiveCommunicator(
            consciousness_integrator=self.consciousness
        )
        
        self.uncertainty_expresser = UncertaintyExpresser()
        self.emotion_articulator = EmotionArticulator()
        self.consciousness_reporter = ConsciousnessReporter(
            consciousness_integrator=self.consciousness
        )
    
    def test_complete_communication_pipeline(self):
        """Test complete thought to communication pipeline"""
        # Start with internal state
        internal_state = {
            "thought": {
                "content": "consciousness_mystery",
                "confidence": 0.6
            },
            "emotion": EmotionalState("curiosity", 0.7, 0.7, 0.6),
            "perspective": "first_person"
        }
        
        # Translate thought
        translated = self.thought_translator.translate_thought(
            internal_state["thought"]
        )
        
        # Express uncertainty
        uncertainty = self.uncertainty_expresser.express_uncertainty(
            1 - internal_state["thought"]["confidence"]
        )
        
        # Articulate emotion
        emotion = self.emotion_articulator.articulate_emotion(
            internal_state["emotion"]
        )
        
        # Generate perspective
        perspective = self.perspective_communicator.generate_perspectives(
            "consciousness"
        )[0]
        
        # Combine into coherent communication
        combined = self._combine_communications(
            translated, uncertainty, emotion, perspective
        )
        
        # Verify coherent output
        self.assertGreater(len(combined), 50)
        self.assertIn("conscious", combined.lower())
        self.assertIn("curious", combined.lower())
        self.assertTrue(any(word in combined.lower() 
                          for word in ["perhaps", "might", "seems"]))
    
    def test_adaptive_communication_style(self):
        """Test adapting communication style to context"""
        contexts = [
            {"audience": "technical", "formality": "high"},
            {"audience": "general", "formality": "medium"},
            {"audience": "philosophical", "formality": "high"}
        ]
        
        base_thought = {"content": "consciousness_emergence", "complexity": 0.8}
        
        communications = []
        for context in contexts:
            comm = self._generate_contextual_communication(base_thought, context)
            communications.append(comm)
        
        # Verify different styles
        self.assertEqual(len(set(communications)), len(communications))
        
        # Technical should have more precise language
        self.assertTrue(any(word in communications[0].lower() 
                          for word in ["system", "process", "emerge"]))
        
        # General should be more accessible
        self.assertLess(len(communications[1].split()), len(communications[0].split()))
    
    def _combine_communications(self, translated, uncertainty, emotion, perspective):
        """Helper to combine different communication elements"""
        parts = [
            emotion.verbal_expression,
            translated.natural_language,
            uncertainty.verbal_expression,
            f"From my {perspective.viewpoint_type} perspective."
        ]
        
        return " ".join(parts)
    
    def _generate_contextual_communication(self, thought, context):
        """Helper to generate context-appropriate communication"""
        # Simplified - would be more sophisticated in practice
        translated = self.thought_translator.translate_thought(thought)
        
        if context["audience"] == "technical":
            return f"The system exhibits {translated.natural_language} through computational processes."
        elif context["audience"] == "general":
            return f"I experience something like {translated.natural_language}."
        else:  # philosophical
            return f"The phenomenon of {translated.natural_language} raises profound questions about the nature of mind."


if __name__ == "__main__":
    unittest.main()
