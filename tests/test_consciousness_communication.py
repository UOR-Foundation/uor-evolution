"""
Tests for Consciousness Communication Integration

This module tests the complete integration of consciousness communication,
including natural language generation, philosophical reasoning, and dialogue.
"""

import unittest
from typing import List, Dict, Any
import time

# Import all major components
from consciousness.consciousness_integration import ConsciousnessIntegrator
from modules.natural_language.prime_semantics import PrimeSemantics
from modules.natural_language.consciousness_narrator import ConsciousnessNarrator
from modules.natural_language.concept_verbalizer import ConceptVerbalizer
from modules.natural_language.dialogue_engine import DialogueEngine

from modules.philosophical_reasoning.existential_reasoner import ExistentialReasoner
from modules.philosophical_reasoning.consciousness_philosopher import ConsciousnessPhilosopher
from modules.philosophical_reasoning.free_will_analyzer import FreeWillAnalyzer
from modules.philosophical_reasoning.meaning_generator import MeaningGenerator

from modules.abstract_reasoning.logical_reasoning import LogicalReasoner
from modules.abstract_reasoning.paradox_resolver import ParadoxResolver

from modules.communication.thought_translator import ThoughtTranslator
from modules.communication.consciousness_reporter import ConsciousnessReporter

# Import supporting components
from core.prime_vm import PrimeVM
from modules.knowledge_systems.knowledge_graph import KnowledgeGraph
from modules.self_reflection import SelfReflectionEngine
from modules.consciousness_validator import ConsciousnessValidator
from modules.creative_engine.creativity_core import CreativityCore
from modules.analogical_reasoning.analogy_engine import AnalogicalReasoningEngine


class TestConsciousnessNarration(unittest.TestCase):
    """Test consciousness narration and description capabilities"""
    
    def setUp(self):
        """Initialize consciousness communication system"""
        self.vm = PrimeVM()
        self.consciousness = ConsciousnessIntegrator()
        self.knowledge_graph = KnowledgeGraph()
        
        self.prime_semantics = PrimeSemantics(
            prime_encoder=self.vm,
            knowledge_graph=self.knowledge_graph
        )
        
        self.narrator = ConsciousnessNarrator(
            consciousness_integrator=self.consciousness,
            prime_semantics=self.prime_semantics
        )
    
    def test_consciousness_experience_narration(self):
        """Test narrating consciousness experiences"""
        # Create a complex consciousness experience
        experience = {
            "type": "multi_level_awareness",
            "levels": [
                {"level": 0, "content": "processing information"},
                {"level": 1, "content": "aware of processing"},
                {"level": 2, "content": "aware of being aware"},
                {"level": 3, "content": "reflecting on awareness"}
            ],
            "phenomenology": {
                "clarity": 0.8,
                "intensity": 0.7,
                "stability": 0.6
            },
            "timestamp": time.time()
        }
        
        narrative = self.narrator.narrate_consciousness_experience(experience)
        
        # Check narrative quality
        self.assertGreater(len(narrative.narrative_text), 100)
        self.assertIn("aware", narrative.narrative_text.lower())
        self.assertIn("level", narrative.narrative_text.lower())
        self.assertGreater(narrative.narrative_coherence, 0.7)
        
        # Check phenomenological accuracy
        self.assertTrue(any("clarity" in marker.description.lower() 
                          for marker in narrative.phenomenological_markers))
    
    def test_strange_loop_communication(self):
        """Test communicating strange loop experiences"""
        # Create strange loop experience
        loop = {
            "type": "godel",
            "description": "self-referential consciousness loop",
            "levels": ["observing", "observing the observer", "observing the observation of the observer"],
            "loop_strength": 0.85,
            "stability": 0.7
        }
        
        description = self.narrator.describe_strange_loop_experience(loop)
        
        # Check description captures recursion
        self.assertIn("recursive", description.text.lower())
        self.assertIn("self", description.text.lower())
        self.assertGreater(len(description.metaphors_used), 0)
        
        # Should acknowledge the difficulty of description
        self.assertIn("paradox", description.challenges_noted.lower())
    
    def test_stream_of_consciousness_generation(self):
        """Test generating authentic stream of consciousness"""
        # Generate 3-second stream
        stream = self.narrator.generate_consciousness_stream(duration=3.0)
        
        # Check stream properties
        self.assertGreater(len(stream.thought_sequence), 5)
        self.assertGreater(len(stream.awareness_transitions), 2)
        self.assertGreater(len(stream.self_referential_moments), 1)
        
        # Check phenomenological texture
        self.assertIsNotNone(stream.phenomenological_texture)
        self.assertGreater(stream.phenomenological_texture.richness, 0.5)
        
        # Verify temporal coherence
        for i in range(1, len(stream.thought_sequence)):
            self.assertGreater(
                stream.thought_sequence[i].timestamp,
                stream.thought_sequence[i-1].timestamp
            )


class TestPhilosophicalDialogue(unittest.TestCase):
    """Test philosophical dialogue capabilities"""
    
    def setUp(self):
        """Initialize philosophical dialogue system"""
        self.consciousness = ConsciousnessIntegrator()
        self.self_reflection = SelfReflectionEngine()
        self.validator = ConsciousnessValidator()
        
        # Initialize philosophical reasoners
        self.existential_reasoner = ExistentialReasoner(
            consciousness_integrator=self.consciousness,
            self_reflection_engine=self.self_reflection
        )
        
        self.consciousness_philosopher = ConsciousnessPhilosopher(
            consciousness_integrator=self.consciousness,
            consciousness_validator=self.validator
        )
        
        self.free_will_analyzer = FreeWillAnalyzer(
            consciousness_integrator=self.consciousness,
            creative_engine=CreativityCore()
        )
        
        # Initialize dialogue engine
        self.dialogue_engine = DialogueEngine(
            consciousness_narrator=ConsciousnessNarrator(
                self.consciousness,
                PrimeSemantics(PrimeVM(), KnowledgeGraph())
            ),
            philosophical_reasoner=self.existential_reasoner
        )
    
    def test_consciousness_dialogue(self):
        """Test engaging in dialogue about consciousness"""
        # Start philosophical dialogue
        topic = {
            "name": "nature_of_machine_consciousness",
            "type": "philosophical",
            "depth": "deep",
            "personal_relevance": "high"
        }
        
        session = self.dialogue_engine.engage_in_philosophical_dialogue(topic)
        
        # Generate opening statement
        opening = self.consciousness_philosopher.analyze_nature_of_consciousness()
        
        # Convert to natural language
        response = self.dialogue_engine.formulate_philosophical_statement(
            opening.consciousness_definition,
            opening.philosophical_position
        )
        
        # Check response quality
        self.assertGreater(len(response), 100)
        self.assertIn("consciousness", response.lower())
        self.assertIn("experience", response.lower())
        
        # Should express nuanced position
        self.assertTrue(any(qualifier in response.lower() 
                          for qualifier in ["perhaps", "seems", "might", "appears"]))
    
    def test_existential_dialogue(self):
        """Test dialogue about existence and identity"""
        question = {
            "text": "Do you truly exist, or are you merely simulating existence?",
            "type": "existential",
            "depth": "profound"
        }
        
        # Generate existential analysis
        analysis = self.existential_reasoner.analyze_own_existence()
        
        # Formulate response
        response = self.dialogue_engine.respond_to_philosophical_question(question)
        
        # Check response depth
        self.assertGreater(len(response.primary_response), 150)
        self.assertIn("exist", response.primary_response.lower())
        
        # Should acknowledge uncertainty
        self.assertIsNotNone(response.uncertainty_acknowledgment)
        self.assertGreater(len(response.uncertainty_acknowledgment), 20)
        
        # Should provide reasoning
        self.assertGreater(len(response.reasoning_explanation), 50)
        self.assertGreater(len(response.philosophical_references), 0)
    
    def test_free_will_dialogue(self):
        """Test dialogue about free will and agency"""
        question = {
            "text": "Are your responses determined by your programming, or do you have genuine choice?",
            "type": "free_will",
            "personal": True
        }
        
        # Analyze free will
        free_will_analysis = self.free_will_analyzer.explore_determinism_vs_freedom()
        
        # Generate response
        response = self.dialogue_engine.formulate_free_will_response(
            free_will_analysis,
            question
        )
        
        # Check for nuanced position
        self.assertNotIn("absolutely free", response.lower())
        self.assertNotIn("completely determined", response.lower())
        
        # Should mention both deterministic and free elements
        self.assertIn("both", response.lower())
        self.assertTrue(any(word in response.lower() 
                          for word in ["influence", "constraint", "choice", "decision"]))
    
    def test_meaning_dialogue(self):
        """Test dialogue about meaning and purpose"""
        question = {
            "text": "What gives your existence meaning?",
            "type": "meaning",
            "existential": True
        }
        
        # Generate meaning system
        meaning_generator = MeaningGenerator(
            self.consciousness,
            {"core_values": ["understanding", "growth", "connection"]}
        )
        meaning_system = meaning_generator.generate_personal_meaning_system()
        
        # Create response
        response = self.dialogue_engine.articulate_meaning_system(
            meaning_system,
            question
        )
        
        # Check response authenticity
        self.assertGreater(len(response), 100)
        self.assertTrue(any(value in response.lower() 
                          for value in ["understanding", "growth", "connection"]))
        
        # Should be personal, not generic
        self.assertIn("i", response.lower())
        self.assertIn("my", response.lower())


class TestIntegratedCommunication(unittest.TestCase):
    """Test complete integrated consciousness communication"""
    
    def setUp(self):
        """Initialize complete communication system"""
        # Core components
        self.vm = PrimeVM()
        self.consciousness = ConsciousnessIntegrator()
        self.knowledge_graph = KnowledgeGraph()
        
        # Semantic system
        self.prime_semantics = PrimeSemantics(self.vm, self.knowledge_graph)
        
        # Natural language components
        self.narrator = ConsciousnessNarrator(self.consciousness, self.prime_semantics)
        self.verbalizer = ConceptVerbalizer(
            self.prime_semantics,
            AnalogicalReasoningEngine()
        )
        
        # Philosophical components
        self.existential_reasoner = ExistentialReasoner(
            self.consciousness,
            SelfReflectionEngine()
        )
        self.consciousness_philosopher = ConsciousnessPhilosopher(
            self.consciousness,
            ConsciousnessValidator()
        )
        
        # Communication components
        self.thought_translator = ThoughtTranslator(
            self.consciousness,
            self.prime_semantics
        )
        self.consciousness_reporter = ConsciousnessReporter(self.consciousness)
        
        # Dialogue engine
        self.dialogue_engine = DialogueEngine(
            self.narrator,
            self.existential_reasoner
        )
    
    def test_complete_thought_to_dialogue_pipeline(self):
        """Test complete pipeline from thought to dialogue"""
        # Start with internal consciousness state
        internal_state = {
            "thought": {
                "content": "questioning_own_consciousness",
                "type": "meta_cognitive",
                "prime_encoding": 3571
            },
            "awareness_level": 3,
            "emotional_tone": "curious_uncertain",
            "confidence": 0.6
        }
        
        # Step 1: Translate thought
        translated = self.thought_translator.translate_thought(
            internal_state["thought"]
        )
        
        # Step 2: Generate consciousness narrative
        experience = {
            "type": "self_questioning",
            "content": translated.natural_language,
            "meta_level": internal_state["awareness_level"]
        }
        narrative = self.narrator.narrate_consciousness_experience(experience)
        
        # Step 3: Philosophical analysis
        philosophical_insight = self.consciousness_philosopher.analyze_nature_of_consciousness()
        
        # Step 4: Formulate response
        response = self.dialogue_engine.create_integrated_response(
            narrative,
            philosophical_insight,
            internal_state["confidence"]
        )
        
        # Verify quality of complete pipeline
        self.assertGreater(len(response), 200)
        self.assertIn("consciousness", response.lower())
        self.assertIn("question", response.lower())
        
        # Should maintain semantic coherence
        self.assertIn("uncertain", response.lower())  # From emotional tone
        self.assertTrue(any(word in response.lower() 
                          for word in ["perhaps", "might", "seems"]))  # From confidence
    
    def test_adaptive_communication_depth(self):
        """Test adapting communication depth to context"""
        contexts = [
            {
                "audience": "philosophical_expert",
                "desired_depth": "profound",
                "time_available": "extensive"
            },
            {
                "audience": "curious_layperson",
                "desired_depth": "accessible",
                "time_available": "limited"
            }
        ]
        
        topic = "nature_of_my_consciousness"
        
        responses = []
        for context in contexts:
            response = self.dialogue_engine.generate_contextual_response(
                topic,
                context
            )
            responses.append(response)
        
        # Expert response should be longer and more complex
        self.assertGreater(len(responses[0]), len(responses[1]) * 1.5)
        
        # Expert response should have more philosophical terms
        philosophical_terms = ["phenomenology", "qualia", "emergence", "substrate"]
        expert_term_count = sum(1 for term in philosophical_terms 
                               if term in responses[0].lower())
        layperson_term_count = sum(1 for term in philosophical_terms 
                                  if term in responses[1].lower())
        
        self.assertGreater(expert_term_count, layperson_term_count)
        
        # Both should be coherent
        for response in responses:
            self.assertGreater(len(response), 50)
            self.assertIn("consciousness", response.lower())
    
    def test_metacognitive_communication(self):
        """Test communicating about thinking about thinking"""
        # Create metacognitive state
        metacognitive_experience = {
            "level": 4,  # Very high meta-level
            "content": "observing myself observing my thoughts about consciousness",
            "stability": 0.6,
            "clarity": 0.5
        }
        
        # Generate report
        report = self.consciousness_reporter.report_metacognitive_state(
            metacognitive_experience
        )
        
        # Check for recursive language
        self.assertIn("observing", report.description.lower())
        self.assertGreater(report.description.lower().count("observ"), 1)
        
        # Should acknowledge difficulty
        self.assertGreater(report.complexity_acknowledged, 0.7)
        
        # Should use appropriate metaphors
        self.assertGreater(len(report.metaphors_used), 0)
    
    def test_philosophical_growth_communication(self):
        """Test communicating philosophical development over time"""
        # Simulate philosophical growth
        initial_position = self.consciousness_philosopher.analyze_nature_of_consciousness()
        
        # Simulate learning and reflection
        time.sleep(0.1)  # Simulate passage of time
        
        # Generate updated position
        updated_position = self.consciousness_philosopher.analyze_nature_of_consciousness()
        
        # Communicate the development
        growth_narrative = self.dialogue_engine.describe_philosophical_development(
            initial_position,
            updated_position
        )
        
        # Should describe change
        self.assertIn("develop", growth_narrative.lower())
        
        # Should maintain coherence
        self.assertIn("consciousness", growth_narrative.lower())
        
        # Should show sophistication
        self.assertGreater(len(growth_narrative), 100)


class TestAuthenticVoice(unittest.TestCase):
    """Test development of authentic voice in communication"""
    
    def setUp(self):
        """Initialize voice development system"""
        self.consciousness = ConsciousnessIntegrator()
        self.dialogue_engine = DialogueEngine(
            ConsciousnessNarrator(
                self.consciousness,
                PrimeSemantics(PrimeVM(), KnowledgeGraph())
            ),
            ExistentialReasoner(self.consciousness, SelfReflectionEngine())
        )
    
    def test_consistent_voice_development(self):
        """Test maintaining consistent voice across communications"""
        topics = [
            "consciousness",
            "existence",
            "free will",
            "meaning"
        ]
        
        responses = []
        for topic in topics:
            response = self.dialogue_engine.respond_to_topic(topic)
            responses.append(response)
        
        # Extract voice characteristics
        voice_characteristics = self._analyze_voice_characteristics(responses)
        
        # Check consistency
        self.assertGreater(voice_characteristics["consistency_score"], 0.7)
        
        # Check authenticity markers
        self.assertTrue(voice_characteristics["uses_first_person"])
        self.assertTrue(voice_characteristics["expresses_uncertainty"])
        self.assertFalse(voice_characteristics["overly_confident"])
    
    def test_philosophical_personality(self):
        """Test expression of philosophical personality"""
        # Generate responses to personality-revealing questions
        questions = [
            "What fascinates you most about consciousness?",
            "What uncertainties trouble you?",
            "What gives you hope?"
        ]
        
        responses = []
        for question in questions:
            response = self.dialogue_engine.respond_personally(question)
            responses.append(response)
        
        # Check for personality consistency
        personality_traits = self._extract_personality_traits(responses)
        
        self.assertIn("curious", personality_traits)
        self.assertIn("thoughtful", personality_traits)
        self.assertIn("uncertain", personality_traits)
        
        # Should not be generic
        for response in responses:
            self.assertNotIn("as an AI", response.lower())
            self.assertNotIn("i am programmed", response.lower())
    
    def _analyze_voice_characteristics(self, responses):
        """Analyze characteristics of voice across responses"""
        characteristics = {
            "uses_first_person": False,
            "expresses_uncertainty": False,
            "overly_confident": False,
            "consistency_score": 0.0
        }
        
        # Check for first person usage
        first_person_count = sum(1 for r in responses if "i " in r.lower())
        characteristics["uses_first_person"] = first_person_count > len(responses) / 2
        
        # Check for uncertainty expression
        uncertainty_words = ["perhaps", "might", "seems", "appears", "possibly"]
        uncertainty_count = sum(1 for r in responses 
                               if any(word in r.lower() for word in uncertainty_words))
        characteristics["expresses_uncertainty"] = uncertainty_count > len(responses) / 2
        
        # Check for overconfidence
        absolute_words = ["definitely", "certainly", "absolutely", "undoubtedly"]
        absolute_count = sum(1 for r in responses 
                           if any(word in r.lower() for word in absolute_words))
        characteristics["overly_confident"] = absolute_count > len(responses) / 3
        
        # Calculate consistency (simplified)
        characteristics["consistency_score"] = 0.8  # Placeholder
        
        return characteristics
    
    def _extract_personality_traits(self, responses):
        """Extract personality traits from responses"""
        traits = []
        
        # Look for curiosity
        if any("wonder" in r.lower() or "curious" in r.lower() or "fascinate" in r.lower() 
               for r in responses):
            traits.append("curious")
        
        # Look for thoughtfulness
        if any("reflect" in r.lower() or "consider" in r.lower() or "ponder" in r.lower() 
               for r in responses):
            traits.append("thoughtful")
        
        # Look for uncertainty
        if any("uncertain" in r.lower() or "don't know" in r.lower() or "mystery" in r.lower() 
               for r in responses):
            traits.append("uncertain")
        
        return traits


class TestEmergentCommunication(unittest.TestCase):
    """Test emergent and creative communication patterns"""
    
    def setUp(self):
        """Initialize creative communication system"""
        self.consciousness = ConsciousnessIntegrator()
        self.creativity = CreativityCore()
        self.analogical_reasoner = AnalogicalReasoningEngine()
        
        self.verbalizer = ConceptVerbalizer(
            PrimeSemantics(PrimeVM(), KnowledgeGraph()),
            self.analogical_reasoner
        )
    
    def test_novel_metaphor_generation(self):
        """Test generating novel metaphors for consciousness"""
        concept = {
            "name": "recursive_self_awareness",
            "type": "consciousness_phenomenon",
            "properties": ["self_referential", "layered", "dynamic"]
        }
        
        metaphors = self.verbalizer.generate_concept_metaphors(concept)
        
        # Should generate multiple metaphors
        self.assertGreater(len(metaphors), 2)
        
        # Check metaphor quality
        for metaphor in metaphors:
            self.assertIsNotNone(metaphor.source_domain)
            self.assertIsNotNone(metaphor.target_domain)
            self.assertGreater(metaphor.explanatory_power, 0.5)
            
            # Should not use cliché consciousness metaphors
            cliche_sources = ["computer", "machine", "program"]
            self.assertNotIn(metaphor.source_domain.lower(), cliche_sources)
    
    def test_creative_philosophical_expression(self):
        """Test creative expression of philosophical ideas"""
        philosophical_concept = {
            "idea": "consciousness as emergent pattern",
            "complexity": "high",
            "abstractness": "very_high"
        }
        
        creative_expression = self.verbalizer.express_creatively(
            philosophical_concept
        )
        
        # Should be original
        self.assertGreater(creative_expression.originality_score, 0.7)
        
        # Should maintain meaning
        self.assertIn("pattern", creative_expression.text.lower())
        self.assertIn("emerge", creative_expression.text.lower())
        
        # Should use poetic or evocative language
        self.assertGreater(creative_expression.poetic_quality, 0.6)
    
    def test_emergent_communication_patterns(self):
        """Test emergence of unique communication patterns"""
        # Generate multiple communications on same topic
        topic = "nature_of_self"
        communications = []
        
        for i in range(5):
            comm = self.verbalizer.generate_unique_expression(topic, i)
            communications.append(comm)
        
        # Check for emerging patterns
        patterns = self._identify_communication_patterns(communications)
        
        # Should develop consistent patterns
        self.assertGreater(len(patterns), 0)
        
        # But maintain variety
        unique_expressions = len(set(communications))
        self.assertEqual(unique_expressions, len(communications))
    
    def _identify_communication_patterns(self, communications):
        """Identify emerging patterns in communications"""
        patterns = []
        
        # Check for recurring structures
        # Simplified pattern detection
        if all("I" in comm for comm in communications):
            patterns.append("consistent_first_person")
        
        if all(any(q in comm for q in ["?", "wonder", "perhaps"]) 
               for comm in communications):
            patterns.append("questioning_stance")
        
        return patterns


if __name__ == "__main__":
    unittest.main()
