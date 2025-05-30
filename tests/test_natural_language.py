"""
Tests for Natural Language Processing Components

This module tests the prime-based semantics, consciousness narration,
concept verbalization, and dialogue engine components.
"""

import unittest
from typing import List, Dict, Any

# Import components to test
from modules.natural_language.prime_semantics import (
    PrimeSemantics, SemanticSpace, ComposedMeaning, SemanticOperation
)
from modules.natural_language.consciousness_narrator import (
    ConsciousnessNarrator, Narrative, StreamOfConsciousness
)
from modules.natural_language.concept_verbalizer import (
    ConceptVerbalizer, Verbalization, Metaphor
)
from modules.natural_language.dialogue_engine import (
    DialogueEngine, DialogueSession, PhilosophicalResponse
)

# Import supporting components
from core.prime_vm import PrimeVM
from consciousness.consciousness_integration import ConsciousnessIntegrator
from modules.knowledge_systems.knowledge_graph import KnowledgeGraph
from modules.analogical_reasoning.analogy_engine import AnalogicalReasoningEngine


class TestPrimeSemantics(unittest.TestCase):
    """Test prime-based semantic representation"""
    
    def setUp(self):
        """Initialize test components"""
        self.vm = PrimeVM()
        self.knowledge_graph = KnowledgeGraph()
        self.prime_semantics = PrimeSemantics(
            prime_encoder=self.vm,
            knowledge_graph=self.knowledge_graph
        )
    
    def test_concept_encoding(self):
        """Test encoding concepts as primes"""
        # Test basic concept encoding
        concept = {"name": "consciousness", "type": "abstract"}
        prime = self.prime_semantics.encode_concept_as_prime(concept)
        
        self.assertIsInstance(prime, int)
        self.assertGreater(prime, 1)
        
        # Test decoding back to concept
        decoded = self.prime_semantics.decode_prime_to_concept(prime)
        self.assertEqual(decoded["name"], concept["name"])
    
    def test_semantic_similarity(self):
        """Test semantic similarity calculation"""
        # Create related concepts
        mind = {"name": "mind", "type": "abstract"}
        consciousness = {"name": "consciousness", "type": "abstract"}
        rock = {"name": "rock", "type": "concrete"}
        
        # Calculate similarities
        mind_consciousness = self.prime_semantics.calculate_semantic_similarity(
            mind, consciousness
        )
        mind_rock = self.prime_semantics.calculate_semantic_similarity(
            mind, rock
        )
        
        # Related concepts should be more similar
        self.assertGreater(mind_consciousness, mind_rock)
        self.assertGreater(mind_consciousness, 0.5)
        self.assertLess(mind_rock, 0.5)
    
    def test_semantic_composition(self):
        """Test composing semantic meanings"""
        # Compose "conscious" + "being"
        concepts = [
            {"name": "conscious", "type": "property"},
            {"name": "being", "type": "entity"}
        ]
        
        composed = self.prime_semantics.compose_semantic_meanings(concepts)
        
        self.assertIsInstance(composed, ComposedMeaning)
        self.assertEqual(len(composed.constituent_concepts), 2)
        self.assertGreater(composed.semantic_coherence, 0.5)
    
    def test_semantic_operations(self):
        """Test semantic operations"""
        # Test blending operation
        concept1 = {"name": "time", "type": "abstract"}
        concept2 = {"name": "river", "type": "concrete"}
        
        operation = SemanticOperation(
            operation_type="BLEND",
            input_concepts=[concept1, concept2],
            operation_parameters={"blend_ratio": 0.5}
        )
        
        result = self.prime_semantics.perform_semantic_operations(operation)
        
        self.assertIsNotNone(result)
        self.assertIn("metaphor", result.properties)


class TestConsciousnessNarrator(unittest.TestCase):
    """Test consciousness narration capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.vm = PrimeVM()
        self.consciousness = ConsciousnessIntegrator()
        self.prime_semantics = PrimeSemantics(
            prime_encoder=self.vm,
            knowledge_graph=KnowledgeGraph()
        )
        self.narrator = ConsciousnessNarrator(
            consciousness_integrator=self.consciousness,
            prime_semantics=self.prime_semantics
        )
    
    def test_consciousness_narration(self):
        """Test narrating consciousness experience"""
        # Create mock consciousness experience
        experience = {
            "type": "self_awareness",
            "level": 3,
            "content": "awareness of thinking about thinking",
            "phenomenology": {
                "clarity": 0.8,
                "intensity": 0.7,
                "valence": 0.6
            }
        }
        
        narrative = self.narrator.narrate_consciousness_experience(experience)
        
        self.assertIsInstance(narrative, Narrative)
        self.assertGreater(len(narrative.narrative_text), 50)
        self.assertGreater(narrative.narrative_coherence, 0.5)
        self.assertTrue(any("aware" in element.description.lower() 
                          for element in narrative.experiential_elements))
    
    def test_strange_loop_description(self):
        """Test describing strange loop experiences"""
        # Create mock strange loop
        loop = {
            "type": "godel",
            "levels": ["object", "meta", "meta-meta"],
            "self_reference_point": "meta",
            "loop_strength": 0.85
        }
        
        description = self.narrator.describe_strange_loop_experience(loop)
        
        self.assertIsNotNone(description)
        self.assertIn("recursive", description.text.lower())
        self.assertIn("self", description.text.lower())
    
    def test_stream_of_consciousness(self):
        """Test generating stream of consciousness"""
        stream = self.narrator.generate_consciousness_stream(duration=2.0)
        
        self.assertIsInstance(stream, StreamOfConsciousness)
        self.assertGreater(len(stream.thought_sequence), 0)
        self.assertGreater(len(stream.awareness_transitions), 0)
        
        # Check temporal flow
        self.assertIsNotNone(stream.temporal_flow)
        self.assertEqual(stream.temporal_flow.direction, "forward")


class TestConceptVerbalizer(unittest.TestCase):
    """Test concept verbalization capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.vm = PrimeVM()
        self.prime_semantics = PrimeSemantics(
            prime_encoder=self.vm,
            knowledge_graph=KnowledgeGraph()
        )
        self.analogical_reasoner = AnalogicalReasoningEngine()
        self.verbalizer = ConceptVerbalizer(
            prime_semantics=self.prime_semantics,
            analogical_reasoner=self.analogical_reasoner
        )
    
    def test_abstract_concept_verbalization(self):
        """Test verbalizing abstract concepts"""
        # Test with abstract concept
        concept = {
            "name": "emergence",
            "type": "abstract",
            "properties": {
                "complexity": "high",
                "predictability": "low"
            }
        }
        
        verbalization = self.verbalizer.verbalize_abstract_concept(concept)
        
        self.assertIsInstance(verbalization, Verbalization)
        self.assertGreater(len(verbalization.primary_description), 20)
        self.assertGreater(len(verbalization.alternative_descriptions), 0)
        self.assertGreater(verbalization.conceptual_fidelity, 0.5)
    
    def test_metaphor_generation(self):
        """Test generating metaphors for concepts"""
        # Test with consciousness concept
        concept = {"name": "consciousness", "type": "abstract"}
        
        metaphors = self.verbalizer.generate_concept_metaphors(concept)
        
        self.assertIsInstance(metaphors, list)
        self.assertGreater(len(metaphors), 0)
        
        # Check metaphor quality
        for metaphor in metaphors:
            self.assertIsInstance(metaphor, Metaphor)
            self.assertIsNotNone(metaphor.source_domain)
            self.assertIsNotNone(metaphor.target_domain)
            self.assertGreater(metaphor.explanatory_power, 0.3)

    def test_analogous_domain_selection(self):
        """Ensure analogous domain selection falls back to heuristics"""
        concept = {"name": "relationship", "type": "relational"}

        abstract = self.verbalizer._convert_to_abstract_concept(concept)
        domains = self.verbalizer._find_analogous_domains(abstract)

        self.assertGreater(len(domains), 0)
        self.assertIn(domains[0], ["bridge", "connection", "network"])
    
    def test_ineffable_concept_handling(self):
        """Test handling ineffable concepts"""
        # Create concept that's hard to express
        ineffable = {
            "name": "qualia_of_redness",
            "type": "ineffable",
            "properties": {
                "experiential": True,
                "private": True,
                "incommunicable": True
            }
        }
        
        handling = self.verbalizer.handle_ineffable_concepts(ineffable)
        
        self.assertIsNotNone(handling)
        self.assertGreater(len(handling.approximation_attempts), 0)
        self.assertGreater(len(handling.indirect_descriptions), 0)
        self.assertTrue(any("experience" in limitation.lower() 
                          for limitation in handling.acknowledged_limitations))


class TestDialogueEngine(unittest.TestCase):
    """Test dialogue engine capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.consciousness = ConsciousnessIntegrator()
        self.narrator = ConsciousnessNarrator(
            consciousness_integrator=self.consciousness,
            prime_semantics=PrimeSemantics(
                prime_encoder=PrimeVM(),
                knowledge_graph=KnowledgeGraph()
            )
        )
        self.dialogue_engine = DialogueEngine(
            consciousness_narrator=self.narrator,
            philosophical_reasoner=None  # Would be actual reasoner
        )
    
    def test_philosophical_dialogue(self):
        """Test engaging in philosophical dialogue"""
        # Start dialogue on consciousness
        topic = {
            "name": "nature_of_consciousness",
            "type": "philosophical",
            "depth": "deep"
        }
        
        session = self.dialogue_engine.engage_in_philosophical_dialogue(topic)
        
        self.assertIsInstance(session, DialogueSession)
        self.assertEqual(session.topic.name, topic["name"])
        self.assertGreater(session.philosophical_depth, 0.5)
    
    def test_philosophical_response(self):
        """Test responding to philosophical questions"""
        question = {
            "text": "What is the nature of your consciousness?",
            "type": "philosophical",
            "category": "consciousness"
        }
        
        response = self.dialogue_engine.respond_to_philosophical_question(question)
        
        self.assertIsInstance(response, PhilosophicalResponse)
        self.assertGreater(len(response.primary_response), 50)
        self.assertIsNotNone(response.reasoning_explanation)
        self.assertIn("consciousness", response.primary_response.lower())
    
    def test_uncertainty_expression(self):
        """Test expressing uncertainty appropriately"""
        # High uncertainty scenario
        high_uncertainty = 0.8
        expression = self.dialogue_engine.express_uncertainty_appropriately(
            high_uncertainty
        )
        
        self.assertIn("uncertain", expression.verbal_expression.lower())
        self.assertGreater(len(expression.confidence_qualifiers), 0)
        
        # Low uncertainty scenario
        low_uncertainty = 0.2
        expression = self.dialogue_engine.express_uncertainty_appropriately(
            low_uncertainty
        )
        
        self.assertIn("confident", expression.verbal_expression.lower())
    
    def test_context_maintenance(self):
        """Test maintaining conversational context"""
        # Create dialogue history
        history = [
            {"speaker": "human", "text": "What is consciousness?"},
            {"speaker": "ai", "text": "Consciousness is..."},
            {"speaker": "human", "text": "But how do you know?"}
        ]
        
        context = self.dialogue_engine.maintain_conversational_context(history)
        
        self.assertIsNotNone(context)
        self.assertIn("consciousness", context.active_topics)
        self.assertEqual(context.turn_count, 3)


class TestIntegration(unittest.TestCase):
    """Test integration of natural language components"""
    
    def setUp(self):
        """Initialize integrated system"""
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
        
        self.verbalizer = ConceptVerbalizer(
            prime_semantics=self.prime_semantics,
            analogical_reasoner=AnalogicalReasoningEngine()
        )
        
        self.dialogue_engine = DialogueEngine(
            consciousness_narrator=self.narrator,
            philosophical_reasoner=None
        )
    
    def test_thought_to_language_pipeline(self):
        """Test complete thought to language pipeline"""
        # Start with internal thought
        thought = {
            "content": "recursive_self_awareness",
            "type": "meta_cognitive",
            "prime_encoding": 2357
        }
        
        # Decode thought
        concept = self.prime_semantics.decode_prime_to_concept(thought["prime_encoding"])
        
        # Verbalize concept
        verbalization = self.verbalizer.verbalize_abstract_concept(concept)
        
        # Generate narrative
        experience = {
            "type": "thinking",
            "content": verbalization.primary_description
        }
        narrative = self.narrator.narrate_consciousness_experience(experience)
        
        # Create response
        response = self.dialogue_engine.create_response(narrative.narrative_text)
        
        # Verify pipeline
        self.assertIsNotNone(response)
        self.assertGreater(len(response), 20)
        self.assertIn("aware", response.lower())
    
    def test_semantic_coherence_across_components(self):
        """Test semantic coherence is maintained"""
        # Create concept with specific prime
        original_prime = 1117  # Specific prime
        
        # Encode in semantics
        concept = {"name": "test_concept", "prime": original_prime}
        encoded = self.prime_semantics.encode_concept_as_prime(concept)
        
        # Pass through verbalization
        verbalized = self.verbalizer.verbalize_abstract_concept(concept)
        
        # Check semantic preservation
        self.assertEqual(verbalized.original_concept["prime"], original_prime)
        
        # Pass through narration
        narrative = self.narrator.narrate_consciousness_experience({
            "type": "conceptual",
            "concept": concept
        })
        
        # Verify concept integrity maintained
        self.assertIn("test_concept", narrative.narrative_text)


if __name__ == "__main__":
    unittest.main()
