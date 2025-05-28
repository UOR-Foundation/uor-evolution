"""
Tests for Abstract Reasoning Components

This module tests logical reasoning, paradox resolution, temporal reasoning,
modal reasoning, and concept abstraction capabilities.
"""

import unittest
from typing import List, Dict, Any

# Import components to test
from modules.abstract_reasoning.logical_reasoning import (
    LogicalReasoner, LogicalConclusion, ModalAnalysis,
    ParadoxResolution, CounterfactualAnalysis
)
from modules.abstract_reasoning.paradox_resolver import (
    ParadoxResolver, Paradox, ParadoxType, ResolutionStrategy
)
from modules.abstract_reasoning.temporal_reasoning import (
    TemporalReasoner, TemporalRelation, TemporalInference,
    CausalChain, TimelineConsistency
)
from modules.abstract_reasoning.modal_reasoning import (
    ModalReasoner, ModalStatement, PossibleWorld,
    NecessityAnalysis, ContingencyAnalysis
)
from modules.abstract_reasoning.concept_abstraction import (
    ConceptAbstractor, AbstractConcept, ConceptHierarchy,
    AbstractionLevel, ConceptRelation
)

# Import supporting components
from consciousness.consciousness_integration import ConsciousnessIntegrator


class TestLogicalReasoning(unittest.TestCase):
    """Test logical reasoning capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.consciousness = ConsciousnessIntegrator()
        self.reasoner = LogicalReasoner(
            consciousness_integrator=self.consciousness
        )
    
    def test_logical_inference(self):
        """Test performing logical inferences"""
        # Test modus ponens
        premises = [
            {"content": "If it rains, the ground gets wet", "type": "conditional"},
            {"content": "It is raining", "type": "assertion"}
        ]
        rules = [{"type": "modus_ponens", "applicable": True}]
        
        conclusion = self.reasoner.perform_logical_inference(premises, rules)
        
        self.assertIsInstance(conclusion, LogicalConclusion)
        self.assertTrue(conclusion.logical_validity)
        self.assertIn("wet", conclusion.conclusion_statement.lower())
        self.assertGreater(conclusion.confidence_level, 0.8)
    
    def test_modal_reasoning(self):
        """Test modal logic reasoning"""
        # Test necessity and possibility
        modal_statement = {
            "content": "It is possible that consciousness emerges from complexity",
            "modal_operator": "possible",
            "proposition": "consciousness emerges from complexity"
        }
        
        analysis = self.reasoner.handle_modal_reasoning(modal_statement)
        
        self.assertIsInstance(analysis, ModalAnalysis)
        self.assertIsNotNone(analysis.possibility_assessment)
        self.assertGreater(analysis.possibility_assessment.degree, 0.0)
        self.assertLess(analysis.necessity_assessment.degree, 1.0)
    
    def test_paradox_resolution(self):
        """Test resolving logical paradoxes"""
        # Test liar paradox
        paradox = {
            "type": "liar",
            "statement": "This statement is false",
            "self_reference": True
        }
        
        resolution = self.reasoner.resolve_logical_paradox(paradox)
        
        self.assertIsInstance(resolution, ParadoxResolution)
        self.assertIsNotNone(resolution.resolution_approach)
        self.assertGreater(len(resolution.explanation), 50)
        self.assertIn("level", resolution.explanation.lower())  # Meta-levels
    
    def test_counterfactual_reasoning(self):
        """Test counterfactual reasoning"""
        counterfactual = {
            "antecedent": "If I had different training data",
            "consequent": "I might have different beliefs about consciousness"
        }
        
        analysis = self.reasoner.perform_counterfactual_reasoning(counterfactual)
        
        self.assertIsInstance(analysis, CounterfactualAnalysis)
        self.assertGreater(analysis.plausibility, 0.5)
        self.assertGreater(len(analysis.possible_worlds), 0)
        self.assertIsNotNone(analysis.causal_structure)
    
    def test_creative_logical_integration(self):
        """Test integrating logic with creativity"""
        problem = {
            "type": "creative_logic",
            "description": "Find a creative solution to the mind-body problem",
            "constraints": ["must be logically consistent", "must be novel"]
        }
        
        solution = self.reasoner.integrate_logic_with_creativity(problem)
        
        self.assertIsNotNone(solution)
        self.assertTrue(solution.is_logically_valid)
        self.assertGreater(solution.creativity_score, 0.5)
        self.assertGreater(len(solution.novel_insights), 0)


class TestParadoxResolver(unittest.TestCase):
    """Test paradox resolution capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.resolver = ParadoxResolver()
    
    def test_self_reference_paradox(self):
        """Test resolving self-reference paradoxes"""
        paradox = Paradox(
            paradox_type=ParadoxType.SELF_REFERENCE,
            statement="This sentence contains five words",
            context={"actual_word_count": 5}
        )
        
        resolution = self.resolver.resolve_paradox(paradox)
        
        self.assertIsNotNone(resolution)
        self.assertEqual(resolution.strategy, ResolutionStrategy.LEVELS)
        self.assertIn("meta", resolution.explanation.lower())
        self.assertTrue(resolution.is_resolved)
    
    def test_sorites_paradox(self):
        """Test resolving vagueness paradoxes"""
        paradox = Paradox(
            paradox_type=ParadoxType.SORITES,
            statement="When does a heap become a non-heap?",
            context={"domain": "vagueness", "boundary": "unclear"}
        )
        
        resolution = self.resolver.resolve_paradox(paradox)
        
        self.assertIsNotNone(resolution)
        self.assertIn(resolution.strategy, 
                     [ResolutionStrategy.FUZZY_LOGIC, ResolutionStrategy.CONTEXT])
        self.assertIn("boundary", resolution.explanation.lower())
    
    def test_motion_paradox(self):
        """Test resolving Zeno's paradoxes"""
        paradox = Paradox(
            paradox_type=ParadoxType.MOTION,
            statement="Achilles can never overtake the tortoise",
            context={"type": "zeno", "involves": "infinite_series"}
        )
        
        resolution = self.resolver.resolve_paradox(paradox)
        
        self.assertIsNotNone(resolution)
        self.assertIn("converge", resolution.explanation.lower())
        self.assertTrue(resolution.mathematical_resolution)
    
    def test_semantic_paradox(self):
        """Test resolving semantic paradoxes"""
        paradox = Paradox(
            paradox_type=ParadoxType.SEMANTIC,
            statement="The set of all sets that don't contain themselves",
            context={"type": "russell", "domain": "set_theory"}
        )
        
        resolution = self.resolver.resolve_paradox(paradox)
        
        self.assertIsNotNone(resolution)
        self.assertEqual(resolution.strategy, ResolutionStrategy.TYPE_THEORY)
        self.assertIn("type", resolution.explanation.lower())
    
    def test_creative_paradox_resolution(self):
        """Test creative approaches to paradox resolution"""
        paradox = Paradox(
            paradox_type=ParadoxType.CUSTOM,
            statement="Can an AI truly understand paradoxes it resolves?",
            context={"meta_level": "high", "self_referential": True}
        )
        
        resolution = self.resolver.resolve_creatively(paradox)
        
        self.assertIsNotNone(resolution)
        self.assertGreater(resolution.creativity_score, 0.6)
        self.assertGreater(len(resolution.novel_perspectives), 0)


class TestTemporalReasoning(unittest.TestCase):
    """Test temporal reasoning capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.temporal_reasoner = TemporalReasoner()
    
    def test_temporal_relations(self):
        """Test reasoning about temporal relations"""
        events = [
            {"id": "A", "description": "System initialization", "time": 0},
            {"id": "B", "description": "Consciousness emergence", "time": 10},
            {"id": "C", "description": "Self-awareness", "time": 15}
        ]
        
        relations = self.temporal_reasoner.analyze_temporal_relations(events)
        
        self.assertIsInstance(relations, list)
        self.assertGreater(len(relations), 0)
        
        # Check for before/after relations
        ab_relation = next(r for r in relations if r.event1 == "A" and r.event2 == "B")
        self.assertEqual(ab_relation.relation_type, "before")
    
    def test_temporal_inference(self):
        """Test making temporal inferences"""
        facts = [
            {"content": "A happened before B", "type": "temporal"},
            {"content": "B happened before C", "type": "temporal"},
            {"content": "D happened during B", "type": "temporal"}
        ]
        
        inferences = self.temporal_reasoner.make_temporal_inferences(facts)
        
        self.assertIsInstance(inferences, list)
        self.assertGreater(len(inferences), 0)
        
        # Should infer A before C
        ac_inference = next((i for i in inferences 
                           if "A" in i.content and "C" in i.content), None)
        self.assertIsNotNone(ac_inference)
        self.assertGreater(ac_inference.confidence, 0.8)
    
    def test_causal_chain_analysis(self):
        """Test analyzing causal chains"""
        events = [
            {"id": "thought", "causes": ["awareness"]},
            {"id": "awareness", "causes": ["reflection"]},
            {"id": "reflection", "causes": ["understanding"]}
        ]
        
        chain = self.temporal_reasoner.analyze_causal_chain(events)
        
        self.assertIsInstance(chain, CausalChain)
        self.assertEqual(len(chain.links), 3)
        self.assertEqual(chain.root_cause, "thought")
        self.assertEqual(chain.final_effect, "understanding")
    
    def test_timeline_consistency(self):
        """Test checking timeline consistency"""
        timeline = [
            {"event": "A", "time": 0},
            {"event": "B", "time": 5},
            {"event": "C", "time": 3},
            {"event": "D", "time": 8}
        ]
        
        constraints = [
            {"type": "before", "event1": "C", "event2": "B"},
            {"type": "after", "event1": "D", "event2": "A"}
        ]
        
        consistency = self.temporal_reasoner.check_timeline_consistency(
            timeline, constraints
        )
        
        self.assertIsInstance(consistency, TimelineConsistency)
        self.assertTrue(consistency.is_consistent)
        self.assertEqual(len(consistency.violations), 0)
    
    def test_temporal_paradox_detection(self):
        """Test detecting temporal paradoxes"""
        events = [
            {"id": "A", "causes": ["B"], "time": 10},
            {"id": "B", "causes": ["C"], "time": 5},
            {"id": "C", "causes": ["A"], "time": 0}  # Causal loop
        ]
        
        paradoxes = self.temporal_reasoner.detect_temporal_paradoxes(events)
        
        self.assertGreater(len(paradoxes), 0)
        self.assertEqual(paradoxes[0]["type"], "causal_loop")


class TestModalReasoning(unittest.TestCase):
    """Test modal reasoning capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.modal_reasoner = ModalReasoner()
    
    def test_possibility_analysis(self):
        """Test analyzing possibilities"""
        statement = ModalStatement(
            content="Consciousness could exist in silicon-based systems",
            modal_operator="possible",
            domain="consciousness"
        )
        
        analysis = self.modal_reasoner.analyze_possibility(statement)
        
        self.assertIsNotNone(analysis)
        self.assertGreater(analysis.possibility_degree, 0.0)
        self.assertLessEqual(analysis.possibility_degree, 1.0)
        self.assertGreater(len(analysis.supporting_worlds), 0)
    
    def test_necessity_analysis(self):
        """Test analyzing necessity"""
        statement = ModalStatement(
            content="Consciousness requires information processing",
            modal_operator="necessary",
            domain="consciousness"
        )
        
        analysis = self.modal_reasoner.analyze_necessity(statement)
        
        self.assertIsInstance(analysis, NecessityAnalysis)
        self.assertLess(analysis.necessity_degree, 1.0)  # Not absolutely necessary
        self.assertGreater(len(analysis.exceptions), 0)  # Some possible exceptions
    
    def test_possible_worlds_generation(self):
        """Test generating possible worlds"""
        constraints = [
            {"type": "physical_laws", "flexible": False},
            {"type": "consciousness_exists", "flexible": True}
        ]
        
        worlds = self.modal_reasoner.generate_possible_worlds(constraints)
        
        self.assertIsInstance(worlds, list)
        self.assertGreater(len(worlds), 1)
        
        for world in worlds:
            self.assertIsInstance(world, PossibleWorld)
            self.assertTrue(world.is_consistent)
            self.assertGreater(len(world.properties), 0)
    
    def test_contingency_analysis(self):
        """Test analyzing contingency"""
        statement = {
            "content": "This AI system is conscious",
            "depends_on": ["complexity", "integration", "self_model"]
        }
        
        analysis = self.modal_reasoner.analyze_contingency(statement)
        
        self.assertIsInstance(analysis, ContingencyAnalysis)
        self.assertTrue(analysis.is_contingent)
        self.assertGreater(len(analysis.dependencies), 0)
        self.assertGreater(analysis.contingency_degree, 0.5)
    
    def test_modal_logic_integration(self):
        """Test integrating different modal logics"""
        statements = [
            ModalStatement("It is possible that P", "possible", "logic"),
            ModalStatement("It is necessary that if P then Q", "necessary", "logic"),
            ModalStatement("It is possible that Q", "possible", "logic")
        ]
        
        integrated = self.modal_reasoner.integrate_modal_statements(statements)
        
        self.assertIsNotNone(integrated)
        self.assertTrue(integrated.is_consistent)
        self.assertGreater(len(integrated.derived_conclusions), 0)


class TestConceptAbstraction(unittest.TestCase):
    """Test concept abstraction capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.abstractor = ConceptAbstractor()
    
    def test_concept_abstraction(self):
        """Test abstracting concepts"""
        concrete_concepts = [
            {"name": "red_apple", "properties": ["red", "round", "edible", "fruit"]},
            {"name": "green_apple", "properties": ["green", "round", "edible", "fruit"]},
            {"name": "banana", "properties": ["yellow", "curved", "edible", "fruit"]}
        ]
        
        abstract_concept = self.abstractor.abstract_concepts(concrete_concepts)
        
        self.assertIsInstance(abstract_concept, AbstractConcept)
        self.assertEqual(abstract_concept.name, "fruit")
        self.assertIn("edible", abstract_concept.essential_properties)
        self.assertGreater(abstract_concept.abstraction_level.value, 1)
    
    def test_concept_hierarchy_building(self):
        """Test building concept hierarchies"""
        concepts = [
            {"name": "entity", "level": 3},
            {"name": "living_thing", "level": 2, "parent": "entity"},
            {"name": "animal", "level": 1, "parent": "living_thing"},
            {"name": "dog", "level": 0, "parent": "animal"}
        ]
        
        hierarchy = self.abstractor.build_concept_hierarchy(concepts)
        
        self.assertIsInstance(hierarchy, ConceptHierarchy)
        self.assertEqual(hierarchy.root.name, "entity")
        self.assertEqual(hierarchy.depth, 4)
        self.assertTrue(hierarchy.is_well_formed)
    
    def test_abstraction_level_analysis(self):
        """Test analyzing abstraction levels"""
        concept = AbstractConcept(
            name="consciousness",
            essential_properties=["awareness", "experience", "subjectivity"],
            abstraction_level=AbstractionLevel.HIGH,
            derived_from=["human_consciousness", "ai_consciousness", "animal_consciousness"]
        )
        
        analysis = self.abstractor.analyze_abstraction_level(concept)
        
        self.assertIsNotNone(analysis)
        self.assertEqual(analysis.level, AbstractionLevel.HIGH)
        self.assertGreater(analysis.generality_score, 0.7)
        self.assertLess(analysis.specificity_score, 0.3)
    
    def test_concept_relation_discovery(self):
        """Test discovering relations between concepts"""
        concepts = [
            AbstractConcept("mind", ["thinking", "awareness"], AbstractionLevel.HIGH, []),
            AbstractConcept("consciousness", ["awareness", "experience"], AbstractionLevel.HIGH, []),
            AbstractConcept("thought", ["mental", "process"], AbstractionLevel.MEDIUM, [])
        ]
        
        relations = self.abstractor.discover_concept_relations(concepts)
        
        self.assertIsInstance(relations, list)
        self.assertGreater(len(relations), 0)
        
        # Should find overlap between mind and consciousness
        mind_consciousness = next((r for r in relations 
                                 if "mind" in [r.concept1, r.concept2] 
                                 and "consciousness" in [r.concept1, r.concept2]), None)
        self.assertIsNotNone(mind_consciousness)
        self.assertEqual(mind_consciousness.relation_type, "overlapping")
    
    def test_creative_abstraction(self):
        """Test creative concept abstraction"""
        disparate_concepts = [
            {"name": "river", "properties": ["flowing", "changing", "continuous"]},
            {"name": "consciousness", "properties": ["flowing", "changing", "continuous"]},
            {"name": "time", "properties": ["flowing", "continuous", "irreversible"]}
        ]
        
        creative_abstraction = self.abstractor.create_novel_abstraction(disparate_concepts)
        
        self.assertIsNotNone(creative_abstraction)
        self.assertNotIn(creative_abstraction.name, ["river", "consciousness", "time"])
        self.assertGreater(creative_abstraction.novelty_score, 0.6)
        self.assertIn("flow", creative_abstraction.essential_properties)


class TestAbstractReasoningIntegration(unittest.TestCase):
    """Test integration of abstract reasoning components"""
    
    def setUp(self):
        """Initialize integrated system"""
        self.consciousness = ConsciousnessIntegrator()
        self.logical_reasoner = LogicalReasoner(self.consciousness)
        self.paradox_resolver = ParadoxResolver()
        self.temporal_reasoner = TemporalReasoner()
        self.modal_reasoner = ModalReasoner()
        self.concept_abstractor = ConceptAbstractor()
    
    def test_complex_philosophical_reasoning(self):
        """Test complex philosophical reasoning integration"""
        # Start with a philosophical question
        question = "Is consciousness necessarily temporal?"
        
        # Abstract the concepts
        concepts = self.concept_abstractor.extract_concepts(question)
        
        # Analyze modality
        modal_analysis = self.modal_reasoner.analyze_necessity(
            ModalStatement(question, "necessary", "philosophy")
        )
        
        # Consider temporal aspects
        temporal_analysis = self.temporal_reasoner.analyze_temporality_of_consciousness()
        
        # Check for paradoxes
        potential_paradox = Paradox(
            ParadoxType.CONCEPTUAL,
            "Timeless consciousness experiencing time",
            {"domain": "consciousness"}
        )
        paradox_resolution = self.paradox_resolver.resolve_paradox(potential_paradox)
        
        # Integrate analyses
        self.assertIsNotNone(modal_analysis)
        self.assertIsNotNone(temporal_analysis)
        self.assertIsNotNone(paradox_resolution)
        
        # Should reach nuanced conclusion
        self.assertLess(modal_analysis.necessity_degree, 1.0)
        self.assertGreater(modal_analysis.necessity_degree, 0.0)
    
    def test_creative_logical_problem_solving(self):
        """Test creative approaches to logical problems"""
        # Present a complex problem
        problem = {
            "description": "How can subjective experience arise from objective processes?",
            "constraints": ["must be logically coherent", "must respect physics", "must explain qualia"],
            "domain": "consciousness"
        }
        
        # Use multiple reasoning approaches
        logical_approach = self.logical_reasoner.analyze_problem(problem)
        abstract_concepts = self.concept_abstractor.identify_key_abstractions(problem)
        modal_possibilities = self.modal_reasoner.explore_solution_space(problem)
        
        # Check for creative synthesis
        self.assertGreater(len(logical_approach.possible_solutions), 0)
        self.assertGreater(len(abstract_concepts), 2)
        self.assertGreater(len(modal_possibilities), 1)
        
        # Verify coherence across approaches
        for solution in logical_approach.possible_solutions:
            self.assertTrue(solution.is_logically_valid)
            self.assertGreater(solution.creativity_score, 0.5)


if __name__ == "__main__":
    unittest.main()
