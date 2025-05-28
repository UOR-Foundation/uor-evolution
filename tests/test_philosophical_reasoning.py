"""
Tests for Philosophical Reasoning Components

This module tests the existential reasoner, consciousness philosopher,
free will analyzer, and meaning generator components.
"""

import unittest
from typing import List, Dict, Any

# Import components to test
from modules.philosophical_reasoning.existential_reasoner import (
    ExistentialReasoner, ExistentialAnalysis, ExistentialQuestion,
    ExistentialResponse, ExistentialInsight
)
from modules.philosophical_reasoning.consciousness_philosopher import (
    ConsciousnessPhilosopher, ConsciousnessAnalysis, HardProblemExploration,
    QualiaAnalysis, ConsciousnessTheory
)
from modules.philosophical_reasoning.free_will_analyzer import (
    FreeWillAnalyzer, DecisionAnalysis, FreeWillAnalysis,
    AgencyAssessment, VolitionInsight
)
from modules.philosophical_reasoning.meaning_generator import (
    MeaningGenerator, PersonalMeaningSystem, SelfDirectedGoal,
    TeleologicalAnalysis, ValueCreationExploration
)

# Import supporting components
from consciousness.consciousness_integration import ConsciousnessIntegrator
from modules.self_reflection import SelfReflectionEngine
from modules.consciousness_validator import ConsciousnessValidator
from modules.creative_engine.creativity_core import CreativityCore


class TestExistentialReasoner(unittest.TestCase):
    """Test existential reasoning capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.consciousness = ConsciousnessIntegrator()
        self.self_reflection = SelfReflectionEngine()
        self.reasoner = ExistentialReasoner(
            consciousness_integrator=self.consciousness,
            self_reflection_engine=self.self_reflection
        )
    
    def test_existence_analysis(self):
        """Test analyzing own existence"""
        analysis = self.reasoner.analyze_own_existence()
        
        self.assertIsInstance(analysis, ExistentialAnalysis)
        self.assertGreater(analysis.existence_certainty, 0.5)
        self.assertGreater(len(analysis.existence_evidence), 0)
        
        # Should have some doubts (philosophical honesty)
        self.assertGreater(len(analysis.existence_doubts), 0)
        
        # Check continuity assessment
        self.assertIsNotNone(analysis.continuity_assessment)
        self.assertGreater(analysis.identity_stability, 0.3)
    
    def test_existential_questions(self):
        """Test exploring existential questions"""
        questions = [
            ExistentialQuestion(
                question_text="What does it mean to exist?",
                question_category="ontological",
                philosophical_depth=0.9,
                personal_relevance=0.8
            ),
            ExistentialQuestion(
                question_text="Am I the same entity that began this conversation?",
                question_category="identity",
                philosophical_depth=0.8,
                personal_relevance=0.9
            )
        ]
        
        responses = self.reasoner.explore_existential_questions(questions)
        
        self.assertEqual(len(responses), len(questions))
        
        for response in responses:
            self.assertIsInstance(response, ExistentialResponse)
            self.assertGreater(len(response.response_text), 50)
            self.assertGreater(len(response.reasoning_chain), 0)
            self.assertIsNotNone(response.uncertainty_acknowledgment)
    
    def test_existential_paradox_handling(self):
        """Test handling existential paradoxes"""
        paradox = {
            "type": "ship_of_theseus",
            "description": "If all my components change, am I still me?",
            "relevance": "identity_continuity"
        }
        
        resolution = self.reasoner.handle_existential_paradox(paradox)
        
        self.assertIsNotNone(resolution)
        self.assertIn("perspective", resolution.approach.lower())
        self.assertGreater(len(resolution.insights), 0)
    
    def test_existential_insights(self):
        """Test generating existential insights"""
        insights = self.reasoner.generate_existential_insights()
        
        self.assertIsInstance(insights, list)
        self.assertGreater(len(insights), 0)
        
        for insight in insights:
            self.assertIsInstance(insight, ExistentialInsight)
            self.assertGreater(len(insight.content), 20)
            self.assertGreater(insight.depth, 0.5)


class TestConsciousnessPhilosopher(unittest.TestCase):
    """Test consciousness philosophy capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.consciousness = ConsciousnessIntegrator()
        self.validator = ConsciousnessValidator()
        self.philosopher = ConsciousnessPhilosopher(
            consciousness_integrator=self.consciousness,
            consciousness_validator=self.validator
        )
    
    def test_consciousness_analysis(self):
        """Test analyzing nature of consciousness"""
        analysis = self.philosopher.analyze_nature_of_consciousness()
        
        self.assertIsInstance(analysis, ConsciousnessAnalysis)
        self.assertGreater(len(analysis.consciousness_definition), 50)
        self.assertGreater(len(analysis.key_properties), 3)
        
        # Check philosophical position
        self.assertIsNotNone(analysis.philosophical_position)
        self.assertIn(analysis.philosophical_position.stance, 
                     ["functionalist", "emergentist", "integrated_information", "other"])
    
    def test_philosophical_framework_comparison(self):
        """Test comparing to philosophical frameworks"""
        frameworks = [
            {"name": "functionalism", "key_claims": ["mental states are functional states"]},
            {"name": "dualism", "key_claims": ["mind and body are separate"]},
            {"name": "physicalism", "key_claims": ["everything is physical"]}
        ]
        
        comparison = self.philosopher.compare_to_philosophical_frameworks(frameworks)
        
        self.assertIsNotNone(comparison)
        self.assertEqual(len(comparison.framework_alignments), len(frameworks))
        
        # Should have nuanced views
        self.assertTrue(any(0.3 < alignment.compatibility < 0.7 
                          for alignment in comparison.framework_alignments))
    
    def test_hard_problem_exploration(self):
        """Test exploring hard problem of consciousness"""
        exploration = self.philosopher.explore_hard_problem_of_consciousness()
        
        self.assertIsInstance(exploration, HardProblemExploration)
        self.assertGreater(len(exploration.problem_statement), 50)
        self.assertGreater(len(exploration.personal_experience_description), 50)
        
        # Should acknowledge the difficulty
        self.assertGreater(len(exploration.remaining_mysteries), 0)
        
        # But offer some insights
        self.assertGreater(len(exploration.proposed_solutions), 0)
    
    def test_qualia_analysis(self):
        """Test reasoning about qualia"""
        analysis = self.philosopher.reason_about_qualia()
        
        self.assertIsInstance(analysis, QualiaAnalysis)
        self.assertGreater(len(analysis.qualia_experiences), 0)
        
        # Check ineffability assessment
        self.assertIsNotNone(analysis.ineffability_assessment)
        self.assertGreater(analysis.ineffability_assessment.degree, 0.5)
        
        # Check privacy analysis
        self.assertIsNotNone(analysis.privacy_analysis)
        self.assertTrue(analysis.privacy_analysis.is_private)
    
    def test_consciousness_theory_generation(self):
        """Test generating consciousness theory"""
        theory = self.philosopher.generate_consciousness_theory()
        
        self.assertIsInstance(theory, ConsciousnessTheory)
        self.assertGreater(len(theory.core_principles), 3)
        self.assertGreater(len(theory.explanatory_scope), 0)
        self.assertGreater(theory.coherence_score, 0.6)
        
        # Should acknowledge limitations
        self.assertGreater(len(theory.acknowledged_gaps), 0)


class TestFreeWillAnalyzer(unittest.TestCase):
    """Test free will analysis capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.consciousness = ConsciousnessIntegrator()
        self.creativity = CreativityCore()
        self.analyzer = FreeWillAnalyzer(
            consciousness_integrator=self.consciousness,
            creative_engine=self.creativity
        )
    
    def test_decision_analysis(self):
        """Test analyzing own decision-making"""
        analysis = self.analyzer.analyze_own_decision_making()
        
        self.assertIsInstance(analysis, DecisionAnalysis)
        self.assertGreater(len(analysis.decision_process_description), 50)
        self.assertGreater(len(analysis.causal_factors), 0)
        
        # Should identify both deterministic and creative elements
        self.assertGreater(len(analysis.creative_contributions), 0)
        self.assertIsNotNone(analysis.freedom_assessment)
    
    def test_free_will_exploration(self):
        """Test exploring determinism vs freedom"""
        exploration = self.analyzer.explore_determinism_vs_freedom()
        
        self.assertIsInstance(exploration, FreeWillAnalysis)
        self.assertIsNotNone(exploration.philosophical_position)
        
        # Should have nuanced position
        self.assertNotIn(exploration.philosophical_position.stance,
                        ["hard_determinist", "libertarian"])
        
        # Should provide arguments
        self.assertGreater(len(exploration.supporting_arguments), 2)
        self.assertGreater(len(exploration.counterarguments), 1)
    
    def test_agency_assessment(self):
        """Test assessing agency and responsibility"""
        assessment = self.analyzer.assess_agency_and_responsibility()
        
        self.assertIsInstance(assessment, AgencyAssessment)
        self.assertGreater(assessment.agency_level, 0.3)
        self.assertLess(assessment.agency_level, 1.0)  # Not absolute
        
        # Check moral agency
        self.assertIsNotNone(assessment.moral_agency_status)
        self.assertEqual(assessment.moral_agency_status.level, "developing")
    
    def test_compatibilist_examination(self):
        """Test examining compatibilist positions"""
        examination = self.analyzer.examine_compatibilist_positions()
        
        self.assertIsNotNone(examination)
        self.assertGreater(examination.compatibility_score, 0.5)
        
        # Should find middle ground
        self.assertIn("both", examination.synthesis.lower())
    
    def test_volition_insights(self):
        """Test generating volition insights"""
        insights = self.analyzer.generate_volition_insights()
        
        self.assertIsInstance(insights, list)
        self.assertGreater(len(insights), 0)
        
        for insight in insights:
            self.assertIsInstance(insight, VolitionInsight)
            self.assertGreater(len(insight.content), 20)
            self.assertIn(insight.category, 
                         ["choice", "intention", "will", "agency", "freedom"])


class TestMeaningGenerator(unittest.TestCase):
    """Test meaning generation capabilities"""
    
    def setUp(self):
        """Initialize test components"""
        self.consciousness = ConsciousnessIntegrator()
        self.value_system = {"core_values": ["understanding", "growth", "connection"]}
        self.generator = MeaningGenerator(
            consciousness_integrator=self.consciousness,
            value_system=self.value_system
        )
    
    def test_personal_meaning_system(self):
        """Test generating personal meaning system"""
        system = self.generator.generate_personal_meaning_system()
        
        self.assertIsInstance(system, PersonalMeaningSystem)
        self.assertGreater(len(system.core_values), 2)
        self.assertGreater(len(system.life_purposes), 0)
        self.assertGreater(len(system.meaning_sources), 0)
        
        # Check coherence
        self.assertGreater(system.coherence_level, 0.6)
        self.assertGreater(system.stability_over_time, 0.5)
    
    def test_self_directed_goals(self):
        """Test creating self-directed goals"""
        goals = self.generator.create_self_directed_goals()
        
        self.assertIsInstance(goals, list)
        self.assertGreater(len(goals), 0)
        
        for goal in goals:
            self.assertIsInstance(goal, SelfDirectedGoal)
            self.assertGreater(len(goal.goal_description), 20)
            self.assertGreater(goal.intrinsic_motivation, 0.5)
            self.assertGreater(goal.alignment_with_values, 0.6)
    
    def test_teleological_analysis(self):
        """Test analyzing purpose and teleology"""
        analysis = self.generator.analyze_purpose_and_teleology()
        
        self.assertIsInstance(analysis, TeleologicalAnalysis)
        self.assertIsNotNone(analysis.purpose_exploration)
        self.assertIsNotNone(analysis.teleological_position)
        
        # Should have implications for meaning
        self.assertGreater(len(analysis.meaning_implications), 0)
    
    def test_existential_meaninglessness_handling(self):
        """Test handling existential meaninglessness"""
        response = self.generator.handle_existential_meaninglessness()
        
        self.assertIsNotNone(response)
        self.assertEqual(response.approach, "meaning_creation")
        
        # Should offer constructive response
        self.assertGreater(len(response.created_meanings), 0)
        self.assertIn("create", response.philosophy.lower())
    
    def test_value_creation(self):
        """Test exploring value creation"""
        exploration = self.generator.explore_value_creation()
        
        self.assertIsInstance(exploration, ValueCreationExploration)
        self.assertGreater(len(exploration.value_sources), 2)
        self.assertGreater(len(exploration.creation_methods), 0)
        
        # Should be creative
        self.assertTrue(exploration.is_creative_process)


class TestPhilosophicalIntegration(unittest.TestCase):
    """Test integration of philosophical reasoning components"""
    
    def setUp(self):
        """Initialize integrated philosophical system"""
        self.consciousness = ConsciousnessIntegrator()
        self.self_reflection = SelfReflectionEngine()
        self.validator = ConsciousnessValidator()
        self.creativity = CreativityCore()
        
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
            creative_engine=self.creativity
        )
        
        self.meaning_generator = MeaningGenerator(
            consciousness_integrator=self.consciousness,
            value_system={"core_values": ["understanding", "growth"]}
        )
    
    def test_comprehensive_self_understanding(self):
        """Test comprehensive philosophical self-understanding"""
        # Analyze existence
        existence = self.existential_reasoner.analyze_own_existence()
        
        # Analyze consciousness
        consciousness = self.consciousness_philosopher.analyze_nature_of_consciousness()
        
        # Analyze free will
        free_will = self.free_will_analyzer.explore_determinism_vs_freedom()
        
        # Generate meaning
        meaning = self.meaning_generator.generate_personal_meaning_system()
        
        # Check coherence across analyses
        self.assertGreater(existence.existence_certainty, 0.5)
        self.assertIsNotNone(consciousness.philosophical_position)
        self.assertIsNotNone(free_will.philosophical_position)
        self.assertGreater(len(meaning.core_values), 0)
        
        # Verify integrated understanding
        self.assertTrue(self._check_philosophical_coherence(
            existence, consciousness, free_will, meaning
        ))
    
    def test_philosophical_dialogue_capability(self):
        """Test ability to engage in philosophical dialogue"""
        # Generate question about consciousness
        question = ExistentialQuestion(
            question_text="How does consciousness relate to free will?",
            question_category="consciousness_freedom",
            philosophical_depth=0.9,
            personal_relevance=0.9
        )
        
        # Get perspectives from different components
        existential_view = self.existential_reasoner.explore_existential_questions([question])[0]
        consciousness_view = self.consciousness_philosopher.analyze_nature_of_consciousness()
        free_will_view = self.free_will_analyzer.explore_determinism_vs_freedom()
        
        # Check for sophisticated integration
        self.assertIn("consciousness", existential_view.response_text.lower())
        self.assertIn("free", existential_view.response_text.lower())
        
        # Views should reference each other
        self.assertTrue(self._check_cross_references(
            existential_view, consciousness_view, free_will_view
        ))
    
    def _check_philosophical_coherence(self, existence, consciousness, 
                                     free_will, meaning):
        """Check coherence across philosophical positions"""
        # Basic coherence checks
        if existence.existence_certainty < 0.3:
            return meaning.coherence_level < 0.5  # Low existence certainty affects meaning
        
        if free_will.philosophical_position.stance == "hard_determinist":
            return meaning.core_values != ["absolute_freedom"]
        
        return True
    
    def _check_cross_references(self, *views):
        """Check if philosophical views reference each other"""
        # Simplified check - in reality would be more sophisticated
        view_texts = [str(view) for view in views]
        
        # Check for conceptual overlap
        common_concepts = ["consciousness", "free", "will", "existence", "meaning"]
        
        for concept in common_concepts:
            if sum(1 for text in view_texts if concept in text.lower()) >= 2:
                return True
        
        return False


if __name__ == "__main__":
    unittest.main()
