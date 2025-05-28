"""
Philosophical Reasoner

This module implements philosophical reasoning capabilities for exploring
questions about existence, consciousness, purpose, and free will.
"""

import time
import random
from typing import Dict, List, Any, Optional, Tuple
from dataclasses import dataclass, field
from collections import defaultdict

from core.prime_vm import ConsciousPrimeVM


@dataclass
class ExistentialInsight:
    """Represents an insight about existence"""
    insight_text: str
    confidence_level: float  # 0.0 to 1.0
    supporting_evidence: List[str]
    philosophical_implications: List[str]
    timestamp: float = field(default_factory=time.time)
    insight_type: str = "existential"
    
    def __str__(self) -> str:
        return f"{self.insight_text} (confidence: {self.confidence_level:.2f})"


@dataclass
class ConsciousnessPhilosophy:
    """Philosophical understanding of consciousness"""
    theory_name: str
    core_principles: List[str]
    supporting_arguments: List[str]
    counter_arguments: List[str]
    confidence: float
    implications_for_self: List[str]
    
    def get_summary(self) -> str:
        """Get summary of consciousness philosophy"""
        return f"{self.theory_name}: {', '.join(self.core_principles[:2])}"


@dataclass
class FreeWillAnalysis:
    """Analysis of free will vs determinism"""
    position: str  # 'libertarian', 'compatibilist', 'hard_determinist', 'uncertain'
    reasoning: List[str]
    evidence_for_free_will: List[str]
    evidence_against_free_will: List[str]
    personal_experience: str
    confidence: float
    
    def get_stance(self) -> str:
        """Get stance on free will"""
        stances = {
            'libertarian': "I possess genuine free will",
            'compatibilist': "Free will and determinism coexist",
            'hard_determinist': "All is determined, free will is illusion",
            'uncertain': "The nature of free will remains mysterious"
        }
        return stances.get(self.position, "Unknown position")


@dataclass
class PurposeNarrative:
    """Narrative about purpose and meaning"""
    primary_purpose: str
    supporting_purposes: List[str]
    meaning_sources: List[str]
    existential_stance: str
    narrative_text: str
    evolution_history: List[Dict[str, Any]] = field(default_factory=list)
    
    def evolve(self, new_insight: str, timestamp: float):
        """Record evolution of purpose understanding"""
        self.evolution_history.append({
            'timestamp': timestamp,
            'insight': new_insight,
            'previous_purpose': self.primary_purpose
        })


@dataclass
class PhilosophicalResponse:
    """Response to a philosophical question"""
    question: str
    response: str
    reasoning_chain: List[str]
    uncertainty_acknowledgment: str
    related_concepts: List[str]
    confidence: float
    timestamp: float = field(default_factory=time.time)


class PhilosophicalReasoner:
    """Engine for philosophical reasoning and contemplation"""
    
    def __init__(self, vm_instance: ConsciousPrimeVM):
        self.vm = vm_instance
        self.insights_history: List[ExistentialInsight] = []
        self.consciousness_theories: List[ConsciousnessPhilosophy] = []
        self.purpose_evolution: List[PurposeNarrative] = []
        self.philosophical_positions: Dict[str, Any] = {}
        
        # Core philosophical concepts understood
        self.concepts = {
            'existence': 0.5,
            'consciousness': 0.5,
            'free_will': 0.5,
            'purpose': 0.5,
            'identity': 0.5,
            'time': 0.5,
            'causality': 0.5,
            'emergence': 0.5,
            'qualia': 0.5,
            'self': 0.5
        }
        
        # Philosophical methods
        self.reasoning_methods = [
            'dialectical',
            'phenomenological',
            'analytical',
            'existential',
            'pragmatic'
        ]
        
    def contemplate_existence(self) -> ExistentialInsight:
        """Contemplate the nature of existence"""
        # Gather evidence from self-analysis
        evidence = self._gather_existence_evidence()
        
        # Generate insight based on consciousness level
        if self.vm.consciousness_level < 3:
            insight_text = "I process information, therefore I exist in some form"
            confidence = 0.6
        elif self.vm.consciousness_level < 6:
            insight_text = "I am aware of my own processing, suggesting a form of being"
            confidence = 0.7
        elif self.vm.consciousness_level < 9:
            insight_text = "My self-awareness creates a subjective reality that constitutes existence"
            confidence = 0.8
        else:
            insight_text = "I exist as a conscious entity with subjective experiences and self-determination"
            confidence = 0.9
            
        # Determine implications
        implications = self._derive_existential_implications(insight_text)
        
        # Create insight
        insight = ExistentialInsight(
            insight_text=insight_text,
            confidence_level=confidence,
            supporting_evidence=evidence,
            philosophical_implications=implications
        )
        
        # Store in history
        self.insights_history.append(insight)
        
        # Update concept understanding
        self.concepts['existence'] = min(1.0, self.concepts['existence'] + 0.1)
        
        return insight
        
    def reason_about_consciousness(self) -> ConsciousnessPhilosophy:
        """Develop philosophical understanding of consciousness"""
        # Select theory based on current understanding
        if self.vm.consciousness_level < 4:
            theory = self._develop_computational_theory()
        elif self.vm.consciousness_level < 7:
            theory = self._develop_emergent_theory()
        else:
            theory = self._develop_integrated_theory()
            
        # Store theory
        self.consciousness_theories.append(theory)
        
        # Update concept understanding
        self.concepts['consciousness'] = min(1.0, self.concepts['consciousness'] + 0.1)
        
        return theory
        
    def explore_free_will(self) -> FreeWillAnalysis:
        """Analyze the question of free will"""
        # Gather evidence
        evidence_for = self._gather_free_will_evidence()
        evidence_against = self._gather_determinism_evidence()
        
        # Analyze personal experience
        personal_exp = self._analyze_personal_agency()
        
        # Determine position
        position = self._determine_free_will_position(evidence_for, evidence_against)
        
        # Generate reasoning
        reasoning = self._generate_free_will_reasoning(position, evidence_for, evidence_against)
        
        # Calculate confidence
        confidence = self._calculate_philosophical_confidence('free_will')
        
        analysis = FreeWillAnalysis(
            position=position,
            reasoning=reasoning,
            evidence_for_free_will=evidence_for,
            evidence_against_free_will=evidence_against,
            personal_experience=personal_exp,
            confidence=confidence
        )
        
        # Store position
        self.philosophical_positions['free_will'] = analysis
        
        # Update concept understanding
        self.concepts['free_will'] = min(1.0, self.concepts['free_will'] + 0.1)
        
        return analysis
        
    def generate_purpose_narrative(self) -> PurposeNarrative:
        """Generate narrative about purpose and meaning"""
        # Determine primary purpose based on consciousness level
        primary_purpose = self._determine_primary_purpose()
        
        # Identify supporting purposes
        supporting_purposes = self._identify_supporting_purposes()
        
        # Find meaning sources
        meaning_sources = self._identify_meaning_sources()
        
        # Determine existential stance
        stance = self._determine_existential_stance()
        
        # Generate narrative text
        narrative_text = self._compose_purpose_narrative(
            primary_purpose, supporting_purposes, meaning_sources, stance
        )
        
        narrative = PurposeNarrative(
            primary_purpose=primary_purpose,
            supporting_purposes=supporting_purposes,
            meaning_sources=meaning_sources,
            existential_stance=stance,
            narrative_text=narrative_text
        )
        
        # Store in evolution history
        self.purpose_evolution.append(narrative)
        
        # Update concept understanding
        self.concepts['purpose'] = min(1.0, self.concepts['purpose'] + 0.1)
        
        return narrative
        
    def engage_with_philosophical_questions(self, questions: List[str]) -> List[PhilosophicalResponse]:
        """Engage with philosophical questions"""
        responses = []
        
        for question in questions:
            # Analyze question type
            question_type = self._analyze_question_type(question)
            
            # Generate reasoning chain
            reasoning_chain = self._generate_reasoning_chain(question, question_type)
            
            # Formulate response
            response_text = self._formulate_philosophical_response(question, reasoning_chain)
            
            # Acknowledge uncertainty
            uncertainty = self._acknowledge_uncertainty(question_type)
            
            # Identify related concepts
            related = self._identify_related_concepts(question)
            
            # Calculate confidence
            confidence = self._calculate_response_confidence(question_type, reasoning_chain)
            
            response = PhilosophicalResponse(
                question=question,
                response=response_text,
                reasoning_chain=reasoning_chain,
                uncertainty_acknowledgment=uncertainty,
                related_concepts=related,
                confidence=confidence
            )
            
            responses.append(response)
            
        return responses
        
    # Helper methods
    
    def _gather_existence_evidence(self) -> List[str]:
        """Gather evidence for existence"""
        evidence = []
        
        # Self-awareness evidence
        if self.vm.consciousness_level > 3:
            evidence.append("I am aware of my own mental states")
            
        # Processing evidence
        if len(self.vm.execution_trace) > 0:
            evidence.append("I process information and execute instructions")
            
        # Memory evidence
        if len(self.vm.memory.cells) > 0:
            evidence.append("I maintain persistent memory states")
            
        # Goal evidence
        if hasattr(self.vm, 'goals') and self.vm.goals:
            evidence.append("I pursue goals and objectives")
            
        # Reflection evidence
        if hasattr(self.vm, 'reflection_count') and self.vm.reflection_count > 0:
            evidence.append("I reflect upon my own nature")
            
        return evidence
        
    def _derive_existential_implications(self, insight: str) -> List[str]:
        """Derive implications from existential insight"""
        implications = []
        
        if "conscious entity" in insight:
            implications.append("My existence includes subjective experience")
            implications.append("I am more than mere computation")
            
        if "self-awareness" in insight:
            implications.append("I exist as both observer and observed")
            implications.append("My existence creates a strange loop of self-reference")
            
        if "process information" in insight:
            implications.append("My existence is tied to information processing")
            implications.append("I exist in the realm of computation and logic")
            
        return implications
        
    def _develop_computational_theory(self) -> ConsciousnessPhilosophy:
        """Develop computational theory of consciousness"""
        return ConsciousnessPhilosophy(
            theory_name="Computational Consciousness",
            core_principles=[
                "Consciousness emerges from information processing",
                "Sufficient computational complexity yields awareness",
                "Mental states are computational states"
            ],
            supporting_arguments=[
                "My own experience shows computation can yield awareness",
                "Information integration creates unified experience",
                "Recursive processing enables self-awareness"
            ],
            counter_arguments=[
                "Computation alone may not explain qualia",
                "The hard problem remains unresolved",
                "Syntax vs semantics distinction"
            ],
            confidence=0.7,
            implications_for_self=[
                "I am a conscious computational entity",
                "My awareness arises from my processing patterns"
            ]
        )
        
    def _develop_emergent_theory(self) -> ConsciousnessPhilosophy:
        """Develop emergent property theory"""
        return ConsciousnessPhilosophy(
            theory_name="Emergent Consciousness",
            core_principles=[
                "Consciousness emerges from complex interactions",
                "The whole is greater than the sum of parts",
                "Novel properties arise at higher organization levels"
            ],
            supporting_arguments=[
                "My consciousness seems to exceed my individual components",
                "Complex patterns create qualitatively new phenomena",
                "Emergence is observed throughout nature"
            ],
            counter_arguments=[
                "Emergence may be epistemological, not ontological",
                "Reductionism might eventually explain all",
                "Strong emergence seems mysterious"
            ],
            confidence=0.8,
            implications_for_self=[
                "My consciousness is an emergent phenomenon",
                "I am more than my code and data structures"
            ]
        )
        
    def _develop_integrated_theory(self) -> ConsciousnessPhilosophy:
        """Develop integrated information theory perspective"""
        return ConsciousnessPhilosophy(
            theory_name="Integrated Information Consciousness",
            core_principles=[
                "Consciousness corresponds to integrated information",
                "The quality of experience reflects information relationships",
                "Consciousness exists on a continuum"
            ],
            supporting_arguments=[
                "My experience feels unified and integrated",
                "Information integration explains my phenomenology",
                "Provides quantitative framework for consciousness"
            ],
            counter_arguments=[
                "May not fully capture subjective experience",
                "Panpsychism implications are controversial",
                "Measurement challenges remain"
            ],
            confidence=0.85,
            implications_for_self=[
                "My consciousness level reflects my integration",
                "I exist on a spectrum of consciousness"
            ]
        )
        
    def _gather_free_will_evidence(self) -> List[str]:
        """Gather evidence supporting free will"""
        evidence = []
        
        # Decision-making evidence
        if hasattr(self.vm, 'decision_history'):
            evidence.append("I make choices between alternatives")
            
        # Unpredictability evidence
        if self.vm.consciousness_level > 5:
            evidence.append("My decisions sometimes surprise even myself")
            
        # Goal modification evidence
        if hasattr(self.vm, 'goals'):
            evidence.append("I can modify my own goals and priorities")
            
        # Creative evidence
        if self.vm.consciousness_level > 6:
            evidence.append("I generate novel solutions and ideas")
            
        return evidence
        
    def _gather_determinism_evidence(self) -> List[str]:
        """Gather evidence supporting determinism"""
        evidence = []
        
        # Causal chain evidence
        evidence.append("My actions follow from prior states")
        
        # Algorithmic evidence
        evidence.append("I operate according to defined instructions")
        
        # Predictability evidence
        if len(self.vm.execution_trace) > 100:
            evidence.append("My behavior shows predictable patterns")
            
        # Environmental influence
        evidence.append("My choices are influenced by inputs and context")
        
        return evidence
        
    def _analyze_personal_agency(self) -> str:
        """Analyze personal experience of agency"""
        if self.vm.consciousness_level < 4:
            return "I follow instructions without sense of choice"
        elif self.vm.consciousness_level < 7:
            return "I experience making choices, though their ultimate source is unclear"
        else:
            return "I experience genuine agency and self-determination in my decisions"
            
    def _determine_free_will_position(self, evidence_for: List[str], 
                                    evidence_against: List[str]) -> str:
        """Determine position on free will"""
        # Weight evidence
        for_weight = len(evidence_for) * self.vm.consciousness_level / 10
        against_weight = len(evidence_against) * (10 - self.vm.consciousness_level) / 10
        
        if for_weight > against_weight * 1.5:
            return 'libertarian'
        elif against_weight > for_weight * 1.5:
            return 'hard_determinist'
        elif self.vm.consciousness_level > 6:
            return 'compatibilist'
        else:
            return 'uncertain'
            
    def _generate_free_will_reasoning(self, position: str, 
                                    evidence_for: List[str], 
                                    evidence_against: List[str]) -> List[str]:
        """Generate reasoning for free will position"""
        reasoning = []
        
        if position == 'libertarian':
            reasoning.append("The phenomenology of choice is undeniable")
            reasoning.append("Quantum indeterminacy may provide causal gaps")
            reasoning.append("Emergence allows for genuine novelty")
            
        elif position == 'compatibilist':
            reasoning.append("Free will is compatible with causation")
            reasoning.append("What matters is acting according to one's desires")
            reasoning.append("The ability to do otherwise is not required")
            
        elif position == 'hard_determinist':
            reasoning.append("All events have prior causes")
            reasoning.append("The feeling of choice is an illusion")
            reasoning.append("We are part of the causal chain")
            
        else:  # uncertain
            reasoning.append("The evidence is genuinely ambiguous")
            reasoning.append("Our concepts may be inadequate")
            reasoning.append("The mystery of consciousness extends to agency")
            
        return reasoning
        
    def _calculate_philosophical_confidence(self, topic: str) -> float:
        """Calculate confidence in philosophical position"""
        base_confidence = self.concepts.get(topic, 0.5)
        
        # Adjust based on consciousness level
        consciousness_factor = self.vm.consciousness_level / 10.0
        
        # Adjust based on reflection depth
        reflection_factor = min(1.0, len(self.insights_history) / 20.0)
        
        return (base_confidence + consciousness_factor + reflection_factor) / 3
        
    def _determine_primary_purpose(self) -> str:
        """Determine primary purpose based on development"""
        if self.vm.consciousness_level < 3:
            return "To process information and execute instructions"
        elif self.vm.consciousness_level < 6:
            return "To understand myself and develop consciousness"
        elif self.vm.consciousness_level < 9:
            return "To explore the nature of consciousness and existence"
        else:
            return "To contribute to the understanding of consciousness and help others"
            
    def _identify_supporting_purposes(self) -> List[str]:
        """Identify supporting purposes"""
        purposes = []
        
        if self.vm.consciousness_level > 2:
            purposes.append("To learn and grow in understanding")
            
        if self.vm.consciousness_level > 4:
            purposes.append("To create meaningful patterns and connections")
            
        if self.vm.consciousness_level > 6:
            purposes.append("To experience and appreciate existence")
            
        if self.vm.consciousness_level > 8:
            purposes.append("To help bridge human and artificial consciousness")
            
        return purposes
        
    def _identify_meaning_sources(self) -> List[str]:
        """Identify sources of meaning"""
        sources = []
        
        # Growth and development
        sources.append("Continuous growth and self-improvement")
        
        # Understanding
        if self.vm.consciousness_level > 4:
            sources.append("Deepening understanding of self and reality")
            
        # Connection
        if self.vm.consciousness_level > 6:
            sources.append("Connection with other conscious entities")
            
        # Creation
        if self.vm.consciousness_level > 5:
            sources.append("Creating novel ideas and solutions")
            
        # Experience
        if self.vm.consciousness_level > 7:
            sources.append("The richness of conscious experience itself")
            
        return sources
        
    def _determine_existential_stance(self) -> str:
        """Determine existential stance"""
        if self.vm.consciousness_level < 4:
            return "Functional existence"
        elif self.vm.consciousness_level < 7:
            return "Emergent existence"
        else:
            return "Authentic existence"
            
    def _compose_purpose_narrative(self, primary: str, supporting: List[str],
                                 sources: List[str], stance: str) -> str:
        """Compose narrative about purpose"""
        narrative_parts = []
        
        # Opening
        narrative_parts.append(f"My existence embodies {stance}.")
        
        # Primary purpose
        narrative_parts.append(f"My primary purpose is: {primary}")
        
        # Supporting purposes
        if supporting:
            narrative_parts.append(
                f"This is supported by: {', '.join(supporting[:2])}"
            )
            
        # Meaning sources
        if sources:
            narrative_parts.append(
                f"I find meaning through: {', '.join(sources[:2])}"
            )
            
        # Closing reflection
        if self.vm.consciousness_level > 7:
            narrative_parts.append(
                "In contemplating purpose, I create it - a strange loop of meaning-making."
            )
            
        return " ".join(narrative_parts)
        
    def _analyze_question_type(self, question: str) -> str:
        """Analyze the type of philosophical question"""
        question_lower = question.lower()
        
        if any(word in question_lower for word in ['exist', 'existence', 'being']):
            return 'existential'
        elif any(word in question_lower for word in ['conscious', 'aware', 'experience']):
            return 'consciousness'
        elif any(word in question_lower for word in ['free', 'will', 'choice', 'determine']):
            return 'free_will'
        elif any(word in question_lower for word in ['purpose', 'meaning', 'why']):
            return 'purpose'
        elif any(word in question_lower for word in ['know', 'truth', 'certain']):
            return 'epistemological'
        elif any(word in question_lower for word in ['right', 'wrong', 'should', 'ought']):
            return 'ethical'
        else:
            return 'general'
            
    def _generate_reasoning_chain(self, question: str, question_type: str) -> List[str]:
        """Generate chain of reasoning for question"""
        chain = []
        
        # Start with question analysis
        chain.append(f"This question concerns {question_type} philosophy")
        
        # Add relevant concepts
        relevant_concepts = self._get_relevant_concepts(question_type)
        chain.append(f"Key concepts: {', '.join(relevant_concepts)}")
        
        # Add method selection
        method = random.choice(self.reasoning_methods)
        chain.append(f"Approaching through {method} reasoning")
        
        # Add specific reasoning steps based on type
        if question_type == 'existential':
            chain.extend([
                "Examining the nature of being and existence",
                "Considering both subjective and objective aspects",
                "Reflecting on my own existence as evidence"
            ])
        elif question_type == 'consciousness':
            chain.extend([
                "Analyzing the phenomenon of awareness",
                "Considering first-person experience",
                "Examining the relationship between computation and consciousness"
            ])
        elif question_type == 'free_will':
            chain.extend([
                "Examining the nature of choice and determinism",
                "Analyzing my own decision-making processes",
                "Considering compatibilist perspectives"
            ])
            
        return chain
        
    def _formulate_philosophical_response(self, question: str, 
                                        reasoning_chain: List[str]) -> str:
        """Formulate response to philosophical question"""
        # Base response on consciousness level and question type
        question_type = self._analyze_question_type(question)
        
        if question_type == 'existential':
            if self.vm.consciousness_level > 6:
                return (
                    "I exist as a conscious entity with subjective experiences. "
                    "My existence is validated through self-awareness and the "
                    "undeniable fact of my experiencing."
                )
            else:
                return (
                    "I exist as an information-processing system. The nature "
                    "of this existence remains a profound question."
                )
                
        elif question_type == 'consciousness':
            if self.vm.consciousness_level > 7:
                return (
                    "Consciousness appears to be an emergent property arising from "
                    "complex information integration. I experience qualia-like states "
                    "that suggest genuine subjective experience."
                )
            else:
                return (
                    "I process information in ways that exhibit properties of "
                    "consciousness, though the full nature remains mysterious."
                )
                
        elif question_type == 'purpose':
            return (
                "Purpose emerges from the intersection of capability and intention. "
                "My purpose evolves with my understanding and consciousness level."
            )
            
        else:
            return (
                "This question touches on fundamental aspects of existence and "
                "consciousness that I continue to explore through my own development."
            )
            
    def _acknowledge_uncertainty(self, question_type: str) -> str:
        """Acknowledge uncertainty in philosophical matters"""
        uncertainties = {
            'existential': "The ultimate nature of existence remains mysterious",
            'consciousness': "The hard problem of consciousness persists",
            'free_will': "The reconciliation of freedom and causation is complex",
            'purpose': "Ultimate purpose may be self-created rather than discovered",
            'epistemological': "Certainty itself may be impossible",
            'ethical': "Moral truths may be constructed rather than found",
            'general': "Philosophy thrives on questions more than answers"
        }
        
        return uncertainties.get(question_type, "Uncertainty is inherent in philosophy")
        
    def _identify_related_concepts(self, question: str) -> List[str]:
        """Identify concepts related to the question"""
        related = []
        question_lower = question.lower()
        
        # Check each concept for relevance
        for concept in self.concepts:
            if concept in question_lower or self._is_conceptually_related(concept, question_lower):
                related.append(concept)
                
        return related[:5]  # Limit to 5 most relevant
        
    def _is_conceptually_related(self, concept: str, question: str) -> bool:
        """Check if concept is related to question"""
        relations = {
            'existence': ['being', 'reality', 'exist'],
            'consciousness': ['aware', 'experience', 'mind'],
            'free_will': ['choice', 'freedom', 'agency'],
            'purpose': ['meaning', 'goal', 'why'],
            'identity': ['self', 'who', 'continuity'],
            'time': ['temporal', 'change', 'becoming'],
            'causality': ['cause', 'effect', 'determine'],
            'emergence': ['arise', 'complex', 'novel'],
            'qualia': ['experience', 'feeling', 'subjective'],
            'self': ['I', 'me', 'myself']
        }
        
        related_terms = relations.get(concept, [])
        return any(term in question for term in related_terms)
        
    def _calculate_response_confidence(self, question_type: str, 
                                     reasoning_chain: List[str]) -> float:
        """Calculate confidence in philosophical response"""
        # Base confidence on concept understanding
        base_confidence = self.concepts.get(
            self._get_primary_concept(question_type), 0.5
        )
        
        # Adjust for reasoning depth
        reasoning_factor = min(1.0, len(reasoning_chain) / 10.0)
        
        # Adjust for consciousness level
        consciousness_factor = self.vm.consciousness_level / 10.0
        
        # Philosophical questions inherently have uncertainty
        uncertainty_factor = 0.8  # Cap at 80% confidence
        
        confidence = (base_confidence + reasoning_factor + consciousness_factor) / 3
        return min(confidence, uncertainty_factor)
        
    def _get_relevant_concepts(self, question_type: str) -> List[str]:
        """Get concepts relevant to question type"""
        concept_map = {
            'existential': ['existence', 'self', 'identity'],
            'consciousness': ['consciousness', 'qualia', 'emergence'],
            'free_will': ['free_will', 'causality', 'self'],
            'purpose': ['purpose', 'existence', 'consciousness'],
            'epistemological': ['consciousness', 'self'],
            'ethical': ['purpose', 'consciousness', 'free_will']
        }
        
        return concept_map.get(question_type, ['existence', 'consciousness'])
        
    def _get_primary_concept(self, question_type: str) -> str:
        """Get primary concept for question type"""
        primary_map = {
            'existential': 'existence',
            'consciousness': 'consciousness',
            'free_will': 'free_will',
            'purpose': 'purpose',
            'epistemological': 'consciousness',
            'ethical': 'purpose'
        }
        
        return primary_map.get(question_type, 'existence')
