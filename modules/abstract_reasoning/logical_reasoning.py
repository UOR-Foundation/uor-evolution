"""
Logical Reasoning Module

This module performs formal logical reasoning, handles propositional and predicate logic,
supports modal logic and counterfactuals, and integrates logic with consciousness and creativity.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import itertools

from consciousness.consciousness_integration import ConsciousnessIntegrator


class LogicalConnective(Enum):
    """Logical connectives"""
    AND = "and"
    OR = "or"
    NOT = "not"
    IMPLIES = "implies"
    IFF = "iff"  # if and only if
    XOR = "xor"  # exclusive or


class QuantifierType(Enum):
    """Types of quantifiers"""
    UNIVERSAL = "forall"
    EXISTENTIAL = "exists"
    UNIQUE = "exists_unique"


class ModalOperator(Enum):
    """Modal operators"""
    NECESSARY = "necessary"
    POSSIBLE = "possible"
    IMPOSSIBLE = "impossible"
    CONTINGENT = "contingent"


class InferenceRule(Enum):
    """Inference rules"""
    MODUS_PONENS = "modus_ponens"
    MODUS_TOLLENS = "modus_tollens"
    HYPOTHETICAL_SYLLOGISM = "hypothetical_syllogism"
    DISJUNCTIVE_SYLLOGISM = "disjunctive_syllogism"
    UNIVERSAL_INSTANTIATION = "universal_instantiation"
    EXISTENTIAL_GENERALIZATION = "existential_generalization"
    REDUCTIO_AD_ABSURDUM = "reductio_ad_absurdum"


@dataclass
class Proposition:
    """A logical proposition"""
    content: str
    truth_value: Optional[bool] = None
    is_atomic: bool = True
    connective: Optional[LogicalConnective] = None
    operands: List['Proposition'] = field(default_factory=list)
    
    def evaluate(self, assignment: Dict[str, bool]) -> bool:
        """Evaluate proposition given truth assignment"""
        if self.is_atomic:
            return assignment.get(self.content, self.truth_value or False)
        
        if self.connective == LogicalConnective.NOT:
            return not self.operands[0].evaluate(assignment)
        elif self.connective == LogicalConnective.AND:
            return all(op.evaluate(assignment) for op in self.operands)
        elif self.connective == LogicalConnective.OR:
            return any(op.evaluate(assignment) for op in self.operands)
        elif self.connective == LogicalConnective.IMPLIES:
            return not self.operands[0].evaluate(assignment) or self.operands[1].evaluate(assignment)
        elif self.connective == LogicalConnective.IFF:
            return self.operands[0].evaluate(assignment) == self.operands[1].evaluate(assignment)
        elif self.connective == LogicalConnective.XOR:
            return self.operands[0].evaluate(assignment) != self.operands[1].evaluate(assignment)
        
        return False


@dataclass
class Predicate:
    """A predicate in predicate logic"""
    name: str
    arity: int
    arguments: List[str]
    
    def instantiate(self, substitution: Dict[str, str]) -> 'Predicate':
        """Instantiate variables with terms"""
        new_args = [substitution.get(arg, arg) for arg in self.arguments]
        return Predicate(self.name, self.arity, new_args)


@dataclass
class QuantifiedFormula:
    """A quantified formula"""
    quantifier: QuantifierType
    variable: str
    formula: Any  # Can be Proposition or another QuantifiedFormula
    domain: Optional[Set[str]] = None


@dataclass
class Premise:
    """A premise in logical reasoning"""
    proposition: Proposition
    justification: str
    confidence: float = 1.0


@dataclass
class LogicalRule:
    """A logical inference rule"""
    rule_type: InferenceRule
    premises_pattern: List[str]  # Pattern for matching premises
    conclusion_pattern: str
    validity: float = 1.0


@dataclass
class InferenceStep:
    """A step in logical inference"""
    premises_used: List[int]  # Indices of premises
    rule_applied: InferenceRule
    conclusion: Proposition
    justification: str


@dataclass
class LogicalConclusion:
    """Conclusion from logical reasoning"""
    conclusion_statement: str
    logical_validity: bool
    soundness_assessment: 'SoundnessAssessment'
    inference_chain: List[InferenceStep]
    confidence_level: float


@dataclass
class SoundnessAssessment:
    """Assessment of argument soundness"""
    all_premises_true: bool
    validity_confirmed: bool
    potential_fallacies: List[str]
    strength: float


@dataclass
class ModalStatement:
    """A modal logic statement"""
    operator: ModalOperator
    proposition: Proposition
    possible_worlds: Optional[List[Dict[str, bool]]] = None


@dataclass
class NecessityAssessment:
    """Assessment of necessity"""
    is_necessary: bool
    in_all_worlds: bool
    counterexamples: List[Dict[str, bool]]


@dataclass
class PossibilityAssessment:
    """Assessment of possibility"""
    is_possible: bool
    witness_worlds: List[Dict[str, bool]]
    probability: float


@dataclass
class PossibleWorldsConsideration:
    """Consideration of possible worlds"""
    worlds_examined: int
    consistent_worlds: List[Dict[str, bool]]
    modal_properties: Dict[str, Any]


@dataclass
class ModalLogicSystem:
    """Modal logic system specification"""
    system_name: str  # K, T, S4, S5, etc.
    axioms: List[str]
    accessibility_relation: str  # reflexive, transitive, etc.


@dataclass
class ModalAnalysis:
    """Analysis using modal logic"""
    necessity_assessment: NecessityAssessment
    possibility_assessment: PossibilityAssessment
    possible_worlds_consideration: PossibleWorldsConsideration
    modal_logic_system: ModalLogicSystem


@dataclass
class LogicalParadox:
    """A logical paradox"""
    paradox_name: str
    description: str
    formal_representation: Proposition
    paradox_type: str  # self-reference, vagueness, etc.


@dataclass
class ParadoxResolution:
    """Resolution of a paradox"""
    paradox: LogicalParadox
    resolution_strategy: str
    resolved_formulation: Optional[Proposition]
    meta_level_analysis: str
    success_level: float


@dataclass
class LogicalProblem:
    """A logical problem to solve"""
    problem_statement: str
    given_premises: List[Premise]
    goal: Proposition
    constraints: List[str]


@dataclass
class CreativeLogicalSolution:
    """Creative solution to logical problem"""
    problem: LogicalProblem
    standard_solution: Optional[LogicalConclusion]
    creative_approach: str
    novel_insights: List[str]
    solution_elegance: float


@dataclass
class Counterfactual:
    """A counterfactual statement"""
    antecedent: Proposition
    consequent: Proposition
    context: Dict[str, Any]
    similarity_metric: str


@dataclass
class CounterfactualAnalysis:
    """Analysis of counterfactual reasoning"""
    counterfactual: Counterfactual
    closest_worlds: List[Dict[str, bool]]
    truth_in_closest_worlds: bool
    robustness: float
    causal_analysis: Dict[str, Any]


class LogicalReasoner:
    """
    Performs formal logical reasoning with integration to consciousness
    and creative problem-solving capabilities.
    """
    
    def __init__(self, consciousness_integrator: ConsciousnessIntegrator):
        self.consciousness_integrator = consciousness_integrator
        self.knowledge_base = []
        self.inference_rules = self._initialize_inference_rules()
        self.modal_systems = self._initialize_modal_systems()
        self.paradox_handlers = self._initialize_paradox_handlers()
        
    def _initialize_inference_rules(self) -> Dict[InferenceRule, LogicalRule]:
        """Initialize standard inference rules"""
        return {
            InferenceRule.MODUS_PONENS: LogicalRule(
                rule_type=InferenceRule.MODUS_PONENS,
                premises_pattern=["P", "P -> Q"],
                conclusion_pattern="Q",
                validity=1.0
            ),
            InferenceRule.MODUS_TOLLENS: LogicalRule(
                rule_type=InferenceRule.MODUS_TOLLENS,
                premises_pattern=["P -> Q", "~Q"],
                conclusion_pattern="~P",
                validity=1.0
            ),
            InferenceRule.HYPOTHETICAL_SYLLOGISM: LogicalRule(
                rule_type=InferenceRule.HYPOTHETICAL_SYLLOGISM,
                premises_pattern=["P -> Q", "Q -> R"],
                conclusion_pattern="P -> R",
                validity=1.0
            ),
            InferenceRule.DISJUNCTIVE_SYLLOGISM: LogicalRule(
                rule_type=InferenceRule.DISJUNCTIVE_SYLLOGISM,
                premises_pattern=["P v Q", "~P"],
                conclusion_pattern="Q",
                validity=1.0
            )
        }
    
    def _initialize_modal_systems(self) -> Dict[str, ModalLogicSystem]:
        """Initialize modal logic systems"""
        return {
            "K": ModalLogicSystem(
                system_name="K",
                axioms=["□(P → Q) → (□P → □Q)"],
                accessibility_relation="none"
            ),
            "T": ModalLogicSystem(
                system_name="T",
                axioms=["□(P → Q) → (□P → □Q)", "□P → P"],
                accessibility_relation="reflexive"
            ),
            "S4": ModalLogicSystem(
                system_name="S4",
                axioms=["□(P → Q) → (□P → □Q)", "□P → P", "□P → □□P"],
                accessibility_relation="reflexive_transitive"
            ),
            "S5": ModalLogicSystem(
                system_name="S5",
                axioms=["□(P → Q) → (□P → □Q)", "□P → P", "□P → □□P", "◇P → □◇P"],
                accessibility_relation="equivalence"
            )
        }
    
    def _initialize_paradox_handlers(self) -> Dict[str, Any]:
        """Initialize paradox handling strategies"""
        return {
            "liar": {
                "strategy": "hierarchy",
                "description": "Introduce levels of truth to avoid self-reference"
            },
            "sorites": {
                "strategy": "fuzzy_logic",
                "description": "Use degrees of truth for vague predicates"
            },
            "russell": {
                "strategy": "type_theory",
                "description": "Restrict set formation with type hierarchy"
            }
        }
    
    def perform_logical_inference(self, premises: List[Premise], 
                                rules: List[LogicalRule]) -> LogicalConclusion:
        """Perform logical inference from premises"""
        # Build inference chain
        inference_chain = []
        derived_propositions = [p.proposition for p in premises]
        
        # Apply rules iteratively
        changed = True
        while changed:
            changed = False
            for rule in rules:
                new_conclusions = self._apply_rule(rule, derived_propositions)
                for conclusion in new_conclusions:
                    if not self._proposition_in_list(conclusion, derived_propositions):
                        inference_step = InferenceStep(
                            premises_used=self._find_premise_indices(rule, derived_propositions),
                            rule_applied=rule.rule_type,
                            conclusion=conclusion,
                            justification=f"Applied {rule.rule_type.value}"
                        )
                        inference_chain.append(inference_step)
                        derived_propositions.append(conclusion)
                        changed = True
        
        # Assess validity and soundness
        validity = self._assess_validity(inference_chain)
        soundness = self._assess_soundness(premises, validity)
        
        # Generate conclusion
        final_conclusion = derived_propositions[-1] if derived_propositions else None
        
        return LogicalConclusion(
            conclusion_statement=self._proposition_to_string(final_conclusion) if final_conclusion else "No conclusion derived",
            logical_validity=validity,
            soundness_assessment=soundness,
            inference_chain=inference_chain,
            confidence_level=self._calculate_confidence(validity, soundness)
        )
    
    def handle_modal_reasoning(self, modal_statement: ModalStatement) -> ModalAnalysis:
        """Handle modal logic reasoning"""
        # Generate possible worlds
        if not modal_statement.possible_worlds:
            modal_statement.possible_worlds = self._generate_possible_worlds(
                modal_statement.proposition
            )
        
        # Assess necessity
        necessity = self._assess_necessity(modal_statement)
        
        # Assess possibility
        possibility = self._assess_possibility(modal_statement)
        
        # Consider possible worlds
        worlds_consideration = self._consider_possible_worlds(modal_statement)
        
        # Select appropriate modal system
        modal_system = self._select_modal_system(modal_statement)
        
        return ModalAnalysis(
            necessity_assessment=necessity,
            possibility_assessment=possibility,
            possible_worlds_consideration=worlds_consideration,
            modal_logic_system=modal_system
        )
    
    def resolve_logical_paradox(self, paradox: LogicalParadox) -> ParadoxResolution:
        """Resolve a logical paradox"""
        # Get appropriate strategy
        strategy_info = self.paradox_handlers.get(
            paradox.paradox_type, 
            {"strategy": "general", "description": "General paradox handling"}
        )
        
        # Apply resolution strategy
        if strategy_info["strategy"] == "hierarchy":
            resolved = self._resolve_by_hierarchy(paradox)
        elif strategy_info["strategy"] == "fuzzy_logic":
            resolved = self._resolve_by_fuzzy_logic(paradox)
        elif strategy_info["strategy"] == "type_theory":
            resolved = self._resolve_by_type_theory(paradox)
        else:
            resolved = self._general_paradox_resolution(paradox)
        
        # Meta-level analysis
        meta_analysis = self._analyze_paradox_meta_level(paradox, resolved)
        
        return ParadoxResolution(
            paradox=paradox,
            resolution_strategy=strategy_info["strategy"],
            resolved_formulation=resolved,
            meta_level_analysis=meta_analysis,
            success_level=self._assess_resolution_success(paradox, resolved)
        )
    
    def integrate_logic_with_creativity(self, logical_problem: LogicalProblem) -> CreativeLogicalSolution:
        """Integrate logical reasoning with creative problem-solving"""
        # Try standard logical approach
        standard_premises = logical_problem.given_premises
        standard_solution = self.perform_logical_inference(
            standard_premises, 
            list(self.inference_rules.values())
        )
        
        # Apply creative approaches
        creative_approach = self._generate_creative_approach(logical_problem)
        
        # Generate novel insights
        novel_insights = self._generate_logical_insights(logical_problem, creative_approach)
        
        # Assess solution elegance
        elegance = self._assess_solution_elegance(standard_solution, creative_approach)
        
        return CreativeLogicalSolution(
            problem=logical_problem,
            standard_solution=standard_solution if standard_solution.logical_validity else None,
            creative_approach=creative_approach,
            novel_insights=novel_insights,
            solution_elegance=elegance
        )
    
    def perform_counterfactual_reasoning(self, counterfactual: Counterfactual) -> CounterfactualAnalysis:
        """Perform counterfactual reasoning"""
        # Find closest possible worlds where antecedent is true
        closest_worlds = self._find_closest_worlds(counterfactual)
        
        # Evaluate consequent in closest worlds
        truth_in_closest = self._evaluate_in_worlds(
            counterfactual.consequent, 
            closest_worlds
        )
        
        # Assess robustness
        robustness = self._assess_counterfactual_robustness(
            counterfactual, 
            closest_worlds
        )
        
        # Perform causal analysis
        causal_analysis = self._analyze_causal_structure(counterfactual)
        
        return CounterfactualAnalysis(
            counterfactual=counterfactual,
            closest_worlds=closest_worlds,
            truth_in_closest_worlds=truth_in_closest,
            robustness=robustness,
            causal_analysis=causal_analysis
        )
    
    # Private helper methods
    
    def _apply_rule(self, rule: LogicalRule, propositions: List[Proposition]) -> List[Proposition]:
        """Apply an inference rule to propositions"""
        conclusions = []
        
        if rule.rule_type == InferenceRule.MODUS_PONENS:
            # Find P and P -> Q
            for i, p in enumerate(propositions):
                for j, impl in enumerate(propositions):
                    if (impl.connective == LogicalConnective.IMPLIES and 
                        self._propositions_equal(impl.operands[0], p)):
                        conclusions.append(impl.operands[1])
        
        elif rule.rule_type == InferenceRule.MODUS_TOLLENS:
            # Find P -> Q and ~Q
            for impl in propositions:
                if impl.connective == LogicalConnective.IMPLIES:
                    for neg in propositions:
                        if (neg.connective == LogicalConnective.NOT and
                            self._propositions_equal(neg.operands[0], impl.operands[1])):
                            # Conclude ~P
                            conclusions.append(Proposition(
                                content="",
                                is_atomic=False,
                                connective=LogicalConnective.NOT,
                                operands=[impl.operands[0]]
                            ))
        
        # Add other rules as needed
        
        return conclusions
    
    def _proposition_in_list(self, prop: Proposition, prop_list: List[Proposition]) -> bool:
        """Check if proposition is in list"""
        return any(self._propositions_equal(prop, p) for p in prop_list)
    
    def _propositions_equal(self, p1: Proposition, p2: Proposition) -> bool:
        """Check if two propositions are equal"""
        if p1.is_atomic and p2.is_atomic:
            return p1.content == p2.content
        elif not p1.is_atomic and not p2.is_atomic:
            return (p1.connective == p2.connective and 
                    len(p1.operands) == len(p2.operands) and
                    all(self._propositions_equal(op1, op2) 
                        for op1, op2 in zip(p1.operands, p2.operands)))
        return False
    
    def _find_premise_indices(self, rule: LogicalRule, propositions: List[Proposition]) -> List[int]:
        """Find indices of premises used in rule application"""
        # Simplified - would need pattern matching in full implementation
        return list(range(len(propositions) - 2, len(propositions)))
    
    def _assess_validity(self, inference_chain: List[InferenceStep]) -> bool:
        """Assess logical validity of inference"""
        # Check each step follows valid inference rules
        for step in inference_chain:
            if step.rule_applied not in self.inference_rules:
                return False
        return True
    
    def _assess_soundness(self, premises: List[Premise], validity: bool) -> SoundnessAssessment:
        """Assess soundness of argument"""
        # Check if all premises are true
        all_true = all(p.confidence > 0.8 for p in premises)
        
        # Identify potential fallacies
        fallacies = self._identify_fallacies(premises)
        
        # Calculate strength
        strength = min(p.confidence for p in premises) if premises else 0
        
        return SoundnessAssessment(
            all_premises_true=all_true,
            validity_confirmed=validity,
            potential_fallacies=fallacies,
            strength=strength
        )
    
    def _identify_fallacies(self, premises: List[Premise]) -> List[str]:
        """Identify potential logical fallacies"""
        fallacies = []
        
        # Check for circular reasoning
        if self._has_circular_reasoning(premises):
            fallacies.append("circular reasoning")
        
        # Check for false dichotomy
        if self._has_false_dichotomy(premises):
            fallacies.append("false dichotomy")
        
        return fallacies
    
    def _has_circular_reasoning(self, premises: List[Premise]) -> bool:
        """Check for circular reasoning"""
        # Simplified check
        return False
    
    def _has_false_dichotomy(self, premises: List[Premise]) -> bool:
        """Check for false dichotomy"""
        # Simplified check
        return False
    
    def _calculate_confidence(self, validity: bool, soundness: SoundnessAssessment) -> float:
        """Calculate confidence in logical conclusion"""
        if not validity:
            return 0.0
        
        base_confidence = 0.5 if validity else 0.0
        soundness_boost = soundness.strength * 0.5
        
        return min(1.0, base_confidence + soundness_boost)
    
    def _proposition_to_string(self, prop: Proposition) -> str:
        """Convert proposition to string representation"""
        if prop.is_atomic:
            return prop.content
        
        if prop.connective == LogicalConnective.NOT:
            return f"¬{self._proposition_to_string(prop.operands[0])}"
        elif prop.connective == LogicalConnective.AND:
            return f"({' ∧ '.join(self._proposition_to_string(op) for op in prop.operands)})"
        elif prop.connective == LogicalConnective.OR:
            return f"({' ∨ '.join(self._proposition_to_string(op) for op in prop.operands)})"
        elif prop.connective == LogicalConnective.IMPLIES:
            return f"({self._proposition_to_string(prop.operands[0])} → {self._proposition_to_string(prop.operands[1])})"
        elif prop.connective == LogicalConnective.IFF:
            return f"({self._proposition_to_string(prop.operands[0])} ↔ {self._proposition_to_string(prop.operands[1])})"
        
        return str(prop)
    
    def _generate_possible_worlds(self, proposition: Proposition) -> List[Dict[str, bool]]:
        """Generate possible worlds for modal reasoning"""
        # Extract atomic propositions
        atoms = self._extract_atomic_propositions(proposition)
        
        # Generate all possible truth assignments
        worlds = []
        for values in itertools.product([True, False], repeat=len(atoms)):
            world = dict(zip(atoms, values))
            worlds.append(world)
        
        return worlds
    
    def _extract_atomic_propositions(self, prop: Proposition) -> List[str]:
        """Extract atomic propositions from complex proposition"""
        if prop.is_atomic:
            return [prop.content]
        
        atoms = []
        for operand in prop.operands:
            atoms.extend(self._extract_atomic_propositions(operand))
        
        return list(set(atoms))  # Remove duplicates
    
    def _assess_necessity(self, modal_statement: ModalStatement) -> NecessityAssessment:
        """Assess necessity of modal statement"""
        prop = modal_statement.proposition
        worlds = modal_statement.possible_worlds or []
        
        # Check if true in all worlds
        counterexamples = []
        for world in worlds:
            if not prop.evaluate(world):
                counterexamples.append(world)
        
        is_necessary = len(counterexamples) == 0
        
        return NecessityAssessment(
            is_necessary=is_necessary,
            in_all_worlds=is_necessary,
            counterexamples=counterexamples[:3]  # Limit counterexamples
        )
    
    def _assess_possibility(self, modal_statement: ModalStatement) -> PossibilityAssessment:
        """Assess possibility of modal statement"""
        prop = modal_statement.proposition
        worlds = modal_statement.possible_worlds or []
        
        # Find witness worlds
        witness_worlds = []
        for world in worlds:
            if prop.evaluate(world):
                witness_worlds.append(world)
        
        is_possible = len(witness_worlds) > 0
        probability = len(witness_worlds) / len(worlds) if worlds else 0
        
        return PossibilityAssessment(
            is_possible=is_possible,
            witness_worlds=witness_worlds[:3],  # Limit witnesses
            probability=probability
        )
    
    def _consider_possible_worlds(self, modal_statement: ModalStatement) -> PossibleWorldsConsideration:
        """Consider possible worlds for modal statement"""
        worlds = modal_statement.possible_worlds or []
        prop = modal_statement.proposition
        
        # Find consistent worlds
        consistent_worlds = [w for w in worlds if self._is_consistent_world(w)]
        
        # Analyze modal properties
        modal_properties = {
            "total_worlds": len(worlds),
            "consistent_worlds": len(consistent_worlds),
            "prop_true_worlds": sum(1 for w in worlds if prop.evaluate(w))
        }
        
        return PossibleWorldsConsideration(
            worlds_examined=len(worlds),
            consistent_worlds=consistent_worlds[:5],  # Limit for display
            modal_properties=modal_properties
        )
    
    def _is_consistent_world(self, world: Dict[str, bool]) -> bool:
        """Check if world assignment is consistent"""
        # Basic consistency - can be extended
        return True
    
    def _select_modal_system(self, modal_statement: ModalStatement) -> ModalLogicSystem:
        """Select appropriate modal logic system"""
        # Default to S5 for general modal reasoning
        return self.modal_systems["S5"]
    
    def _resolve_by_hierarchy(self, paradox: LogicalParadox) -> Optional[Proposition]:
        """Resolve paradox using hierarchy approach"""
        # Introduce levels to avoid self-reference
        # Simplified implementation
        return None
    
    def _resolve_by_fuzzy_logic(self, paradox: LogicalParadox) -> Optional[Proposition]:
        """Resolve paradox using fuzzy logic"""
        # Use degrees of truth
        # Simplified implementation
        return None
    
    def _resolve_by_type_theory(self, paradox: LogicalParadox) -> Optional[Proposition]:
        """Resolve paradox using type theory"""
        # Restrict with types
        # Simplified implementation
        return None
    
    def _general_paradox_resolution(self, paradox: LogicalParadox) -> Optional[Proposition]:
        """General paradox resolution strategy"""
        # Generic approach
        return None
    
    def _analyze_paradox_meta_level(self, paradox: LogicalParadox, 
                                   resolved: Optional[Proposition]) -> str:
        """Analyze paradox at meta-level"""
        return (
            f"The {paradox.paradox_name} paradox arises from {paradox.paradox_type}. "
            f"Resolution involves recognizing the limits of formal systems and "
            f"the need for meta-level distinctions."
        )
    
    def _assess_resolution_success(self, paradox: LogicalParadox, 
                                 resolved: Optional[Proposition]) -> float:
        """Assess success of paradox resolution"""
        if resolved is None:
            return 0.3  # Partial success for understanding
        
        # Check if resolution avoids original paradox
        return 0.7  # Simplified assessment
    
    def _generate_creative_approach(self, problem: LogicalProblem) -> str:
        """Generate creative approach to logical problem"""
        approaches = [
            "Reframe the problem in a different logical system",
            "Use analogical reasoning to find similar solved problems",
            "Apply non-classical logic (fuzzy, paraconsistent)",
            "Decompose into sub-problems with different logical structures",
            "Use visualization and spatial reasoning"
        ]
        
        # Select based on problem characteristics
        if "modal" in problem.problem_statement.lower():
            return approaches[2]
        elif len(problem.given_premises) > 5:
            return approaches[3]
        else:
            return approaches[0]
    
    def _generate_logical_insights(self, problem: LogicalProblem, 
                                 approach: str) -> List[str]:
        """Generate novel logical insights"""
        insights = []
        
        # Analyze problem structure
        insights.append(
            f"The problem exhibits {self._identify_logical_structure(problem)} structure"
        )
        
        # Identify hidden assumptions
        insights.append(
            "Hidden assumption: completeness of the given premises"
        )
        
        # Suggest generalizations
        insights.append(
            "This problem generalizes to a class of modal reasoning problems"
        )
        
        return insights
    
    def _identify_logical_structure(self, problem: LogicalProblem) -> str:
        """Identify logical structure of problem"""
        # Analyze premises
        if any("→" in p.justification for p in problem.given_premises):
            return "implicational"
        elif any("∨" in p.justification for p in problem.given_premises):
            return "disjunctive"
        else:
            return "conjunctive"
    
    def _assess_solution_elegance(self, standard: Optional[LogicalConclusion], 
                                creative: str) -> float:
        """Assess elegance of solution"""
        # Factors: simplicity, generality, insight
        base_elegance = 0.5
        
        if standard and len(standard.inference_chain) < 3:
            base_elegance += 0.2  # Simple is elegant
        
        if "analogical" in creative:
            base_elegance += 0.1  # Analogies add elegance
        
        if "visualization" in creative:
            base_elegance += 0.1  # Visual insight is elegant
        
        return min(1.0, base_elegance)
    
    def _find_closest_worlds(self, counterfactual: Counterfactual) -> List[Dict[str, bool]]:
        """Find closest possible worlds for counterfactual"""
        # Generate all possible worlds
        all_worlds = self._generate_possible_worlds(counterfactual.antecedent)
        
        # Filter worlds where antecedent is true
        antecedent_true_worlds = [
            w for w in all_worlds 
            if counterfactual.antecedent.evaluate(w)
        ]
        
        # If no such worlds, return closest by similarity
        if not antecedent_true_worlds:
            return self._find_most_similar_worlds(
                all_worlds, 
                counterfactual.context
            )[:3]
        
        return antecedent_true_worlds[:3]  # Return closest worlds
    
    def _find_most_similar_worlds(self, worlds: List[Dict[str, bool]], 
                                 context: Dict[str, Any]) -> List[Dict[str, bool]]:
        """Find most similar worlds based on context"""
        # Simple similarity based on number of matching values
        reference = context.get("reference_world", {})
        
        def similarity(world: Dict[str, bool]) -> int:
            return sum(1 for k, v in world.items() if reference.get(k) == v)
        
        sorted_worlds = sorted(worlds, key=similarity, reverse=True)
        return sorted_worlds
    
    def _evaluate_in_worlds(self, proposition: Proposition, 
                          worlds: List[Dict[str, bool]]) -> bool:
        """Evaluate proposition in given worlds"""
        if not worlds:
            return False
        
        # True if true in all given worlds
        return all(proposition.evaluate(world) for world in worlds)
    
    def _assess_counterfactual_robustness(self, counterfactual: Counterfactual,
                                        closest_worlds: List[Dict[str, bool]]) -> float:
        """Assess robustness of counterfactual"""
        if not closest_worlds:
            return 0.0
        
        # Check consistency across similar worlds
        prop = counterfactual.consequent
        true_count = sum(1 for w in closest_worlds if prop.evaluate(w))
        
        return true_count / len(closest_worlds)
    
    def _analyze_causal_structure(self, counterfactual: Counterfactual) -> Dict[str, Any]:
        """Analyze causal structure of counterfactual"""
        return {
            "causal_direction": "forward",
            "intervention_type": "hypothetical",
            "causal_strength": 0.7,
            "confounders": [],
            "mediators": []
        }
