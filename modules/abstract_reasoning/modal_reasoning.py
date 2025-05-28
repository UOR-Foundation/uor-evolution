"""
Modal Reasoning Module

This module handles modal logic reasoning about possibility, necessity,
knowledge, belief, and other modal concepts.
"""

from typing import List, Dict, Any, Optional, Set, Tuple
from dataclasses import dataclass, field
from enum import Enum
import itertools

from consciousness.consciousness_integration import ConsciousnessIntegrator
from modules.abstract_reasoning.logical_reasoning import Proposition, LogicalConnective


class ModalType(Enum):
    """Types of modalities"""
    ALETHIC = "alethic"  # Necessity/possibility
    EPISTEMIC = "epistemic"  # Knowledge/belief
    DEONTIC = "deontic"  # Obligation/permission
    TEMPORAL = "temporal"  # Always/sometimes
    DOXASTIC = "doxastic"  # Belief
    DYNAMIC = "dynamic"  # Action/ability


class ModalOperator(Enum):
    """Modal operators"""
    # Alethic modality
    NECESSARY = "□"  # Box - necessary
    POSSIBLE = "◇"  # Diamond - possible
    
    # Epistemic modality
    KNOWS = "K"  # Knowledge
    BELIEVES = "B"  # Belief
    
    # Deontic modality
    OBLIGATORY = "O"  # Obligation
    PERMITTED = "P"  # Permission
    FORBIDDEN = "F"  # Forbidden
    
    # Temporal modality
    ALWAYS = "G"  # Always/globally
    EVENTUALLY = "F"  # Eventually/future
    
    # Dynamic modality
    CAN_DO = "⟨⟩"  # Ability
    AFTER_DOING = "[]"  # After action


class AccessibilityRelation(Enum):
    """Types of accessibility relations"""
    REFLEXIVE = "reflexive"  # Every world accesses itself
    SYMMETRIC = "symmetric"  # If w1→w2 then w2→w1
    TRANSITIVE = "transitive"  # If w1→w2 and w2→w3 then w1→w3
    EUCLIDEAN = "euclidean"  # If w1→w2 and w1→w3 then w2→w3
    SERIAL = "serial"  # Every world accesses at least one world
    EQUIVALENCE = "equivalence"  # Reflexive + symmetric + transitive


@dataclass
class PossibleWorld:
    """A possible world in modal logic"""
    world_id: str
    propositions: Dict[str, bool]  # Truth values of propositions
    properties: Dict[str, Any] = field(default_factory=dict)
    
    def evaluate_proposition(self, prop: Proposition) -> bool:
        """Evaluate proposition in this world"""
        if prop.is_atomic:
            return self.propositions.get(prop.content, False)
        
        # Handle complex propositions
        if prop.connective == LogicalConnective.NOT:
            return not self.evaluate_proposition(prop.operands[0])
        elif prop.connective == LogicalConnective.AND:
            return all(self.evaluate_proposition(op) for op in prop.operands)
        elif prop.connective == LogicalConnective.OR:
            return any(self.evaluate_proposition(op) for op in prop.operands)
        elif prop.connective == LogicalConnective.IMPLIES:
            return (not self.evaluate_proposition(prop.operands[0]) or 
                    self.evaluate_proposition(prop.operands[1]))
        
        return False


@dataclass
class ModalFormula:
    """A modal logic formula"""
    operator: ModalOperator
    operand: Any  # Can be Proposition or another ModalFormula
    agent: Optional[str] = None  # For multi-agent modalities
    action: Optional[str] = None  # For dynamic modalities


@dataclass
class KripkeModel:
    """Kripke model for modal logic"""
    worlds: Dict[str, PossibleWorld]
    accessibility: Dict[str, Set[str]]  # World -> accessible worlds
    actual_world: str
    properties: AccessibilityRelation = AccessibilityRelation.SERIAL


@dataclass
class ModalContext:
    """Context for modal reasoning"""
    modal_type: ModalType
    agents: List[str] = field(default_factory=list)
    time_points: List[float] = field(default_factory=list)
    constraints: List[str] = field(default_factory=list)


@dataclass
class ModalInference:
    """An inference in modal logic"""
    premises: List[ModalFormula]
    conclusion: ModalFormula
    inference_rule: str
    validity: bool
    counterexample: Optional[KripkeModel] = None


@dataclass
class BeliefState:
    """Belief state of an agent"""
    agent: str
    beliefs: Set[Proposition]
    knowledge: Set[Proposition]
    uncertainties: Dict[Proposition, float]  # Proposition -> confidence


@dataclass
class ObligationState:
    """Deontic state"""
    obligations: Set[Proposition]
    permissions: Set[Proposition]
    prohibitions: Set[Proposition]
    conflicts: List[Tuple[Proposition, Proposition]]


class ModalReasoner:
    """
    Performs modal reasoning about possibility, necessity, knowledge, belief, etc.
    """
    
    def __init__(self, consciousness_integrator: ConsciousnessIntegrator):
        self.consciousness_integrator = consciousness_integrator
        self.modal_axioms = self._initialize_modal_axioms()
        self.inference_rules = self._initialize_inference_rules()
        self.kripke_models = {}
        
    def _initialize_modal_axioms(self) -> Dict[str, Dict[str, Any]]:
        """Initialize modal logic axioms"""
        return {
            "K": {  # Distribution axiom
                "formula": "□(p → q) → (□p → □q)",
                "systems": ["K", "T", "S4", "S5"],
                "description": "Necessity distributes over implication"
            },
            "T": {  # Reflexivity axiom
                "formula": "□p → p",
                "systems": ["T", "S4", "S5"],
                "description": "What is necessary is true"
            },
            "4": {  # Transitivity axiom
                "formula": "□p → □□p",
                "systems": ["S4", "S5"],
                "description": "Necessity implies necessary necessity"
            },
            "5": {  # Euclidean axiom
                "formula": "◇p → □◇p",
                "systems": ["S5"],
                "description": "Possibility implies necessary possibility"
            },
            "D": {  # Serial axiom
                "formula": "□p → ◇p",
                "systems": ["D", "KD45"],
                "description": "Necessity implies possibility"
            }
        }
    
    def _initialize_inference_rules(self) -> Dict[str, Any]:
        """Initialize modal inference rules"""
        return {
            "necessitation": {
                "pattern": "If ⊢ p then ⊢ □p",
                "description": "Theorems are necessary"
            },
            "modal_modus_ponens": {
                "pattern": "□(p → q), □p ⊢ □q",
                "description": "Modus ponens under necessity"
            },
            "possibility_introduction": {
                "pattern": "p ⊢ ◇p",
                "description": "Truth implies possibility"
            },
            "knowledge_distribution": {
                "pattern": "K(p → q), Kp ⊢ Kq",
                "description": "Knowledge is closed under implication"
            }
        }
    
    def create_kripke_model(self, worlds_data: Dict[str, Dict[str, bool]],
                          accessibility_data: Dict[str, List[str]],
                          properties: AccessibilityRelation = AccessibilityRelation.SERIAL) -> KripkeModel:
        """Create a Kripke model"""
        # Create worlds
        worlds = {}
        for world_id, props in worlds_data.items():
            worlds[world_id] = PossibleWorld(world_id, props)
        
        # Create accessibility relation
        accessibility = {w: set(accessible) for w, accessible in accessibility_data.items()}
        
        # Enforce properties
        accessibility = self._enforce_accessibility_properties(accessibility, properties)
        
        # Select actual world (first by default)
        actual_world = list(worlds.keys())[0] if worlds else "w0"
        
        model = KripkeModel(
            worlds=worlds,
            accessibility=accessibility,
            actual_world=actual_world,
            properties=properties
        )
        
        # Store model
        model_id = f"model_{len(self.kripke_models)}"
        self.kripke_models[model_id] = model
        
        return model
    
    def evaluate_modal_formula(self, formula: ModalFormula, 
                             model: KripkeModel,
                             world: Optional[str] = None) -> bool:
        """Evaluate modal formula in a world"""
        if world is None:
            world = model.actual_world
        
        # Handle different modal operators
        if formula.operator == ModalOperator.NECESSARY:
            # True if operand is true in all accessible worlds
            accessible_worlds = model.accessibility.get(world, set())
            return all(
                self._evaluate_in_world(formula.operand, model, w)
                for w in accessible_worlds
            )
        
        elif formula.operator == ModalOperator.POSSIBLE:
            # True if operand is true in at least one accessible world
            accessible_worlds = model.accessibility.get(world, set())
            return any(
                self._evaluate_in_world(formula.operand, model, w)
                for w in accessible_worlds
            )
        
        elif formula.operator == ModalOperator.KNOWS:
            # Epistemic logic - knowledge
            return self._evaluate_knowledge(formula, model, world)
        
        elif formula.operator == ModalOperator.BELIEVES:
            # Doxastic logic - belief
            return self._evaluate_belief(formula, model, world)
        
        elif formula.operator == ModalOperator.OBLIGATORY:
            # Deontic logic - obligation
            return self._evaluate_obligation(formula, model, world)
        
        return False
    
    def check_modal_validity(self, formula: ModalFormula,
                           modal_system: str = "S5") -> Tuple[bool, Optional[KripkeModel]]:
        """Check if formula is valid in given modal system"""
        # Generate all possible Kripke models up to certain size
        max_worlds = 3  # Limit for computational feasibility
        
        for num_worlds in range(1, max_worlds + 1):
            # Generate all possible truth assignments
            props = ["p", "q", "r"]  # Basic propositions
            
            for world_assignments in itertools.product([True, False], 
                                                      repeat=num_worlds * len(props)):
                # Create worlds
                worlds_data = {}
                for i in range(num_worlds):
                    world_props = {}
                    for j, prop in enumerate(props):
                        world_props[prop] = world_assignments[i * len(props) + j]
                    worlds_data[f"w{i}"] = world_props
                
                # Generate accessibility relations
                for accessibility_pattern in self._generate_accessibility_patterns(
                    num_worlds, modal_system
                ):
                    model = self.create_kripke_model(
                        worlds_data, 
                        accessibility_pattern,
                        self._get_system_properties(modal_system)
                    )
                    
                    # Check formula in all worlds
                    for world in model.worlds:
                        if not self.evaluate_modal_formula(formula, model, world):
                            return False, model  # Found counterexample
        
        return True, None  # Valid
    
    def reason_about_knowledge(self, agent: str, 
                             proposition: Proposition,
                             model: KripkeModel) -> Dict[str, Any]:
        """Reason about agent's knowledge"""
        # Create knowledge formula
        knows_formula = ModalFormula(
            operator=ModalOperator.KNOWS,
            operand=proposition,
            agent=agent
        )
        
        # Check if agent knows proposition
        knows = self.evaluate_modal_formula(knows_formula, model)
        
        # Check if agent knows that they know (positive introspection)
        knows_knows_formula = ModalFormula(
            operator=ModalOperator.KNOWS,
            operand=knows_formula,
            agent=agent
        )
        knows_that_knows = self.evaluate_modal_formula(knows_knows_formula, model)
        
        # Check if agent knows what they don't know (negative introspection)
        not_knows_formula = ModalFormula(
            operator=ModalOperator.KNOWS,
            operand=Proposition("", is_atomic=False, 
                              connective=LogicalConnective.NOT,
                              operands=[knows_formula]),
            agent=agent
        )
        knows_that_not_knows = self.evaluate_modal_formula(not_knows_formula, model)
        
        return {
            "knows": knows,
            "knows_that_knows": knows_that_knows,
            "knows_that_not_knows": knows_that_not_knows,
            "knowledge_complete": knows_that_knows or knows_that_not_knows
        }
    
    def analyze_belief_revision(self, agent: str,
                              current_beliefs: BeliefState,
                              new_information: Proposition) -> BeliefState:
        """Analyze belief revision with new information"""
        # AGM belief revision postulates
        new_beliefs = current_beliefs.beliefs.copy()
        new_knowledge = current_beliefs.knowledge.copy()
        
        # Check consistency with current beliefs
        if self._is_consistent_with_beliefs(new_information, current_beliefs):
            # Simple expansion
            new_beliefs.add(new_information)
            
            # Update uncertainties
            new_uncertainties = current_beliefs.uncertainties.copy()
            new_uncertainties[new_information] = 0.9  # High confidence in new info
        else:
            # Belief revision needed
            # Remove conflicting beliefs (simplified)
            conflicting = self._find_conflicting_beliefs(new_information, current_beliefs)
            for belief in conflicting:
                new_beliefs.discard(belief)
            
            # Add new information
            new_beliefs.add(new_information)
            
            # Update uncertainties
            new_uncertainties = self._revise_uncertainties(
                current_beliefs.uncertainties,
                conflicting,
                new_information
            )
        
        return BeliefState(
            agent=agent,
            beliefs=new_beliefs,
            knowledge=new_knowledge,
            uncertainties=new_uncertainties
        )
    
    def check_deontic_consistency(self, obligations: Set[Proposition],
                                permissions: Set[Proposition],
                                prohibitions: Set[Proposition]) -> ObligationState:
        """Check consistency of deontic modalities"""
        conflicts = []
        
        # Check obligation-prohibition conflicts
        for obligation in obligations:
            for prohibition in prohibitions:
                if self._propositions_equivalent(obligation, prohibition):
                    conflicts.append((obligation, prohibition))
        
        # Check permission-prohibition conflicts
        for permission in permissions:
            for prohibition in prohibitions:
                if self._propositions_equivalent(permission, prohibition):
                    conflicts.append((permission, prohibition))
        
        # Ought implies can (obligation implies permission)
        for obligation in obligations:
            if obligation not in permissions:
                permissions.add(obligation)
        
        return ObligationState(
            obligations=obligations,
            permissions=permissions,
            prohibitions=prohibitions,
            conflicts=conflicts
        )
    
    def generate_possible_worlds(self, propositions: List[str],
                               constraints: List[ModalFormula] = None) -> List[PossibleWorld]:
        """Generate possible worlds satisfying constraints"""
        worlds = []
        
        # Generate all truth assignments
        for assignment in itertools.product([True, False], repeat=len(propositions)):
            world_props = dict(zip(propositions, assignment))
            world = PossibleWorld(
                world_id=f"w{len(worlds)}",
                propositions=world_props
            )
            
            # Check constraints
            if constraints:
                # Create temporary model with just this world
                temp_model = KripkeModel(
                    worlds={world.world_id: world},
                    accessibility={world.world_id: {world.world_id}},
                    actual_world=world.world_id
                )
                
                # Check all constraints
                satisfies_all = all(
                    self.evaluate_modal_formula(constraint, temp_model)
                    for constraint in constraints
                )
                
                if satisfies_all:
                    worlds.append(world)
            else:
                worlds.append(world)
        
        return worlds
    
    def analyze_modal_collapse(self, model: KripkeModel) -> Dict[str, bool]:
        """Analyze if modalities collapse in the model"""
        # In S5, □p ↔ □□p and ◇p ↔ □◇p
        
        # Test with sample proposition
        test_prop = Proposition("p", is_atomic=True)
        
        # Check if necessity collapses
        necessary_p = ModalFormula(ModalOperator.NECESSARY, test_prop)
        necessary_necessary_p = ModalFormula(ModalOperator.NECESSARY, necessary_p)
        
        necessity_collapse = True
        for world in model.worlds:
            val1 = self.evaluate_modal_formula(necessary_p, model, world)
            val2 = self.evaluate_modal_formula(necessary_necessary_p, model, world)
            if val1 != val2:
                necessity_collapse = False
                break
        
        # Check if possibility collapses
        possible_p = ModalFormula(ModalOperator.POSSIBLE, test_prop)
        necessary_possible_p = ModalFormula(ModalOperator.NECESSARY, possible_p)
        
        possibility_collapse = True
        for world in model.worlds:
            val1 = self.evaluate_modal_formula(possible_p, model, world)
            val2 = self.evaluate_modal_formula(necessary_possible_p, model, world)
            if val1 != val2:
                possibility_collapse = False
                break
        
        return {
            "necessity_collapses": necessity_collapse,
            "possibility_collapses": possibility_collapse,
            "is_s5_like": necessity_collapse and possibility_collapse
        }
    
    # Private helper methods
    
    def _enforce_accessibility_properties(self, accessibility: Dict[str, Set[str]],
                                        properties: AccessibilityRelation) -> Dict[str, Set[str]]:
        """Enforce accessibility relation properties"""
        worlds = set(accessibility.keys())
        
        # Add all worlds to accessibility if not present
        for world in worlds:
            if world not in accessibility:
                accessibility[world] = set()
        
        if properties == AccessibilityRelation.REFLEXIVE or \
           properties == AccessibilityRelation.EQUIVALENCE:
            # Add reflexive edges
            for world in worlds:
                accessibility[world].add(world)
        
        if properties == AccessibilityRelation.SYMMETRIC or \
           properties == AccessibilityRelation.EQUIVALENCE:
            # Add symmetric edges
            for w1 in worlds:
                for w2 in list(accessibility[w1]):
                    accessibility[w2].add(w1)
        
        if properties == AccessibilityRelation.TRANSITIVE or \
           properties == AccessibilityRelation.EQUIVALENCE:
            # Add transitive edges (Floyd-Warshall style)
            for k in worlds:
                for i in worlds:
                    for j in worlds:
                        if k in accessibility[i] and j in accessibility[k]:
                            accessibility[i].add(j)
        
        if properties == AccessibilityRelation.EUCLIDEAN:
            # Add Euclidean edges
            for w1 in worlds:
                accessible_from_w1 = list(accessibility[w1])
                for w2 in accessible_from_w1:
                    for w3 in accessible_from_w1:
                        accessibility[w2].add(w3)
        
        if properties == AccessibilityRelation.SERIAL:
            # Ensure each world accesses at least one world
            for world in worlds:
                if not accessibility[world]:
                    accessibility[world].add(world)  # Access itself
        
        return accessibility
    
    def _evaluate_in_world(self, formula: Any, model: KripkeModel, world: str) -> bool:
        """Evaluate formula in specific world"""
        if isinstance(formula, Proposition):
            return model.worlds[world].evaluate_proposition(formula)
        elif isinstance(formula, ModalFormula):
            return self.evaluate_modal_formula(formula, model, world)
        return False
    
    def _evaluate_knowledge(self, formula: ModalFormula, 
                          model: KripkeModel, world: str) -> bool:
        """Evaluate knowledge formula"""
        # Knowledge requires truth in all epistemically accessible worlds
        # For simplicity, use same accessibility as alethic modality
        accessible_worlds = model.accessibility.get(world, set())
        
        # Knowledge implies truth (factivity)
        if not self._evaluate_in_world(formula.operand, model, world):
            return False
        
        # Check all accessible worlds
        return all(
            self._evaluate_in_world(formula.operand, model, w)
            for w in accessible_worlds
        )
    
    def _evaluate_belief(self, formula: ModalFormula,
                       model: KripkeModel, world: str) -> bool:
        """Evaluate belief formula"""
        # Belief doesn't require factivity
        accessible_worlds = model.accessibility.get(world, set())
        
        return all(
            self._evaluate_in_world(formula.operand, model, w)
            for w in accessible_worlds
        )
    
    def _evaluate_obligation(self, formula: ModalFormula,
                           model: KripkeModel, world: str) -> bool:
        """Evaluate obligation formula"""
        # Obligation true in all deontically ideal worlds
        # For simplicity, use same accessibility
        accessible_worlds = model.accessibility.get(world, set())
        
        return all(
            self._evaluate_in_world(formula.operand, model, w)
            for w in accessible_worlds
        )
    
    def _generate_accessibility_patterns(self, num_worlds: int, 
                                       modal_system: str) -> List[Dict[str, List[str]]]:
        """Generate accessibility patterns for modal system"""
        worlds = [f"w{i}" for i in range(num_worlds)]
        patterns = []
        
        # Generate based on system properties
        properties = self._get_system_properties(modal_system)
        
        if num_worlds == 1:
            # Single world must access itself in reflexive systems
            if properties in [AccessibilityRelation.REFLEXIVE, 
                            AccessibilityRelation.EQUIVALENCE]:
                patterns.append({"w0": ["w0"]})
            else:
                patterns.append({"w0": []})
                patterns.append({"w0": ["w0"]})
        else:
            # For simplicity, generate a few representative patterns
            # Full generation would be exponential
            
            # Empty relation
            patterns.append({w: [] for w in worlds})
            
            # Full relation
            patterns.append({w: worlds.copy() for w in worlds})
            
            # Linear relation
            linear = {}
            for i, w in enumerate(worlds):
                if i < len(worlds) - 1:
                    linear[w] = [worlds[i + 1]]
                else:
                    linear[w] = []
            patterns.append(linear)
        
        return patterns
    
    def _get_system_properties(self, modal_system: str) -> AccessibilityRelation:
        """Get accessibility properties for modal system"""
        system_map = {
            "K": AccessibilityRelation.SERIAL,
            "T": AccessibilityRelation.REFLEXIVE,
            "S4": AccessibilityRelation.REFLEXIVE,  # Also transitive
            "S5": AccessibilityRelation.EQUIVALENCE,
            "D": AccessibilityRelation.SERIAL,
            "KD45": AccessibilityRelation.SERIAL  # Also transitive and Euclidean
        }
        
        return system_map.get(modal_system, AccessibilityRelation.SERIAL)
    
    def _is_consistent_with_beliefs(self, new_info: Proposition,
                                  beliefs: BeliefState) -> bool:
        """Check if new information is consistent with beliefs"""
        # Simplified consistency check
        for belief in beliefs.beliefs:
            if self._directly_contradicts(new_info, belief):
                return False
        return True
    
    def _directly_contradicts(self, prop1: Proposition, prop2: Proposition) -> bool:
        """Check if propositions directly contradict"""
        # Check if one is negation of other
        if prop1.is_atomic and prop2.is_atomic:
            return False
        
        if not prop1.is_atomic and prop1.connective == LogicalConnective.NOT:
            return self._propositions_equivalent(prop1.operands[0], prop2)
        
        if not prop2.is_atomic and prop2.connective == LogicalConnective.NOT:
            return self._propositions_equivalent(prop2.operands[0], prop1)
        
        return False
    
    def _propositions_equivalent(self, prop1: Proposition, prop2: Proposition) -> bool:
        """Check if propositions are equivalent"""
        if prop1.is_atomic and prop2.is_atomic:
            return prop1.content == prop2.content
        
        if prop1.is_atomic != prop2.is_atomic:
            return False
        
        # Both complex
        return (prop1.connective == prop2.connective and
                len(prop1.operands) == len(prop2.operands) and
                all(self._propositions_equivalent(op1, op2)
                    for op1, op2 in zip(prop1.operands, prop2.operands)))
    
    def _find_conflicting_beliefs(self, new_info: Proposition,
                                beliefs: BeliefState) -> Set[Proposition]:
        """Find beliefs that conflict with new information"""
        conflicting = set()
        
        for belief in beliefs.beliefs:
            if self._directly_contradicts(new_info, belief):
                conflicting.add(belief)
        
        return conflicting
    
    def _revise_uncertainties(self, old_uncertainties: Dict[Proposition, float],
                            removed_beliefs: Set[Proposition],
                            new_info: Proposition) -> Dict[Proposition, float]:
        """Revise uncertainty values after belief revision"""
        new_uncertainties = old_uncertainties.copy()
        
        # Remove uncertainties for removed beliefs
        for belief in removed_beliefs:
            new_uncertainties.pop(belief, None)
        
        # Add uncertainty for new information
        new_uncertainties[new_info] = 0.9
        
        # Slightly decrease confidence in remaining beliefs
        for prop in new_uncertainties:
            if prop != new_info:
                new_uncertainties[prop] *= 0.95
        
        return new_uncertainties
