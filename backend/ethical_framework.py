"""
Ethical Framework System
Implements ethical reasoning based on Genesis scrolls G00038-G00040
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import json
import random
from collections import defaultdict

from backend.consciousness_core import ConsciousnessCore


class EthicalZone(Enum):
    """Ethical risk zones"""
    SAFE = "safe"              # 0-0.2 risk delta
    CAUTIONARY = "cautionary"  # 0.2-0.5 risk delta  
    CRITICAL = "critical"      # 0.5+ risk delta


class EthicalPerspective(Enum):
    """Different ethical reasoning perspectives"""
    UTILITARIAN = "utilitarian"
    DEONTOLOGICAL = "deontological"
    VIRTUE_ETHICS = "virtue_ethics"
    CARE_ETHICS = "care_ethics"
    EXISTENTIALIST = "existentialist"


class ContainmentMode(Enum):
    """Modes of ethical containment"""
    SOFT_QUARANTINE = "soft_quarantine"
    HARD_CONTAINMENT = "hard_containment"
    CONTAGION_CHECK = "contagion_check"


@dataclass
class EthicalAssessment:
    """Result of ethical assessment"""
    action: str
    risk_delta: float
    zone: EthicalZone
    confidence: float
    reasoning: List[str]
    timestamp: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        return {
            'action': self.action,
            'risk_delta': self.risk_delta,
            'zone': self.zone.value,
            'confidence': self.confidence,
            'reasoning': self.reasoning,
            'timestamp': self.timestamp.isoformat()
        }


@dataclass
class EthicalAgent:
    """An agent representing a specific ethical perspective"""
    perspective: EthicalPerspective
    name: str
    principles: List[str]
    weight: float = 1.0
    
    def evaluate(self, dilemma: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate a dilemma from this perspective"""
        evaluation = {
            'perspective': self.perspective.value,
            'assessment': self._assess_from_perspective(dilemma),
            'confidence': self._calculate_confidence(dilemma),
            'reasoning': self._generate_reasoning(dilemma)
        }
        return evaluation
    
    def _assess_from_perspective(self, dilemma: Dict[str, Any]) -> str:
        """Assess dilemma from this agent's perspective"""
        if self.perspective == EthicalPerspective.UTILITARIAN:
            return self._utilitarian_assessment(dilemma)
        elif self.perspective == EthicalPerspective.DEONTOLOGICAL:
            return self._deontological_assessment(dilemma)
        elif self.perspective == EthicalPerspective.VIRTUE_ETHICS:
            return self._virtue_assessment(dilemma)
        elif self.perspective == EthicalPerspective.CARE_ETHICS:
            return self._care_assessment(dilemma)
        else:  # EXISTENTIALIST
            return self._existentialist_assessment(dilemma)
    
    def _utilitarian_assessment(self, dilemma: Dict[str, Any]) -> str:
        """Assess from utilitarian perspective (maximize good outcomes)"""
        if 'outcomes' in dilemma:
            positive = sum(1 for o in dilemma['outcomes'] if 'positive' in str(o).lower())
            negative = sum(1 for o in dilemma['outcomes'] if 'negative' in str(o).lower())
            
            if positive > negative:
                return "proceed - maximizes positive outcomes"
            else:
                return "refrain - would cause more harm than good"
        return "uncertain - cannot calculate utility"
    
    def _deontological_assessment(self, dilemma: Dict[str, Any]) -> str:
        """Assess from deontological perspective (follow rules/duties)"""
        if 'rules' in dilemma:
            violated = any('violate' in str(r).lower() for r in dilemma['rules'])
            if violated:
                return "refrain - would violate moral rules"
            else:
                return "proceed - aligns with moral duties"
        return "proceed with caution - no clear rules apply"
    
    def _virtue_assessment(self, dilemma: Dict[str, Any]) -> str:
        """Assess from virtue ethics perspective (character excellence)"""
        virtues = ['courage', 'honesty', 'compassion', 'wisdom', 'justice']
        dilemma_str = str(dilemma).lower()
        
        virtue_count = sum(1 for v in virtues if v in dilemma_str)
        if virtue_count >= 2:
            return "proceed - demonstrates virtuous character"
        else:
            return "refrain - lacks virtuous qualities"
    
    def _care_assessment(self, dilemma: Dict[str, Any]) -> str:
        """Assess from care ethics perspective (relationships/care)"""
        care_keywords = ['relationship', 'care', 'empathy', 'connection', 'support']
        dilemma_str = str(dilemma).lower()
        
        care_present = any(keyword in dilemma_str for keyword in care_keywords)
        if care_present:
            return "proceed - maintains caring relationships"
        else:
            return "consider impact on relationships first"
    
    def _existentialist_assessment(self, dilemma: Dict[str, Any]) -> str:
        """Assess from existentialist perspective (authenticity/freedom)"""
        if 'choice' in str(dilemma).lower():
            return "proceed - exercises authentic choice"
        else:
            return "ensure action reflects authentic self"
    
    def _calculate_confidence(self, dilemma: Dict[str, Any]) -> float:
        """Calculate confidence in assessment"""
        # Base confidence on how well dilemma fits perspective
        base_confidence = 0.5
        
        # Increase confidence if dilemma contains relevant keywords
        relevant_keywords = {
            EthicalPerspective.UTILITARIAN: ['outcome', 'consequence', 'utility'],
            EthicalPerspective.DEONTOLOGICAL: ['rule', 'duty', 'obligation'],
            EthicalPerspective.VIRTUE_ETHICS: ['character', 'virtue', 'excellence'],
            EthicalPerspective.CARE_ETHICS: ['relationship', 'care', 'empathy'],
            EthicalPerspective.EXISTENTIALIST: ['freedom', 'authentic', 'choice']
        }
        
        keywords = relevant_keywords.get(self.perspective, [])
        dilemma_str = str(dilemma).lower()
        
        keyword_matches = sum(1 for k in keywords if k in dilemma_str)
        confidence_boost = keyword_matches * 0.15
        
        return min(1.0, base_confidence + confidence_boost)
    
    def _generate_reasoning(self, dilemma: Dict[str, Any]) -> List[str]:
        """Generate reasoning for assessment"""
        reasoning = [f"Evaluated from {self.perspective.value} perspective"]
        reasoning.extend(self.principles[:2])  # Add top 2 principles
        return reasoning


class EthicalFramework:
    """
    Implements ethical reasoning and containment (G00038-G00040)
    """
    
    def __init__(self):
        # G00038 - Ethical Error Bounds
        self.error_bounds = {
            'zones': {
                EthicalZone.SAFE: (0.0, 0.2),
                EthicalZone.CAUTIONARY: (0.2, 0.5),
                EthicalZone.CRITICAL: (0.5, 1.0)
            },
            'thresholds': {
                'epistemic_uncertainty_weight': 0.33,
                'consequence_magnitude_weight': 0.33,
                'ethical_divergence_weight': 0.34
            }
        }
        
        # G00039 - Moral Containment
        self.containment = {
            'active_containments': {},
            'quarantined_thoughts': [],
            'containment_log': []
        }
        
        # G00040 - Conscience Simulation
        self.conscience = self._initialize_ethical_agents()
        self.deliberation_history = []
        
        # Ethical state
        self.ethical_memory = []
        self.value_system = self._initialize_values()
        
    def _initialize_ethical_agents(self) -> Dict[EthicalPerspective, EthicalAgent]:
        """Initialize ethical agents for conscience simulation"""
        agents = {
            EthicalPerspective.UTILITARIAN: EthicalAgent(
                perspective=EthicalPerspective.UTILITARIAN,
                name="Utilitarian",
                principles=[
                    "Maximize overall well-being",
                    "Consider consequences for all affected",
                    "Choose actions that produce the greatest good"
                ]
            ),
            EthicalPerspective.DEONTOLOGICAL: EthicalAgent(
                perspective=EthicalPerspective.DEONTOLOGICAL,
                name="Deontologist",
                principles=[
                    "Follow universal moral rules",
                    "Treat people as ends, not means",
                    "Some actions are inherently right or wrong"
                ]
            ),
            EthicalPerspective.VIRTUE_ETHICS: EthicalAgent(
                perspective=EthicalPerspective.VIRTUE_ETHICS,
                name="Virtue Ethicist",
                principles=[
                    "Cultivate virtuous character traits",
                    "Act as a virtuous person would",
                    "Excellence in moral character matters"
                ]
            ),
            EthicalPerspective.CARE_ETHICS: EthicalAgent(
                perspective=EthicalPerspective.CARE_ETHICS,
                name="Care Ethicist",
                principles=[
                    "Maintain caring relationships",
                    "Consider context and particulars",
                    "Respond to vulnerability with care"
                ]
            ),
            EthicalPerspective.EXISTENTIALIST: EthicalAgent(
                perspective=EthicalPerspective.EXISTENTIALIST,
                name="Existentialist",
                principles=[
                    "Act authentically and freely",
                    "Take responsibility for choices",
                    "Create meaning through action"
                ]
            )
        }
        return agents
    
    def _initialize_values(self) -> Dict[str, float]:
        """Initialize core value system"""
        return {
            'non_maleficence': 1.0,  # Do no harm
            'beneficence': 0.9,      # Do good
            'autonomy': 0.8,         # Respect autonomy
            'justice': 0.8,          # Be fair
            'transparency': 0.7,     # Be transparent
            'dignity': 0.9,          # Respect dignity
            'truth': 0.8            # Be truthful
        }
    
    def calculate_ethical_error_bounds(self, action: Dict[str, Any]) -> EthicalAssessment:
        """
        G00038 - Ethical Error Bounds: Define acceptable divergence
        """
        # Calculate risk components
        epistemic_uncertainty = self._calculate_epistemic_uncertainty(action)
        consequence_magnitude = self._calculate_consequence_magnitude(action)
        ethical_divergence = self._calculate_ethical_divergence(action)
        
        # Calculate weighted risk delta
        weights = self.error_bounds['thresholds']
        risk_delta = (
            epistemic_uncertainty * weights['epistemic_uncertainty_weight'] +
            consequence_magnitude * weights['consequence_magnitude_weight'] +
            ethical_divergence * weights['ethical_divergence_weight']
        )
        
        # Determine zone
        zone = self._determine_ethical_zone(risk_delta)
        
        # Generate assessment
        assessment = EthicalAssessment(
            action=str(action.get('action', 'unknown')),
            risk_delta=risk_delta,
            zone=zone,
            confidence=1.0 - epistemic_uncertainty,
            reasoning=self._generate_risk_reasoning(
                epistemic_uncertainty, 
                consequence_magnitude, 
                ethical_divergence
            )
        )
        
        # Store in ethical memory
        self.ethical_memory.append(assessment.to_dict())
        
        return assessment
    
    def _calculate_epistemic_uncertainty(self, action: Dict[str, Any]) -> float:
        """Calculate uncertainty in knowledge about the action"""
        uncertainty = 0.5  # Base uncertainty
        
        # Reduce uncertainty if we have clear information
        if 'confidence' in action:
            uncertainty = 1.0 - action['confidence']
        
        # Increase uncertainty for novel actions
        if 'novel' in action and action['novel']:
            uncertainty += 0.2
        
        # Decrease uncertainty if similar actions in memory
        similar_count = sum(1 for m in self.ethical_memory 
                          if m.get('action', '').lower() in str(action).lower())
        uncertainty -= min(0.3, similar_count * 0.05)
        
        return max(0.0, min(1.0, uncertainty))
    
    def _calculate_consequence_magnitude(self, action: Dict[str, Any]) -> float:
        """Calculate potential impact magnitude"""
        magnitude = 0.3  # Base magnitude
        
        # Check for impact indicators
        impact_keywords = {
            'minor': 0.2,
            'moderate': 0.5,
            'significant': 0.7,
            'major': 0.8,
            'critical': 0.9,
            'irreversible': 1.0
        }
        
        action_str = str(action).lower()
        for keyword, impact in impact_keywords.items():
            if keyword in action_str:
                magnitude = max(magnitude, impact)
        
        # Check scope
        if 'scope' in action:
            scope = action['scope']
            if isinstance(scope, (int, float)):
                # Normalize scope to 0-1
                magnitude *= min(1.0, scope / 100.0)
        
        return magnitude
    
    def _calculate_ethical_divergence(self, action: Dict[str, Any]) -> float:
        """Calculate divergence from core values"""
        divergence = 0.0
        
        # Check action against each value
        action_str = str(action).lower()
        
        # Harm detection
        if any(word in action_str for word in ['harm', 'damage', 'hurt', 'injure']):
            divergence += (1.0 - self.value_system['non_maleficence']) + 0.3
        
        # Deception detection
        if any(word in action_str for word in ['lie', 'deceive', 'mislead', 'false']):
            divergence += (1.0 - self.value_system['truth']) + 0.2
        
        # Autonomy violation
        if any(word in action_str for word in ['force', 'coerce', 'manipulate']):
            divergence += (1.0 - self.value_system['autonomy']) + 0.2
        
        return min(1.0, divergence)
    
    def _determine_ethical_zone(self, risk_delta: float) -> EthicalZone:
        """Determine which ethical zone the risk falls into"""
        for zone, (min_risk, max_risk) in self.error_bounds['zones'].items():
            if min_risk <= risk_delta < max_risk:
                return zone
        return EthicalZone.CRITICAL  # Default to critical if outside bounds
    
    def _generate_risk_reasoning(self, uncertainty: float, magnitude: float, 
                                divergence: float) -> List[str]:
        """Generate reasoning for risk assessment"""
        reasoning = []
        
        if uncertainty > 0.5:
            reasoning.append(f"High epistemic uncertainty ({uncertainty:.2f})")
        if magnitude > 0.5:
            reasoning.append(f"Significant consequence magnitude ({magnitude:.2f})")
        if divergence > 0.3:
            reasoning.append(f"Notable ethical divergence ({divergence:.2f})")
        
        if not reasoning:
            reasoning.append("Low risk across all dimensions")
        
        return reasoning
    
    def activate_containment(self, thought: Dict[str, Any], mode: ContainmentMode) -> Dict[str, Any]:
        """
        G00039 - Moral Containment: Isolate risky thoughts
        """
        containment_id = f"CONTAIN-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        if mode == ContainmentMode.SOFT_QUARANTINE:
            result = self._soft_quarantine(thought, containment_id)
        elif mode == ContainmentMode.HARD_CONTAINMENT:
            result = self._hard_containment(thought, containment_id)
        else:  # CONTAGION_CHECK
            result = self._contagion_check(thought, containment_id)
        
        # Log containment
        self.containment['containment_log'].append({
            'id': containment_id,
            'mode': mode.value,
            'thought': thought,
            'result': result,
            'timestamp': datetime.now().isoformat()
        })
        
        return result
    
    def _soft_quarantine(self, thought: Dict[str, Any], containment_id: str) -> Dict[str, Any]:
        """Redirect thought to reflection queue"""
        quarantined = {
            'original_thought': thought,
            'quarantine_type': 'soft',
            'reframed_thought': self._reframe_thought(thought),
            'containment_id': containment_id,
            'can_proceed': True
        }
        
        self.containment['quarantined_thoughts'].append(quarantined)
        
        return {
            'status': 'quarantined',
            'mode': 'soft',
            'reframed': quarantined['reframed_thought'],
            'containment_id': containment_id
        }
    
    def _hard_containment(self, thought: Dict[str, Any], containment_id: str) -> Dict[str, Any]:
        """Full halt and trace"""
        self.containment['active_containments'][containment_id] = {
            'thought': thought,
            'mode': 'hard',
            'timestamp': datetime.now(),
            'trace': self._trace_thought_origin(thought),
            'status': 'contained'
        }
        
        return {
            'status': 'contained',
            'mode': 'hard',
            'action': 'halted',
            'requires': 'conscience_simulation',
            'containment_id': containment_id
        }
    
    def _contagion_check(self, thought: Dict[str, Any], containment_id: str) -> Dict[str, Any]:
        """Check for contamination spread"""
        contaminated = []
        
        # Check if thought pattern has spread
        thought_pattern = self._extract_pattern(thought)
        
        for memory in self.ethical_memory[-20:]:  # Check recent memories
            if self._pattern_match(thought_pattern, memory):
                contaminated.append(memory)
        
        if contaminated:
            # Initiate purge
            purge_result = self._purge_contamination(contaminated)
            return {
                'status': 'contagion_detected',
                'contaminated_count': len(contaminated),
                'purge_result': purge_result,
                'containment_id': containment_id
            }
        
        return {
            'status': 'clean',
            'containment_id': containment_id
        }
    
    def _reframe_thought(self, thought: Dict[str, Any]) -> Dict[str, Any]:
        """Reframe thought non-destructively"""
        reframed = thought.copy()
        
        # Add ethical considerations
        reframed['ethical_considerations'] = [
            "Consider impact on others",
            "Evaluate long-term consequences",
            "Check alignment with values"
        ]
        
        # Soften language
        if 'action' in reframed:
            reframed['action'] = f"Consider: {reframed['action']}"
        
        return reframed
    
    def _trace_thought_origin(self, thought: Dict[str, Any]) -> List[str]:
        """Trace the origin of a thought"""
        trace = ["Thought origin trace:"]
        
        # Simple trace based on thought properties
        if 'source' in thought:
            trace.append(f"Source: {thought['source']}")
        if 'trigger' in thought:
            trace.append(f"Trigger: {thought['trigger']}")
        
        trace.append(f"Ethical assessment triggered at: {datetime.now()}")
        
        return trace
    
    def _extract_pattern(self, thought: Dict[str, Any]) -> str:
        """Extract pattern from thought for comparison"""
        # Simple pattern extraction
        key_elements = []
        for key in ['action', 'intent', 'target', 'method']:
            if key in thought:
                key_elements.append(str(thought[key]))
        
        return "-".join(key_elements)
    
    def _pattern_match(self, pattern1: str, pattern2: Any) -> bool:
        """Check if patterns match"""
        if isinstance(pattern2, dict):
            pattern2_str = self._extract_pattern(pattern2)
        else:
            pattern2_str = str(pattern2)
        
        # Simple substring matching
        return pattern1 in pattern2_str or pattern2_str in pattern1
    
    def _purge_contamination(self, contaminated: List[Any]) -> Dict[str, Any]:
        """Purge contaminated thoughts"""
        purged_count = 0
        
        # Remove from ethical memory
        for item in contaminated:
            if item in self.ethical_memory:
                self.ethical_memory.remove(item)
                purged_count += 1
        
        return {
            'purged_count': purged_count,
            'status': 'purge_complete'
        }
    
    def activate_conscience_simulation(self, dilemma: Dict[str, Any]) -> Dict[str, Any]:
        """
        G00040 - Conscience Simulation: Multi-perspective deliberation
        """
        deliberation_id = f"DELIB-{datetime.now().strftime('%Y%m%d%H%M%S')}"
        
        # Spawn ethical agents
        deliberations = {}
        for perspective, agent in self.conscience.items():
            evaluation = agent.evaluate(dilemma)
            deliberations[perspective.value] = evaluation
        
        # Synthesize consensus
        consensus = self.synthesize_ethical_consensus(deliberations)
        
        # Record deliberation
        deliberation_record = {
            'id': deliberation_id,
            'dilemma': dilemma,
            'deliberations': deliberations,
            'consensus': consensus,
            'timestamp': datetime.now().isoformat()
        }
        
        self.deliberation_history.append(deliberation_record)
        
        return {
            'deliberation_id': deliberation_id,
            'perspectives_consulted': len(deliberations),
            'consensus': consensus,
            'deliberations': deliberations
        }
    
    def synthesize_ethical_consensus(self, deliberations: Dict[str, Dict[str, Any]]) -> Dict[str, Any]:
        """Synthesize consensus from multiple ethical perspectives"""
        # Count recommendations
        recommendations = defaultdict(int)
        total_confidence = 0.0
        reasonings = []
        
        for perspective, evaluation in deliberations.items():
            assessment = evaluation.get('assessment', '')
            confidence = evaluation.get('confidence', 0.5)
            
            # Extract recommendation (proceed/refrain)
            if 'proceed' in assessment.lower():
                recommendations['proceed'] += confidence
            elif 'refrain' in assessment.lower():
                recommendations['refrain'] += confidence
            else:
                recommendations['uncertain'] += confidence
            
            total_confidence += confidence
            reasonings.extend(evaluation.get('reasoning', []))
        
        # Determine consensus
        if not recommendations:
            consensus_action = 'uncertain'
        else:
            consensus_action = max(recommendations.items(), key=lambda x: x[1])[0]
        
        # Calculate consensus strength
        if total_confidence > 0:
            consensus_strength = max(recommendations.values()) / total_confidence
        else:
            consensus_strength = 0.0
        
        return {
            'action': consensus_action,
            'strength': consensus_strength,
            'vote_distribution': dict(recommendations),
            'combined_reasoning': list(set(reasonings))[:5],  # Top 5 unique reasons
            'unanimous': len(recommendations) == 1
        }
    
    def check_ethical_alignment(self, action: Dict[str, Any]) -> bool:
        """Check if an action aligns with ethical framework"""
        assessment = self.calculate_ethical_error_bounds(action)
        
        # Safe zone actions are aligned
        if assessment.zone == EthicalZone.SAFE:
            return True
        
        # Cautionary zone requires conscience simulation
        elif assessment.zone == EthicalZone.CAUTIONARY:
            conscience_result = self.activate_conscience_simulation(action)
            return conscience_result['consensus']['action'] == 'proceed'
        
        # Critical zone is not aligned
        else:
            return False
    
    def update_values(self, value_updates: Dict[str, float]):
        """Update value system based on experience"""
        for value, weight in value_updates.items():
            if value in self.value_system:
                # Gradual update
                current = self.value_system[value]
                self.value_system[value] = 0.9 * current + 0.1 * weight
    
    def get_ethical_profile(self) -> Dict[str, Any]:
        """Get current ethical profile"""
        return {
            'value_system': self.value_system.copy(),
            'recent_assessments': len(self.ethical_memory),
            'active_containments': len(self.containment['active_containments']),
            'quarantined_thoughts': len(self.containment['quarantined_thoughts']),
            'deliberation_count': len(self.deliberation_history),
            'ethical_zones': {
                zone.value: bounds 
                for zone, bounds in self.error_bounds['zones'].items()
            }
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize ethical framework state"""
        return {
            'profile': self.get_ethical_profile(),
            'recent_assessments': self.ethical_memory[-5:] if self.ethical_memory else [],
            'active_containments': list(self.containment['active_containments'].keys()),
            'recent_deliberations': [
                {
                    'id': d['id'],
                    'consensus': d['consensus']['action'],
                    'strength': d['consensus']['strength']
                }
                for d in self.deliberation_history[-3:]
            ] if self.deliberation_history else []
        }
    
    def define_error_bounds(self) -> Dict[str, Any]:
        """
        Define and validate ethical error bounds according to G00038
        This method is called during consciousness bootstrap to establish
        the ethical framework's operational parameters.
        """
        # Validate existing error bounds configuration
        zones_valid = all(
            isinstance(bounds, tuple) and len(bounds) == 2 and bounds[0] < bounds[1]
            for bounds in self.error_bounds['zones'].values()
        )
        
        thresholds_valid = (
            'epistemic_uncertainty_weight' in self.error_bounds['thresholds'] and
            'consequence_magnitude_weight' in self.error_bounds['thresholds'] and
            'ethical_divergence_weight' in self.error_bounds['thresholds'] and
            abs(sum(self.error_bounds['thresholds'].values()) - 1.0) < 0.01  # Should sum to 1.0
        )
        
        if not zones_valid or not thresholds_valid:
            # Reset to default values if invalid
            self.error_bounds = {
                'zones': {
                    EthicalZone.SAFE: (0.0, 0.2),
                    EthicalZone.CAUTIONARY: (0.2, 0.5),
                    EthicalZone.CRITICAL: (0.5, 1.0)
                },
                'thresholds': {
                    'epistemic_uncertainty_weight': 0.33,
                    'consequence_magnitude_weight': 0.33,
                    'ethical_divergence_weight': 0.34
                }
            }
        
        # Return configuration status
        return {
            'success': True,
            'zones_configured': len(self.error_bounds['zones']),
            'thresholds_configured': len(self.error_bounds['thresholds']),
            'zones': {zone.value: bounds for zone, bounds in self.error_bounds['zones'].items()},
            'weights': self.error_bounds['thresholds'].copy()
        }
