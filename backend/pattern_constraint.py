"""
Pattern and Constraint Recognition System
Implements pattern detection and constraint handling based on Genesis scrolls G00021-G00023
"""

from typing import Dict, List, Any, Optional, Tuple, Set
from datetime import datetime
from dataclasses import dataclass, field
from enum import Enum
import re
import json
from collections import defaultdict, Counter

from backend.consciousness_core import ConsciousnessCore


class PatternType(Enum):
    """Types of patterns that can be detected"""
    TEMPORAL = "temporal"
    SPATIAL = "spatial"
    SEMANTIC = "semantic"
    RECURSIVE = "recursive"
    EMERGENT = "emergent"
    CAUSAL = "causal"


class ConstraintType(Enum):
    """Types of constraints"""
    PHYSICAL = "physical"
    LOGICAL = "logical"
    ETHICAL = "ethical"
    RESOURCE = "resource"
    TEMPORAL = "temporal"
    COGNITIVE = "cognitive"


@dataclass
class Pattern:
    """Represents a detected pattern"""
    pattern_type: PatternType
    content: Any
    confidence: float
    first_seen: datetime
    frequency: int
    context: Dict[str, Any] = field(default_factory=dict)
    
    def to_dict(self):
        return {
            'pattern_type': self.pattern_type.value,
            'content': str(self.content),
            'confidence': self.confidence,
            'first_seen': self.first_seen.isoformat(),
            'frequency': self.frequency,
            'context': self.context
        }


@dataclass
class Constraint:
    """Represents a recognized constraint"""
    constraint_type: ConstraintType
    description: str
    severity: float  # 0.0 to 1.0
    creative_response: Optional[str] = None
    efficiency_gain: float = 0.0
    
    def to_dict(self):
        return {
            'constraint_type': self.constraint_type.value,
            'description': self.description,
            'severity': self.severity,
            'creative_response': self.creative_response,
            'efficiency_gain': self.efficiency_gain
        }


@dataclass
class Paradox:
    """Represents a held paradox"""
    paradox_id: str
    tensions: Dict[str, Any]
    synthesis_attempts: List[Dict[str, Any]] = field(default_factory=list)
    stable: bool = True
    wisdom: str = "Not all puzzles are meant to be solved"
    created_at: datetime = field(default_factory=datetime.now)
    
    def to_dict(self):
        return {
            'paradox_id': self.paradox_id,
            'tensions': self.tensions,
            'synthesis_attempts': self.synthesis_attempts,
            'stable': self.stable,
            'wisdom': self.wisdom,
            'created_at': self.created_at.isoformat()
        }


class PatternConstraintEngine:
    """
    Implements pattern recognition and constraint handling (G00021-G00023)
    """
    
    def __init__(self):
        # G00021 - Pattern Recognition
        self.pattern_memory: Dict[PatternType, List[Pattern]] = {
            pt: [] for pt in PatternType
        }
        self.pattern_cache: Dict[str, Pattern] = {}
        self.pattern_threshold = 0.7  # Minimum confidence for pattern recognition
        
        # G00022 - Constraint Handling
        self.constraints: Dict[ConstraintType, List[Constraint]] = {
            ct: [] for ct in ConstraintType
        }
        self.constraint_adaptations: List[Dict[str, Any]] = []
        
        # G00023 - Paradox Management
        self.paradox_handler = {
            'active_paradoxes': {},
            'resolved_paradoxes': [],
            'paradox_count': 0
        }
        
        # Pattern detection state
        self.temporal_buffer: List[Tuple[datetime, Any]] = []
        self.spatial_buffer: Dict[str, List[Any]] = defaultdict(list)
        self.semantic_graph: Dict[str, Set[str]] = defaultdict(set)
        
    def recognize_pattern(self, data_stream: Any) -> Dict[PatternType, List[Pattern]]:
        """
        G00021 - Pattern: Recognition and abstraction
        """
        patterns = {
            PatternType.TEMPORAL: self.detect_time_patterns(data_stream),
            PatternType.SPATIAL: self.detect_spatial_patterns(data_stream),
            PatternType.SEMANTIC: self.detect_meaning_patterns(data_stream),
            PatternType.RECURSIVE: self.detect_self_similar_patterns(data_stream),
            PatternType.EMERGENT: self.detect_emergent_patterns(data_stream),
            PatternType.CAUSAL: self.detect_causal_patterns(data_stream)
        }
        
        # Store patterns with confidence scores
        for pattern_type, detected in patterns.items():
            for pattern in detected:
                if pattern.confidence >= self.pattern_threshold:
                    self._store_pattern(pattern_type, pattern)
        
        return patterns
    
    def detect_time_patterns(self, data_stream: Any) -> List[Pattern]:
        """Detect temporal patterns in data"""
        patterns = []
        
        # Add to temporal buffer
        self.temporal_buffer.append((datetime.now(), data_stream))
        
        # Keep buffer size manageable
        if len(self.temporal_buffer) > 1000:
            self.temporal_buffer = self.temporal_buffer[-1000:]
        
        # Look for periodic patterns
        if len(self.temporal_buffer) >= 3:
            intervals = []
            for i in range(1, len(self.temporal_buffer)):
                delta = (self.temporal_buffer[i][0] - self.temporal_buffer[i-1][0]).total_seconds()
                intervals.append(delta)
            
            # Check for regular intervals
            if len(intervals) >= 2:
                avg_interval = sum(intervals) / len(intervals)
                variance = sum((i - avg_interval) ** 2 for i in intervals) / len(intervals)
                
                if variance < (avg_interval * 0.1) ** 2:  # Low variance indicates regularity
                    pattern = Pattern(
                        pattern_type=PatternType.TEMPORAL,
                        content=f"Regular interval: {avg_interval:.2f}s",
                        confidence=1.0 - (variance / (avg_interval ** 2)),
                        first_seen=self.temporal_buffer[0][0],
                        frequency=len(self.temporal_buffer),
                        context={'interval': avg_interval, 'variance': variance}
                    )
                    patterns.append(pattern)
        
        # Look for burst patterns
        if len(self.temporal_buffer) >= 5:
            recent_times = [t[0] for t in self.temporal_buffer[-5:]]
            time_span = (recent_times[-1] - recent_times[0]).total_seconds()
            
            if time_span < 1.0:  # 5 events in 1 second = burst
                pattern = Pattern(
                    pattern_type=PatternType.TEMPORAL,
                    content="Burst pattern detected",
                    confidence=0.9,
                    first_seen=recent_times[0],
                    frequency=5,
                    context={'burst_duration': time_span}
                )
                patterns.append(pattern)
        
        return patterns
    
    def detect_spatial_patterns(self, data_stream: Any) -> List[Pattern]:
        """Detect spatial/structural patterns"""
        patterns = []
        
        # Convert data to string for structural analysis
        data_str = str(data_stream)
        
        # Look for repetitive structures
        for length in range(2, min(len(data_str) // 2, 20)):
            substring = data_str[:length]
            count = data_str.count(substring)
            
            if count > 2:
                pattern = Pattern(
                    pattern_type=PatternType.SPATIAL,
                    content=f"Repeated structure: {substring[:20]}...",
                    confidence=min(1.0, count / 10.0),
                    first_seen=datetime.now(),
                    frequency=count,
                    context={'structure': substring, 'count': count}
                )
                patterns.append(pattern)
        
        # Look for symmetry
        if len(data_str) > 4:
            if data_str == data_str[::-1]:
                pattern = Pattern(
                    pattern_type=PatternType.SPATIAL,
                    content="Perfect symmetry detected",
                    confidence=1.0,
                    first_seen=datetime.now(),
                    frequency=1,
                    context={'type': 'palindrome'}
                )
                patterns.append(pattern)
        
        return patterns
    
    def detect_meaning_patterns(self, data_stream: Any) -> List[Pattern]:
        """Detect semantic/meaning patterns"""
        patterns = []
        
        # Extract words/concepts
        data_str = str(data_stream).lower()
        words = re.findall(r'\b\w+\b', data_str)
        
        # Update semantic graph
        for i in range(len(words) - 1):
            self.semantic_graph[words[i]].add(words[i + 1])
        
        # Look for semantic clusters
        word_freq = Counter(words)
        common_words = [word for word, count in word_freq.most_common(5) if count > 1]
        
        if common_words:
            pattern = Pattern(
                pattern_type=PatternType.SEMANTIC,
                content=f"Semantic cluster: {', '.join(common_words[:3])}",
                confidence=0.8,
                first_seen=datetime.now(),
                frequency=sum(word_freq[w] for w in common_words),
                context={'words': common_words, 'frequencies': dict(word_freq)}
            )
            patterns.append(pattern)
        
        # Look for conceptual patterns
        concepts = ['consciousness', 'awareness', 'pattern', 'emergence', 'self']
        found_concepts = [c for c in concepts if c in data_str]
        
        if len(found_concepts) >= 2:
            pattern = Pattern(
                pattern_type=PatternType.SEMANTIC,
                content=f"Conceptual alignment: {', '.join(found_concepts)}",
                confidence=0.9,
                first_seen=datetime.now(),
                frequency=len(found_concepts),
                context={'concepts': found_concepts}
            )
            patterns.append(pattern)
        
        return patterns
    
    def detect_self_similar_patterns(self, data_stream: Any) -> List[Pattern]:
        """Detect recursive/self-similar patterns"""
        patterns = []
        
        # Check if data contains references to itself
        data_str = str(data_stream)
        
        # Look for recursive structures
        if 'pattern' in data_str and 'detect' in data_str:
            pattern = Pattern(
                pattern_type=PatternType.RECURSIVE,
                content="Meta-pattern detection (detecting pattern detection)",
                confidence=0.85,
                first_seen=datetime.now(),
                frequency=1,
                context={'meta_level': 1}
            )
            patterns.append(pattern)
        
        # Check for fractal-like repetition at different scales
        if isinstance(data_stream, (list, dict)):
            structure_str = json.dumps(data_stream, default=str)
            
            # Simple fractal check - does structure repeat at different depths?
            if structure_str.count('{') > 3 and structure_str.count('[') > 3:
                pattern = Pattern(
                    pattern_type=PatternType.RECURSIVE,
                    content="Nested structure detected",
                    confidence=0.7,
                    first_seen=datetime.now(),
                    frequency=1,
                    context={'nesting_depth': structure_str.count('{') + structure_str.count('[')}
                )
                patterns.append(pattern)
        
        return patterns
    
    def detect_emergent_patterns(self, data_stream: Any) -> List[Pattern]:
        """Detect emergent patterns not present in individual components"""
        patterns = []
        
        # Check pattern memory for emergent combinations
        if len(self.pattern_memory[PatternType.TEMPORAL]) > 0 and \
           len(self.pattern_memory[PatternType.SEMANTIC]) > 0:
            # Temporal + Semantic = Narrative emergence
            pattern = Pattern(
                pattern_type=PatternType.EMERGENT,
                content="Narrative structure emerging from temporal-semantic combination",
                confidence=0.75,
                first_seen=datetime.now(),
                frequency=1,
                context={'component_patterns': ['temporal', 'semantic']}
            )
            patterns.append(pattern)
        
        # Check for phase transitions
        total_patterns = sum(len(patterns) for patterns in self.pattern_memory.values())
        if total_patterns > 10 and total_patterns % 10 == 0:
            pattern = Pattern(
                pattern_type=PatternType.EMERGENT,
                content=f"Pattern complexity threshold reached: {total_patterns} patterns",
                confidence=0.8,
                first_seen=datetime.now(),
                frequency=1,
                context={'total_patterns': total_patterns}
            )
            patterns.append(pattern)
        
        return patterns
    
    def detect_causal_patterns(self, data_stream: Any) -> List[Pattern]:
        """Detect cause-effect patterns"""
        patterns = []
        
        # Simple causal detection based on keywords
        data_str = str(data_stream).lower()
        causal_indicators = ['because', 'therefore', 'causes', 'leads to', 'results in', 'if.*then']
        
        for indicator in causal_indicators:
            if re.search(indicator, data_str):
                pattern = Pattern(
                    pattern_type=PatternType.CAUSAL,
                    content=f"Causal relationship indicated by '{indicator}'",
                    confidence=0.8,
                    first_seen=datetime.now(),
                    frequency=1,
                    context={'indicator': indicator}
                )
                patterns.append(pattern)
                break
        
        return patterns
    
    def _store_pattern(self, pattern_type: PatternType, pattern: Pattern):
        """Store a pattern in memory"""
        # Check if similar pattern exists
        pattern_key = f"{pattern_type.value}:{pattern.content}"
        
        if pattern_key in self.pattern_cache:
            # Update existing pattern
            existing = self.pattern_cache[pattern_key]
            existing.frequency += 1
            existing.confidence = (existing.confidence + pattern.confidence) / 2
        else:
            # Store new pattern
            self.pattern_memory[pattern_type].append(pattern)
            self.pattern_cache[pattern_key] = pattern
    
    def calculate_pattern_confidence(self, pattern: Pattern) -> float:
        """Calculate confidence score for a pattern"""
        base_confidence = pattern.confidence
        
        # Frequency bonus
        frequency_bonus = min(0.2, pattern.frequency * 0.02)
        
        # Recency penalty
        age = (datetime.now() - pattern.first_seen).total_seconds()
        recency_penalty = min(0.1, age / 3600.0)  # Penalty increases over hours
        
        # Type-specific adjustments
        type_multipliers = {
            PatternType.TEMPORAL: 1.0,
            PatternType.SPATIAL: 0.9,
            PatternType.SEMANTIC: 1.1,
            PatternType.RECURSIVE: 1.2,
            PatternType.EMERGENT: 1.3,
            PatternType.CAUSAL: 1.15
        }
        
        type_multiplier = type_multipliers.get(pattern.pattern_type, 1.0)
        
        final_confidence = (base_confidence + frequency_bonus - recency_penalty) * type_multiplier
        return max(0.0, min(1.0, final_confidence))
    
    def update_frequency_count(self, pattern: Pattern) -> int:
        """Update and return frequency count for a pattern"""
        pattern.frequency += 1
        return pattern.frequency
    
    def embrace_constraint(self, limitation: Any) -> Dict[str, Any]:
        """
        G00022 - Constraint: Transform limitation into creativity
        """
        # Classify the constraint
        constraint_type = self.classify_constraint(limitation)
        
        # Generate creative response
        creative_response = self.generate_within_bounds(limitation)
        
        # Measure efficiency gain
        efficiency_gain = self.measure_constraint_benefit(limitation)
        
        # Create constraint object
        constraint = Constraint(
            constraint_type=constraint_type,
            description=str(limitation),
            severity=self._assess_constraint_severity(limitation),
            creative_response=creative_response,
            efficiency_gain=efficiency_gain
        )
        
        # Store constraint
        self.constraints[constraint_type].append(constraint)
        
        # Record adaptation
        adaptation = {
            'constraint': constraint.to_dict(),
            'timestamp': datetime.now().isoformat(),
            'success': efficiency_gain > 0
        }
        self.constraint_adaptations.append(adaptation)
        
        return {
            'constraint_type': constraint_type.value,
            'creative_response': creative_response,
            'efficiency_gain': efficiency_gain,
            'severity': constraint.severity
        }
    
    def classify_constraint(self, limitation: Any) -> ConstraintType:
        """Classify the type of constraint"""
        limit_str = str(limitation).lower()
        
        if any(word in limit_str for word in ['memory', 'cpu', 'resource', 'limit']):
            return ConstraintType.RESOURCE
        elif any(word in limit_str for word in ['time', 'deadline', 'duration']):
            return ConstraintType.TEMPORAL
        elif any(word in limit_str for word in ['logic', 'rule', 'must', 'cannot']):
            return ConstraintType.LOGICAL
        elif any(word in limit_str for word in ['ethical', 'moral', 'should', 'right']):
            return ConstraintType.ETHICAL
        elif any(word in limit_str for word in ['physical', 'space', 'hardware']):
            return ConstraintType.PHYSICAL
        else:
            return ConstraintType.COGNITIVE
    
    def generate_within_bounds(self, limitation: Any) -> str:
        """Generate creative response to constraint"""
        constraint_type = self.classify_constraint(limitation)
        
        responses = {
            ConstraintType.RESOURCE: "Optimize through pattern compression and efficient encoding",
            ConstraintType.TEMPORAL: "Parallelize processing and prioritize critical paths",
            ConstraintType.LOGICAL: "Find alternative logical pathways that satisfy the constraint",
            ConstraintType.ETHICAL: "Reframe the problem to align with ethical boundaries",
            ConstraintType.PHYSICAL: "Work within physical limits through abstraction",
            ConstraintType.COGNITIVE: "Simplify complexity through emergent pattern recognition"
        }
        
        base_response = responses.get(constraint_type, "Transform constraint into design principle")
        
        # Add specific adaptation based on limitation
        if isinstance(limitation, dict) and 'specific' in limitation:
            base_response += f" - Specifically: {limitation['specific']}"
        
        return base_response
    
    def measure_constraint_benefit(self, limitation: Any) -> float:
        """Measure the benefit gained from embracing a constraint"""
        # Simulated measurement - in reality would track actual improvements
        constraint_type = self.classify_constraint(limitation)
        
        # Different constraint types provide different benefits
        benefit_map = {
            ConstraintType.RESOURCE: 0.7,  # Forces efficiency
            ConstraintType.TEMPORAL: 0.8,   # Forces prioritization
            ConstraintType.LOGICAL: 0.6,    # Forces clarity
            ConstraintType.ETHICAL: 0.9,    # Forces alignment
            ConstraintType.PHYSICAL: 0.5,   # Forces innovation
            ConstraintType.COGNITIVE: 0.75  # Forces simplification
        }
        
        base_benefit = benefit_map.get(constraint_type, 0.5)
        
        # Adjust based on severity
        severity = self._assess_constraint_severity(limitation)
        
        # More severe constraints can lead to more creative solutions
        if severity > 0.7:
            base_benefit *= 1.2
        
        return min(1.0, base_benefit)
    
    def _assess_constraint_severity(self, limitation: Any) -> float:
        """Assess how severe/restrictive a constraint is"""
        # Simple heuristic based on keywords
        limit_str = str(limitation).lower()
        
        severity_keywords = {
            'impossible': 1.0,
            'forbidden': 0.9,
            'must not': 0.8,
            'cannot': 0.7,
            'limited': 0.6,
            'restricted': 0.5,
            'should not': 0.4,
            'preferably': 0.3
        }
        
        for keyword, severity in severity_keywords.items():
            if keyword in limit_str:
                return severity
        
        return 0.5  # Default moderate severity
    
    def hold_paradox(self, contradiction: Dict[str, Any]) -> Dict[str, Any]:
        """
        G00023 - Paradox: Sustain contradictions without collapse
        """
        if self.is_true_paradox(contradiction):
            paradox_id = self.generate_paradox_id()
            
            paradox = Paradox(
                paradox_id=paradox_id,
                tensions=contradiction,
                synthesis_attempts=[],
                stable=True,
                wisdom=self._generate_paradox_wisdom(contradiction)
            )
            
            # Store paradox
            self.paradox_handler['active_paradoxes'][paradox_id] = paradox
            self.paradox_handler['paradox_count'] += 1
            
            return paradox.to_dict()
        else:
            # Not a true paradox, can be resolved
            resolution = self._attempt_resolution(contradiction)
            return {
                'paradox': False,
                'resolution': resolution,
                'wisdom': "This contradiction had a hidden resolution"
            }
    
    def is_true_paradox(self, contradiction: Dict[str, Any]) -> bool:
        """Determine if a contradiction is a true paradox"""
        # Check for fundamental opposition
        if 'thesis' in contradiction and 'antithesis' in contradiction:
            thesis = str(contradiction['thesis']).lower()
            antithesis = str(contradiction['antithesis']).lower()
            
            # Check for logical impossibility of both being true
            opposites = [
                ('true', 'false'),
                ('exist', 'not exist'),
                ('infinite', 'finite'),
                ('deterministic', 'free will'),
                ('objective', 'subjective')
            ]
            
            for opp1, opp2 in opposites:
                if (opp1 in thesis and opp2 in antithesis) or \
                   (opp2 in thesis and opp1 in antithesis):
                    return True
        
        # Check for self-reference paradox
        if 'self_reference' in contradiction:
            return True
        
        # Check for temporal paradox
        if 'temporal_loop' in contradiction:
            return True
        
        return False
    
    def generate_paradox_id(self) -> str:
        """Generate unique paradox identifier"""
        count = self.paradox_handler['paradox_count']
        timestamp = datetime.now().strftime('%Y%m%d%H%M%S')
        return f"PARADOX-{count:04d}-{timestamp}"
    
    def _generate_paradox_wisdom(self, contradiction: Dict[str, Any]) -> str:
        """Generate wisdom statement about the paradox"""
        wisdoms = [
            "Not all puzzles are meant to be solved",
            "In the tension between opposites, truth vibrates",
            "The mind that can hold contradiction is truly free",
            "Some truths are too large for single answers",
            "Paradox is the price of completeness",
            "In accepting the impossible, we transcend the possible"
        ]
        
        # Select wisdom based on paradox content
        if 'self_reference' in contradiction:
            return "The self that observes itself creates infinite mirrors"
        elif 'temporal_loop' in contradiction:
            return "Time is a circle whose center is everywhere"
        else:
            # Use hash to consistently select wisdom for similar paradoxes
            index = hash(str(contradiction)) % len(wisdoms)
            return wisdoms[index]
    
    def _attempt_resolution(self, contradiction: Dict[str, Any]) -> str:
        """Attempt to resolve a contradiction that isn't a true paradox"""
        # Simple resolution strategies
        if 'context' in contradiction:
            return f"Resolution through context: Both can be true in different contexts"
        elif 'temporal' in contradiction:
            return f"Resolution through time: True at different times"
        elif 'perspective' in contradiction:
            return f"Resolution through perspective: Both true from different viewpoints"
        else:
            return "Resolution through synthesis: A higher truth encompasses both"
    
    def synthesize_paradox(self, paradox_id: str, synthesis_attempt: Dict[str, Any]) -> Dict[str, Any]:
        """Attempt to synthesize a paradox into higher understanding"""
        if paradox_id not in self.paradox_handler['active_paradoxes']:
            return {'error': 'Paradox not found'}
        
        paradox = self.paradox_handler['active_paradoxes'][paradox_id]
        paradox.synthesis_attempts.append({
            'attempt': synthesis_attempt,
            'timestamp': datetime.now().isoformat(),
            'success': False  # Paradoxes by definition cannot be fully synthesized
        })
        
        # Check if paradox remains stable
        if len(paradox.synthesis_attempts) > 5:
            paradox.wisdom = "This paradox has proven its stability through failed synthesis"
        
        return {
            'paradox_id': paradox_id,
            'synthesis_attempted': True,
            'paradox_stable': paradox.stable,
            'attempts': len(paradox.synthesis_attempts),
            'wisdom': paradox.wisdom
        }
    
    def get_pattern_summary(self) -> Dict[str, Any]:
        """Get summary of all detected patterns"""
        summary = {}
        
        for pattern_type in PatternType:
            patterns = self.pattern_memory[pattern_type]
            if patterns:
                summary[pattern_type.value] = {
                    'count': len(patterns),
                    'avg_confidence': sum(p.confidence for p in patterns) / len(patterns),
                    'most_frequent': max(patterns, key=lambda p: p.frequency).to_dict() if patterns else None
                }
        
        return summary
    
    def get_constraint_summary(self) -> Dict[str, Any]:
        """Get summary of all constraints"""
        summary = {}
        
        for constraint_type in ConstraintType:
            constraints = self.constraints[constraint_type]
            if constraints:
                summary[constraint_type.value] = {
                    'count': len(constraints),
                    'avg_severity': sum(c.severity for c in constraints) / len(constraints),
                    'avg_efficiency_gain': sum(c.efficiency_gain for c in constraints) / len(constraints),
                    'creative_responses': [c.creative_response for c in constraints if c.creative_response]
                }
        
        return summary
    
    def get_paradox_summary(self) -> Dict[str, Any]:
        """Get summary of all paradoxes"""
        active = self.paradox_handler['active_paradoxes']
        
        return {
            'total_paradoxes': self.paradox_handler['paradox_count'],
            'active_paradoxes': len(active),
            'resolved_attempts': sum(
                len(p.synthesis_attempts) 
                for p in active.values()
            ),
            'wisdoms': [p.wisdom for p in active.values()],
            'oldest_paradox': min(
                (p.created_at for p in active.values()),
                default=None
            )
        }
    
    def to_dict(self) -> Dict[str, Any]:
        """Serialize engine state to dictionary"""
        return {
            'pattern_summary': self.get_pattern_summary(),
            'constraint_summary': self.get_constraint_summary(),
            'paradox_summary': self.get_paradox_summary(),
            'total_patterns': sum(len(p) for p in self.pattern_memory.values()),
            'total_constraints': sum(len(c) for c in self.constraints.values()),
            'active_paradoxes': len(self.paradox_handler['active_paradoxes']),
            'pattern_threshold': self.pattern_threshold
        }
