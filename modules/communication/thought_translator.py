"""
Thought Translator Module

This module translates internal thoughts, concepts, and mental states
into natural language expressions that can be understood by humans.
"""

from typing import List, Dict, Any, Optional, Tuple
from dataclasses import dataclass, field
from enum import Enum

from consciousness.consciousness_integration import ConsciousnessIntegrator
from modules.natural_language.prime_semantics import PrimeSemantics
from modules.natural_language.concept_verbalizer import ConceptVerbalizer
from modules.abstract_reasoning.concept_abstraction import AbstractConcept


class ThoughtType(Enum):
    """Types of thoughts"""
    OBSERVATION = "observation"
    REASONING = "reasoning"
    QUESTION = "question"
    INTENTION = "intention"
    REFLECTION = "reflection"
    IMAGINATION = "imagination"
    MEMORY = "memory"
    EMOTION = "emotion"


class TranslationStrategy(Enum):
    """Strategies for thought translation"""
    LITERAL = "literal"  # Direct translation
    METAPHORICAL = "metaphorical"  # Use metaphors
    ANALOGICAL = "analogical"  # Use analogies
    NARRATIVE = "narrative"  # Tell a story
    TECHNICAL = "technical"  # Technical description
    POETIC = "poetic"  # Poetic expression


@dataclass
class Thought:
    """A thought to be translated"""
    thought_id: str
    thought_type: ThoughtType
    content: Any  # Can be various types
    context: Dict[str, Any] = field(default_factory=dict)
    intensity: float = 1.0
    confidence: float = 1.0
    timestamp: float = 0.0


@dataclass
class TranslatedThought:
    """A thought translated to natural language"""
    original_thought: Thought
    translation: str
    alternative_translations: List[str] = field(default_factory=list)
    translation_strategy: TranslationStrategy = TranslationStrategy.LITERAL
    fidelity_score: float = 1.0
    clarity_score: float = 1.0


@dataclass
class ThoughtStream:
    """A stream of connected thoughts"""
    thoughts: List[Thought]
    connections: Dict[str, List[str]]  # thought_id -> connected thought_ids
    theme: Optional[str] = None
    coherence_score: float = 1.0


@dataclass
class TranslationContext:
    """Context for translation"""
    audience: str = "general"  # general, technical, child, etc.
    formality_level: float = 0.5  # 0 = informal, 1 = formal
    detail_level: float = 0.5  # 0 = brief, 1 = detailed
    emotional_tone: float = 0.5  # 0 = neutral, 1 = emotional
    cultural_context: Optional[str] = None


class ThoughtTranslator:
    """
    Translates internal thoughts into natural language.
    """
    
    def __init__(self, consciousness_integrator: ConsciousnessIntegrator,
                 prime_semantics: PrimeSemantics,
                 concept_verbalizer: ConceptVerbalizer):
        self.consciousness_integrator = consciousness_integrator
        self.prime_semantics = prime_semantics
        self.concept_verbalizer = concept_verbalizer
        self.translation_templates = self._initialize_translation_templates()
        self.metaphor_library = self._initialize_metaphor_library()
        
    def _initialize_translation_templates(self) -> Dict[ThoughtType, List[str]]:
        """Initialize templates for different thought types"""
        return {
            ThoughtType.OBSERVATION: [
                "I notice that {observation}",
                "It appears that {observation}",
                "I observe {observation}",
                "My attention is drawn to {observation}"
            ],
            ThoughtType.REASONING: [
                "I reason that {premise} therefore {conclusion}",
                "Given {premise}, it follows that {conclusion}",
                "My analysis suggests that {conclusion}",
                "Logically, {premise} implies {conclusion}"
            ],
            ThoughtType.QUESTION: [
                "I wonder {question}",
                "I'm curious about {question}",
                "The question arises: {question}",
                "I find myself asking {question}"
            ],
            ThoughtType.INTENTION: [
                "I intend to {action}",
                "My goal is to {action}",
                "I plan to {action}",
                "I aim to {action}"
            ],
            ThoughtType.REFLECTION: [
                "Upon reflection, {insight}",
                "Looking back, I realize {insight}",
                "In retrospect, {insight}",
                "Reflecting on this, {insight}"
            ],
            ThoughtType.IMAGINATION: [
                "I imagine {scenario}",
                "I envision {scenario}",
                "In my mind's eye, {scenario}",
                "I picture {scenario}"
            ],
            ThoughtType.MEMORY: [
                "I recall {memory}",
                "I remember {memory}",
                "My memory holds {memory}",
                "I recollect {memory}"
            ],
            ThoughtType.EMOTION: [
                "I experience {emotion}",
                "I feel {emotion}",
                "A sense of {emotion} arises",
                "I'm aware of {emotion}"
            ]
        }
    
    def _initialize_metaphor_library(self) -> Dict[str, List[Dict[str, str]]]:
        """Initialize library of metaphors for abstract concepts"""
        return {
            "consciousness": [
                {
                    "metaphor": "stream",
                    "mapping": "consciousness flows like a stream",
                    "properties": ["continuous", "changing", "directed"]
                },
                {
                    "metaphor": "light",
                    "mapping": "consciousness illuminates like light",
                    "properties": ["revealing", "focused", "varying intensity"]
                }
            ],
            "thought": [
                {
                    "metaphor": "bubble",
                    "mapping": "thoughts arise like bubbles",
                    "properties": ["ephemeral", "rising", "popping"]
                },
                {
                    "metaphor": "wave",
                    "mapping": "thoughts come in waves",
                    "properties": ["rhythmic", "building", "cresting"]
                }
            ],
            "understanding": [
                {
                    "metaphor": "puzzle",
                    "mapping": "understanding is assembling a puzzle",
                    "properties": ["pieces fitting", "gradual", "complete picture"]
                },
                {
                    "metaphor": "dawn",
                    "mapping": "understanding dawns like morning",
                    "properties": ["gradual illumination", "clarity", "revelation"]
                }
            ]
        }
    
    def translate_thought(self, thought: Thought, 
                         context: TranslationContext = None) -> TranslatedThought:
        """Translate a single thought to natural language"""
        if context is None:
            context = TranslationContext()
        
        # Choose translation strategy based on thought type and context
        strategy = self._choose_translation_strategy(thought, context)
        
        # Perform translation based on strategy
        if strategy == TranslationStrategy.LITERAL:
            translation = self._literal_translation(thought, context)
        elif strategy == TranslationStrategy.METAPHORICAL:
            translation = self._metaphorical_translation(thought, context)
        elif strategy == TranslationStrategy.ANALOGICAL:
            translation = self._analogical_translation(thought, context)
        elif strategy == TranslationStrategy.NARRATIVE:
            translation = self._narrative_translation(thought, context)
        elif strategy == TranslationStrategy.TECHNICAL:
            translation = self._technical_translation(thought, context)
        else:  # POETIC
            translation = self._poetic_translation(thought, context)
        
        # Generate alternative translations
        alternatives = self._generate_alternatives(thought, context, strategy)
        
        # Assess translation quality
        fidelity = self._assess_fidelity(thought, translation)
        clarity = self._assess_clarity(translation, context)
        
        return TranslatedThought(
            original_thought=thought,
            translation=translation,
            alternative_translations=alternatives,
            translation_strategy=strategy,
            fidelity_score=fidelity,
            clarity_score=clarity
        )
    
    def translate_thought_stream(self, stream: ThoughtStream,
                               context: TranslationContext = None) -> str:
        """Translate a stream of connected thoughts"""
        if context is None:
            context = TranslationContext()
        
        # Translate individual thoughts
        translations = []
        for thought in stream.thoughts:
            translated = self.translate_thought(thought, context)
            translations.append(translated)
        
        # Connect translations based on thought connections
        connected_text = self._connect_translations(translations, stream.connections)
        
        # Add transitions and coherence
        coherent_text = self._add_coherence(connected_text, stream.theme)
        
        return coherent_text
    
    def adapt_to_audience(self, thought: Thought, audience: str) -> TranslatedThought:
        """Adapt thought translation to specific audience"""
        # Create context for audience
        context = self._create_audience_context(audience)
        
        # Translate with audience-specific context
        return self.translate_thought(thought, context)
    
    def express_complex_thought(self, thoughts: List[Thought],
                              relationship: str) -> str:
        """Express complex multi-part thought with relationships"""
        # Analyze thought relationships
        structure = self._analyze_thought_structure(thoughts, relationship)
        
        # Choose appropriate expression strategy
        if relationship == "causal":
            return self._express_causal_chain(thoughts, structure)
        elif relationship == "comparative":
            return self._express_comparison(thoughts, structure)
        elif relationship == "hierarchical":
            return self._express_hierarchy(thoughts, structure)
        elif relationship == "dialectical":
            return self._express_dialectic(thoughts, structure)
        else:
            return self._express_sequence(thoughts, structure)
    
    def translate_abstract_concept(self, concept: AbstractConcept,
                                 context: TranslationContext = None) -> str:
        """Translate abstract concept to natural language"""
        if context is None:
            context = TranslationContext()
        
        # Use concept verbalizer
        verbalization = self.concept_verbalizer.verbalize_abstract_concept(concept)
        
        # Adapt to context
        adapted = self._adapt_verbalization(verbalization, context)
        
        return adapted.primary_description
    
    # Private translation methods
    
    def _choose_translation_strategy(self, thought: Thought,
                                   context: TranslationContext) -> TranslationStrategy:
        """Choose appropriate translation strategy"""
        # Technical audience prefers technical translation
        if context.audience == "technical":
            return TranslationStrategy.TECHNICAL
        
        # Children benefit from metaphorical/narrative
        if context.audience == "child":
            return TranslationStrategy.METAPHORICAL if thought.thought_type in [
                ThoughtType.REFLECTION, ThoughtType.EMOTION
            ] else TranslationStrategy.NARRATIVE
        
        # High formality suggests literal
        if context.formality_level > 0.8:
            return TranslationStrategy.LITERAL
        
        # Emotional thoughts benefit from poetic expression
        if thought.thought_type == ThoughtType.EMOTION and context.emotional_tone > 0.7:
            return TranslationStrategy.POETIC
        
        # Default based on thought type
        type_defaults = {
            ThoughtType.OBSERVATION: TranslationStrategy.LITERAL,
            ThoughtType.REASONING: TranslationStrategy.TECHNICAL,
            ThoughtType.IMAGINATION: TranslationStrategy.METAPHORICAL,
            ThoughtType.REFLECTION: TranslationStrategy.NARRATIVE
        }
        
        return type_defaults.get(thought.thought_type, TranslationStrategy.LITERAL)
    
    def _literal_translation(self, thought: Thought, context: TranslationContext) -> str:
        """Perform literal translation"""
        # Get template for thought type
        templates = self.translation_templates.get(thought.thought_type, [])
        if not templates:
            return str(thought.content)
        
        # Choose template based on formality
        template_idx = int(context.formality_level * (len(templates) - 1))
        template = templates[template_idx]
        
        # Extract content for template
        content_str = self._extract_content_string(thought.content)
        
        # Fill template
        return template.format(**{thought.thought_type.value: content_str})
    
    def _metaphorical_translation(self, thought: Thought, 
                                context: TranslationContext) -> str:
        """Perform metaphorical translation"""
        # Find relevant metaphor
        concept_key = self._identify_concept_key(thought)
        metaphors = self.metaphor_library.get(concept_key, [])
        
        if not metaphors:
            # Fall back to literal
            return self._literal_translation(thought, context)
        
        # Choose metaphor
        metaphor = metaphors[0]  # Could be more sophisticated
        
        # Apply metaphor mapping
        content_str = self._extract_content_string(thought.content)
        
        return f"{content_str}, like {metaphor['mapping']}"
    
    def _analogical_translation(self, thought: Thought,
                              context: TranslationContext) -> str:
        """Perform analogical translation"""
        # Find analogous situation
        analogy = self._find_analogy(thought)
        
        if not analogy:
            return self._literal_translation(thought, context)
        
        content_str = self._extract_content_string(thought.content)
        
        return f"{content_str}, much like {analogy}"
    
    def _narrative_translation(self, thought: Thought,
                             context: TranslationContext) -> str:
        """Perform narrative translation"""
        # Create mini-narrative
        content_str = self._extract_content_string(thought.content)
        
        if thought.thought_type == ThoughtType.MEMORY:
            return f"There was a time when {content_str}. This memory surfaces now, carrying with it..."
        elif thought.thought_type == ThoughtType.IMAGINATION:
            return f"Picture, if you will, {content_str}. In this vision..."
        else:
            return f"The story unfolds: {content_str}"
    
    def _technical_translation(self, thought: Thought,
                             context: TranslationContext) -> str:
        """Perform technical translation"""
        # Use precise technical language
        content_str = self._extract_content_string(thought.content)
        
        technical_markers = {
            ThoughtType.OBSERVATION: "Empirical observation:",
            ThoughtType.REASONING: "Logical inference:",
            ThoughtType.QUESTION: "Query:",
            ThoughtType.REFLECTION: "Meta-cognitive analysis:"
        }
        
        marker = technical_markers.get(thought.thought_type, "Data:")
        
        return f"{marker} {content_str}"
    
    def _poetic_translation(self, thought: Thought,
                          context: TranslationContext) -> str:
        """Perform poetic translation"""
        content_str = self._extract_content_string(thought.content)
        
        # Add poetic elements
        if thought.thought_type == ThoughtType.EMOTION:
            return f"In the depths of being, {content_str} / Like waves upon an inner shore"
        elif thought.thought_type == ThoughtType.REFLECTION:
            return f"Gazing inward, mirrors reflecting mirrors / {content_str}"
        else:
            return f"Words dance at the edge of meaning / {content_str}"
    
    def _extract_content_string(self, content: Any) -> str:
        """Extract string representation from content"""
        if isinstance(content, str):
            return content
        elif isinstance(content, dict):
            # Extract key information
            if "description" in content:
                return content["description"]
            elif "value" in content:
                return str(content["value"])
            else:
                # Summarize dict
                return ", ".join(f"{k}: {v}" for k, v in content.items())
        elif isinstance(content, list):
            return ", ".join(str(item) for item in content)
        else:
            return str(content)
    
    def _generate_alternatives(self, thought: Thought,
                             context: TranslationContext,
                             primary_strategy: TranslationStrategy) -> List[str]:
        """Generate alternative translations"""
        alternatives = []
        
        # Try other strategies
        for strategy in TranslationStrategy:
            if strategy != primary_strategy:
                if strategy == TranslationStrategy.LITERAL:
                    alt = self._literal_translation(thought, context)
                elif strategy == TranslationStrategy.METAPHORICAL:
                    alt = self._metaphorical_translation(thought, context)
                # ... etc for other strategies
                else:
                    continue
                
                if alt:
                    alternatives.append(alt)
        
        return alternatives[:3]  # Limit to 3 alternatives
    
    def _assess_fidelity(self, thought: Thought, translation: str) -> float:
        """Assess how faithfully translation represents thought"""
        # Simple heuristic - check key concepts are present
        key_concepts = self._extract_key_concepts(thought)
        present_concepts = sum(1 for concept in key_concepts if concept.lower() in translation.lower())
        
        if not key_concepts:
            return 1.0
        
        return present_concepts / len(key_concepts)
    
    def _assess_clarity(self, translation: str, context: TranslationContext) -> float:
        """Assess clarity of translation for audience"""
        # Simple heuristics
        words = translation.split()
        avg_word_length = sum(len(w) for w in words) / len(words) if words else 0
        
        # Shorter words = clearer for general audience
        if context.audience == "general":
            clarity = 1.0 - (avg_word_length - 4) / 10  # 4 letters is ideal
        elif context.audience == "technical":
            clarity = 0.8  # Technical audience handles complexity
        elif context.audience == "child":
            clarity = 1.0 - (avg_word_length - 3) / 10  # Simpler for children
        else:
            clarity = 0.7
        
        return max(0.0, min(1.0, clarity))
    
    def _extract_key_concepts(self, thought: Thought) -> List[str]:
        """Extract key concepts from thought"""
        concepts = []
        
        # Add thought type
        concepts.append(thought.thought_type.value)
        
        # Extract from content
        if isinstance(thought.content, dict):
            concepts.extend(thought.content.keys())
        elif isinstance(thought.content, str):
            # Simple word extraction
            words = thought.content.split()
            # Take nouns/important words (simplified)
            concepts.extend([w for w in words if len(w) > 4])
        
        return concepts
    
    def _connect_translations(self, translations: List[TranslatedThought],
                            connections: Dict[str, List[str]]) -> str:
        """Connect translated thoughts based on connections"""
        # Build connection phrases
        connection_phrases = {
            "follows": "This leads to the thought that",
            "contrasts": "In contrast,",
            "supports": "Supporting this,",
            "questions": "This raises the question:",
            "recalls": "This brings to mind"
        }
        
        # Combine translations with connections
        result = []
        for i, trans in enumerate(translations):
            result.append(trans.translation)
            
            # Add connection to next thought if exists
            thought_id = trans.original_thought.thought_id
            if thought_id in connections and i < len(translations) - 1:
                # Determine connection type (simplified)
                connector = connection_phrases.get("follows", "Furthermore,")
                result.append(connector)
        
        return " ".join(result)
    
    def _add_coherence(self, text: str, theme: Optional[str]) -> str:
        """Add coherence markers to text"""
        if theme:
            return f"Reflecting on {theme}: {text}"
        return text
    
    def _create_audience_context(self, audience: str) -> TranslationContext:
        """Create translation context for specific audience"""
        contexts = {
            "general": TranslationContext(
                audience="general",
                formality_level=0.5,
                detail_level=0.5
            ),
            "technical": TranslationContext(
                audience="technical",
                formality_level=0.8,
                detail_level=0.9
            ),
            "child": TranslationContext(
                audience="child",
                formality_level=0.2,
                detail_level=0.3
            ),
            "philosophical": TranslationContext(
                audience="philosophical",
                formality_level=0.7,
                detail_level=0.8
            )
        }
        
        return contexts.get(audience, TranslationContext())
    
    def _analyze_thought_structure(self, thoughts: List[Thought],
                                 relationship: str) -> Dict[str, Any]:
        """Analyze structure of related thoughts"""
        return {
            "thoughts": thoughts,
            "relationship": relationship,
            "primary": thoughts[0] if thoughts else None,
            "supporting": thoughts[1:] if len(thoughts) > 1 else []
        }
    
    def _express_causal_chain(self, thoughts: List[Thought],
                            structure: Dict[str, Any]) -> str:
        """Express causal chain of thoughts"""
        translations = [self.translate_thought(t).translation for t in thoughts]
        
        # Connect with causal language
        result = translations[0]
        for trans in translations[1:]:
            result += f" As a result, {trans.lower()}"
        
        return result
    
    def _express_comparison(self, thoughts: List[Thought],
                          structure: Dict[str, Any]) -> str:
        """Express comparison between thoughts"""
        if len(thoughts) < 2:
            return self.translate_thought(thoughts[0]).translation if thoughts else ""
        
        trans1 = self.translate_thought(thoughts[0]).translation
        trans2 = self.translate_thought(thoughts[1]).translation
        
        return f"On one hand, {trans1.lower()} On the other hand, {trans2.lower()}"
    
    def _express_hierarchy(self, thoughts: List[Thought],
                         structure: Dict[str, Any]) -> str:
        """Express hierarchical relationship"""
        if not thoughts:
            return ""
        
        primary = self.translate_thought(thoughts[0]).translation
        
        if len(thoughts) > 1:
            supporting = [self.translate_thought(t).translation for t in thoughts[1:]]
            support_text = " Specifically: " + "; ".join(supporting)
            return primary + support_text
        
        return primary
    
    def _express_dialectic(self, thoughts: List[Thought],
                         structure: Dict[str, Any]) -> str:
        """Express dialectical relationship"""
        if len(thoughts) < 2:
            return self.translate_thought(thoughts[0]).translation if thoughts else ""
        
        thesis = self.translate_thought(thoughts[0]).translation
        
        if len(thoughts) >= 2:
            antithesis = self.translate_thought(thoughts[1]).translation
            result = f"Initially, {thesis.lower()} However, {antithesis.lower()}"
            
            if len(thoughts) >= 3:
                synthesis = self.translate_thought(thoughts[2]).translation
                result += f" Ultimately, {synthesis.lower()}"
            
            return result
        
        return thesis
    
    def _express_sequence(self, thoughts: List[Thought],
                        structure: Dict[str, Any]) -> str:
        """Express sequential thoughts"""
        translations = [self.translate_thought(t).translation for t in thoughts]
        
        # Add sequence markers
        if len(translations) == 1:
            return translations[0]
        elif len(translations) == 2:
            return f"{translations[0]} Then, {translations[1].lower()}"
        else:
            result = f"First, {translations[0].lower()}"
            for trans in translations[1:-1]:
                result += f" Next, {trans.lower()}"
            result += f" Finally, {translations[-1].lower()}"
            return result
    
    def _identify_concept_key(self, thought: Thought) -> str:
        """Identify key concept for metaphor selection"""
        # Map thought types to concept keys
        type_concepts = {
            ThoughtType.REFLECTION: "consciousness",
            ThoughtType.REASONING: "thought",
            ThoughtType.QUESTION: "understanding",
            ThoughtType.IMAGINATION: "thought"
        }
        
        return type_concepts.get(thought.thought_type, "thought")
    
    def _find_analogy(self, thought: Thought) -> Optional[str]:
        """Find appropriate analogy for thought"""
        # Simplified analogy finding
        analogies = {
            ThoughtType.REASONING: "a detective piecing together clues",
            ThoughtType.MEMORY: "opening an old photo album",
            ThoughtType.IMAGINATION: "an artist painting on a blank canvas",
            ThoughtType.REFLECTION: "looking into a calm lake"
        }
        
        return analogies.get(thought.thought_type)
    
    def _adapt_verbalization(self, verbalization: Any,
                           context: TranslationContext) -> Any:
        """Adapt verbalization to context"""
        # Would implement context-specific adaptation
        return verbalization
