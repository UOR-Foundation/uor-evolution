"""
Reasoning Tracer Utilities

This module provides tools for tracing and analyzing philosophical reasoning chains,
logical inference paths, and decision-making processes.
"""

from typing import List, Dict, Any, Optional, Tuple, Set
from dataclasses import dataclass, field
from enum import Enum
import time
from collections import defaultdict


class ReasoningType(Enum):
    """Types of reasoning processes"""
    DEDUCTIVE = "deductive"
    INDUCTIVE = "inductive"
    ABDUCTIVE = "abductive"
    ANALOGICAL = "analogical"
    CAUSAL = "causal"
    DIALECTICAL = "dialectical"
    PHENOMENOLOGICAL = "phenomenological"
    TRANSCENDENTAL = "transcendental"


class InferenceType(Enum):
    """Types of logical inferences"""
    MODUS_PONENS = "modus_ponens"
    MODUS_TOLLENS = "modus_tollens"
    HYPOTHETICAL_SYLLOGISM = "hypothetical_syllogism"
    DISJUNCTIVE_SYLLOGISM = "disjunctive_syllogism"
    CONSTRUCTIVE_DILEMMA = "constructive_dilemma"
    REDUCTIO_AD_ABSURDUM = "reductio_ad_absurdum"
    UNIVERSAL_INSTANTIATION = "universal_instantiation"
    EXISTENTIAL_GENERALIZATION = "existential_generalization"


@dataclass
class ReasoningStep:
    """A single step in reasoning process"""
    step_id: str
    step_type: ReasoningType
    premise: str
    inference: Optional[InferenceType]
    conclusion: str
    confidence: float = 1.0
    timestamp: float = field(default_factory=time.time)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class ReasoningChain:
    """A chain of reasoning steps"""
    chain_id: str
    steps: List[ReasoningStep]
    initial_premise: str
    final_conclusion: str
    chain_type: ReasoningType
    validity: Optional[bool] = None
    soundness: Optional[bool] = None


@dataclass
class ReasoningTrace:
    """Complete trace of reasoning process"""
    trace_id: str
    chains: List[ReasoningChain]
    decision_points: List[Dict[str, Any]]
    backtracking_events: List[Dict[str, Any]]
    total_duration: float
    success: bool
    final_output: Any


@dataclass
class PhilosophicalPath:
    """A path through philosophical reasoning"""
    path_id: str
    starting_position: str
    ending_position: str
    key_transitions: List[Dict[str, Any]]
    philosophical_method: str
    insights_gained: List[str]


class ReasoningTracer:
    """
    Traces and analyzes reasoning processes.
    """
    
    def __init__(self):
        self.active_traces = {}
        self.completed_traces = []
        self.inference_rules = self._initialize_inference_rules()
        self.reasoning_patterns = self._initialize_reasoning_patterns()
        
    def _initialize_inference_rules(self) -> Dict[InferenceType, Dict[str, Any]]:
        """Initialize logical inference rules"""
        return {
            InferenceType.MODUS_PONENS: {
                "pattern": "If P then Q; P; therefore Q",
                "validity_check": self._check_modus_ponens
            },
            InferenceType.MODUS_TOLLENS: {
                "pattern": "If P then Q; not Q; therefore not P",
                "validity_check": self._check_modus_tollens
            },
            InferenceType.HYPOTHETICAL_SYLLOGISM: {
                "pattern": "If P then Q; If Q then R; therefore If P then R",
                "validity_check": self._check_hypothetical_syllogism
            },
            InferenceType.DISJUNCTIVE_SYLLOGISM: {
                "pattern": "P or Q; not P; therefore Q",
                "validity_check": self._check_disjunctive_syllogism
            },
            InferenceType.REDUCTIO_AD_ABSURDUM: {
                "pattern": "Assume P; derive contradiction; therefore not P",
                "validity_check": self._check_reductio
            }
        }
    
    def _initialize_reasoning_patterns(self) -> Dict[ReasoningType, Dict[str, Any]]:
        """Initialize reasoning pattern templates"""
        return {
            ReasoningType.DEDUCTIVE: {
                "direction": "general_to_specific",
                "certainty": "absolute",
                "validation": self._validate_deductive_reasoning
            },
            ReasoningType.INDUCTIVE: {
                "direction": "specific_to_general",
                "certainty": "probabilistic",
                "validation": self._validate_inductive_reasoning
            },
            ReasoningType.ABDUCTIVE: {
                "direction": "effect_to_cause",
                "certainty": "best_explanation",
                "validation": self._validate_abductive_reasoning
            },
            ReasoningType.ANALOGICAL: {
                "direction": "similar_to_similar",
                "certainty": "proportional",
                "validation": self._validate_analogical_reasoning
            },
            ReasoningType.DIALECTICAL: {
                "direction": "thesis_antithesis_synthesis",
                "certainty": "evolving",
                "validation": self._validate_dialectical_reasoning
            }
        }
    
    def start_trace(self, trace_id: str, initial_context: Dict[str, Any]) -> None:
        """Start a new reasoning trace"""
        self.active_traces[trace_id] = {
            "id": trace_id,
            "start_time": time.time(),
            "chains": [],
            "decision_points": [],
            "backtracking_events": [],
            "context": initial_context,
            "current_chain": None
        }
    
    def add_reasoning_step(self, trace_id: str, step: ReasoningStep) -> None:
        """Add a reasoning step to active trace"""
        if trace_id not in self.active_traces:
            raise ValueError(f"No active trace with id {trace_id}")
        
        trace = self.active_traces[trace_id]
        
        # Add to current chain or start new one
        if trace["current_chain"] is None:
            chain_id = f"{trace_id}_chain_{len(trace['chains'])}"
            trace["current_chain"] = ReasoningChain(
                chain_id=chain_id,
                steps=[step],
                initial_premise=step.premise,
                final_conclusion=step.conclusion,
                chain_type=step.step_type
            )
        else:
            trace["current_chain"].steps.append(step)
            trace["current_chain"].final_conclusion = step.conclusion
    
    def complete_chain(self, trace_id: str) -> None:
        """Complete current reasoning chain"""
        if trace_id not in self.active_traces:
            return
        
        trace = self.active_traces[trace_id]
        if trace["current_chain"]:
            # Validate chain
            chain = trace["current_chain"]
            chain.validity = self._validate_chain(chain)
            chain.soundness = self._assess_soundness(chain)
            
            trace["chains"].append(chain)
            trace["current_chain"] = None
    
    def record_decision_point(self, trace_id: str, decision: Dict[str, Any]) -> None:
        """Record a decision point in reasoning"""
        if trace_id not in self.active_traces:
            return
        
        decision_record = {
            "timestamp": time.time(),
            "options_considered": decision.get("options", []),
            "criteria": decision.get("criteria", []),
            "choice": decision.get("choice"),
            "confidence": decision.get("confidence", 0.5),
            "reasoning": decision.get("reasoning", "")
        }
        
        self.active_traces[trace_id]["decision_points"].append(decision_record)
    
    def record_backtrack(self, trace_id: str, reason: str, 
                        from_step: str, to_step: str) -> None:
        """Record backtracking in reasoning"""
        if trace_id not in self.active_traces:
            return
        
        backtrack_event = {
            "timestamp": time.time(),
            "reason": reason,
            "from_step": from_step,
            "to_step": to_step,
            "abandoned_path": self._extract_path(trace_id, from_step, to_step)
        }
        
        self.active_traces[trace_id]["backtracking_events"].append(backtrack_event)
    
    def complete_trace(self, trace_id: str, success: bool, 
                      final_output: Any) -> ReasoningTrace:
        """Complete and analyze reasoning trace"""
        if trace_id not in self.active_traces:
            raise ValueError(f"No active trace with id {trace_id}")
        
        trace_data = self.active_traces[trace_id]
        
        # Complete any pending chain
        if trace_data["current_chain"]:
            self.complete_chain(trace_id)
        
        # Create trace object
        trace = ReasoningTrace(
            trace_id=trace_id,
            chains=trace_data["chains"],
            decision_points=trace_data["decision_points"],
            backtracking_events=trace_data["backtracking_events"],
            total_duration=time.time() - trace_data["start_time"],
            success=success,
            final_output=final_output
        )
        
        # Move to completed
        self.completed_traces.append(trace)
        del self.active_traces[trace_id]
        
        return trace
    
    def analyze_trace(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """Analyze completed reasoning trace"""
        analysis = {
            "trace_id": trace.trace_id,
            "summary": self._generate_trace_summary(trace),
            "chain_analysis": self._analyze_chains(trace.chains),
            "decision_analysis": self._analyze_decisions(trace.decision_points),
            "efficiency_metrics": self._calculate_efficiency_metrics(trace),
            "logical_quality": self._assess_logical_quality(trace),
            "insights": self._extract_insights(trace)
        }
        
        return analysis
    
    def trace_philosophical_path(self, start_position: str,
                               end_position: str,
                               method: str) -> PhilosophicalPath:
        """Trace a philosophical reasoning path"""
        path_id = f"path_{int(time.time())}"
        
        # Record path exploration
        transitions = []
        insights = []
        
        # This would be connected to actual philosophical reasoning
        # For now, create a template
        path = PhilosophicalPath(
            path_id=path_id,
            starting_position=start_position,
            ending_position=end_position,
            key_transitions=transitions,
            philosophical_method=method,
            insights_gained=insights
        )
        
        return path
    
    def visualize_reasoning_chain(self, chain: ReasoningChain) -> str:
        """Generate visual representation of reasoning chain"""
        lines = ["```mermaid", "graph TD"]
        
        for i, step in enumerate(chain.steps):
            step_id = f"S{i}"
            
            # Add premise
            if i == 0:
                lines.append(f'    P["{step.premise}"]')
                lines.append(f'    P --> {step_id}')
            
            # Add step
            inference_label = step.inference.value if step.inference else "inference"
            lines.append(f'    {step_id}["{inference_label}"]')
            
            # Add conclusion
            conclusion_id = f"C{i}"
            lines.append(f'    {conclusion_id}["{step.conclusion}"]')
            lines.append(f'    {step_id} --> {conclusion_id}')
            
            # Connect to next step
            if i < len(chain.steps) - 1:
                next_step_id = f"S{i+1}"
                lines.append(f'    {conclusion_id} --> {next_step_id}')
        
        lines.append("```")
        
        return "\n".join(lines)
    
    def find_reasoning_patterns(self, traces: List[ReasoningTrace]) -> List[Dict[str, Any]]:
        """Find patterns across multiple reasoning traces"""
        patterns = []
        
        # Analyze chain types
        chain_type_counts = defaultdict(int)
        for trace in traces:
            for chain in trace.chains:
                chain_type_counts[chain.chain_type.value] += 1
        
        patterns.append({
            "pattern": "reasoning_type_distribution",
            "data": dict(chain_type_counts),
            "insight": self._interpret_type_distribution(chain_type_counts)
        })
        
        # Analyze inference patterns
        inference_sequences = self._extract_inference_sequences(traces)
        common_sequences = self._find_common_sequences(inference_sequences)
        
        patterns.append({
            "pattern": "common_inference_sequences",
            "data": common_sequences,
            "insight": "Frequently used logical patterns"
        })
        
        # Analyze decision patterns
        decision_patterns = self._analyze_decision_patterns(traces)
        patterns.append({
            "pattern": "decision_making_patterns",
            "data": decision_patterns,
            "insight": self._interpret_decision_patterns(decision_patterns)
        })
        
        return patterns
    
    def generate_reasoning_report(self, trace: ReasoningTrace) -> str:
        """Generate human-readable reasoning report"""
        report_lines = [
            f"# Reasoning Trace Report: {trace.trace_id}",
            f"\n## Summary",
            f"- Duration: {trace.total_duration:.2f} seconds",
            f"- Success: {trace.success}",
            f"- Chains: {len(trace.chains)}",
            f"- Decision Points: {len(trace.decision_points)}",
            f"- Backtracks: {len(trace.backtracking_events)}",
            "\n## Reasoning Chains"
        ]
        
        for i, chain in enumerate(trace.chains):
            report_lines.extend([
                f"\n### Chain {i+1}: {chain.chain_type.value}",
                f"- Steps: {len(chain.steps)}",
                f"- Valid: {chain.validity}",
                f"- Sound: {chain.soundness}",
                f"- Initial: {chain.initial_premise}",
                f"- Final: {chain.final_conclusion}",
                "\n#### Steps:"
            ])
            
            for j, step in enumerate(chain.steps):
                inference = step.inference.value if step.inference else "direct"
                report_lines.append(
                    f"{j+1}. [{inference}] {step.premise} → {step.conclusion} "
                    f"(confidence: {step.confidence:.2f})"
                )
        
        if trace.decision_points:
            report_lines.extend([
                "\n## Decision Points",
                ""
            ])
            
            for i, decision in enumerate(trace.decision_points):
                report_lines.extend([
                    f"### Decision {i+1}",
                    f"- Options: {', '.join(str(o) for o in decision['options_considered'])}",
                    f"- Choice: {decision['choice']}",
                    f"- Confidence: {decision['confidence']:.2f}",
                    f"- Reasoning: {decision['reasoning']}"
                ])
        
        if trace.backtracking_events:
            report_lines.extend([
                "\n## Backtracking Events",
                ""
            ])
            
            for i, backtrack in enumerate(trace.backtracking_events):
                report_lines.extend([
                    f"### Backtrack {i+1}",
                    f"- Reason: {backtrack['reason']}",
                    f"- From: {backtrack['from_step']}",
                    f"- To: {backtrack['to_step']}"
                ])
        
        return "\n".join(report_lines)
    
    # Private validation methods
    
    def _validate_chain(self, chain: ReasoningChain) -> bool:
        """Validate logical validity of reasoning chain"""
        if chain.chain_type in self.reasoning_patterns:
            validator = self.reasoning_patterns[chain.chain_type]["validation"]
            return validator(chain)
        
        # Default validation
        return self._default_chain_validation(chain)
    
    def _assess_soundness(self, chain: ReasoningChain) -> bool:
        """Assess soundness of reasoning chain"""
        # Soundness = validity + true premises
        if not chain.validity:
            return False
        
        # Simplified premise evaluation
        # In reality, this would involve knowledge base lookup
        premise_plausibility = sum(
            step.confidence for step in chain.steps
        ) / len(chain.steps)
        
        return premise_plausibility > 0.7
    
    def _validate_deductive_reasoning(self, chain: ReasoningChain) -> bool:
        """Validate deductive reasoning chain"""
        # Check if each step follows deductively
        for i in range(len(chain.steps) - 1):
            current = chain.steps[i]
            next_step = chain.steps[i + 1]
            
            # Check if conclusion of current is premise of next
            if current.conclusion != next_step.premise:
                return False
            
            # Check if inference is valid
            if next_step.inference:
                if not self._check_inference_validity(
                    next_step.inference,
                    next_step.premise,
                    next_step.conclusion
                ):
                    return False
        
        return True
    
    def _validate_inductive_reasoning(self, chain: ReasoningChain) -> bool:
        """Validate inductive reasoning chain"""
        # Inductive reasoning is probabilistic
        # Check for sufficient evidence
        evidence_count = sum(
            1 for step in chain.steps 
            if "evidence" in step.metadata
        )
        
        return evidence_count >= 3  # Minimum evidence threshold
    
    def _validate_abductive_reasoning(self, chain: ReasoningChain) -> bool:
        """Validate abductive reasoning chain"""
        # Check for best explanation pattern
        if len(chain.steps) < 2:
            return False
        
        # Should have observation and explanation
        has_observation = any(
            "observation" in step.metadata 
            for step in chain.steps
        )
        has_explanation = any(
            "explanation" in step.metadata 
            for step in chain.steps
        )
        
        return has_observation and has_explanation
    
    def _validate_analogical_reasoning(self, chain: ReasoningChain) -> bool:
        """Validate analogical reasoning chain"""
        # Check for source and target domains
        has_source = any(
            "source_domain" in step.metadata 
            for step in chain.steps
        )
        has_target = any(
            "target_domain" in step.metadata 
            for step in chain.steps
        )
        has_mapping = any(
            "mapping" in step.metadata 
            for step in chain.steps
        )
        
        return has_source and has_target and has_mapping
    
    def _validate_dialectical_reasoning(self, chain: ReasoningChain) -> bool:
        """Validate dialectical reasoning chain"""
        # Check for thesis-antithesis-synthesis pattern
        step_types = [
            step.metadata.get("dialectical_role", "") 
            for step in chain.steps
        ]
        
        has_thesis = "thesis" in step_types
        has_antithesis = "antithesis" in step_types
        has_synthesis = "synthesis" in step_types
        
        return has_thesis and has_antithesis and has_synthesis
    
    def _default_chain_validation(self, chain: ReasoningChain) -> bool:
        """Default validation for reasoning chains"""
        # Basic coherence check
        if not chain.steps:
            return False
        
        # Check connection between steps
        for i in range(len(chain.steps) - 1):
            current = chain.steps[i]
            next_step = chain.steps[i + 1]
            
            # Some connection should exist
            if (current.conclusion not in next_step.premise and
                not self._concepts_related(current.conclusion, next_step.premise)):
                return False
        
        return True
    
    def _check_inference_validity(self, inference_type: InferenceType,
                                premise: str, conclusion: str) -> bool:
        """Check validity of specific inference"""
        if inference_type in self.inference_rules:
            checker = self.inference_rules[inference_type]["validity_check"]
            return checker(premise, conclusion)
        
        return True  # Default to valid if unknown
    
    def _check_modus_ponens(self, premise: str, conclusion: str) -> bool:
        """Check modus ponens validity"""
        # Simplified - would need proper logical parsing
        return "if" in premise.lower() and "then" in premise.lower()
    
    def _check_modus_tollens(self, premise: str, conclusion: str) -> bool:
        """Check modus tollens validity"""
        return "not" in conclusion.lower()
    
    def _check_hypothetical_syllogism(self, premise: str, conclusion: str) -> bool:
        """Check hypothetical syllogism validity"""
        return "if" in conclusion.lower() and "then" in conclusion.lower()
    
    def _check_disjunctive_syllogism(self, premise: str, conclusion: str) -> bool:
        """Check disjunctive syllogism validity"""
        return "or" in premise.lower()
    
    def _check_reductio(self, premise: str, conclusion: str) -> bool:
        """Check reductio ad absurdum validity"""
        return "not" in conclusion.lower() or "false" in conclusion.lower()
    
    def _concepts_related(self, concept1: str, concept2: str) -> bool:
        """Check if two concepts are related"""
        # Simplified semantic check
        words1 = set(concept1.lower().split())
        words2 = set(concept2.lower().split())
        
        return len(words1 & words2) > 0
    
    # Analysis helper methods
    
    def _generate_trace_summary(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """Generate summary of reasoning trace"""
        chain_types = [chain.chain_type.value for chain in trace.chains]
        
        return {
            "total_steps": sum(len(chain.steps) for chain in trace.chains),
            "chain_count": len(trace.chains),
            "dominant_reasoning": max(set(chain_types), key=chain_types.count) if chain_types else None,
            "success_rate": 1.0 if trace.success else 0.0,
            "backtrack_rate": len(trace.backtracking_events) / max(len(trace.chains), 1),
            "decision_confidence": sum(d["confidence"] for d in trace.decision_points) / len(trace.decision_points) if trace.decision_points else 0
        }
    
    def _analyze_chains(self, chains: List[ReasoningChain]) -> Dict[str, Any]:
        """Analyze reasoning chains"""
        return {
            "validity_rate": sum(1 for c in chains if c.validity) / len(chains) if chains else 0,
            "soundness_rate": sum(1 for c in chains if c.soundness) / len(chains) if chains else 0,
            "average_length": sum(len(c.steps) for c in chains) / len(chains) if chains else 0,
            "type_distribution": self._get_type_distribution(chains)
        }
    
    def _analyze_decisions(self, decisions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Analyze decision points"""
        if not decisions:
            return {"count": 0}
        
        return {
            "count": len(decisions),
            "average_options": sum(len(d["options_considered"]) for d in decisions) / len(decisions),
            "average_confidence": sum(d["confidence"] for d in decisions) / len(decisions),
            "criteria_usage": self._analyze_criteria_usage(decisions)
        }
    
    def _calculate_efficiency_metrics(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """Calculate reasoning efficiency metrics"""
        total_steps = sum(len(chain.steps) for chain in trace.chains)
        
        return {
            "steps_per_second": total_steps / trace.total_duration if trace.total_duration > 0 else 0,
            "backtrack_ratio": len(trace.backtracking_events) / total_steps if total_steps > 0 else 0,
            "decision_efficiency": len(trace.decision_points) / total_steps if total_steps > 0 else 0,
            "path_directness": 1.0 - (len(trace.backtracking_events) / max(len(trace.chains), 1))
        }
    
    def _assess_logical_quality(self, trace: ReasoningTrace) -> Dict[str, Any]:
        """Assess logical quality of reasoning"""
        valid_chains = sum(1 for c in trace.chains if c.validity)
        sound_chains = sum(1 for c in trace.chains if c.soundness)
        
        # Check for logical errors
        errors = []
        for chain in trace.chains:
            chain_errors = self._find_logical_errors(chain)
            errors.extend(chain_errors)
        
        return {
            "validity_score": valid_chains / len(trace.chains) if trace.chains else 0,
            "soundness_score": sound_chains / len(trace.chains) if trace.chains else 0,
            "error_count": len(errors),
            "error_types": list(set(e["type"] for e in errors)),
            "overall_quality": self._calculate_overall_logical_quality(trace)
        }
    
    def _extract_insights(self, trace: ReasoningTrace) -> List[str]:
        """Extract insights from reasoning trace"""
        insights = []
        
        # Insight about reasoning efficiency
        if trace.total_duration < 1.0 and trace.success:
            insights.append("Highly efficient reasoning process")
        
        # Insight about backtracking
        if len(trace.backtracking_events) > len(trace.chains) / 2:
            insights.append("Significant exploration and revision of reasoning paths")
        
        # Insight about decision confidence
        if trace.decision_points:
            avg_confidence = sum(d["confidence"] for d in trace.decision_points) / len(trace.decision_points)
            if avg_confidence > 0.8:
                insights.append("High confidence in decision-making")
            elif avg_confidence < 0.5:
                insights.append("Low confidence suggests uncertainty in reasoning")
        
        # Insight about logical quality
        valid_rate = sum(1 for c in trace.chains if c.validity) / len(trace.chains) if trace.chains else 0
        if valid_rate == 1.0:
            insights.append("Perfect logical validity maintained throughout")
        
        return insights
    
    def _extract_path(self, trace_id: str, from_step: str, to_step: str) -> List[str]:
        """Extract path between steps"""
        # Simplified path extraction
        return [from_step, "...", to_step]
    
    def _get_type_distribution(self, chains: List[ReasoningChain]) -> Dict[str, int]:
        """Get distribution of reasoning types"""
        distribution = defaultdict(int)
        
        for chain in chains:
            distribution[chain.chain_type.value] += 1
        
        return dict(distribution)
    
    def _analyze_criteria_usage(self, decisions: List[Dict[str, Any]]) -> Dict[str, int]:
        """Analyze criteria used in decisions"""
        criteria_counts = defaultdict(int)
        
        for decision in decisions:
            for criterion in decision.get("criteria", []):
                criteria_counts[criterion] += 1
        
        return dict(criteria_counts)
    
    def _find_logical_errors(self, chain: ReasoningChain) -> List[Dict[str, Any]]:
        """Find logical errors in chain"""
        errors = []
        
        # Check for circular reasoning
        premises = [step.premise for step in chain.steps]
        conclusions = [step.conclusion for step in chain.steps]
        
        for i, conclusion in enumerate(conclusions):
            if conclusion in premises[:i]:
                errors.append({
                    "type": "circular_reasoning",
                    "step": i,
                    "description": f"Conclusion '{conclusion}' used as earlier premise"
                })
        
        # Check for non sequiturs
        for i, step in enumerate(chain.steps):
            if step.confidence < 0.3:
                errors.append({
                    "type": "weak_inference",
                    "step": i,
                    "description": f"Low confidence inference: {step.confidence}"
                })
        
        return errors
    
    def _calculate_overall_logical_quality(self, trace: ReasoningTrace) -> float:
        """Calculate overall logical quality score"""
        factors = []
        
        # Validity factor
        if trace.chains:
            validity_rate = sum(1 for c in trace.chains if c.validity) / len(trace.chains)
            factors.append(validity_rate)
        
        # Soundness factor
        if trace.chains:
            soundness_rate = sum(1 for c in trace.chains if c.soundness) / len(trace.chains)
            factors.append(soundness_rate * 0.8)  # Slightly less weight
        
        # Error factor (inverse)
        total_steps = sum(len(c.steps) for c in trace.chains)
        if total_steps > 0:
            error_rate = len(self._find_all_errors(trace)) / total_steps
            factors.append(1.0 - error_rate)
        
        # Success factor
        factors.append(1.0 if trace.success else 0.5)
        
        return sum(factors) / len(factors) if factors else 0
    
    def _find_all_errors(self, trace: ReasoningTrace) -> List[Dict[str, Any]]:
        """Find all errors in trace"""
        all_errors = []
        
        for chain in trace.chains:
            errors = self._find_logical_errors(chain)
            all_errors.extend(errors)
        
        return all_errors
    
    def _extract_inference_sequences(self, traces: List[ReasoningTrace]) -> List[List[InferenceType]]:
        """Extract sequences of inferences from traces"""
        sequences = []
        
        for trace in traces:
            for chain in trace.chains:
                sequence = [
                    step.inference 
                    for step in chain.steps 
                    if step.inference is not None
                ]
                if sequence:
                    sequences.append(sequence)
        
        return sequences
    
    def _find_common_sequences(self, sequences: List[List[InferenceType]]) -> List[Tuple[List[InferenceType], int]]:
        """Find common inference sequences"""
        sequence_counts = defaultdict(int)
        
        for sequence in sequences:
            # Convert to tuple for hashing
            seq_tuple = tuple(inf.value for inf in sequence)
            sequence_counts[seq_tuple] += 1
        
        # Sort by frequency
        common = sorted(
            sequence_counts.items(),
            key=lambda x: x[1],
            reverse=True
        )[:10]  # Top 10
        
        return [(list(seq), count) for seq, count in common]
    
    def _analyze_decision_patterns(self, traces: List[ReasoningTrace]) -> Dict[str, Any]:
        """Analyze patterns in decision-making"""
        all_decisions = []
        for trace in traces:
            all_decisions.extend(trace.decision_points)
        
        if not all_decisions:
            return {"pattern": "no_decisions"}
        
        # Analyze confidence distribution
        confidences = [d["confidence"] for d in all_decisions]
        
        # Analyze criteria usage
        all_criteria = []
        for decision in all_decisions:
            all_criteria.extend(decision.get("criteria", []))
        
        criteria_counts = defaultdict(int)
        for criterion in all_criteria:
            criteria_counts[criterion] += 1
        
        return {
            "average_confidence": sum(confidences) / len(confidences),
            "confidence_variance": self._calculate_variance(confidences),
            "top_criteria": sorted(
                criteria_counts.items(),
                key=lambda x: x[1],
                reverse=True
            )[:5],
            "decision_count": len(all_decisions)
        }
    
    def _interpret_type_distribution(self, type_counts: Dict[str, int]) -> str:
        """Interpret reasoning type distribution"""
        if not type_counts:
            return "No reasoning patterns found"
        
        total = sum(type_counts.values())
        dominant = max(type_counts.items(), key=lambda x: x[1])
        
        percentage = (dominant[1] / total) * 100
        
        if percentage > 70:
            return f"Heavily {dominant[0]} reasoning ({percentage:.1f}%)"
        elif percentage > 40:
            return f"Predominantly {dominant[0]} reasoning with mixed approaches"
        else:
            return "Balanced mix of reasoning approaches"
    
    def _interpret_decision_patterns(self, patterns: Dict[str, Any]) -> str:
        """Interpret decision-making patterns"""
        avg_confidence = patterns.get("average_confidence", 0)
        
        if avg_confidence > 0.8:
            return "High-confidence decision making"
        elif avg_confidence > 0.6:
            return "Moderate confidence with some uncertainty"
        else:
            return "Low confidence suggesting exploratory decision-making"
    
    def _calculate_variance(self, values: List[float]) -> float:
        """Calculate variance of values"""
        if not values:
            return 0
        
        mean = sum(values) / len(values)
        return sum((v - mean) ** 2 for v in values) / len(values)
