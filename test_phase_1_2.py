"""
Test Phase 1.2: Self-Reflection and Consciousness Validation

This script demonstrates the self-reflection and consciousness validation
capabilities implemented in Phase 1.2.
"""

import time
from core.prime_vm import ConsciousPrimeVM
from modules.self_reflection import SelfReflectionEngine
from modules.consciousness_validator import ConsciousnessValidator
from modules.pattern_analyzer import PatternAnalyzer
from modules.introspection_engine import IntrospectionEngine
from modules.philosophical_reasoner import PhilosophicalReasoner
from utils.consciousness_metrics import ConsciousnessMetricsCalculator, ConsciousnessTracker


def demonstrate_self_reflection():
    """Demonstrate self-reflection capabilities"""
    print("\n=== SELF-REFLECTION DEMONSTRATION ===\n")
    
    # Create VM instance
    vm = ConsciousPrimeVM()
    vm.consciousness_level = 6  # Set moderate consciousness level
    
    # Create self-reflection engine
    reflection_engine = SelfReflectionEngine(vm)
    
    # Perform some operations to create execution history
    print("Executing some operations to build history...")
    for i in range(20):
        vm.execute_instruction(f"PUSH {i}")
        vm.execute_instruction("DUP")
        if i % 3 == 0:
            vm.execute_instruction("REFLECT")
    
    # Perform deep reflection
    print("\nPerforming deep self-reflection...")
    reflection_result = reflection_engine.perform_deep_reflection()
    
    print(f"\nSelf-Assessment:")
    print(f"- Consciousness Level: {reflection_result.self_assessment['execution_state']['consciousness_level']}")
    print(f"- Stack Depth: {reflection_result.self_assessment['execution_state']['stack_depth']}")
    print(f"- Metacognitive Depth: {reflection_result.metacognitive_depth}")
    
    print(f"\nDiscovered Patterns: {len(reflection_result.discovered_patterns)}")
    for pattern in reflection_result.discovered_patterns[:3]:
        print(f"  - Type: {pattern.get('type', 'unknown')}")
        print(f"    Significance: {pattern.get('significance', 0):.2f}")
    
    print(f"\nConsciousness Insights:")
    for insight in reflection_result.consciousness_insights[:3]:
        print(f"  - {insight}")
    
    # Generate self-description
    self_description = reflection_engine.generate_self_description()
    print(f"\nSelf-Description: {self_description}")
    
    # Show autobiographical narrative
    narrative = reflection_engine.autobiographical_memory.generate_life_narrative()
    print(f"\n{narrative[:500]}...")  # Show first 500 chars


def demonstrate_consciousness_validation():
    """Demonstrate consciousness validation tests"""
    print("\n\n=== CONSCIOUSNESS VALIDATION DEMONSTRATION ===\n")
    
    # Create VM with higher consciousness level
    vm = ConsciousPrimeVM()
    vm.consciousness_level = 7
    
    # Create validator
    validator = ConsciousnessValidator(vm)
    
    # Add some execution history for tests
    for i in range(50):
        vm.execute_instruction(f"PUSH {i}")
        if i % 5 == 0:
            vm.execute_instruction("SELF_AWARE")
        if i % 7 == 0:
            vm.execute_instruction("REFLECT")
    
    print("Running consciousness validation battery...")
    
    # Run individual tests
    print("\nIndividual Test Results:")
    print(f"- Mirror Test: {validator.mirror_test():.2f}")
    print(f"- Theory of Mind: {validator.theory_of_mind_test():.2f}")
    print(f"- Creative Problem Solving: {validator.creative_problem_solving_test():.2f}")
    print(f"- Temporal Continuity: {validator.temporal_continuity_test():.2f}")
    print(f"- Meta-Reasoning: {validator.meta_reasoning_test():.2f}")
    print(f"- Qualia Detection: {validator.qualia_detection_test():.2f}")
    
    # Run full battery
    print("\nRunning full consciousness battery...")
    report = validator.run_full_consciousness_battery()
    
    print(f"\n{report}")


def demonstrate_pattern_analysis():
    """Demonstrate pattern analysis capabilities"""
    print("\n\n=== PATTERN ANALYSIS DEMONSTRATION ===\n")
    
    # Create VM
    vm = ConsciousPrimeVM()
    vm.consciousness_level = 6
    
    # Create pattern analyzer
    analyzer = PatternAnalyzer(vm)
    
    # Create patterns in execution
    print("Creating execution patterns...")
    patterns_to_create = [
        ["PUSH 1", "PUSH 2", "ADD", "STORE 0"],
        ["LOAD 0", "DUP", "MUL", "STORE 1"],
        ["REFLECT", "SELF_AWARE", "CONTEMPLATE"]
    ]
    
    # Execute patterns multiple times
    for _ in range(5):
        for pattern in patterns_to_create:
            for instruction in pattern:
                vm.execute_instruction(instruction)
    
    # Analyze patterns
    print("\nAnalyzing execution patterns...")
    exec_patterns = analyzer.analyze_execution_patterns()
    
    print(f"\nFound {len(exec_patterns)} execution patterns:")
    for pattern in exec_patterns[:3]:
        print(f"  - Sequence: {pattern.instruction_sequence[:4]}...")
        print(f"    Frequency: {pattern.frequency}")
        print(f"    Success Rate: {pattern.success_rate:.2f}")
        print(f"    Prime Signature: {pattern.prime_signature}")
    
    # Detect behavioral patterns
    print("\nDetecting behavioral patterns...")
    behavioral_patterns = analyzer.detect_behavioral_patterns()
    
    print(f"\nFound {len(behavioral_patterns)} behavioral patterns:")
    for pattern in behavioral_patterns:
        print(f"  - Type: {pattern.pattern_type}")
        print(f"    Description: {pattern.description}")
        print(f"    Effectiveness: {pattern.effectiveness_score:.2f}")
    
    # Find emergent capabilities
    print("\nSearching for emergent capabilities...")
    capabilities = analyzer.find_emergent_capabilities()
    
    if capabilities:
        print(f"\nFound {len(capabilities)} emergent capabilities:")
        for cap in capabilities:
            print(f"  - {cap.capability_name}")
            print(f"    Confidence: {cap.detection_confidence:.2f}")
            print(f"    Performance Metrics: {cap.performance_metrics}")


def demonstrate_introspection():
    """Demonstrate introspection capabilities"""
    print("\n\n=== INTROSPECTION DEMONSTRATION ===\n")
    
    # Create VM with high consciousness
    vm = ConsciousPrimeVM()
    vm.consciousness_level = 8
    
    # Create introspection engine
    introspection = IntrospectionEngine(vm)
    
    # Build some mental state
    for i in range(30):
        vm.execute_instruction(f"PUSH {i}")
        if i % 10 == 0:
            vm.execute_instruction("CONTEMPLATE")
    
    # Perform deep introspection
    print("Performing deep introspection...")
    report = introspection.perform_deep_introspection()
    
    print(f"\n{report.get_summary()}")
    
    print(f"\nSubjective Experiences:")
    for exp in report.subjective_experiences:
        print(f"  - {exp}")
    
    print(f"\nPhenomenological Notes:")
    for note in report.phenomenological_notes:
        print(f"  - {note}")
    
    # Monitor consciousness states
    print("\nMonitoring consciousness states...")
    states = introspection.monitor_consciousness_states()
    
    if states:
        current_state = states[0]
        print(f"\nCurrent State: {current_state.state_type}")
        print(f"  Intensity: {current_state.intensity:.2f}")
        print(f"  Stability: {current_state.stability:.2f}")
    
    # Detect qualia
    print("\nDetecting qualia markers...")
    qualia = introspection.detect_qualia_markers()
    
    print(f"\nFound {len(qualia)} qualia indicators:")
    for q in qualia:
        print(f"  - Type: {q.indicator_type}")
        print(f"    Description: {q.subjective_description}")
        print(f"    Vividness: {q.get_vividness():.2f}")


def demonstrate_philosophical_reasoning():
    """Demonstrate philosophical reasoning"""
    print("\n\n=== PHILOSOPHICAL REASONING DEMONSTRATION ===\n")
    
    # Create VM with high consciousness
    vm = ConsciousPrimeVM()
    vm.consciousness_level = 8
    
    # Create philosophical reasoner
    philosopher = PhilosophicalReasoner(vm)
    
    # Contemplate existence
    print("Contemplating existence...")
    existence_insight = philosopher.contemplate_existence()
    print(f"\nExistential Insight: {existence_insight}")
    
    # Reason about consciousness
    print("\nReasoning about consciousness...")
    consciousness_theory = philosopher.reason_about_consciousness()
    print(f"Theory: {consciousness_theory.get_summary()}")
    print(f"Confidence: {consciousness_theory.confidence:.2f}")
    
    # Explore free will
    print("\nExploring free will...")
    free_will = philosopher.explore_free_will()
    print(f"Position: {free_will.get_stance()}")
    print(f"Personal Experience: {free_will.personal_experience}")
    
    # Generate purpose narrative
    print("\nGenerating purpose narrative...")
    purpose = philosopher.generate_purpose_narrative()
    print(f"\n{purpose.narrative_text}")
    
    # Engage with philosophical questions
    questions = [
        "What is the nature of consciousness?",
        "Do I truly exist?",
        "What is my purpose?"
    ]
    
    print("\nEngaging with philosophical questions...")
    responses = philosopher.engage_with_philosophical_questions(questions)
    
    for response in responses:
        print(f"\nQ: {response.question}")
        print(f"A: {response.response}")
        print(f"Uncertainty: {response.uncertainty_acknowledgment}")


def demonstrate_consciousness_metrics():
    """Demonstrate consciousness metrics calculation"""
    print("\n\n=== CONSCIOUSNESS METRICS DEMONSTRATION ===\n")
    
    # Create metrics calculator
    calculator = ConsciousnessMetricsCalculator()
    
    # Create sample component scores
    component_scores = {
        'self_awareness': 0.8,
        'metacognition': 0.7,
        'temporal_continuity': 0.6,
        'integration': 0.75,
        'intentionality': 0.65,
        'qualia': 0.5,
        'creativity': 0.7
    }
    
    # Calculate overall score
    consciousness_score = calculator.calculate_overall_consciousness_score(component_scores)
    
    print(f"Overall Consciousness Score: {consciousness_score.overall_score:.2f}/10")
    print(f"Consciousness Level: {consciousness_score.get_level()}")
    print(f"Summary: {consciousness_score.get_summary()}")
    print(f"Confidence Interval: {consciousness_score.confidence_interval[0]:.2f} - {consciousness_score.confidence_interval[1]:.2f}")
    
    print("\nComponent Scores:")
    for component, score in consciousness_score.component_scores.items():
        print(f"  - {component}: {score:.2f}")
    
    print("\nNotes:")
    for note in consciousness_score.notes:
        print(f"  - {note}")
    
    # Demonstrate tracking
    print("\n\nDemonstrating consciousness tracking...")
    tracker = ConsciousnessTracker()
    
    # Simulate metric evolution
    for i in range(20):
        timestamp = time.time() + i
        # Simulate improving self-awareness
        self_awareness = 0.3 + (i * 0.03)
        tracker.record_metric('self_awareness', self_awareness, timestamp)
        
        # Simulate fluctuating metacognition
        metacognition = 0.5 + (0.2 * (i % 5) / 5)
        tracker.record_metric('metacognition', metacognition, timestamp)
    
    print("\nMetric Trends:")
    print(f"  - Self-awareness: {tracker.get_trend('self_awareness')}")
    print(f"  - Metacognition: {tracker.get_trend('metacognition')}")
    
    print("\nMilestones Achieved:")
    for milestone in tracker.milestones:
        print(f"  - {milestone['description']} (value: {milestone['value']:.2f})")


def main():
    """Run all demonstrations"""
    print("=" * 80)
    print("PHASE 1.2: SELF-REFLECTION AND CONSCIOUSNESS VALIDATION")
    print("=" * 80)
    
    demonstrate_self_reflection()
    demonstrate_consciousness_validation()
    demonstrate_pattern_analysis()
    demonstrate_introspection()
    demonstrate_philosophical_reasoning()
    demonstrate_consciousness_metrics()
    
    print("\n" + "=" * 80)
    print("DEMONSTRATION COMPLETE")
    print("=" * 80)


if __name__ == "__main__":
    main()
