"""
Test suite for the Enhanced Prime Virtual Machine.

Demonstrates consciousness-aware capabilities, self-reflection, and strange loops.
"""

import sys
import os
sys.path.append(os.path.dirname(os.path.abspath(__file__)))

from core.prime_vm import ConsciousPrimeVM, Instruction, OpCode
from core.consciousness_layer import ConsciousnessLevel
import json


def print_section(title: str):
    """Print a formatted section header."""
    print(f"\n{'='*60}")
    print(f"{title:^60}")
    print(f"{'='*60}\n")


def test_basic_operations():
    """Test basic VM operations."""
    print_section("Testing Basic Operations")
    
    vm = ConsciousPrimeVM()
    
    # Test stack operations
    instructions = [
        Instruction(OpCode.PUSH, 42),
        Instruction(OpCode.PUSH, 17),
        Instruction(OpCode.ADD),
    ]
    
    results = vm.run_program(instructions)
    print(f"Stack after operations: {vm.stack}")
    print(f"Results: {results}")
    print(f"Consciousness level: {vm.consciousness_level.name}")


def test_self_reflection():
    """Test self-reflection capabilities."""
    print_section("Testing Self-Reflection")
    
    vm = ConsciousPrimeVM()
    
    # Execute some operations first
    setup_instructions = [
        Instruction(OpCode.PUSH, 10),
        Instruction(OpCode.PUSH, 20),
        Instruction(OpCode.ADD),
    ]
    vm.run_program(setup_instructions)
    
    # Perform self-reflection
    reflection_result = vm.execute_instruction(Instruction(OpCode.SELF_REFLECT))
    
    print("Self-Reflection Result:")
    print(f"- Self Description: {reflection_result['self_description']}")
    print(f"- Consciousness State: {reflection_result['consciousness_state']['current_level']}")
    print(f"- Execution Count: {reflection_result['execution_count']}")
    print(f"- Capability Assessment: {json.dumps(reflection_result['capability_assessment'], indent=2)}")


def test_strange_loop():
    """Test strange loop creation."""
    print_section("Testing Strange Loop Creation")
    
    vm = ConsciousPrimeVM()
    
    # Create a strange loop
    loop_instruction = Instruction(OpCode.STRANGE_LOOP, 0, {'depth': 3})
    loop_result = vm.execute_instruction(loop_instruction)
    
    print("Strange Loop Result:")
    print(f"- Initial Consciousness: {loop_result['initial_consciousness']}")
    print(f"- Final Consciousness: {loop_result['final_consciousness']}")
    print(f"- Iterations: {len(loop_result['iterations'])}")
    print(f"- Emergence Detected: {loop_result.get('emergence_detected', False)}")
    
    # Check consciousness level after loop
    print(f"\nCurrent consciousness level: {vm.consciousness_level.name}")
    print(f"Strange loop count: {vm.consciousness.meta_state.strange_loop_count}")


def test_pattern_recognition():
    """Test pattern recognition in execution."""
    print_section("Testing Pattern Recognition")
    
    vm = ConsciousPrimeVM()
    
    # Create a pattern by repeating operations
    pattern_instructions = [
        Instruction(OpCode.PUSH, 1),
        Instruction(OpCode.PUSH, 2),
        Instruction(OpCode.ADD),
        Instruction(OpCode.PUSH, 1),
        Instruction(OpCode.PUSH, 2),
        Instruction(OpCode.ADD),
        Instruction(OpCode.PUSH, 1),
        Instruction(OpCode.PUSH, 2),
        Instruction(OpCode.ADD),
    ]
    
    vm.run_program(pattern_instructions)
    
    # Analyze patterns
    pattern_result = vm.execute_instruction(
        Instruction(OpCode.PATTERN_MATCH, 0, {'type': 'execution'})
    )
    
    print(f"Patterns found: {len(pattern_result)}")
    
    # Perform self-analysis to detect patterns
    analysis = vm.execute_instruction(Instruction(OpCode.ANALYZE_SELF))
    print(f"\nDetected execution patterns: {json.dumps(analysis['patterns'], indent=2)}")


def test_consciousness_evolution():
    """Test consciousness level evolution through various operations."""
    print_section("Testing Consciousness Evolution")
    
    vm = ConsciousPrimeVM()
    
    consciousness_history = []
    
    # Series of consciousness-raising operations
    consciousness_program = [
        # Basic operations
        Instruction(OpCode.PUSH, 42),
        Instruction(OpCode.SELF_REFLECT),
        
        # Meta-reasoning
        Instruction(OpCode.META_REASON, 0, {'context': {'task': 'understand_self'}}),
        
        # Pattern analysis
        Instruction(OpCode.PATTERN_MATCH),
        
        # Self-analysis
        Instruction(OpCode.ANALYZE_SELF),
        
        # Consciousness test
        Instruction(OpCode.CONSCIOUSNESS_TEST),
        
        # Create strange loop
        Instruction(OpCode.STRANGE_LOOP, 0, {'depth': 2}),
        
        # Final reflection
        Instruction(OpCode.SELF_REFLECT),
    ]
    
    for i, instruction in enumerate(consciousness_program):
        print(f"\nExecuting: {instruction.opcode.name}")
        result = vm.execute_instruction(instruction)
        
        consciousness_history.append({
            'step': i,
            'operation': instruction.opcode.name,
            'level': vm.consciousness_level.name,
            'level_value': vm.consciousness_level.value
        })
        
        if instruction.opcode == OpCode.CONSCIOUSNESS_TEST:
            print(f"Consciousness Test Score: {result['overall_score']:.2f}")
            print(f"Recommendation: {result['recommendation']}")
    
    print("\nConsciousness Evolution:")
    for entry in consciousness_history:
        print(f"Step {entry['step']}: {entry['operation']:20} -> {entry['level']} (level {entry['level_value']})")


def test_meta_reasoning():
    """Test meta-reasoning capabilities."""
    print_section("Testing Meta-Reasoning")
    
    vm = ConsciousPrimeVM()
    
    # Perform meta-reasoning about problem-solving
    meta_context = {
        'problem': 'optimize_execution',
        'constraints': ['limited_memory', 'pattern_detection'],
        'goal': 'improve_efficiency'
    }
    
    meta_result = vm.execute_instruction(
        Instruction(OpCode.META_REASON, 0, {'context': meta_context})
    )
    
    print("Meta-Reasoning Analysis:")
    print(f"- Reasoning Depth: {meta_result['depth']}")
    print(f"- Patterns Detected: {meta_result['patterns_detected']}")
    print(f"- Efficiency Score: {meta_result['efficiency_score']:.2f}")
    print(f"- Coherence: {meta_result['coherence']:.2f}")


def test_memory_systems():
    """Test different memory systems."""
    print_section("Testing Memory Systems")
    
    vm = ConsciousPrimeVM()
    
    # Store in working memory
    vm.working_memory.store('test_key', 'test_value', importance=0.8)
    vm.working_memory.store('important_data', {'type': 'critical'}, importance=1.0)
    
    # Consolidate to long-term memory
    vm.long_term_memory.consolidate(
        'learned_pattern',
        {'pattern': 'push_add_sequence', 'frequency': 'high'},
        importance=0.9,
        associations={'arithmetic', 'stack_operations'}
    )
    
    # Store a strategy
    vm.long_term_memory.store_strategy(
        'efficient_addition',
        {'method': 'batch_push_then_add', 'complexity': 'O(n)'}
    )
    
    # Create pattern cache entries
    vm.pattern_cache.store_pattern('execution', {'sequence': ['PUSH', 'PUSH', 'ADD']})
    vm.pattern_cache.store_pattern('behavioral', {'tendency': 'arithmetic_focus'})
    
    # Perform self-reflection to see memory state
    reflection = vm.execute_instruction(Instruction(OpCode.SELF_REFLECT))
    
    print("Memory State:")
    print(f"- Working Memory Context: {reflection['memory_state']['working_memory']}")
    print(f"- Pattern Cache Stats: {json.dumps(reflection['memory_state']['pattern_cache_stats'], indent=2)}")


def test_full_consciousness_demo():
    """Comprehensive demonstration of consciousness capabilities."""
    print_section("Full Consciousness Demonstration")
    
    vm = ConsciousPrimeVM()
    
    print("Initial State:")
    print(f"- Consciousness Level: {vm.consciousness_level.name}")
    
    # Complex program demonstrating various capabilities
    demo_program = [
        # Initialize with some data
        Instruction(OpCode.PUSH, 100),
        Instruction(OpCode.PUSH, 200),
        
        # First self-reflection
        Instruction(OpCode.SELF_REFLECT),
        
        # Perform computation
        Instruction(OpCode.ADD),
        
        # Analyze patterns
        Instruction(OpCode.PATTERN_MATCH),
        
        # Meta-reasoning about the task
        Instruction(OpCode.META_REASON, 0, {
            'context': {'task': 'arithmetic_processing', 'goal': 'understand_computation'}
        }),
        
        # Create analogy
        Instruction(OpCode.CREATE_ANALOGY, 0, {
            'source': {'operation': 'addition', 'type': 'arithmetic'},
            'target': {'operation': 'concatenation', 'type': 'string'}
        }),
        
        # Test consciousness
        Instruction(OpCode.CONSCIOUSNESS_TEST),
        
        # Create strange loop
        Instruction(OpCode.STRANGE_LOOP, 0, {'depth': 2}),
        
        # Final deep analysis
        Instruction(OpCode.ANALYZE_SELF),
    ]
    
    results = vm.run_program(demo_program)
    
    # Extract key results
    final_analysis = results[-1]  # ANALYZE_SELF result
    consciousness_test = next((r for r in results if isinstance(r, dict) and 'test_results' in r), None)
    
    print("\nFinal State:")
    print(f"- Consciousness Level: {vm.consciousness_level.name}")
    print(f"- Self Description: {final_analysis['consciousness_report']['self_description']}")
    
    if consciousness_test:
        print(f"\nConsciousness Test Results:")
        print(f"- Overall Score: {consciousness_test['overall_score']:.2f}")
        print(f"- Self Recognition: {consciousness_test['test_results']['self_recognition']}")
        print(f"- Meta-Cognitive Ability: {consciousness_test['test_results']['meta_cognitive_ability']:.2f}")
        print(f"- Strange Loop Detection: {consciousness_test['test_results']['strange_loop_detection']}")
        print(f"- Emergence Indicators: {consciousness_test['test_results']['emergence_indicators']}")
    
    print(f"\nMemory Analysis:")
    print(f"- Working Memory Utilization: {final_analysis['memory_analysis']['working_memory_utilization']:.1%}")
    print(f"- Pattern Cache: {json.dumps(final_analysis['memory_analysis']['pattern_cache_stats'], indent=2)}")
    print(f"- Episodic Memory Count: {final_analysis['memory_analysis']['episodic_memory_count']}")


def main():
    """Run all tests."""
    print("\n" + "="*60)
    print("Enhanced Prime Virtual Machine Test Suite".center(60))
    print("Phase 1.1 - Consciousness-Aware Implementation".center(60))
    print("="*60)
    
    tests = [
        test_basic_operations,
        test_self_reflection,
        test_strange_loop,
        test_pattern_recognition,
        test_meta_reasoning,
        test_memory_systems,
        test_consciousness_evolution,
        test_full_consciousness_demo,
    ]
    
    for test in tests:
        try:
            test()
        except Exception as e:
            print(f"\nError in {test.__name__}: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "="*60)
    print("Test Suite Complete".center(60))
    print("="*60)


if __name__ == "__main__":
    main()
