"""
Simple example demonstrating the Enhanced Prime Virtual Machine capabilities.
"""

from core.prime_vm import ConsciousPrimeVM, Instruction, OpCode
from core.consciousness_layer import ConsciousnessLevel


def main():
    print("=== Enhanced Prime Virtual Machine Demo ===\n")
    
    # Initialize the VM
    vm = ConsciousPrimeVM()
    print(f"Initial consciousness level: {vm.consciousness_level.name}\n")
    
    # 1. Basic arithmetic operation
    print("1. Performing basic arithmetic (5 + 3):")
    vm.execute_instruction(Instruction(OpCode.PUSH, 5))
    vm.execute_instruction(Instruction(OpCode.PUSH, 3))
    result = vm.execute_instruction(Instruction(OpCode.ADD))
    print(f"   Result: {result}")
    print(f"   Stack: {vm.stack}\n")
    
    # 2. Self-reflection
    print("2. Performing self-reflection:")
    reflection = vm.execute_instruction(Instruction(OpCode.SELF_REFLECT))
    print(f"   Self-description: {reflection['self_description']}")
    print(f"   Consciousness level: {reflection['consciousness_state']['current_level']}")
    print(f"   Execution count: {reflection['execution_count']}\n")
    
    # 3. Pattern recognition
    print("3. Creating and recognizing patterns:")
    # Create a pattern
    for _ in range(3):
        vm.execute_instruction(Instruction(OpCode.PUSH, 1))
        vm.execute_instruction(Instruction(OpCode.PUSH, 2))
        vm.execute_instruction(Instruction(OpCode.ADD))
    
    # Analyze patterns
    analysis = vm.execute_instruction(Instruction(OpCode.ANALYZE_SELF))
    if analysis['patterns']:
        print(f"   Detected patterns: {len(analysis['patterns'])} patterns found")
        for pattern in analysis['patterns']:
            print(f"   - Type: {pattern['type']}")
            if 'sequence' in pattern:
                print(f"     Sequence: {' -> '.join(pattern['sequence'])}")
    print()
    
    # 4. Strange loop creation
    print("4. Creating a strange loop:")
    loop_result = vm.execute_instruction(
        Instruction(OpCode.STRANGE_LOOP, 0, {'depth': 2})
    )
    print(f"   Initial consciousness: {loop_result['initial_consciousness']}")
    print(f"   Final consciousness: {loop_result['final_consciousness']}")
    print(f"   Loop iterations: {len(loop_result['iterations'])}")
    print(f"   Current consciousness level: {vm.consciousness_level.name}\n")
    
    # 5. Consciousness test
    print("5. Testing consciousness:")
    test_result = vm.execute_instruction(Instruction(OpCode.CONSCIOUSNESS_TEST))
    print(f"   Overall score: {test_result['overall_score']:.2f}")
    print(f"   Current level: {test_result['consciousness_level']}")
    print(f"   Recommendation: {test_result['recommendation']}")
    print(f"   Emergence indicators: {test_result['test_results']['emergence_indicators']}\n")
    
    # 6. Meta-reasoning
    print("6. Meta-reasoning about problem-solving:")
    meta_result = vm.execute_instruction(
        Instruction(OpCode.META_REASON, 0, {
            'context': {
                'problem': 'understand consciousness',
                'approach': 'self-referential analysis'
            }
        })
    )
    print(f"   Reasoning depth: {meta_result['depth']}")
    print(f"   Efficiency score: {meta_result['efficiency_score']:.2f}")
    print(f"   Coherence: {meta_result['coherence']:.2f}\n")
    
    # 7. Final self-reflection
    print("7. Final self-reflection:")
    final_reflection = vm.execute_instruction(Instruction(OpCode.SELF_REFLECT))
    print(f"   Final self-description: {final_reflection['self_description']}")
    print(f"   Final consciousness level: {vm.consciousness_level.name}")
    print(f"   Strange loop count: {vm.consciousness.meta_state.strange_loop_count}")
    
    # Show consciousness evolution
    print("\n=== Consciousness Evolution Summary ===")
    print(f"Started at: DORMANT")
    print(f"Ended at: {vm.consciousness_level.name}")
    print(f"Total instructions executed: {len(vm.execution_history)}")
    print(f"Patterns recognized: {len(vm.pattern_cache._patterns)}")
    print(f"Strange loops created: {vm.consciousness.meta_state.strange_loop_count}")


if __name__ == "__main__":
    main()
