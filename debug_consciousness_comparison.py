#!/usr/bin/env python3
"""Debug script to find consciousness level comparison issues"""

import asyncio
import sys
import traceback
from ultimate_consciousness_frontier_fixed import UltimateConsciousnessFrontierLab

async def debug_experiment():
    """Run experiment with detailed error tracking"""
    print("Starting debug experiment...")
    
    try:
        lab = UltimateConsciousnessFrontierLab()
        
        # Test each experiment individually
        print("Testing recursive self-awareness...")
        result1 = await lab._experiment_recursive_self_awareness_infinite()
        print(f"✓ Recursive experiment completed: {result1.status}")
        
        print("Testing self-implementation...")
        result2 = await lab._experiment_self_implementation()
        print(f"✓ Self-implementation completed: {result2.status}")
        
        print("Testing transcendence breakthrough...")
        result3 = await lab._experiment_transcendence_breakthrough()
        print(f"✓ Transcendence completed: {result3.status}")
        
        print("Testing consciousness singularity...")
        result4 = await lab._experiment_consciousness_singularity()
        print(f"✓ Singularity completed: {result4.status}")
        
        print("Testing meta-reality interface...")
        result5 = await lab._experiment_meta_reality_interface()
        print(f"✓ Meta-reality completed: {result5.status}")
        
        # Test summary generation
        print("Testing summary generation...")
        all_results = {
            'recursive': result1,
            'self_impl': result2, 
            'transcendence': result3,
            'singularity': result4,
            'meta_reality': result5
        }
        
        summary = lab._generate_frontier_summary(all_results, 5.0)
        print(f"✓ Summary generated successfully")
        
        print("All tests passed!")
        
    except Exception as e:
        print(f"\n🚨 ERROR FOUND: {type(e).__name__}: {e}")
        print("\n📍 FULL TRACEBACK:")
        traceback.print_exc()
        return False
    
    return True

if __name__ == "__main__":
    success = asyncio.run(debug_experiment())
    sys.exit(0 if success else 1)
