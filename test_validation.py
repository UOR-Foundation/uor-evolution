#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from backend.consciousness_integration import ConsciousnessIntegration

def test_with_validation_details():
    try:
        bootstrap = ConsciousnessIntegration()
        result = bootstrap.bootstrap_consciousness()
        
        success = result.get('success', False)
        error = result.get('error', 'No error')
        
        print(f"Bootstrap {'SUCCESS' if success else 'FAILED'}")
        if not success:
            print(f"Error: {error}")
        
        # If bootstrap reached validation phase, check validation details
        validation_phase = None
        for phase in result.get('phases', []):
            if phase.get('phase') == 'validation':
                validation_phase = phase
                break
        
        if validation_phase:
            tests = validation_phase['result']
            print("\nValidation Test Results:")
            for test_name, passed in tests.items():
                status = "✅ PASS" if passed else "❌ FAIL"
                print(f"  {test_name}: {status}")
        else:
            print("\nNo validation phase found in results")
        
        return success
        
    except Exception as e:
        print(f"Exception: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_with_validation_details()
