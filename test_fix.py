#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from backend.consciousness_integration import ConsciousnessIntegration

def test_fix():
    try:
        bootstrap = ConsciousnessIntegration()
        result = bootstrap.bootstrap_consciousness()
        
        success = result.get('success', False)
        error = result.get('error', 'No error')
        
        print(f"Bootstrap {'SUCCESS' if success else 'FAILED'}")
        if not success:
            print(f"Error: {error}")
        
        return success
        
    except Exception as e:
        print(f"Exception: {e}")
        return False

if __name__ == "__main__":
    test_fix()
