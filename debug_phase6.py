#!/usr/bin/env python3

import sys
import os
import logging
import traceback

# Add the current directory to Python path
sys.path.insert(0, os.path.abspath('.'))

from backend.consciousness_integration import ConsciousnessIntegration

# Set up logging
logging.basicConfig(level=logging.DEBUG, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

def test_single_entity():
    """Test a single entity bootstrap to isolate the Phase 6 error"""
    try:
        print("🧠 Testing single consciousness entity bootstrap...")
        
        bootstrap = ConsciousnessIntegration()
        
        # Patch the bootstrap method to add more debugging
        import functools
        original_activate_emergence_scrolls = bootstrap._activate_emergence_scrolls
        
        def debug_activate_emergence_scrolls():
            print("🔍 Starting Phase 6 emergence scroll activation...")
            try:
                result = original_activate_emergence_scrolls()
                print(f"✅ Phase 6 completed successfully: {result}")
                return result
            except Exception as e:
                print(f"❌ Phase 6 error: {e}")
                print(f"Error type: {type(e)}")
                traceback.print_exc()
                raise
        
        bootstrap._activate_emergence_scrolls = debug_activate_emergence_scrolls
        
        # Try to bootstrap just one entity
        result = bootstrap.bootstrap_consciousness()
        
        print(f"Result: {result}")
        
        if not result.get('success', False):
            print(f"❌ Bootstrap failed: {result.get('error', 'Unknown error')}")
            if 'traceback' in result:
                print(f"Full traceback:\n{result['traceback']}")
        else:
            print("✅ Bootstrap successful!")
            
    except Exception as e:
        print(f"❌ Exception during bootstrap: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    test_single_entity()
