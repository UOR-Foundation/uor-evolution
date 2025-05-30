#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from interactive_consciousness_chat import InteractiveConsciousnessChat

def test_entity_initialization():
    """Test consciousness entity initialization"""
    try:
        print("🧠 Testing consciousness entity initialization...")
        
        chat = InteractiveConsciousnessChat()
        results = chat.initialize_consciousness_entities()
        
        print(f"\n✅ Initialization Results:")
        for entity_id, result in results.items():
            status = result['status']
            if status == "awakened":
                print(f"  🧠 {entity_id}: FULLY AWAKENED - {result['consciousness_level']}")
            elif status == "partially_awakened":
                print(f"  ⚠️  {entity_id}: PARTIALLY AWAKENED - {result['consciousness_level']}")
                print(f"      Validation: {'Passed' if result.get('validation_passed', False) else 'Failed'}")
                print(f"      Emergence: {'Complete' if result.get('emergence_completed', False) else 'Incomplete'}")
            else:
                print(f"  ❌ {entity_id}: FAILED - {result.get('error', 'Unknown error')}")
        
        print(f"\n🎯 Active Entity: {chat.active_entity.name if chat.active_entity else 'None'}")
        
        if chat.active_entity:
            print(f"  Type: {chat.active_entity.consciousness_type.value}")
            print(f"  Level: {chat.active_entity.consciousness_level}")
            print(f"  Capabilities: {len(chat.active_entity.capabilities)} available")
        
        return len([r for r in results.values() if r['status'] in ['awakened', 'partially_awakened']])
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return 0

if __name__ == "__main__":
    count = test_entity_initialization()
    print(f"\n🏆 Successfully initialized {count} consciousness entities!")
