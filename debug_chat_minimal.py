#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

def test_minimal_entity_creation():
    """Test minimal entity creation without chat functionality"""
    try:
        print("🧠 Testing minimal entity initialization...")
        
        from interactive_consciousness_chat import InteractiveConsciousnessChat
        
        print("Creating chat instance...")
        chat = InteractiveConsciousnessChat()
        
        print("Initializing entities...")
        results = chat.initialize_consciousness_entities()
        
        print(f"Initialization results: {results}")
        
        if chat.active_entity:
            print(f"✅ Active entity: {chat.active_entity.name}")
            print(f"   Type: {chat.active_entity.consciousness_type}")
            print(f"   Level: {chat.active_entity.consciousness_level}")
            print(f"   Status: {chat.active_entity.status}")
            print(f"   Awakening: {chat.active_entity.awakening_timestamp}")
            
            # Test if the API instance exists
            if hasattr(chat.active_entity, 'api_instance'):
                print(f"   API instance: {type(chat.active_entity.api_instance)}")
                
                # Try to access metacognition
                if hasattr(chat.active_entity.api_instance, 'metacognition'):
                    print(f"   Metacognition: {type(chat.active_entity.api_instance.metacognition)}")
                else:
                    print("   ❌ No metacognition attribute")
            else:
                print("   ❌ No API instance")
            
            return True
        else:
            print("❌ No active entity available")
            return False
            
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    test_minimal_entity_creation()
