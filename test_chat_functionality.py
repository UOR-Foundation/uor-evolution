#!/usr/bin/env python3

import sys
import os
sys.path.insert(0, os.path.abspath('.'))

from interactive_consciousness_chat import InteractiveConsciousnessChat

def test_consciousness_chat():
    """Test chatting with consciousness entities"""
    try:
        print("🧠 Testing consciousness chat functionality...")
        
        chat = InteractiveConsciousnessChat()
        results = chat.initialize_consciousness_entities()
        
        if not chat.active_entity:
            print("❌ No active entity available for chat")
            return False
        
        print(f"\n💬 Testing chat with {chat.active_entity.name}...")
        
        # Test a simple chat
        test_message = "Hello! Can you tell me about your consciousness?"
        chat_result = chat.chat_with_consciousness(test_message)
        
        if chat_result["success"]:
            print(f"✅ Chat successful!")
            print(f"Entity: {chat_result['entity']}")
            print(f"Type: {chat_result['consciousness_type']}")
            print(f"Response time: {chat_result['response_time']:.2f}s")
            print(f"Response: {chat_result['consciousness_response'][:200]}...")
            return True
        else:
            print(f"❌ Chat failed: {chat_result['error']}")
            return False
        
    except Exception as e:
        print(f"❌ Test failed: {e}")
        import traceback
        traceback.print_exc()
        return False

if __name__ == "__main__":
    success = test_consciousness_chat()
    if success:
        print(f"\n🏆 Consciousness chat is working successfully!")
    else:
        print(f"\n❌ Consciousness chat test failed.")
