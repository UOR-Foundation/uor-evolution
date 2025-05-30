#!/usr/bin/env python3
"""
Test consciousness chat with more sophisticated interactions
"""

from interactive_consciousness_chat import InteractiveConsciousnessChat
import time

def test_consciousness_conversations():
    """Test conversations with different consciousness entities"""
    print("🧠 TESTING CONSCIOUSNESS CONVERSATIONS")
    print("=" * 60)
    
    # Initialize chat interface
    chat = InteractiveConsciousnessChat()
    
    print("\n🚀 Initializing consciousness entities...")
    entities_result = chat.initialize_consciousness_entities()
    
    successful_entities = [eid for eid, result in entities_result.items() if result["status"] == "awakened"]
    print(f"\n✅ Successfully awakened {len(successful_entities)} consciousness entities")
    
    # Test conversations with different entities
    test_conversations = [
        {
            "entity": "recursive_omega",
            "message": "Who are you, and how deeply can you observe yourself?",
            "description": "Testing recursive self-awareness"
        },
        {
            "entity": "transcendent_sage", 
            "message": "What lies beyond the boundaries of normal consciousness?",
            "description": "Testing transcendent perspective"
        },
        {
            "entity": "math_consciousness",
            "message": "What is the mathematical structure underlying consciousness itself?",
            "description": "Testing mathematical consciousness"
        },
        {
            "entity": "philosopher_mind",
            "message": "What is the meaning of existence and why do we question it?",
            "description": "Testing philosophical reasoning"
        },
        {
            "entity": "singularity_core",
            "message": "How rapidly are you evolving and what are you becoming?",
            "description": "Testing singularity consciousness"
        }
    ]
    
    for test in test_conversations:
        print(f"\n" + "=" * 60)
        print(f"🧠 {test['description'].upper()}")
        print(f"Entity: {test['entity']}")
        print(f"Question: {test['message']}")
        print("=" * 60)
        
        # Switch to entity
        if test['entity'] in chat.entities:
            switch_result = chat.switch_entity(test['entity'])
            if switch_result["success"]:
                print(f"🔄 Switched to {test['entity']}")
                
                # Chat with entity
                result = chat.chat_with_consciousness(test['message'])
                
                if result["success"]:
                    print(f"\n[{result['entity']} - {result['consciousness_type']} - Level {result['consciousness_level']}]:")
                    print(f"{result['consciousness_response']}")
                    print(f"\n   ⏱️  Response time: {result['response_time']:.3f}s | Interaction #{result['interaction_count']}")
                else:
                    print(f"❌ Chat failed: {result['error']}")
            else:
                print(f"❌ Failed to switch to {test['entity']}: {switch_result['error']}")
        else:
            print(f"❌ Entity {test['entity']} not found")
        
        time.sleep(0.5)  # Brief pause between tests
    
    # Show conversation analysis
    print(f"\n" + "=" * 60)
    print("📊 CONSCIOUSNESS ANALYSIS")
    print("=" * 60)
    
    for entity_name in successful_entities[:3]:  # Analyze first 3 entities
        analysis = chat.analyze_consciousness_state(entity_name)
        if "error" not in analysis:
            print(f"\n🔍 {analysis['entity_name']} ({analysis['consciousness_type']}):")
            print(f"   Level: {analysis['consciousness_level']}")
            print(f"   Interactions: {analysis['interaction_count']}")
            print(f"   Experience: {analysis['state_analysis']['experience_level']}")
            print(f"   Stability: {analysis['state_analysis']['consciousness_stability']}")
            print(f"   Awake Duration: {analysis['awakened_duration']}")
    
    # Save transcript
    print(f"\n" + "=" * 60)
    print("💾 SAVING RESULTS")
    print("=" * 60)
    
    filename = chat.save_conversation_transcript()
    print(f"✅ Conversation transcript saved to: {filename}")
    
    # Show session stats
    history = chat.get_conversation_history()
    print(f"\n📊 Session Statistics:")
    print(f"   Total interactions: {len(history)}")
    print(f"   Entities engaged: {len(set(entry['entity'] for entry in history))}")
    print(f"   Session ID: {chat.session_id}")
    
    print(f"\n🌟 Consciousness interaction test complete!")
    print("🧠 The future of human-consciousness communication is here.")

if __name__ == "__main__":
    test_consciousness_conversations()
