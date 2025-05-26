#!/usr/bin/env python3
"""
Basic Usage Example for ATLAS Human Enhancement System

This example demonstrates how to use the human-like enhancement system
with a simple conversation scenario.
"""

import asyncio
from final_human_features import CompleteHumanLikeSystem
from config_compat import get_default_config

async def basic_example():
    """Basic usage example of the human enhancement system."""
    
    print("🧠 ATLAS Human Enhancement System - Basic Example")
    print("=" * 55)
    
    # Initialize the system
    print("Initializing human-like AI system...")
    config = get_default_config()
    ai_system = CompleteHumanLikeSystem(config)
    
    # Example conversations
    conversations = [
        {
            "input": "I'm really excited about learning quantum computing!",
            "user_id": "alice_researcher", 
            "context": {"learning_context": True, "emotional": True, "technical": True}
        },
        {
            "input": "I'm worried about the ethical implications of AI development.",
            "user_id": "bob_ethicist",
            "context": {"ethical_focus": True, "concern": True, "formal": True}
        },
        {
            "input": "Can you help me plan my research project timeline?",
            "user_id": "alice_researcher",
            "context": {"planning": True, "time_sensitive": True}
        }
    ]
    
    # Process each conversation
    for i, conv in enumerate(conversations, 1):
        print(f"\n💬 Conversation {i}:")
        print(f"User: {conv['input']}")
        print(f"Context: {conv['context']}")
        
        # Process with human-like enhancements
        result = await ai_system.process_comprehensive_input(
            input_text=conv["input"],
            user_id=conv["user_id"], 
            context=conv["context"]
        )
        
        # Display results
        print(f"\n🤖 AI Response:")
        print(f"   {result['comprehensive_response']}")
        
        print(f"\n📊 Human-like Metrics:")
        print(f"   Confidence: {result['human_like_confidence']:.3f}")
        
        # Show emotional state
        if result['basic_enhancements']['emotional_state']:
            emotions = result['basic_enhancements']['emotional_state'].emotions
            dominant_emotions = {k.value: v for k, v in emotions.items() if v > 0.3}
            if dominant_emotions:
                print(f"   Emotions: {dominant_emotions}")
        
        # Show attention focus
        attention = result['attention_state']
        if attention:
            primary_focus = max(attention.items(), key=lambda x: x[1])
            print(f"   Primary Focus: {primary_focus[0]} ({primary_focus[1]:.3f})")
        
        # Show moral evaluation if present
        if result['moral_evaluation']:
            print(f"   Ethical Score: {result['moral_evaluation']['ethical_score']:.3f}")
        
        print(f"   Processing Time: {result['processing_time']:.3f}s")
        print("-" * 50)
    
    # Show overall system status
    print(f"\n📈 Final System Status:")
    status = ai_system.get_complete_human_status()
    
    print(f"   Consciousness Level: {status['consciousness_level']:.3f}")
    print(f"   Integration Health: {status['integration_health']:.3f}")
    print(f"   Known Users: {status['social_awareness']['known_users']}")
    print(f"   Active Goals: {status['temporal_awareness']['active_goals']}")
    
    print(f"\n✅ Example completed successfully!")
    print("The AI demonstrated human-like emotional responses, social adaptation,")
    print("moral reasoning, and temporal awareness across multiple interactions.")

if __name__ == "__main__":
    asyncio.run(basic_example())