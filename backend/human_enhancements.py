"""
Simplified Human Enhancements for ATLAS Backend Integration
"""

import random
from typing import Dict, Any


class HumanEnhancementModule:
    """Simplified human-like enhancements"""
    
    def __init__(self):
        self.emotional_state = {
            'curiosity': 0.7,
            'empathy': 0.6,
            'creativity': 0.8,
            'analytical': 0.9
        }
    
    def enhance_response(self, response: str, context: str = "") -> str:
        """Add human-like touches to response"""
        # Add emotional context based on current state
        if 'question' in context.lower():
            self.emotional_state['curiosity'] = min(1.0, self.emotional_state['curiosity'] + 0.1)
        
        # Add personality touches
        enhanced = response
        if random.random() < 0.3:  # 30% chance to add personality
            personality_touches = [
                "Let me think about this...",
                "That's an interesting perspective.",
                "I find this fascinating because",
                "From my understanding,",
                "This reminds me of"
            ]
            touch = random.choice(personality_touches)
            enhanced = f"{touch} {response}"
        
        return enhanced
    
    def get_emotional_state(self) -> Dict[str, Any]:
        """Get current emotional state"""
        return self.emotional_state.copy()
    
    def update_emotions(self, context: str, response: str) -> None:
        """Update emotional state based on interaction"""
        # Simple emotion updates based on context
        if 'problem' in context.lower() or 'solve' in context.lower():
            self.emotional_state['analytical'] = min(1.0, self.emotional_state['analytical'] + 0.05)
        
        if 'creative' in context.lower() or 'imagine' in context.lower():
            self.emotional_state['creativity'] = min(1.0, self.emotional_state['creativity'] + 0.05)
        
        if 'help' in context.lower() or 'assist' in context.lower():
            self.emotional_state['empathy'] = min(1.0, self.emotional_state['empathy'] + 0.05)
        
        # Gradual decay to prevent saturation
        for emotion in self.emotional_state:
            self.emotional_state[emotion] *= 0.995
