"""
Simplified Consciousness Monitor for ATLAS Backend Integration
"""

import random
import time
from typing import Dict, Any, List
from datetime import datetime


class ConsciousnessMonitor:
    """Simplified consciousness monitoring"""
    
    def __init__(self, hidden_dim: int = 512, i2c_units: int = 8):
        self.hidden_dim = hidden_dim
        self.i2c_units = i2c_units
        self.current_level = 0.0
        self.history = []
    
    def update_consciousness(self, text: str, tokens: List[int] = None) -> float:
        """Update consciousness level based on generated text"""
        # Simple consciousness metric based on text complexity and length
        complexity = len(set(text.split())) / max(len(text.split()), 1)  # Vocabulary diversity
        length_factor = min(len(text) / 1000, 1.0)  # Normalize by length
        random_factor = random.uniform(0.8, 1.2)  # Add some variation
        
        consciousness_level = (complexity * 0.5 + length_factor * 0.3 + random_factor * 0.2)
        consciousness_level = max(0.0, min(1.0, consciousness_level))
        
        self.current_level = consciousness_level
        self.history.append({
            "level": consciousness_level,
            "timestamp": datetime.now().isoformat(),
            "text_length": len(text)
        })
        
        # Keep only last 100 entries
        if len(self.history) > 100:
            self.history = self.history[-100:]
        
        return consciousness_level
    
    def get_current_state(self) -> Dict[str, Any]:
        """Get current consciousness state"""
        return {
            "consciousness_level": self.current_level,
            "i2c_activations": [random.uniform(0.0, 1.0) for _ in range(self.i2c_units)],
            "attention_patterns": {
                "self_attention": random.uniform(0.4, 0.8),
                "environmental_attention": random.uniform(0.2, 0.6),
                "memory_attention": random.uniform(0.3, 0.7)
            },
            "history_length": len(self.history)
        }


# For backward compatibility with original ATLAS files
QwenConsciousnessMonitor = ConsciousnessMonitor
