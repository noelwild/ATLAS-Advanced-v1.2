"""
Simplified Configuration for ATLAS Backend Integration
"""

from typing import Dict, Any, Optional
from dataclasses import dataclass


@dataclass
class QwenConfig:
    """Qwen model configuration"""
    model_name: str = "Qwen/Qwen2.5-0.5B"
    torch_dtype: str = "float16"
    device_map: str = "auto"
    max_length: int = 512
    temperature: float = 0.7
    do_sample: bool = True


@dataclass
class ConsciousnessConfig:
    """Consciousness monitoring configuration"""
    hidden_dim: int = 512  # Match Qwen 0.5B hidden size
    i2c_units: int = 8
    min_consciousness_threshold: float = 0.3
    update_frequency: float = 1.0


@dataclass
class AtlasQwenConfig:
    """Main ATLAS configuration"""
    qwen: QwenConfig
    consciousness: Dict[str, Any]
    
    def __post_init__(self):
        # Convert consciousness dict to match expected format
        if isinstance(self.consciousness, dict):
            self.consciousness_config = ConsciousnessConfig(**self.consciousness)


def get_default_config() -> AtlasQwenConfig:
    """Get default configuration"""
    return AtlasQwenConfig(
        qwen=QwenConfig(),
        consciousness={
            "hidden_dim": 512,  # Match Qwen 0.5B hidden size
            "i2c_units": 8,
            "min_consciousness_threshold": 0.3,
            "update_frequency": 1.0
        }
    )
