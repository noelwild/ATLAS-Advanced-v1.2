"""
Configuration and utility functions for ATLAS Human Enhancement System
Compatibility layer for import management
"""

# Check if we're in the ATLAS environment
try:
    from config import get_default_config, AtlasQwenConfig, QwenConfig, LoRAConfig, StreamConfig, TagConfig
except ImportError:
    # Create minimal config classes for standalone usage
    from dataclasses import dataclass
    from typing import Dict, Any, Optional
    
    @dataclass
    class QwenConfig:
        model_name: str = "Qwen/Qwen2.5-32B"
        device: str = "auto"
        torch_dtype: str = "auto"
        trust_remote_code: bool = True
        
    @dataclass
    class LoRAConfig:
        r: int = 8
        lora_alpha: int = 32
        target_modules: list = None
        lora_dropout: float = 0.1
        
        def __post_init__(self):
            if self.target_modules is None:
                self.target_modules = ["q_proj", "v_proj"]
    
    @dataclass 
    class StreamConfig:
        context_window_size: int = 4096
        memory_injection_size: int = 512
        max_response_length: int = 1024
        temperature: float = 0.7
        
    @dataclass
    class TagConfig:
        thought_tags: bool = True
        memory_tags: bool = True
        hidden_tags: bool = True
        python_tags: bool = True
        recall_tags: bool = True
        
    @dataclass
    class AtlasQwenConfig:
        qwen: QwenConfig
        lora: LoRAConfig
        stream: StreamConfig
        tags: TagConfig
        consciousness: Optional[Dict[str, Any]] = None
        memory: Optional[Dict[str, Any]] = None
        code_execution: Optional[Dict[str, Any]] = None
        
        def __post_init__(self):
            if self.consciousness is None:
                self.consciousness = {
                    "hidden_dim": 4096,
                    "i2c_units": 8,
                    "consciousness_threshold": 0.5,
                    "phi_integration": True
                }
            
            if self.memory is None:
                self.memory = {
                    "embedding_dim": 4096,
                    "max_memories": 10000,
                    "similarity_threshold": 0.75,
                    "chunk_size": 512
                }
            
            if self.code_execution is None:
                self.code_execution = {
                    "timeout": 30,
                    "max_output_length": 10000,
                    "environment_name": "atlas_env",
                    "allowed_imports": ["math", "numpy", "matplotlib", "pandas", "scipy"],
                    "sandbox_enabled": True
                }
    
    def get_default_config() -> AtlasQwenConfig:
        """Get default configuration for standalone usage"""
        return AtlasQwenConfig(
            qwen=QwenConfig(),
            lora=LoRAConfig(),
            stream=StreamConfig(),
            tags=TagConfig()
        )

__all__ = [
    'get_default_config',
    'AtlasQwenConfig', 
    'QwenConfig',
    'LoRAConfig',
    'StreamConfig',
    'TagConfig'
]