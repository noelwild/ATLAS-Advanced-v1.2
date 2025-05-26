"""
ATLAS-Qwen Configuration
Configuration for Qwen 3 32B integration with ATLAS consciousness system
"""

from dataclasses import dataclass
from typing import Dict, Any, List
import yaml


@dataclass
class QwenConfig:
    """Qwen model configuration"""
    model_name: str = "Qwen/Qwen2.5-32B"
    use_flash_attention: bool = True
    torch_dtype: str = "bfloat16"
    device_map: str = "auto"
    max_context_length: int = 32768
    generation_config: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.generation_config is None:
            self.generation_config = {
                "max_new_tokens": 2048,
                "temperature": 0.7,
                "top_p": 0.9,
                "do_sample": True,
                "repetition_penalty": 1.1,
                "pad_token_id": 151643,  # Qwen pad token
                "eos_token_id": 151645,  # Qwen eos token
            }


@dataclass
class LoRAConfig:
    """QLoRA configuration for fine-tuning"""
    r: int = 64
    lora_alpha: int = 128
    lora_dropout: float = 0.1
    bias: str = "none"
    task_type: str = "CAUSAL_LM"
    target_modules: List[str] = None
    
    def __post_init__(self):
        if self.target_modules is None:
            # Qwen 3 32B attention and MLP modules
            self.target_modules = [
                "q_proj", "k_proj", "v_proj", "o_proj",
                "gate_proj", "up_proj", "down_proj"
            ]


@dataclass
class StreamConfig:
    """Continuous stream configuration"""
    max_stream_length: int = 8192
    context_window_size: int = 24576  # Leave room for memory injection
    memory_injection_size: int = 4096
    stream_update_interval: float = 0.1  # seconds
    consciousness_update_interval: float = 2.0  # seconds
    hidden_thought_probability: float = 0.3
    memory_storage_threshold: float = 0.7


@dataclass
class TagConfig:
    """Tag system configuration"""
    thought_tag: str = "thought"
    memory_tag: str = "memory"
    recall_tag: str = "recall"
    consciousness_tag: str = "consciousness"
    stream_context_tag: str = "stream_context"
    hidden_tag: str = "hidden"
    code_tag: str = "code"
    python_tag: str = "python"
    
    # Tag patterns for extraction
    tag_patterns: Dict[str, str] = None
    
    def __post_init__(self):
        if self.tag_patterns is None:
            self.tag_patterns = {
                "thought": r'<thought>(.*?)</thought>',
                "memory": r'<memory key="([^"]*)">(.*?)</memory>',
                "recall": r'<recall query="([^"]*)"/>',
                "consciousness": r'<consciousness>(.*?)</consciousness>',
                "stream_context": r'<stream_context>(.*?)</stream_context>',
                "hidden": r'<hidden>(.*?)</hidden>',
                "code": r'<code(?:\s+language="([^"]*)")?\s*>(.*?)</code>',
                "python": r'<python>(.*?)</python>'
            }


@dataclass
class AtlasQwenConfig:
    """Main ATLAS-Qwen configuration"""
    qwen: QwenConfig
    lora: LoRAConfig
    stream: StreamConfig
    tags: TagConfig
    consciousness: Dict[str, Any] = None
    memory: Dict[str, Any] = None
    code_execution: Dict[str, Any] = None
    
    def __post_init__(self):
        if self.consciousness is None:
            self.consciousness = {
                "hidden_dim": 4096,  # Match Qwen hidden size
                "i2c_units": 8,
                "i2c_position": 16,  # Insert at middle layer
                "min_consciousness_threshold": 0.3
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
    """Get default ATLAS-Qwen configuration"""
    return AtlasQwenConfig(
        qwen=QwenConfig(),
        lora=LoRAConfig(),
        stream=StreamConfig(),
        tags=TagConfig()
    )


def save_config(config: AtlasQwenConfig, path: str):
    """Save configuration to YAML file"""
    config_dict = {
        'qwen': {
            'model_name': config.qwen.model_name,
            'use_flash_attention': config.qwen.use_flash_attention,
            'torch_dtype': config.qwen.torch_dtype,
            'device_map': config.qwen.device_map,
            'max_context_length': config.qwen.max_context_length,
            'generation_config': config.qwen.generation_config
        },
        'lora': {
            'r': config.lora.r,
            'lora_alpha': config.lora.lora_alpha,
            'lora_dropout': config.lora.lora_dropout,
            'bias': config.lora.bias,
            'task_type': config.lora.task_type,
            'target_modules': config.lora.target_modules
        },
        'stream': {
            'max_stream_length': config.stream.max_stream_length,
            'context_window_size': config.stream.context_window_size,
            'memory_injection_size': config.stream.memory_injection_size,
            'stream_update_interval': config.stream.stream_update_interval,
            'consciousness_update_interval': config.stream.consciousness_update_interval,
            'hidden_thought_probability': config.stream.hidden_thought_probability,
            'memory_storage_threshold': config.stream.memory_storage_threshold
        },
        'tags': {
            'thought_tag': config.tags.thought_tag,
            'memory_tag': config.tags.memory_tag,
            'recall_tag': config.tags.recall_tag,
            'consciousness_tag': config.tags.consciousness_tag,
            'stream_context_tag': config.tags.stream_context_tag,
            'hidden_tag': config.tags.hidden_tag,
            'tag_patterns': config.tags.tag_patterns
        },
        'consciousness': config.consciousness,
        'memory': config.memory,
        'code_execution': config.code_execution
    }
    
    with open(path, 'w') as f:
        yaml.dump(config_dict, f, default_flow_style=False, indent=2)


def load_config(path: str) -> AtlasQwenConfig:
    """Load configuration from YAML file"""
    with open(path, 'r') as f:
        config_dict = yaml.safe_load(f)
    
    return AtlasQwenConfig(
        qwen=QwenConfig(**config_dict['qwen']),
        lora=LoRAConfig(**config_dict['lora']),
        stream=StreamConfig(**config_dict['stream']),
        tags=TagConfig(**config_dict['tags']),
        consciousness=config_dict.get('consciousness'),
        memory=config_dict.get('memory'),
        code_execution=config_dict.get('code_execution')
    )
