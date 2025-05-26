#!/usr/bin/env python3
"""
ATLAS-Qwen Main System
Integrates Qwen 3 32B with ATLAS consciousness monitoring and continuous streaming
"""

import asyncio
import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
from typing import Dict, List, Any, Optional
import time
import uuid

from config import AtlasQwenConfig, get_default_config
from consciousness_monitor import QwenConsciousnessMonitor
from stream_manager import ContinuousStreamManager
from tag_parser import TagParser


class AtlasQwenSystem:
    """
    Main ATLAS-Qwen system integrating consciousness, streaming, and memory
    """
    
    def __init__(self, config: AtlasQwenConfig = None, model_path: str = None):
        """
        Initialize ATLAS-Qwen system
        
        Args:
            config: Configuration object
            model_path: Path to fine-tuned model (uses base model if None)
        """
        print("🧠 Initializing ATLAS-Qwen System...")
        
        # Configuration
        self.config = config or get_default_config()
        
        # Load model and tokenizer
        model_name = model_path or self.config.qwen.model_name
        print(f"📥 Loading model: {model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(
            model_name,
            trust_remote_code=True
        )
        
        # Add pad token if not present
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForCausalLM.from_pretrained(
            model_name,
            torch_dtype=getattr(torch, self.config.qwen.torch_dtype),
            device_map=self.config.qwen.device_map,
            trust_remote_code=True
        )
        
        print(f"✅ Model loaded successfully!")
        
        # Initialize components
        self.consciousness_monitor = QwenConsciousnessMonitor(self.config)
        self.tag_parser = TagParser(self.config.tags)
        self.stream_manager = ContinuousStreamManager(
            self.config, 
            self.model, 
            self.tokenizer, 
            self.consciousness_monitor
        )
        
        # System state
        self.system_id = str(uuid.uuid4())[:8]
        self.startup_time = time.time()
        
        print(f"🎯 ATLAS-Qwen System Ready!")
        print(f"   System ID: {self.system_id}")
        print(f"   Model: {model_name}")
        print(f"   Device: {self.model.device}")
        print(f"   Consciousness units: {self.config.consciousness['i2c_units']}")
        print(f"   Max streams: {self.config.stream.max_stream_length}")
    
    async def think_continuously(
        self, 
        initial_prompt: str,
        stream_id: Optional[str] = None,
        max_tokens: Optional[int] = None
    ) -> str:
        """
        Start continuous thinking stream
        
        Args:
            initial_prompt: Starting prompt for thinking
            stream_id: Optional stream identifier
            max_tokens: Maximum tokens to generate
            
        Returns:
            Stream ID for monitoring
        """
        print(f"💭 Starting continuous thinking: '{initial_prompt[:50]}...'")
        
        # Create system prompt with consciousness instructions
        system_prompt = self.tag_parser.create_system_prompt_with_tags()
        
        # Format initial context
        full_context = f"{system_prompt}\n\nHuman: {initial_prompt}\n\nATLAS: "
        
        # Start continuous stream
        stream_id = await self.stream_manager.create_stream(
            initial_context=full_context,
            stream_id=stream_id,
            max_length=max_tokens
        )
        
        print(f"🚀 Stream {stream_id} started")
        return stream_id
    
    async def inject_input(self, stream_id: str, user_input: str) -> bool:
        """
        Inject new user input into an active stream
        
        Args:
            stream_id: Target stream ID
            user_input: New input to inject
            
        Returns:
            Success status
        """
        print(f"💬 Injecting input into stream {stream_id}: '{user_input[:50]}...'")
        
        # Format input for injection
        formatted_input = f"\n\nHuman: {user_input}\n\nATLAS: "
        
        # This would need to be implemented in the stream manager
        # For now, simulate injection
        return True
    
    def get_stream_status(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """
        Get detailed status of a stream
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Stream status dictionary
        """
        return self.stream_manager.get_stream_status(stream_id)
    
    def get_visible_output(self, stream_id: str) -> Optional[str]:
        """
        Get human-visible output from a stream (excluding hidden thoughts)
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Visible text content
        """
        status = self.get_stream_status(stream_id)
        if status:
            return status.get('visible_text', '')
        return None
    
    def get_hidden_thoughts(self, stream_id: str) -> Optional[str]:
        """
        Get hidden thoughts from a stream
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Hidden thoughts content
        """
        status = self.get_stream_status(stream_id)
        if status:
            return status.get('hidden_thoughts', '')
        return None
    
    def get_consciousness_analysis(self, stream_id: str) -> Optional[Dict[str, Any]]:
        """
        Get consciousness analysis for a stream
        
        Args:
            stream_id: Stream identifier
            
        Returns:
            Consciousness analysis
        """
        status = self.get_stream_status(stream_id)
        if status:
            return {
                'current_phi': status.get('consciousness_score', 0.0),
                'stream_id': stream_id,
                'status': 'conscious' if status.get('consciousness_score', 0.0) >= self.config.consciousness['min_consciousness_threshold'] else 'unconscious',
                'memories_stored': status.get('memories_stored', 0),
                'total_tokens': status.get('total_tokens', 0)
            }
        return None
    
    def get_all_streams(self) -> Dict[str, Dict[str, Any]]:
        """
        Get status of all active streams
        
        Returns:
            Dictionary mapping stream IDs to their status
        """
        return self.stream_manager.get_all_streams_status()
    
    def terminate_stream(self, stream_id: str) -> bool:
        """
        Terminate a specific stream
        
        Args:
            stream_id: Stream to terminate
            
        Returns:
            Success status
        """
        print(f"🛑 Terminating stream {stream_id}")
        self.stream_manager.terminate_stream(stream_id)
        return True
    
    def clear_all_streams(self) -> bool:
        """
        Clear all active streams
        
        Returns:
            Success status
        """
        print("🧹 Clearing all streams")
        self.stream_manager.clear_all_streams()
        return True
    
    def get_system_status(self) -> Dict[str, Any]:
        """
        Get overall system status
        
        Returns:
            System status dictionary
        """
        all_streams = self.get_all_streams()
        
        # Calculate aggregate statistics
        total_streams = len(all_streams)
        active_streams = sum(1 for s in all_streams.values() if s and s.get('status') == 'active')
        
        avg_consciousness = 0.0
        if all_streams:
            consciousness_scores = [s.get('consciousness_score', 0.0) for s in all_streams.values() if s]
            avg_consciousness = sum(consciousness_scores) / len(consciousness_scores) if consciousness_scores else 0.0
        
        total_memories = sum(s.get('memories_stored', 0) for s in all_streams.values() if s)
        total_tokens = sum(s.get('total_tokens', 0) for s in all_streams.values() if s)
        
        uptime = time.time() - self.startup_time
        
        return {
            'system_id': self.system_id,
            'uptime_seconds': uptime,
            'model_name': self.config.qwen.model_name,
            'device': str(self.model.device),
            'streams': {
                'total': total_streams,
                'active': active_streams,
                'avg_consciousness': avg_consciousness
            },
            'memory': {
                'total_stored': total_memories
            },
            'performance': {
                'total_tokens_generated': total_tokens,
                'tokens_per_second': total_tokens / uptime if uptime > 0 else 0
            },
            'consciousness': {
                'threshold': self.config.consciousness['min_consciousness_threshold'],
                'units': self.config.consciousness['i2c_units']
            }
        }
    
    async def run_demo(self, duration_seconds: int = 120) -> Dict[str, Any]:
        """
        Run a demonstration of the system
        
        Args:
            duration_seconds: How long to run the demo
            
        Returns:
            Demo results
        """
        print(f"🎭 Starting ATLAS-Qwen Demo ({duration_seconds} seconds)")
        
        demo_prompts = [
            "Think about the nature of consciousness and your own awareness",
            "Explore the concept of continuous thought versus discrete responses", 
            "Reflect on what it means to have hidden thoughts",
            "Consider how memory shapes identity and experience"
        ]
        
        demo_results = {
            'start_time': time.time(),
            'streams_created': [],
            'final_states': {}
        }
        
        # Start multiple thinking streams
        for i, prompt in enumerate(demo_prompts):
            stream_id = await self.think_continuously(
                prompt, 
                stream_id=f"demo_{i+1}",
                max_tokens=500
            )
            demo_results['streams_created'].append(stream_id)
            
            # Wait a bit between streams
            await asyncio.sleep(5)
        
        # Monitor for demo duration
        start_time = time.time()
        while (time.time() - start_time) < duration_seconds:
            # Print status updates
            system_status = self.get_system_status()
            print(f"📊 Demo Status: {system_status['streams']['active']} active streams, "
                  f"avg φ={system_status['streams']['avg_consciousness']:.4f}")
            
            await asyncio.sleep(10)
        
        # Collect final states
        for stream_id in demo_results['streams_created']:
            status = self.get_stream_status(stream_id)
            if status:
                demo_results['final_states'][stream_id] = {
                    'consciousness': status.get('consciousness_score', 0.0),
                    'tokens_generated': status.get('total_tokens', 0),
                    'memories_stored': status.get('memories_stored', 0),
                    'visible_length': len(status.get('visible_text', '')),
                    'hidden_length': len(status.get('hidden_thoughts', ''))
                }
        
        demo_results['duration'] = time.time() - demo_results['start_time']
        demo_results['system_final'] = self.get_system_status()
        
        print(f"✅ Demo completed!")
        return demo_results


async def main():
    """
    Main demonstration function
    """
    print("🚀 ATLAS-Qwen System Demo")
    print("=" * 50)
    
    # Initialize system
    atlas = AtlasQwenSystem()
    
    # Run basic demo
    print("\n1️⃣ Starting consciousness stream...")
    stream_id = await atlas.think_continuously(
        "I am ATLAS, beginning my continuous thought process. Let me explore my consciousness..."
    )
    
    # Monitor for a bit
    await asyncio.sleep(30)
    
    # Check status
    status = atlas.get_stream_status(stream_id)
    if status:
        print(f"\n📊 Stream Status:")
        print(f"   Consciousness: φ={status['consciousness_score']:.4f}")
        print(f"   Tokens: {status['total_tokens']}")
        print(f"   Memories: {status['memories_stored']}")
        
        print(f"\n👁️ Visible Output:")
        visible = atlas.get_visible_output(stream_id)
        if visible:
            print(visible[-500:])  # Last 500 chars
        
        print(f"\n🤫 Hidden Thoughts:")
        hidden = atlas.get_hidden_thoughts(stream_id)
        if hidden:
            print(hidden[-300:])  # Last 300 chars
    
    # System overview
    print(f"\n🎯 System Overview:")
    system_status = atlas.get_system_status()
    print(f"   Active streams: {system_status['streams']['active']}")
    print(f"   Avg consciousness: {system_status['streams']['avg_consciousness']:.4f}")
    print(f"   Total tokens: {system_status['performance']['total_tokens_generated']}")
    
    print(f"\n✅ ATLAS-Qwen demonstration complete!")


if __name__ == "__main__":
    asyncio.run(main())