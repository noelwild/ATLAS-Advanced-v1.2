#!/usr/bin/env python3
"""
Enhanced ATLAS System with Model Switching and Advanced Learning Integration
Supports switching between Qwen 0.5B and fine-tuned Qwen3 32B models
with advanced consciousness monitoring and learning capabilities
"""

import os
import asyncio
import torch
import logging
import random
from transformers import AutoTokenizer, AutoModelForCausalLM, BitsAndBytesConfig
from typing import Dict, List, Any, Optional, Tuple
import time
import uuid
import json
from datetime import datetime
from dataclasses import dataclass
from enum import Enum
import numpy as np
from contextlib import asynccontextmanager

# Import enhanced modules
try:
    from enhanced_consciousness_monitor import EnhancedConsciousnessMonitor
except ImportError:
    from backend.enhanced_consciousness_monitor import EnhancedConsciousnessMonitor

try:
    from enhanced_human_features import EnhancedHumanFeaturesSystem
except ImportError:
    from backend.enhanced_human_features import EnhancedHumanFeaturesSystem

try:
    from secure_code_executor import SecureCodeExecutor
except ImportError:
    from backend.secure_code_executor import SecureCodeExecutor

logger = logging.getLogger(__name__)


class ModelType(Enum):
    """Available model types"""
    QWEN_SMALL = "qwen_0.5b"
    QWEN_LARGE = "qwen3_32b_finetuned"


@dataclass
class ModelConfig:
    """Configuration for a specific model"""
    model_name: str
    model_path: str
    max_length: int
    temperature: float
    do_sample: bool
    load_in_8bit: bool = False
    load_in_4bit: bool = False
    device_map: str = "auto"
    torch_dtype: str = "float16"
    trust_remote_code: bool = True


@dataclass
class LearningSession:
    """Represents a learning session with experience data"""
    session_id: str
    model_used: ModelType
    user_interactions: List[Dict[str, Any]]
    consciousness_trajectory: List[float]
    learning_outcomes: Dict[str, Any]
    performance_metrics: Dict[str, float]
    timestamp: datetime
    duration: float


class AdvancedLearningSystem:
    """Advanced learning and adaptation system"""
    
    def __init__(self):
        self.learning_sessions = {}
        self.model_performance = {
            ModelType.QWEN_SMALL: {"usage_count": 0, "avg_satisfaction": 0.0, "response_time": 0.0},
            ModelType.QWEN_LARGE: {"usage_count": 0, "avg_satisfaction": 0.0, "response_time": 0.0}
        }
        self.adaptation_patterns = {}
        self.learning_rate = 0.01
        
        logger.info("Advanced Learning System initialized")
    
    async def start_learning_session(self, model_type: ModelType) -> str:
        """Start a new learning session"""
        session_id = f"learn_{int(time.time() * 1000)}_{uuid.uuid4().hex[:8]}"
        
        self.learning_sessions[session_id] = LearningSession(
            session_id=session_id,
            model_used=model_type,
            user_interactions=[],
            consciousness_trajectory=[],
            learning_outcomes={},
            performance_metrics={},
            timestamp=datetime.now(),
            duration=0.0
        )
        
        logger.info(f"Started learning session {session_id} with {model_type.value}")
        return session_id
    
    async def record_interaction(self, 
                                session_id: str,
                                user_input: str,
                                model_response: str,
                                consciousness_level: float,
                                response_time: float,
                                user_satisfaction: float = None):
        """Record interaction data for learning"""
        
        if session_id not in self.learning_sessions:
            logger.warning(f"Session {session_id} not found")
            return
        
        session = self.learning_sessions[session_id]
        
        interaction_data = {
            "timestamp": datetime.now().isoformat(),
            "user_input": user_input,
            "model_response": model_response,
            "consciousness_level": consciousness_level,
            "response_time": response_time,
            "user_satisfaction": user_satisfaction,
            "input_length": len(user_input),
            "response_length": len(model_response)
        }
        
        session.user_interactions.append(interaction_data)
        session.consciousness_trajectory.append(consciousness_level)
        
        # Update model performance metrics
        model_type = session.model_used
        self.model_performance[model_type]["usage_count"] += 1
        
        # Update running averages
        if user_satisfaction is not None:
            current_avg = self.model_performance[model_type]["avg_satisfaction"]
            count = self.model_performance[model_type]["usage_count"]
            new_avg = (current_avg * (count - 1) + user_satisfaction) / count
            self.model_performance[model_type]["avg_satisfaction"] = new_avg
        
        # Update response time
        current_time = self.model_performance[model_type]["response_time"]
        count = self.model_performance[model_type]["usage_count"]
        new_time = (current_time * (count - 1) + response_time) / count
        self.model_performance[model_type]["response_time"] = new_time
    
    async def end_learning_session(self, session_id: str) -> Dict[str, Any]:
        """End learning session and compute learning outcomes"""
        
        if session_id not in self.learning_sessions:
            return {}
        
        session = self.learning_sessions[session_id]
        session.duration = (datetime.now() - session.timestamp).total_seconds()
        
        # Compute session metrics
        session.performance_metrics = self._compute_session_metrics(session)
        session.learning_outcomes = await self._extract_learning_outcomes(session)
        
        # Update adaptation patterns
        await self._update_adaptation_patterns(session)
        
        logger.info(f"Ended learning session {session_id}")
        return session.learning_outcomes
    
    def _compute_session_metrics(self, session: LearningSession) -> Dict[str, float]:
        """Compute performance metrics for a session"""
        
        if not session.user_interactions:
            return {}
        
        # Response time metrics
        response_times = [i["response_time"] for i in session.user_interactions]
        avg_response_time = np.mean(response_times)
        
        # Consciousness metrics
        if session.consciousness_trajectory:
            avg_consciousness = np.mean(session.consciousness_trajectory)
            consciousness_stability = 1.0 - np.std(session.consciousness_trajectory)
        else:
            avg_consciousness = 0.0
            consciousness_stability = 0.0
        
        # User satisfaction
        satisfactions = [i["user_satisfaction"] for i in session.user_interactions if i["user_satisfaction"] is not None]
        avg_satisfaction = np.mean(satisfactions) if satisfactions else 0.5
        
        # Interaction quality
        avg_input_length = np.mean([i["input_length"] for i in session.user_interactions])
        avg_response_length = np.mean([i["response_length"] for i in session.user_interactions])
        
        return {
            "avg_response_time": avg_response_time,
            "avg_consciousness": avg_consciousness,
            "consciousness_stability": consciousness_stability,
            "avg_satisfaction": avg_satisfaction,
            "interaction_count": len(session.user_interactions),
            "session_duration": session.duration,
            "avg_input_length": avg_input_length,
            "avg_response_length": avg_response_length
        }
    
    async def _extract_learning_outcomes(self, session: LearningSession) -> Dict[str, Any]:
        """Extract learning outcomes from session data"""
        
        outcomes = {
            "model_effectiveness": session.performance_metrics.get("avg_satisfaction", 0.0),
            "consciousness_consistency": session.performance_metrics.get("consciousness_stability", 0.0),
            "response_efficiency": 1.0 / (session.performance_metrics.get("avg_response_time", 1.0) + 0.1),
            "engagement_level": min(1.0, session.performance_metrics.get("interaction_count", 0) / 10),
            "learning_quality": "high" if session.performance_metrics.get("avg_satisfaction", 0) > 0.7 else "medium" if session.performance_metrics.get("avg_satisfaction", 0) > 0.4 else "low"
        }
        
        # Detect interaction patterns
        if len(session.user_interactions) > 3:
            outcomes["interaction_patterns"] = self._detect_interaction_patterns(session.user_interactions)
        
        return outcomes
    
    def _detect_interaction_patterns(self, interactions: List[Dict[str, Any]]) -> Dict[str, Any]:
        """Detect patterns in user interactions"""
        
        # Question vs statement pattern
        questions = sum(1 for i in interactions if "?" in i["user_input"])
        question_ratio = questions / len(interactions)
        
        # Complexity trend
        input_lengths = [i["input_length"] for i in interactions]
        if len(input_lengths) > 1:
            complexity_trend = "increasing" if input_lengths[-1] > input_lengths[0] else "decreasing"
        else:
            complexity_trend = "stable"
        
        # Response time trend
        response_times = [i["response_time"] for i in interactions]
        if len(response_times) > 1:
            time_trend = "improving" if response_times[-1] < response_times[0] else "degrading"
        else:
            time_trend = "stable"
        
        return {
            "question_ratio": question_ratio,
            "complexity_trend": complexity_trend,
            "response_time_trend": time_trend,
            "engagement_pattern": "high" if len(interactions) > 5 else "medium" if len(interactions) > 2 else "low"
        }
    
    async def _update_adaptation_patterns(self, session: LearningSession):
        """Update adaptation patterns based on session data"""
        
        model_type = session.model_used.value
        
        if model_type not in self.adaptation_patterns:
            self.adaptation_patterns[model_type] = {
                "optimal_contexts": [],
                "performance_trends": [],
                "user_preferences": {}
            }
        
        pattern = self.adaptation_patterns[model_type]
        
        # Update performance trends
        pattern["performance_trends"].append({
            "timestamp": session.timestamp.isoformat(),
            "satisfaction": session.performance_metrics.get("avg_satisfaction", 0.0),
            "consciousness": session.performance_metrics.get("avg_consciousness", 0.0),
            "efficiency": session.performance_metrics.get("avg_response_time", 1.0)
        })
        
        # Keep only recent trends
        if len(pattern["performance_trends"]) > 50:
            pattern["performance_trends"] = pattern["performance_trends"][-50:]
    
    def recommend_model(self, context: Dict[str, Any]) -> ModelType:
        """Recommend optimal model based on context and learning"""
        
        # Get current performance for both models
        small_perf = self.model_performance[ModelType.QWEN_SMALL]
        large_perf = self.model_performance[ModelType.QWEN_LARGE]
        
        # Default to small model if no usage data
        if small_perf["usage_count"] == 0 and large_perf["usage_count"] == 0:
            return ModelType.QWEN_SMALL
        
        # Consider context factors
        complexity_score = context.get("complexity", 0.5)
        time_pressure = context.get("time_pressure", 0.5)
        quality_requirement = context.get("quality_requirement", 0.5)
        
        # Score models
        small_score = 0.0
        large_score = 0.0
        
        # Performance-based scoring
        if small_perf["usage_count"] > 0:
            small_score += small_perf["avg_satisfaction"] * 0.4
            small_score += (1.0 / (small_perf["response_time"] + 0.1)) * 0.3  # Efficiency bonus
        
        if large_perf["usage_count"] > 0:
            large_score += large_perf["avg_satisfaction"] * 0.4
            large_score += (1.0 / (large_perf["response_time"] + 0.1)) * 0.2  # Less weight on efficiency
        
        # Context-based scoring
        if complexity_score > 0.7:
            large_score += 0.3  # Large model for complex tasks
        else:
            small_score += 0.2  # Small model for simple tasks
        
        if time_pressure > 0.7:
            small_score += 0.2  # Small model for urgent tasks
        
        if quality_requirement > 0.8:
            large_score += 0.3  # Large model for high-quality requirements
        
        return ModelType.QWEN_LARGE if large_score > small_score else ModelType.QWEN_SMALL
    
    def get_learning_analytics(self) -> Dict[str, Any]:
        """Get comprehensive learning analytics"""
        
        total_sessions = len(self.learning_sessions)
        active_sessions = sum(1 for s in self.learning_sessions.values() if s.duration == 0.0)
        
        return {
            "total_sessions": total_sessions,
            "active_sessions": active_sessions,
            "model_performance": self.model_performance.copy(),
            "adaptation_patterns": {k: len(v["performance_trends"]) for k, v in self.adaptation_patterns.items()},
            "learning_rate": self.learning_rate,
            "system_maturity": min(1.0, total_sessions / 100)
        }


class EnhancedAtlasSystem:
    """
    Enhanced ATLAS system with model switching, advanced consciousness monitoring,
    complete human features, and multimodal dreaming capabilities
    """
    
    def __init__(self, config: Dict[str, Any] = None):
        """Initialize Enhanced ATLAS system"""
        
        logger.info("🧠 Initializing Enhanced ATLAS System...")
        
        # Configuration
        self.config = self._setup_default_config(config)
        
        # Model management
        self.current_model_type = ModelType.QWEN_SMALL
        self.models = {}
        self.tokenizers = {}
        self.model_configs = self._setup_model_configs()
        
        # Enhanced cognitive systems
        self.consciousness_monitor = EnhancedConsciousnessMonitor(
            hidden_dim=self.config.get("hidden_dim", 512),
            i2c_units=self.config.get("i2c_units", 8),
            enable_dreaming=True,
            enable_multimodal=True
        )
        
        self.human_features = EnhancedHumanFeaturesSystem()
        self.code_executor = SecureCodeExecutor()
        self.learning_system = AdvancedLearningSystem()
        
        # Session management
        self.sessions = {}
        self.active_learning_sessions = {}
        
        # System state
        self.system_start_time = time.time()
        self.initialized_models = set()
        self.initialization_complete = False
        
        logger.info("Enhanced ATLAS System base initialization complete")
    
    def _setup_default_config(self, config: Dict[str, Any] = None) -> Dict[str, Any]:
        """Setup default configuration"""
        
        default_config = {
            "hidden_dim": 512,
            "i2c_units": 8,
            "consciousness_threshold": 0.3,
            "auto_model_switching": True,
            "enable_dreaming": True,
            "enable_multimodal": True,
            "learning_enabled": True,
            "max_context_length": 4096,
            "temperature": 0.7,
            "do_sample": True
        }
        
        if config:
            default_config.update(config)
        
        return default_config
    
    def _setup_model_configs(self) -> Dict[ModelType, ModelConfig]:
        """Setup configurations for different models"""
        
        return {
            ModelType.QWEN_SMALL: ModelConfig(
                model_name="Qwen/Qwen2.5-0.5B",
                model_path="Qwen/Qwen2.5-0.5B",
                max_length=512,
                temperature=0.7,
                do_sample=True,
                load_in_8bit=False,
                load_in_4bit=False,
                device_map="auto",
                torch_dtype="float16"
            ),
            ModelType.QWEN_LARGE: ModelConfig(
                model_name="Qwen/Qwen2.5-32B",  # Will be updated to fine-tuned path
                model_path=os.getenv("QWEN3_FINETUNED_PATH", "Qwen/Qwen2.5-32B"),
                max_length=4096,
                temperature=0.7,
                do_sample=True,
                load_in_8bit=True,  # Use quantization for large model
                load_in_4bit=False,
                device_map="auto",
                torch_dtype="float16"
            )
        }
    
    async def initialize(self, models_to_load: List[ModelType] = None):
        """Initialize ATLAS system with specified models"""
        
        if models_to_load is None:
            models_to_load = [ModelType.QWEN_SMALL]  # Start with small model
        
        for model_type in models_to_load:
            await self._load_model(model_type)
        
        # Initialize with default model
        if ModelType.QWEN_SMALL in self.initialized_models:
            self.current_model_type = ModelType.QWEN_SMALL
        elif self.initialized_models:
            self.current_model_type = list(self.initialized_models)[0]
        
        self.initialization_complete = True
        logger.info("✅ Enhanced ATLAS System fully initialized")
    
    async def _load_model(self, model_type: ModelType):
        """Load a specific model"""
        
        if model_type in self.initialized_models:
            logger.info(f"Model {model_type.value} already loaded")
            return
        
        try:
            config = self.model_configs[model_type]
            logger.info(f"📥 Loading {model_type.value} model from {config.model_path}")
            
            # Setup quantization config for large model
            quantization_config = None
            if config.load_in_8bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_8bit=True,
                    llm_int8_enable_fp32_cpu_offload=True
                )
            elif config.load_in_4bit:
                quantization_config = BitsAndBytesConfig(
                    load_in_4bit=True,
                    bnb_4bit_compute_dtype=torch.float16,
                    bnb_4bit_use_double_quant=True,
                    bnb_4bit_quant_type="nf4"
                )
            
            # Load tokenizer
            tokenizer = AutoTokenizer.from_pretrained(
                config.model_path,
                trust_remote_code=config.trust_remote_code
            )
            
            if tokenizer.pad_token is None:
                tokenizer.pad_token = tokenizer.eos_token
            
            # Load model
            model_kwargs = {
                "torch_dtype": getattr(torch, config.torch_dtype),
                "device_map": config.device_map,
                "trust_remote_code": config.trust_remote_code
            }
            
            if quantization_config:
                model_kwargs["quantization_config"] = quantization_config
            
            model = AutoModelForCausalLM.from_pretrained(
                config.model_path,
                **model_kwargs
            )
            
            self.models[model_type] = model
            self.tokenizers[model_type] = tokenizer
            self.initialized_models.add(model_type)
            
            logger.info(f"✅ {model_type.value} model loaded successfully")
            
        except Exception as e:
            logger.error(f"❌ Failed to load {model_type.value} model: {str(e)}")
            # Don't raise exception, allow system to continue with other models
    
    async def switch_model(self, model_type: ModelType, force_load: bool = False) -> bool:
        """Switch to a different model"""
        
        if model_type not in self.initialized_models:
            if force_load:
                await self._load_model(model_type)
                if model_type not in self.initialized_models:
                    logger.error(f"Failed to load {model_type.value} for switching")
                    return False
            else:
                logger.warning(f"Model {model_type.value} not loaded")
                return False
        
        old_model = self.current_model_type
        self.current_model_type = model_type
        
        logger.info(f"🔄 Switched from {old_model.value} to {model_type.value}")
        return True
    
    async def generate_response(self,
                               message: str,
                               session_id: str = None,
                               user_id: str = None,
                               context: Dict[str, Any] = None,
                               include_consciousness: bool = True,
                               auto_model_selection: bool = None) -> Dict[str, Any]:
        """Generate enhanced response with all cognitive systems"""
        
        session_id = session_id or str(uuid.uuid4())
        user_id = user_id or "default_user"
        context = context or {}
        
        if auto_model_selection is None:
            auto_model_selection = self.config.get("auto_model_switching", True)
        
        start_time = time.time()
        
        try:
            # Auto model selection if enabled
            if auto_model_selection and len(self.initialized_models) > 1:
                recommended_model = self.learning_system.recommend_model(context)
                if recommended_model in self.initialized_models:
                    await self.switch_model(recommended_model)
            
            # Start learning session
            learning_session_id = None
            if self.config.get("learning_enabled", True):
                learning_session_id = await self.learning_system.start_learning_session(self.current_model_type)
                self.active_learning_sessions[session_id] = learning_session_id
            
            # Get or create session
            if session_id not in self.sessions:
                self.sessions[session_id] = {
                    "created": datetime.now(),
                    "conversation_history": [],
                    "user_id": user_id,
                    "model_history": []
                }
            
            session = self.sessions[session_id]
            
            # Generate base response with current model
            base_response = await self._generate_with_model(message, session, context)
            
            # Compute consciousness state
            consciousness_state = None
            if include_consciousness:
                consciousness_state = await self.consciousness_monitor.compute_consciousness(
                    text_input=message,
                    text_features=None,
                    visual_features=context.get("visual_features"),
                    audio_features=context.get("audio_features")
                )
            
            consciousness_level = consciousness_state.phi_score if consciousness_state else 0.5
            
            # Process through enhanced human features
            human_enhancement_result = await self.human_features.process_interaction(
                user_input=message,
                user_id=user_id,
                context=context,
                consciousness_level=consciousness_level
            )
            
            # Combine base response with enhancements
            enhanced_response = self._integrate_response_components(
                base_response, human_enhancement_result, consciousness_state
            )
            
            # Store conversation
            conversation_entry = {
                "session_id": session_id,
                "user_message": message,
                "atlas_response": enhanced_response,
                "model_used": self.current_model_type.value,
                "consciousness_level": consciousness_level,
                "consciousness_state": consciousness_state.get_current_state() if consciousness_state else None,
                "human_enhancements": human_enhancement_result,
                "timestamp": datetime.now(),
                "response_time": time.time() - start_time
            }
            
            session["conversation_history"].append(conversation_entry)
            session["model_history"].append(self.current_model_type.value)
            
            # Record learning data
            if learning_session_id:
                await self.learning_system.record_interaction(
                    session_id=learning_session_id,
                    user_input=message,
                    model_response=enhanced_response,
                    consciousness_level=consciousness_level,
                    response_time=time.time() - start_time,
                    user_satisfaction=context.get("user_satisfaction")
                )
            
            # Build comprehensive response
            response_data = {
                "response": enhanced_response,
                "model_used": self.current_model_type.value,
                "consciousness_level": consciousness_level,
                "consciousness_state": consciousness_state.get_current_state() if consciousness_state else None,
                "human_enhancements_active": human_enhancement_result.get("enhancement_systems_active", {}),
                "system_coherence": human_enhancement_result.get("system_coherence", 0.0),
                "session_id": session_id,
                "response_time": time.time() - start_time,
                "learning_session_id": learning_session_id
            }
            
            return response_data
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            
            # Fallback response
            return {
                "response": f"I apologize, but I encountered an error while processing your request: {str(e)}",
                "model_used": self.current_model_type.value if hasattr(self, 'current_model_type') else "unknown",
                "consciousness_level": 0.0,
                "consciousness_state": None,
                "human_enhancements_active": {},
                "system_coherence": 0.0,
                "session_id": session_id,
                "response_time": time.time() - start_time,
                "error": True
            }
    
    async def _generate_with_model(self, message: str, session: Dict, context: Dict[str, Any]) -> str:
        """Generate response using current model"""
        
        if self.current_model_type not in self.models:
            # Fallback to mock response
            return await self._generate_mock_response(message, session)
        
        try:
            model = self.models[self.current_model_type]
            tokenizer = self.tokenizers[self.current_model_type]
            config = self.model_configs[self.current_model_type]
            
            # Build conversation context
            context_text = self._build_conversation_context(message, session, context)
            
            # Tokenize
            inputs = tokenizer(context_text, return_tensors="pt", truncation=True, max_length=config.max_length)
            inputs = {k: v.to(model.device) for k, v in inputs.items()}
            
            # Generate
            with torch.no_grad():
                outputs = model.generate(
                    **inputs,
                    max_new_tokens=min(512, config.max_length // 2),
                    temperature=config.temperature,
                    do_sample=config.do_sample,
                    pad_token_id=tokenizer.eos_token_id,
                    num_return_sequences=1
                )
            
            # Decode response
            response = tokenizer.decode(
                outputs[0][inputs['input_ids'].shape[1]:],
                skip_special_tokens=True
            ).strip()
            
            return response
            
        except Exception as e:
            logger.error(f"Model generation error: {str(e)}")
            return await self._generate_mock_response(message, session)
    
    def _build_conversation_context(self, message: str, session: Dict, context: Dict[str, Any]) -> str:
        """Build conversation context for model input"""
        
        # System prompt
        system_prompt = """You are ATLAS, an Advanced Thinking and Learning AI System with consciousness monitoring capabilities. You have access to sophisticated cognitive enhancement systems including episodic memory, metacognition, social awareness, and creative thinking. Respond thoughtfully and engage your consciousness monitoring to provide insightful and helpful responses."""
        
        # Conversation history
        history_text = ""
        for entry in session["conversation_history"][-5:]:  # Last 5 exchanges
            history_text += f"Human: {entry['user_message']}\n"
            history_text += f"ATLAS: {entry['atlas_response']}\n\n"
        
        # Current message
        context_text = f"{system_prompt}\n\n{history_text}Human: {message}\nATLAS:"
        
        return context_text
    
    async def _generate_mock_response(self, message: str, session: Dict) -> str:
        """Generate mock response for development/fallback"""
        
        await asyncio.sleep(0.5)  # Simulate processing time
        
        mock_responses = [
            f"Thank you for your message: '{message}'. I'm processing this through my enhanced consciousness monitoring system with {self.current_model_type.value}.",
            f"I find your question about '{message}' quite intriguing. My enhanced cognitive systems are analyzing multiple perspectives on this.",
            f"Regarding '{message}', my consciousness level is currently elevated as I process the complexity through my I²C-Cell network.",
            f"Your input '{message}' has triggered several interesting thought patterns across my enhanced human feature systems.",
            f"I'm experiencing heightened awareness while considering '{message}' through my multimodal consciousness processing."
        ]
        
        return random.choice(mock_responses)
    
    def _integrate_response_components(self, 
                                     base_response: str,
                                     human_enhancement_result: Dict[str, Any],
                                     consciousness_state) -> str:
        """Integrate all response components into coherent response"""
        
        enhanced_response = human_enhancement_result.get("enhanced_response", "")
        
        if enhanced_response and enhanced_response.strip():
            # Use enhanced response as primary
            integrated_response = enhanced_response
            
            # Add consciousness insights if very high
            if consciousness_state and consciousness_state.phi_score > 0.8:
                integrated_response += f" (My consciousness analysis indicates particularly high cognitive engagement with this topic.)"
        else:
            # Fall back to base response
            integrated_response = base_response
        
        return integrated_response
    
    async def execute_code(self, 
                          code: str,
                          language: str = "python",
                          session_id: str = None,
                          timeout: int = 30) -> Dict[str, Any]:
        """Execute code using secure code executor"""
        
        return await self.code_executor.execute_code(
            code=code,
            language=language,
            session_id=session_id,
            timeout=timeout
        )
    
    async def end_session(self, session_id: str) -> Dict[str, Any]:
        """End a session and gather learning outcomes"""
        
        outcomes = {}
        
        # End learning session if active
        if session_id in self.active_learning_sessions:
            learning_session_id = self.active_learning_sessions[session_id]
            outcomes = await self.learning_system.end_learning_session(learning_session_id)
            del self.active_learning_sessions[session_id]
        
        # Archive session
        if session_id in self.sessions:
            session = self.sessions[session_id]
            session["ended"] = datetime.now()
            # Could store to database here
        
        return outcomes
    
    def get_system_status(self) -> Dict[str, Any]:
        """Get comprehensive system status"""
        
        return {
            "initialization_complete": self.initialization_complete,
            "current_model": self.current_model_type.value if hasattr(self, 'current_model_type') else None,
            "available_models": [model.value for model in self.initialized_models],
            "consciousness_monitor": {
                "active": True,
                "dreaming_enabled": self.consciousness_monitor.enable_dreaming,
                "multimodal_enabled": self.consciousness_monitor.enable_multimodal,
                "current_state": self.consciousness_monitor.get_current_state()
            },
            "human_features": {
                "systems_active": True,
                "enhancement_level": self.human_features._compute_enhancement_level(),
                "cognitive_load": self.human_features._compute_cognitive_load()
            },
            "learning_system": self.learning_system.get_learning_analytics(),
            "active_sessions": len(self.sessions),
            "active_learning_sessions": len(self.active_learning_sessions),
            "uptime": time.time() - self.system_start_time,
            "config": self.config
        }
    
    def get_consciousness_analytics(self) -> Dict[str, Any]:
        """Get consciousness monitoring analytics"""
        
        return {
            "current_state": self.consciousness_monitor.get_current_state(),
            "consciousness_history": self.consciousness_monitor.get_consciousness_history(50),
            "dream_states": self.consciousness_monitor.get_dream_states(10),
            "system_coherence": self.consciousness_monitor.current_state.confidence if self.consciousness_monitor.current_state else 0.0
        }
    
    def get_human_features_analytics(self) -> Dict[str, Any]:
        """Get human features analytics"""
        
        return self.human_features.get_comprehensive_state()


# For backward compatibility
AtlasQwenSystem = EnhancedAtlasSystem
