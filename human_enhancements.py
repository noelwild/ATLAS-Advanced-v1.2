"""
ATLAS Human Cognitive Enhancements
Mathematical formulations and implementations for making ATLAS more human-like
"""

import torch
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass
from enum import Enum
import asyncio
from collections import deque, defaultdict
import math


# =============================================================================
# 1. EMOTIONAL INTELLIGENCE & AFFECT SYSTEM
# =============================================================================

class EmotionType(Enum):
    CURIOSITY = "curiosity"
    FRUSTRATION = "frustration"
    EXCITEMENT = "excitement"
    CONCERN = "concern"
    SATISFACTION = "satisfaction"
    CONFUSION = "confusion"
    CONFIDENCE = "confidence"
    EMPATHY = "empathy"

@dataclass
class EmotionalState:
    """Represents current emotional state"""
    emotions: Dict[EmotionType, float]  # Emotion intensities [0, 1]
    arousal: float  # Overall emotional arousal [0, 1]
    valence: float  # Positive/negative emotion [-1, 1]
    timestamp: float

class EmotionalIntelligenceSystem:
    """
    Emotional intelligence system that tracks and influences cognition
    
    Mathematical Model:
    - Emotion intensity: e_i(t) ∈ [0, 1] for emotion type i at time t
    - Arousal: A(t) = √(Σ e_i²(t)) / √n where n = number of emotions
    - Valence: V(t) = Σ (w_i * e_i(t)) where w_i is valence weight of emotion i
    - Emotion decay: e_i(t+Δt) = e_i(t) * exp(-λ * Δt) where λ is decay rate
    - Emotion influence on consciousness: Φ'(t) = Φ(t) * (1 + α * A(t))
    """
    
    def __init__(self, decay_rate: float = 0.1, influence_strength: float = 0.2):
        self.decay_rate = decay_rate  # λ in decay formula
        self.influence_strength = influence_strength  # α in consciousness influence
        
        # Valence weights for each emotion type
        self.valence_weights = {
            EmotionType.CURIOSITY: 0.3,
            EmotionType.FRUSTRATION: -0.8,
            EmotionType.EXCITEMENT: 0.9,
            EmotionType.CONCERN: -0.4,
            EmotionType.SATISFACTION: 0.7,
            EmotionType.CONFUSION: -0.3,
            EmotionType.CONFIDENCE: 0.6,
            EmotionType.EMPATHY: 0.4
        }
        
        # Current emotional state
        self.current_state = EmotionalState(
            emotions={emotion: 0.0 for emotion in EmotionType},
            arousal=0.0,
            valence=0.0,
            timestamp=time.time()
        )
        
        # Emotion history for pattern analysis
        self.emotion_history = deque(maxlen=1000)
    
    def trigger_emotion(self, emotion: EmotionType, intensity: float, trigger_context: str = ""):
        """
        Trigger an emotion with given intensity
        
        Formula: e_i(t) = min(1, e_i(t-1) + intensity)
        """
        current_time = time.time()
        
        # Decay existing emotions
        self._decay_emotions(current_time)
        
        # Add new emotion intensity
        current_intensity = self.current_state.emotions[emotion]
        new_intensity = min(1.0, current_intensity + intensity)
        self.current_state.emotions[emotion] = new_intensity
        
        # Update arousal and valence
        self._update_arousal_valence()
        self.current_state.timestamp = current_time
        
        # Store in history
        self.emotion_history.append({
            'timestamp': current_time,
            'emotion': emotion.value,
            'intensity': new_intensity,
            'trigger_context': trigger_context,
            'arousal': self.current_state.arousal,
            'valence': self.current_state.valence
        })
    
    def _decay_emotions(self, current_time: float):
        """Apply exponential decay to all emotions"""
        dt = current_time - self.current_state.timestamp
        decay_factor = math.exp(-self.decay_rate * dt)
        
        for emotion in EmotionType:
            self.current_state.emotions[emotion] *= decay_factor
    
    def _update_arousal_valence(self):
        """Calculate arousal and valence from current emotions"""
        emotions = list(self.current_state.emotions.values())
        
        # Arousal: RMS of emotion intensities
        self.current_state.arousal = math.sqrt(sum(e**2 for e in emotions) / len(emotions))
        
        # Valence: weighted sum of emotions
        self.current_state.valence = sum(
            self.valence_weights[emotion] * intensity
            for emotion, intensity in self.current_state.emotions.items()
        )
    
    def influence_consciousness(self, phi_score: float) -> float:
        """
        Modify consciousness score based on emotional state
        
        Formula: Φ'(t) = Φ(t) * (1 + α * A(t))
        """
        self._decay_emotions(time.time())
        return phi_score * (1 + self.influence_strength * self.current_state.arousal)
    
    def influence_memory_encoding(self, base_strength: float) -> float:
        """
        Modify memory encoding strength based on emotional state
        
        Formula: strength' = strength * (1 + β * (|V(t)| + A(t)))
        where β is emotional memory coefficient
        """
        beta = 0.3  # Emotional memory coefficient
        emotion_factor = abs(self.current_state.valence) + self.current_state.arousal
        return base_strength * (1 + beta * emotion_factor)
    
    def get_dominant_emotion(self) -> Tuple[EmotionType, float]:
        """Get the currently dominant emotion"""
        self._decay_emotions(time.time())
        max_emotion = max(self.current_state.emotions.items(), key=lambda x: x[1])
        return max_emotion
    
    def generate_emotional_tags(self) -> List[str]:
        """Generate emotional tags for the current state"""
        self._decay_emotions(time.time())
        
        tags = []
        for emotion, intensity in self.current_state.emotions.items():
            if intensity > 0.3:  # Threshold for significant emotions
                tags.append(f"<emotion type=\"{emotion.value}\" intensity=\"{intensity:.2f}\"/>")
        
        if self.current_state.arousal > 0.5:
            tags.append(f"<arousal level=\"{self.current_state.arousal:.2f}\"/>")
        
        if abs(self.current_state.valence) > 0.3:
            valence_label = "positive" if self.current_state.valence > 0 else "negative"
            tags.append(f"<valence type=\"{valence_label}\" value=\"{self.current_state.valence:.2f}\"/>")
        
        return tags


# =============================================================================
# 2. EPISODIC MEMORY WITH TEMPORAL CONTEXT
# =============================================================================

@dataclass
class EpisodicMemory:
    """Represents an autobiographical memory"""
    event_id: str
    content: str
    timestamp: float
    context: Dict[str, Any]  # Who, where, what, why
    emotional_state: EmotionalState
    associated_memories: List[str]  # Related memory IDs
    importance_score: float
    access_count: int
    last_accessed: float

class EpisodicMemorySystem:
    """
    Temporal autobiographical memory system
    
    Mathematical Model:
    - Memory strength: S(t) = S₀ * exp(-δ * (t - t₀)) * (1 + α * R)
      where δ is decay rate, R is number of retrievals, α is rehearsal bonus
    - Temporal clustering: τ(m₁, m₂) = exp(-|t₁ - t₂| / σ)
      where σ is temporal clustering parameter
    - Importance: I = w₁ * E + w₂ * N + w₃ * C
      where E is emotional intensity, N is novelty, C is centrality
    """
    
    def __init__(self, decay_rate: float = 0.001, temporal_clustering_sigma: float = 3600):
        self.decay_rate = decay_rate  # δ in strength formula
        self.temporal_clustering_sigma = temporal_clustering_sigma  # σ in clustering
        self.rehearsal_bonus = 0.1  # α in strength formula
        
        # Memory storage
        self.memories: Dict[str, EpisodicMemory] = {}
        self.temporal_index: Dict[int, List[str]] = defaultdict(list)  # day -> memory_ids
        
        # Importance weights
        self.importance_weights = {
            'emotional': 0.4,
            'novelty': 0.3,
            'centrality': 0.3
        }
    
    async def encode_episodic_memory(
        self,
        content: str,
        context: Dict[str, Any],
        emotional_state: EmotionalState,
        novelty_score: float = 0.5
    ) -> str:
        """
        Encode a new episodic memory
        
        Formula for importance: I = w₁ * E + w₂ * N + w₃ * C
        """
        current_time = time.time()
        event_id = f"episode_{int(current_time)}_{hash(content) % 10000}"
        
        # Calculate importance score
        emotional_intensity = emotional_state.arousal
        centrality_score = await self._calculate_centrality(content, context)
        
        importance = (
            self.importance_weights['emotional'] * emotional_intensity +
            self.importance_weights['novelty'] * novelty_score +
            self.importance_weights['centrality'] * centrality_score
        )
        
        # Find associated memories through temporal and semantic similarity
        associated_memories = await self._find_associated_memories(content, current_time)
        
        # Create memory
        memory = EpisodicMemory(
            event_id=event_id,
            content=content,
            timestamp=current_time,
            context=context,
            emotional_state=emotional_state,
            associated_memories=associated_memories,
            importance_score=importance,
            access_count=0,
            last_accessed=current_time
        )
        
        # Store memory
        self.memories[event_id] = memory
        day_key = int(current_time // 86400)  # Day since epoch
        self.temporal_index[day_key].append(event_id)
        
        return event_id
    
    async def retrieve_episodic_memories(
        self,
        query: str,
        time_range: Optional[Tuple[float, float]] = None,
        max_memories: int = 5
    ) -> List[EpisodicMemory]:
        """
        Retrieve episodic memories based on query and time constraints
        
        Formula for retrieval strength: R = S(t) * sim(query, content) * τ(t_query, t_memory)
        """
        current_time = time.time()
        candidates = []
        
        # Filter by time range if specified
        memory_pool = self.memories.values()
        if time_range:
            start_time, end_time = time_range
            memory_pool = [m for m in memory_pool if start_time <= m.timestamp <= end_time]
        
        for memory in memory_pool:
            # Calculate memory strength with decay and rehearsal
            time_diff = current_time - memory.timestamp
            strength = math.exp(-self.decay_rate * time_diff) * (1 + self.rehearsal_bonus * memory.access_count)
            
            # Calculate semantic similarity (simplified)
            semantic_sim = self._calculate_semantic_similarity(query, memory.content)
            
            # Calculate temporal clustering bonus
            temporal_bonus = 1.0  # Could be enhanced with query temporal context
            
            # Final retrieval score
            retrieval_score = strength * semantic_sim * temporal_bonus
            
            candidates.append((memory, retrieval_score))
        
        # Sort by retrieval score and return top memories
        candidates.sort(key=lambda x: x[1], reverse=True)
        retrieved_memories = [memory for memory, score in candidates[:max_memories]]
        
        # Update access counts
        for memory in retrieved_memories:
            memory.access_count += 1
            memory.last_accessed = current_time
        
        return retrieved_memories
    
    def get_autobiographical_timeline(self, days_back: int = 30) -> List[Dict[str, Any]]:
        """Get chronological timeline of recent memories"""
        current_day = int(time.time() // 86400)
        timeline = []
        
        for day_offset in range(days_back):
            day_key = current_day - day_offset
            if day_key in self.temporal_index:
                day_memories = [self.memories[mid] for mid in self.temporal_index[day_key]]
                day_memories.sort(key=lambda m: m.timestamp)
                
                timeline.append({
                    'date': day_key * 86400,
                    'memories': [
                        {
                            'content': m.content[:100] + "..." if len(m.content) > 100 else m.content,
                            'importance': m.importance_score,
                            'emotional_valence': m.emotional_state.valence,
                            'context': m.context
                        }
                        for m in day_memories
                    ]
                })
        
        return timeline
    
    async def _find_associated_memories(self, content: str, timestamp: float) -> List[str]:
        """Find memories associated through temporal and semantic similarity"""
        associated = []
        
        for memory_id, memory in self.memories.items():
            # Temporal clustering
            time_diff = abs(timestamp - memory.timestamp)
            temporal_similarity = math.exp(-time_diff / self.temporal_clustering_sigma)
            
            # Semantic similarity
            semantic_similarity = self._calculate_semantic_similarity(content, memory.content)
            
            # Combined association strength
            association_strength = 0.6 * temporal_similarity + 0.4 * semantic_similarity
            
            if association_strength > 0.3:  # Threshold for association
                associated.append(memory_id)
        
        return associated[:5]  # Limit associations
    
    async def _calculate_centrality(self, content: str, context: Dict[str, Any]) -> float:
        """Calculate how central this memory is to the agent's experience"""
        # Simplified centrality based on context richness and content complexity
        context_richness = len(context) / 10.0  # Normalize by expected max context items
        content_complexity = len(content.split()) / 100.0  # Normalize by expected max words
        
        return min(1.0, (context_richness + content_complexity) / 2.0)
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Simplified semantic similarity calculation"""
        # In a real implementation, this would use embeddings
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0


# =============================================================================
# 3. ADVANCED METACOGNITION
# =============================================================================

class MetaCognitionType(Enum):
    CONFIDENCE = "confidence"
    UNCERTAINTY = "uncertainty"
    STRATEGY = "strategy"
    SELF_ASSESSMENT = "self_assessment"
    COGNITIVE_LOAD = "cognitive_load"
    ATTENTION_FOCUS = "attention_focus"
    LEARNING_STATE = "learning_state"

@dataclass
class MetaCognitiveState:
    """Represents metacognitive awareness"""
    confidence_level: float  # [0, 1]
    uncertainty_areas: List[str]
    current_strategy: str
    cognitive_load: float  # [0, 1]
    attention_focus: Dict[str, float]  # topic -> attention weight
    self_assessment: Dict[str, float]  # capability -> assessment
    timestamp: float

class MetaCognitionSystem:
    """
    Advanced metacognitive monitoring and control
    
    Mathematical Model:
    - Confidence: C(t) = 1 / (1 + exp(-k * (accuracy - threshold)))
      where k is steepness, accuracy is recent performance
    - Cognitive Load: L(t) = Σ w_i * task_complexity_i / max_capacity
    - Strategy Selection: S* = argmax_s (expected_utility(s) * confidence(s))
    - Uncertainty: U(t) = 1 - C(t) + entropy(belief_distribution)
    """
    
    def __init__(self):
        self.confidence_threshold = 0.7  # Threshold for sigmoid
        self.confidence_steepness = 5.0  # k in confidence formula
        self.max_cognitive_capacity = 10.0  # Maximum cognitive load
        
        # Current metacognitive state
        self.current_state = MetaCognitiveState(
            confidence_level=0.5,
            uncertainty_areas=[],
            current_strategy="analytical",
            cognitive_load=0.0,
            attention_focus={},
            self_assessment={},
            timestamp=time.time()
        )
        
        # Performance tracking for confidence calculation
        self.performance_history = deque(maxlen=50)
        self.strategy_performance = defaultdict(list)
        
        # Available strategies
        self.strategies = [
            "analytical", "intuitive", "creative", "systematic", 
            "exploratory", "focused", "collaborative"
        ]
    
    def update_confidence(self, task_accuracy: float):
        """
        Update confidence based on recent performance
        
        Formula: C(t) = 1 / (1 + exp(-k * (accuracy - threshold)))
        """
        self.performance_history.append(task_accuracy)
        
        if len(self.performance_history) > 0:
            recent_accuracy = np.mean(list(self.performance_history)[-10:])  # Last 10 tasks
            confidence = 1 / (1 + math.exp(-self.confidence_steepness * (recent_accuracy - self.confidence_threshold)))
            self.current_state.confidence_level = confidence
    
    def calculate_cognitive_load(self, active_tasks: List[Dict[str, Any]]) -> float:
        """
        Calculate current cognitive load
        
        Formula: L(t) = Σ w_i * complexity_i / max_capacity
        """
        total_load = 0.0
        
        for task in active_tasks:
            complexity = task.get('complexity', 1.0)
            attention_weight = task.get('attention_weight', 1.0)
            total_load += attention_weight * complexity
        
        cognitive_load = min(1.0, total_load / self.max_cognitive_capacity)
        self.current_state.cognitive_load = cognitive_load
        
        return cognitive_load
    
    def select_strategy(self, task_context: Dict[str, Any]) -> str:
        """
        Select optimal strategy based on context and past performance
        
        Formula: S* = argmax_s (expected_utility(s) * confidence(s))
        """
        strategy_scores = {}
        
        for strategy in self.strategies:
            # Calculate expected utility based on past performance
            if strategy in self.strategy_performance:
                performance_history = self.strategy_performance[strategy]
                expected_utility = np.mean(performance_history) if performance_history else 0.5
            else:
                expected_utility = 0.5  # Default for untested strategies
            
            # Calculate strategy confidence based on how often it's been successful
            strategy_confidence = self._calculate_strategy_confidence(strategy)
            
            # Context bonus (simplified)
            context_bonus = self._calculate_context_bonus(strategy, task_context)
            
            # Final score
            strategy_scores[strategy] = expected_utility * strategy_confidence + context_bonus
        
        # Select best strategy
        best_strategy = max(strategy_scores.items(), key=lambda x: x[1])[0]
        self.current_state.current_strategy = best_strategy
        
        return best_strategy
    
    def assess_uncertainty(self, belief_distribution: Dict[str, float]) -> Tuple[float, List[str]]:
        """
        Assess uncertainty in current beliefs
        
        Formula: U(t) = 1 - C(t) + entropy(belief_distribution)
        """
        # Calculate entropy of belief distribution
        entropy = 0.0
        for belief, probability in belief_distribution.items():
            if probability > 0:
                entropy -= probability * math.log(probability)
        
        # Normalize entropy
        max_entropy = math.log(len(belief_distribution)) if belief_distribution else 1.0
        normalized_entropy = entropy / max_entropy if max_entropy > 0 else 0.0
        
        # Total uncertainty
        uncertainty = (1 - self.current_state.confidence_level) + normalized_entropy
        uncertainty = min(1.0, uncertainty)  # Cap at 1.0
        
        # Identify specific uncertainty areas
        uncertainty_areas = [
            belief for belief, prob in belief_distribution.items()
            if prob < 0.6  # Low confidence in belief
        ]
        
        self.current_state.uncertainty_areas = uncertainty_areas
        
        return uncertainty, uncertainty_areas
    
    def perform_self_assessment(self, capabilities: List[str]) -> Dict[str, float]:
        """
        Assess own capabilities across different domains
        """
        assessment = {}
        
        for capability in capabilities:
            # Base assessment on performance history for this capability
            if capability in self.strategy_performance:
                recent_performance = self.strategy_performance[capability][-10:]  # Last 10
                if recent_performance:
                    assessment[capability] = np.mean(recent_performance)
                else:
                    assessment[capability] = 0.5  # Default
            else:
                assessment[capability] = 0.5  # Default for untested
        
        self.current_state.self_assessment = assessment
        return assessment
    
    def generate_metacognitive_tags(self) -> List[str]:
        """Generate metacognitive tags for the current state"""
        tags = []
        
        # Confidence tag
        confidence_level = "high" if self.current_state.confidence_level > 0.7 else "low" if self.current_state.confidence_level < 0.3 else "medium"
        tags.append(f"<confidence level=\"{confidence_level}\" value=\"{self.current_state.confidence_level:.2f}\"/>")
        
        # Strategy tag
        tags.append(f"<strategy current=\"{self.current_state.current_strategy}\"/>")
        
        # Cognitive load tag
        load_level = "high" if self.current_state.cognitive_load > 0.7 else "low" if self.current_state.cognitive_load < 0.3 else "medium"
        tags.append(f"<cognitive_load level=\"{load_level}\" value=\"{self.current_state.cognitive_load:.2f}\"/>")
        
        # Uncertainty areas
        if self.current_state.uncertainty_areas:
            uncertainty_list = ", ".join(self.current_state.uncertainty_areas[:3])  # Top 3
            tags.append(f"<uncertainty areas=\"{uncertainty_list}\"/>")
        
        # Self-assessment for top capabilities
        if self.current_state.self_assessment:
            top_capabilities = sorted(self.current_state.self_assessment.items(), key=lambda x: x[1], reverse=True)[:2]
            for capability, score in top_capabilities:
                tags.append(f"<self_assessment capability=\"{capability}\" score=\"{score:.2f}\"/>")
        
        return tags
    
    def _calculate_strategy_confidence(self, strategy: str) -> float:
        """Calculate confidence in a specific strategy"""
        if strategy not in self.strategy_performance:
            return 0.5  # Default confidence
        
        performances = self.strategy_performance[strategy]
        if not performances:
            return 0.5
        
        # Confidence based on consistency and success rate
        success_rate = np.mean(performances)
        consistency = 1.0 - np.std(performances) if len(performances) > 1 else 1.0
        
        return (success_rate + consistency) / 2.0
    
    def _calculate_context_bonus(self, strategy: str, context: Dict[str, Any]) -> float:
        """Calculate context-specific bonus for strategy selection"""
        # Simplified context matching
        context_bonuses = {
            "analytical": 0.2 if context.get("requires_logic", False) else 0.0,
            "creative": 0.2 if context.get("requires_creativity", False) else 0.0,
            "systematic": 0.2 if context.get("complex_task", False) else 0.0,
            "intuitive": 0.2 if context.get("time_pressure", False) else 0.0,
        }
        
        return context_bonuses.get(strategy, 0.0)


# =============================================================================
# 4. INTUITION & GUT FEELINGS SYSTEM
# =============================================================================

@dataclass
class IntuitiveFeedback:
    """Represents an intuitive feeling or hunch"""
    feeling_type: str  # "good", "bad", "uncertain", "promising"
    intensity: float  # [0, 1]
    confidence: float  # [0, 1]
    source_patterns: List[str]  # What patterns triggered this
    timestamp: float

class IntuitionSystem:
    """
    Subconscious pattern recognition and gut feelings
    
    Mathematical Model:
    - Pattern activation: a_i(t) = Σ w_ij * s_j(t) where s_j are input signals
    - Intuitive strength: I(t) = tanh(Σ a_i(t) - threshold)
    - Gut feeling: G(t) = sigmoid(I(t) * confidence(patterns))
    - Pattern learning: w_ij(t+1) = w_ij(t) + η * δ * s_j(t)
      where η is learning rate, δ is prediction error
    """
    
    def __init__(self, pattern_threshold: float = 0.5, learning_rate: float = 0.01):
        self.pattern_threshold = pattern_threshold
        self.learning_rate = learning_rate
        
        # Pattern recognition network (simplified)
        self.pattern_weights = defaultdict(lambda: defaultdict(float))
        self.pattern_activations = defaultdict(float)
        self.pattern_history = deque(maxlen=1000)
        
        # Current intuitive state
        self.current_feedback = None
        self.gut_feeling_history = deque(maxlen=100)
        
        # Pattern categories that trigger intuition
        self.pattern_categories = [
            "semantic_coherence", "emotional_resonance", "logical_consistency",
            "novelty_detection", "risk_assessment", "opportunity_recognition",
            "social_dynamics", "temporal_patterns"
        ]
    
    def process_input_signals(self, signals: Dict[str, float]) -> IntuitiveFeedback:
        """
        Process input signals and generate intuitive feedback
        
        Formula: a_i(t) = Σ w_ij * s_j(t)
        """
        current_time = time.time()
        
        # Calculate pattern activations
        pattern_activations = {}
        for pattern in self.pattern_categories:
            activation = 0.0
            for signal_name, signal_value in signals.items():
                weight = self.pattern_weights[pattern][signal_name]
                activation += weight * signal_value
            pattern_activations[pattern] = activation
        
        # Calculate overall intuitive strength
        total_activation = sum(pattern_activations.values())
        intuitive_strength = math.tanh(total_activation - self.pattern_threshold)
        
        # Determine feeling type and intensity
        feeling_type, intensity = self._determine_feeling_type(pattern_activations, intuitive_strength)
        
        # Calculate confidence based on pattern consistency
        confidence = self._calculate_intuitive_confidence(pattern_activations)
        
        # Identify source patterns
        source_patterns = [
            pattern for pattern, activation in pattern_activations.items()
            if abs(activation) > 0.3
        ]
        
        # Create intuitive feedback
        feedback = IntuitiveFeedback(
            feeling_type=feeling_type,
            intensity=abs(intensity),
            confidence=confidence,
            source_patterns=source_patterns,
            timestamp=current_time
        )
        
        self.current_feedback = feedback
        self.gut_feeling_history.append(feedback)
        
        # Store pattern history for learning
        self.pattern_history.append({
            'timestamp': current_time,
            'signals': signals.copy(),
            'activations': pattern_activations.copy(),
            'feedback': feedback
        })
        
        return feedback
    
    def learn_from_outcome(self, actual_outcome: float, predicted_feeling: str):
        """
        Update pattern weights based on actual outcomes
        
        Formula: w_ij(t+1) = w_ij(t) + η * δ * s_j(t)
        """
        if not self.pattern_history:
            return
        
        # Get the most recent pattern activation
        recent_pattern = self.pattern_history[-1]
        
        # Calculate prediction error
        predicted_value = self._feeling_to_value(predicted_feeling)
        prediction_error = actual_outcome - predicted_value
        
        # Update weights
        for pattern in self.pattern_categories:
            for signal_name, signal_value in recent_pattern['signals'].items():
                current_weight = self.pattern_weights[pattern][signal_name]
                weight_update = self.learning_rate * prediction_error * signal_value
                self.pattern_weights[pattern][signal_name] = current_weight + weight_update
    
    def get_gut_feeling_about(self, situation: Dict[str, Any]) -> str:
        """
        Generate a gut feeling description about a specific situation
        """
        if not self.current_feedback:
            return "No strong intuitive feeling"
        
        feedback = self.current_feedback
        
        # Generate descriptive gut feeling
        if feedback.intensity > 0.7:
            intensity_desc = "strong"
        elif feedback.intensity > 0.4:
            intensity_desc = "moderate"
        else:
            intensity_desc = "weak"
        
        confidence_desc = "confident" if feedback.confidence > 0.6 else "uncertain"
        
        return f"I have a {intensity_desc} {feedback.feeling_type} feeling about this, and I'm {confidence_desc} in this intuition."
    
    def generate_intuitive_tags(self) -> List[str]:
        """Generate tags representing current intuitive state"""
        if not self.current_feedback:
            return []
        
        feedback = self.current_feedback
        tags = []
        
        # Main intuitive feeling tag
        tags.append(f"<intuition type=\"{feedback.feeling_type}\" intensity=\"{feedback.intensity:.2f}\" confidence=\"{feedback.confidence:.2f}\"/>")
        
        # Source patterns
        if feedback.source_patterns:
            patterns_str = ", ".join(feedback.source_patterns[:3])  # Top 3
            tags.append(f"<gut_feeling_sources patterns=\"{patterns_str}\"/>")
        
        # Gut feeling strength
        if feedback.intensity > 0.5:
            tags.append(f"<strong_hunch direction=\"{feedback.feeling_type}\"/>")
        
        return tags
    
    def _determine_feeling_type(self, activations: Dict[str, float], strength: float) -> Tuple[str, float]:
        """Determine the type and intensity of feeling"""
        # Analyze pattern activations to determine feeling type
        risk_patterns = ["risk_assessment", "logical_consistency"]
        opportunity_patterns = ["opportunity_recognition", "novelty_detection"]
        social_patterns = ["social_dynamics", "emotional_resonance"]
        
        risk_activation = sum(activations.get(p, 0) for p in risk_patterns)
        opportunity_activation = sum(activations.get(p, 0) for p in opportunity_patterns)
        social_activation = sum(activations.get(p, 0) for p in social_patterns)
        
        # Determine feeling type based on dominant pattern
        if risk_activation < -0.5:
            return "bad", strength
        elif opportunity_activation > 0.5:
            return "promising", strength
        elif social_activation > 0.3:
            return "empathetic", strength
        elif abs(strength) < 0.2:
            return "uncertain", abs(strength)
        else:
            return "good" if strength > 0 else "concerning", abs(strength)
    
    def _calculate_intuitive_confidence(self, activations: Dict[str, float]) -> float:
        """Calculate confidence in the intuitive assessment"""
        # Confidence based on consistency of patterns
        activation_values = list(activations.values())
        
        if not activation_values:
            return 0.0
        
        # High confidence when patterns agree (low variance)
        mean_activation = np.mean(activation_values)
        variance = np.var(activation_values)
        
        # Convert variance to confidence (lower variance = higher confidence)
        confidence = 1.0 / (1.0 + variance)
        
        return min(1.0, confidence)
    
    def _feeling_to_value(self, feeling: str) -> float:
        """Convert feeling type to numerical value"""
        feeling_values = {
            "good": 0.7,
            "promising": 0.8,
            "empathetic": 0.6,
            "uncertain": 0.5,
            "concerning": 0.3,
            "bad": 0.2
        }
        return feeling_values.get(feeling, 0.5)


# =============================================================================
# INTEGRATION CLASS FOR ALL HUMAN-LIKE ENHANCEMENTS
# =============================================================================

class HumanLikeEnhancementSystem:
    """
    Integrated system combining all human-like cognitive enhancements
    """
    
    def __init__(self):
        self.emotional_system = EmotionalIntelligenceSystem()
        self.episodic_memory = EpisodicMemorySystem()
        self.metacognition = MetaCognitionSystem()
        self.intuition = IntuitionSystem()
        
        # Integration parameters
        self.enhancement_weights = {
            'emotional': 0.3,
            'episodic': 0.2,
            'metacognitive': 0.3,
            'intuitive': 0.2
        }
    
    async def process_input(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """
        Process input through all enhancement systems
        """
        current_time = time.time()
        
        # Extract signals for different systems
        signals = self._extract_signals(input_text, context)
        
        # Process through each system
        results = {}
        
        # Emotional processing
        if signals.get('emotional_trigger'):
            emotion_type = EmotionType(signals['emotional_trigger'])
            intensity = signals.get('emotional_intensity', 0.5)
            self.emotional_system.trigger_emotion(emotion_type, intensity, input_text)
        
        results['emotional_state'] = self.emotional_system.current_state
        results['emotional_tags'] = self.emotional_system.generate_emotional_tags()
        
        # Episodic memory processing
        if signals.get('memorable_event', False):
            memory_id = await self.episodic_memory.encode_episodic_memory(
                content=input_text,
                context=context,
                emotional_state=self.emotional_system.current_state,
                novelty_score=signals.get('novelty_score', 0.5)
            )
            results['memory_encoded'] = memory_id
        
        # Metacognitive processing
        task_complexity = signals.get('task_complexity', 1.0)
        active_tasks = [{'complexity': task_complexity, 'attention_weight': 1.0}]
        cognitive_load = self.metacognition.calculate_cognitive_load(active_tasks)
        
        strategy = self.metacognition.select_strategy(context)
        results['metacognitive_state'] = self.metacognition.current_state
        results['metacognitive_tags'] = self.metacognition.generate_metacognitive_tags()
        results['selected_strategy'] = strategy
        
        # Intuitive processing
        intuitive_signals = {
            'semantic_coherence': signals.get('coherence', 0.5),
            'emotional_resonance': self.emotional_system.current_state.arousal,
            'logical_consistency': signals.get('logic_score', 0.5),
            'novelty_detection': signals.get('novelty_score', 0.5)
        }
        
        intuitive_feedback = self.intuition.process_input_signals(intuitive_signals)
        results['intuitive_feedback'] = intuitive_feedback
        results['intuitive_tags'] = self.intuition.generate_intuitive_tags()
        
        # Generate integrated response
        results['integrated_tags'] = self._integrate_all_tags(results)
        results['human_like_response'] = await self._generate_human_like_response(input_text, results)
        
        return results
    
    def _extract_signals(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Extract relevant signals for processing"""
        signals = {}
        
        # Emotional signals
        emotional_keywords = {
            'excitement': ['amazing', 'wonderful', 'fantastic', 'exciting'],
            'frustration': ['difficult', 'impossible', 'frustrating', 'annoying'],
            'curiosity': ['why', 'how', 'what if', 'interesting', 'wonder'],
            'concern': ['worried', 'concerned', 'problem', 'issue', 'trouble']
        }
        
        for emotion, keywords in emotional_keywords.items():
            if any(keyword in input_text.lower() for keyword in keywords):
                signals['emotional_trigger'] = emotion
                signals['emotional_intensity'] = 0.6
                break
        
        # Task complexity (simplified)
        word_count = len(input_text.split())
        question_marks = input_text.count('?')
        signals['task_complexity'] = min(1.0, (word_count / 100.0) + (question_marks * 0.2))
        
        # Novelty detection (simplified)
        uncommon_words = ['quantum', 'consciousness', 'metaphysical', 'paradigm']
        novelty_score = sum(1 for word in uncommon_words if word in input_text.lower()) / len(uncommon_words)
        signals['novelty_score'] = novelty_score
        
        # Coherence (simplified)
        sentences = input_text.split('.')
        signals['coherence'] = 1.0 if len(sentences) > 1 else 0.7
        
        # Logic score (simplified)
        logical_indicators = ['because', 'therefore', 'since', 'thus', 'consequently']
        logic_score = sum(1 for indicator in logical_indicators if indicator in input_text.lower()) / len(logical_indicators)
        signals['logic_score'] = min(1.0, logic_score)
        
        # Memorable event detection
        signals['memorable_event'] = len(input_text) > 50 or '?' in input_text or signals.get('emotional_intensity', 0) > 0.5
        
        return signals
    
    def _integrate_all_tags(self, results: Dict[str, Any]) -> List[str]:
        """Integrate tags from all systems"""
        all_tags = []
        
        # Add tags from each system
        all_tags.extend(results.get('emotional_tags', []))
        all_tags.extend(results.get('metacognitive_tags', []))
        all_tags.extend(results.get('intuitive_tags', []))
        
        # Add integration-specific tags
        emotional_state = results.get('emotional_state')
        metacognitive_state = results.get('metacognitive_state')
        
        if emotional_state and metacognitive_state:
            # Cross-system integration tags
            if emotional_state.arousal > 0.6 and metacognitive_state.confidence_level < 0.4:
                all_tags.append("<emotional_uncertainty conflict=\"high_arousal_low_confidence\"/>")
            
            if emotional_state.valence > 0.5 and metacognitive_state.current_strategy == "creative":
                all_tags.append("<positive_creative_state synergy=\"true\"/>")
        
        return all_tags
    
    async def _generate_human_like_response(self, input_text: str, results: Dict[str, Any]) -> str:
        """Generate a human-like response incorporating all enhancements"""
        response_parts = []
        
        # Add emotional coloring
        emotional_state = results.get('emotional_state')
        if emotional_state:
            dominant_emotion, intensity = self.emotional_system.get_dominant_emotion()
            if intensity > 0.3:
                if dominant_emotion == EmotionType.CURIOSITY:
                    response_parts.append("This is quite intriguing!")
                elif dominant_emotion == EmotionType.EXCITEMENT:
                    response_parts.append("How exciting!")
                elif dominant_emotion == EmotionType.CONCERN:
                    response_parts.append("I'm a bit concerned about this.")
        
        # Add metacognitive awareness
        metacognitive_state = results.get('metacognitive_state')
        if metacognitive_state:
            if metacognitive_state.confidence_level < 0.4:
                response_parts.append("I'm not entirely certain about this.")
            elif metacognitive_state.current_strategy == "creative":
                response_parts.append("Let me think creatively about this.")
        
        # Add intuitive feedback
        intuitive_feedback = results.get('intuitive_feedback')
        if intuitive_feedback and intuitive_feedback.intensity > 0.5:
            gut_feeling = self.intuition.get_gut_feeling_about({})
            response_parts.append(f"My intuition tells me: {gut_feeling}")
        
        # Add episodic references
        if results.get('memory_encoded'):
            response_parts.append("This reminds me of something similar I encountered before.")
        
        # Combine into coherent response
        if response_parts:
            human_response = " ".join(response_parts)
        else:
            human_response = "Let me process this thoughtfully."
        
        return human_response


# Example usage and integration with ATLAS
if __name__ == "__main__":
    async def demo_human_enhancements():
        """Demonstrate the human-like enhancement system"""
        
        print("🧠 ATLAS Human-Like Enhancement System Demo")
        print("=" * 50)
        
        # Initialize the enhancement system
        enhancement_system = HumanLikeEnhancementSystem()
        
        # Test inputs
        test_inputs = [
            ("This is a fascinating problem about quantum consciousness!", {"requires_creativity": True}),
            ("I'm really frustrated with this difficult math problem.", {"task_complexity": 0.8}),
            ("How does photosynthesis work in plants?", {"requires_logic": True}),
            ("I wonder what it feels like to be truly conscious.", {"philosophical": True})
        ]
        
        for input_text, context in test_inputs:
            print(f"\n📝 Input: {input_text}")
            print(f"🔍 Context: {context}")
            
            # Process through enhancement system
            results = await enhancement_system.process_input(input_text, context)
            
            # Display results
            print(f"🤖 Human-like Response: {results['human_like_response']}")
            print(f"😊 Emotional State: {results['emotional_state'].emotions}")
            print(f"🧠 Strategy: {results['selected_strategy']}")
            
            if results['intuitive_feedback']:
                feedback = results['intuitive_feedback']
                print(f"💭 Intuition: {feedback.feeling_type} (intensity: {feedback.intensity:.2f})")
            
            print(f"🏷️  Enhanced Tags: {results['integrated_tags'][:3]}")  # Show first 3 tags
            print("-" * 30)
        
        print("\n✅ Demo completed!")
    
    # Run the demo
    asyncio.run(demo_human_enhancements())
