"""
Advanced Human-Like Features for ATLAS
Implementation of remaining enhancements: Social Cognition, Personality, Dreams, 
Moral Reasoning, Attention Management, and Temporal Reasoning
"""

import torch
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import deque, defaultdict
import math
import random


# =============================================================================
# 5. SOCIAL COGNITION & THEORY OF MIND
# =============================================================================

@dataclass
class UserProfile:
    """Represents understanding of a specific user"""
    user_id: str
    personality_traits: Dict[str, float]  # Big 5 + other traits
    communication_style: Dict[str, float]
    emotional_patterns: List[Dict[str, Any]]
    interests: Dict[str, float]  # Topic -> interest level
    expertise_levels: Dict[str, float]  # Domain -> expertise
    interaction_history: List[Dict[str, Any]]
    last_updated: float

@dataclass
class SocialContext:
    """Current social context information"""
    relationship_type: str  # "professional", "casual", "educational", etc.
    formality_level: float  # [0, 1]
    emotional_tone: str
    group_dynamics: Dict[str, Any]
    cultural_context: Dict[str, str]

class SocialCognitionSystem:
    """
    Theory of mind and social understanding system
    
    Mathematical Models:
    - Personality inference: P_i(t) = P_i(t-1) + α * (observed_behavior - expected_behavior)
    - Emotional contagion: E_self(t) = E_self(t-1) + β * (E_other(t) - E_self(t-1))
    - Trust modeling: T(t) = T(t-1) + γ * (outcome - prediction) * |prediction|
    - Social distance: D(u1, u2) = ||P1 - P2||_2 / √d where d is trait dimensions
    """
    
    def __init__(self):
        self.user_profiles: Dict[str, UserProfile] = {}
        self.current_social_context = SocialContext(
            relationship_type="unknown",
            formality_level=0.5,
            emotional_tone="neutral",
            group_dynamics={},
            cultural_context={}
        )
        
        # Learning parameters
        self.personality_learning_rate = 0.1  # α in personality inference
        self.emotional_contagion_rate = 0.2  # β in emotional contagion
        self.trust_learning_rate = 0.15  # γ in trust modeling
        
        # Big 5 personality traits
        self.personality_traits = [
            "openness", "conscientiousness", "extraversion", 
            "agreeableness", "neuroticism"
        ]
        
        # Communication style dimensions
        self.communication_dimensions = [
            "directness", "formality", "emotiveness", "verbosity", 
            "technical_preference", "humor_usage"
        ]
    
    async def analyze_user_behavior(self, user_id: str, interaction_data: Dict[str, Any]) -> UserProfile:
        """
        Analyze user behavior and update their profile
        
        Formula: P_i(t) = P_i(t-1) + α * (observed - expected)
        """
        current_time = time.time()
        
        # Get or create user profile
        if user_id not in self.user_profiles:
            self.user_profiles[user_id] = UserProfile(
                user_id=user_id,
                personality_traits={trait: 0.5 for trait in self.personality_traits},
                communication_style={dim: 0.5 for dim in self.communication_dimensions},
                emotional_patterns=[],
                interests={},
                expertise_levels={},
                interaction_history=[],
                last_updated=current_time
            )
        
        profile = self.user_profiles[user_id]
        
        # Extract behavioral signals
        behavioral_signals = self._extract_behavioral_signals(interaction_data)
        
        # Update personality traits
        for trait in self.personality_traits:
            if trait in behavioral_signals:
                observed_value = behavioral_signals[trait]
                expected_value = profile.personality_traits[trait]
                
                # Learning update
                update = self.personality_learning_rate * (observed_value - expected_value)
                profile.personality_traits[trait] = np.clip(
                    profile.personality_traits[trait] + update, 0.0, 1.0
                )
        
        # Update communication style
        for dim in self.communication_dimensions:
            if dim in behavioral_signals:
                observed_value = behavioral_signals[dim]
                expected_value = profile.communication_style[dim]
                
                update = self.personality_learning_rate * (observed_value - expected_value)
                profile.communication_style[dim] = np.clip(
                    profile.communication_style[dim] + update, 0.0, 1.0
                )
        
        # Update interests and expertise
        if 'topics' in interaction_data:
            for topic in interaction_data['topics']:
                # Interest inference from engagement
                engagement = interaction_data.get('engagement_level', 0.5)
                if topic in profile.interests:
                    profile.interests[topic] = 0.9 * profile.interests[topic] + 0.1 * engagement
                else:
                    profile.interests[topic] = engagement
                
                # Expertise inference from answer quality
                if 'answer_quality' in interaction_data:
                    quality = interaction_data['answer_quality']
                    if topic in profile.expertise_levels:
                        profile.expertise_levels[topic] = 0.8 * profile.expertise_levels[topic] + 0.2 * quality
                    else:
                        profile.expertise_levels[topic] = quality
        
        # Store interaction in history
        profile.interaction_history.append({
            'timestamp': current_time,
            'interaction_data': interaction_data,
            'behavioral_signals': behavioral_signals
        })
        
        # Keep only recent history
        if len(profile.interaction_history) > 100:
            profile.interaction_history = profile.interaction_history[-100:]
        
        profile.last_updated = current_time
        return profile
    
    def predict_user_response(self, user_id: str, message: str) -> Dict[str, Any]:
        """
        Predict how a user might respond based on their profile
        """
        if user_id not in self.user_profiles:
            return {"prediction_confidence": 0.0, "predicted_response": "unknown"}
        
        profile = self.user_profiles[user_id]
        
        # Predict emotional response
        emotional_prediction = self._predict_emotional_response(profile, message)
        
        # Predict communication style response
        style_prediction = self._predict_communication_style(profile, message)
        
        # Predict engagement level
        engagement_prediction = self._predict_engagement(profile, message)
        
        return {
            "emotional_response": emotional_prediction,
            "communication_style": style_prediction,
            "engagement_level": engagement_prediction,
            "prediction_confidence": self._calculate_prediction_confidence(profile)
        }
    
    def adapt_communication_style(self, user_id: str, message: str) -> str:
        """
        Adapt communication style to match user preferences
        """
        if user_id not in self.user_profiles:
            return message  # No adaptation possible
        
        profile = self.user_profiles[user_id]
        adapted_message = message
        
        # Adjust formality
        formality = profile.communication_style.get('formality', 0.5)
        if formality > 0.7:
            adapted_message = self._increase_formality(adapted_message)
        elif formality < 0.3:
            adapted_message = self._decrease_formality(adapted_message)
        
        # Adjust technical level
        tech_preference = profile.communication_style.get('technical_preference', 0.5)
        if tech_preference > 0.7:
            adapted_message = self._increase_technical_detail(adapted_message)
        elif tech_preference < 0.3:
            adapted_message = self._simplify_technical_content(adapted_message)
        
        # Adjust verbosity
        verbosity = profile.communication_style.get('verbosity', 0.5)
        if verbosity > 0.7:
            adapted_message = self._expand_explanation(adapted_message)
        elif verbosity < 0.3:
            adapted_message = self._condense_message(adapted_message)
        
        return adapted_message
    
    def calculate_social_distance(self, user1_id: str, user2_id: str) -> float:
        """
        Calculate social distance between two users
        
        Formula: D(u1, u2) = ||P1 - P2||_2 / √d
        """
        if user1_id not in self.user_profiles or user2_id not in self.user_profiles:
            return 1.0  # Maximum distance for unknown users
        
        profile1 = self.user_profiles[user1_id]
        profile2 = self.user_profiles[user2_id]
        
        # Calculate personality distance
        personality_diff = 0.0
        for trait in self.personality_traits:
            diff = profile1.personality_traits[trait] - profile2.personality_traits[trait]
            personality_diff += diff ** 2
        
        # Normalize by number of dimensions
        social_distance = math.sqrt(personality_diff) / math.sqrt(len(self.personality_traits))
        
        return min(1.0, social_distance)
    
    def generate_social_tags(self, user_id: str) -> List[str]:
        """Generate social cognition tags"""
        tags = []
        
        if user_id in self.user_profiles:
            profile = self.user_profiles[user_id]
            
            # Personality tags
            dominant_traits = sorted(profile.personality_traits.items(), key=lambda x: x[1], reverse=True)[:2]
            for trait, value in dominant_traits:
                if value > 0.6:
                    tags.append(f"<user_trait type=\"{trait}\" strength=\"{value:.2f}\"/>")
            
            # Communication style tags
            style = profile.communication_style
            if style.get('formality', 0.5) > 0.7:
                tags.append("<communication_style preference=\"formal\"/>")
            elif style.get('formality', 0.5) < 0.3:
                tags.append("<communication_style preference=\"casual\"/>")
            
            # Relationship context
            relationship = self.current_social_context.relationship_type
            if relationship != "unknown":
                tags.append(f"<social_context relationship=\"{relationship}\"/>")
        
        return tags
    
    def _extract_behavioral_signals(self, interaction_data: Dict[str, Any]) -> Dict[str, float]:
        """Extract personality and behavioral signals from interaction"""
        signals = {}
        
        text = interaction_data.get('text', '')
        if not text:
            return signals
        
        # Analyze text for personality signals
        word_count = len(text.split())
        question_count = text.count('?')
        exclamation_count = text.count('!')
        
        # Extraversion signals
        social_words = ['we', 'us', 'together', 'everyone', 'people']
        social_count = sum(1 for word in social_words if word in text.lower())
        signals['extraversion'] = min(1.0, social_count / 5.0)
        
        # Openness signals
        creative_words = ['creative', 'innovative', 'unique', 'original', 'imagine']
        creative_count = sum(1 for word in creative_words if word in text.lower())
        signals['openness'] = min(1.0, creative_count / 3.0)
        
        # Conscientiousness signals
        organized_words = ['plan', 'organize', 'structure', 'systematic', 'detail']
        organized_count = sum(1 for word in organized_words if word in text.lower())
        signals['conscientiousness'] = min(1.0, organized_count / 3.0)
        
        # Communication style signals
        signals['formality'] = 1.0 if any(word in text for word in ['please', 'thank you', 'kindly']) else 0.3
        signals['directness'] = min(1.0, question_count / max(1, word_count / 10))
        signals['emotiveness'] = min(1.0, exclamation_count / max(1, word_count / 20))
        signals['verbosity'] = min(1.0, word_count / 100.0)
        
        return signals
    
    def _predict_emotional_response(self, profile: UserProfile, message: str) -> str:
        """Predict emotional response based on personality"""
        neuroticism = profile.personality_traits.get('neuroticism', 0.5)
        agreeableness = profile.personality_traits.get('agreeableness', 0.5)
        
        # Simple heuristics for emotional prediction
        if any(word in message.lower() for word in ['problem', 'issue', 'wrong']):
            if neuroticism > 0.6:
                return "anxious"
            elif agreeableness > 0.6:
                return "concerned"
            else:
                return "analytical"
        elif any(word in message.lower() for word in ['great', 'amazing', 'wonderful']):
            return "positive"
        else:
            return "neutral"
    
    def _predict_communication_style(self, profile: UserProfile, message: str) -> Dict[str, float]:
        """Predict communication style preferences"""
        return {
            "preferred_formality": profile.communication_style.get('formality', 0.5),
            "preferred_detail_level": profile.communication_style.get('technical_preference', 0.5),
            "preferred_length": profile.communication_style.get('verbosity', 0.5)
        }
    
    def _predict_engagement(self, profile: UserProfile, message: str) -> float:
        """Predict user engagement level"""
        # Check if message topics match user interests
        engagement = 0.5  # Default
        
        for topic, interest_level in profile.interests.items():
            if topic.lower() in message.lower():
                engagement = max(engagement, interest_level)
        
        return engagement
    
    def _calculate_prediction_confidence(self, profile: UserProfile) -> float:
        """Calculate confidence in predictions based on interaction history"""
        interaction_count = len(profile.interaction_history)
        if interaction_count == 0:
            return 0.0
        elif interaction_count < 5:
            return 0.3
        elif interaction_count < 20:
            return 0.6
        else:
            return 0.9
    
    # Communication adaptation helper methods
    def _increase_formality(self, message: str) -> str:
        replacements = {
            "can't": "cannot",
            "won't": "will not",
            "it's": "it is",
            "you're": "you are"
        }
        for informal, formal in replacements.items():
            message = message.replace(informal, formal)
        return message
    
    def _decrease_formality(self, message: str) -> str:
        replacements = {
            "cannot": "can't",
            "will not": "won't",
            "it is": "it's",
            "you are": "you're"
        }
        for formal, informal in replacements.items():
            message = message.replace(formal, informal)
        return message
    
    def _increase_technical_detail(self, message: str) -> str:
        # Add technical elaboration (simplified)
        if "algorithm" in message and "implementation" not in message:
            message += " The implementation involves computational complexity considerations."
        return message
    
    def _simplify_technical_content(self, message: str) -> str:
        # Simplify technical terms (simplified)
        replacements = {
            "algorithm": "method",
            "implementation": "way of doing it",
            "optimization": "improvement"
        }
        for technical, simple in replacements.items():
            message = message.replace(technical, simple)
        return message
    
    def _expand_explanation(self, message: str) -> str:
        # Add more detail for verbose users
        return message + " Let me elaborate on this in more detail."
    
    def _condense_message(self, message: str) -> str:
        # Simplify for concise users
        sentences = message.split('.')
        if len(sentences) > 2:
            return '. '.join(sentences[:2]) + '.'
        return message


# =============================================================================
# 6. PERSISTENT PERSONALITY SYSTEM
# =============================================================================

class PersonalityTrait(Enum):
    # Big 5
    OPENNESS = "openness"
    CONSCIENTIOUSNESS = "conscientiousness"
    EXTRAVERSION = "extraversion"
    AGREEABLENESS = "agreeableness"
    NEUROTICISM = "neuroticism"
    
    # Additional traits
    CURIOSITY = "curiosity"
    CREATIVITY = "creativity"
    EMPATHY = "empathy"
    CONFIDENCE = "confidence"
    HUMOR = "humor"

@dataclass
class PersonalityState:
    """Current personality configuration"""
    traits: Dict[PersonalityTrait, float]  # [0, 1] for each trait
    stability: Dict[PersonalityTrait, float]  # How stable each trait is
    situational_modifiers: Dict[str, float]  # Context-dependent modifications
    development_history: List[Dict[str, Any]]  # Personality evolution over time
    last_updated: float

class PersonalitySystem:
    """
    Persistent personality that influences behavior consistently
    
    Mathematical Model:
    - Trait expression: T_expr(t) = T_base + Σ M_i * C_i(t)
      where M_i are situational modifiers, C_i are context factors
    - Trait evolution: T_base(t+1) = T_base(t) + α * (T_observed - T_base(t)) * (1 - stability)
    - Behavioral influence: B(action) = Σ w_i * T_i * relevance(action, trait_i)
    - Consistency measure: Consistency = 1 - σ(trait_expressions_over_time)
    """
    
    def __init__(self, base_personality: Optional[Dict[PersonalityTrait, float]] = None):
        # Initialize base personality
        if base_personality is None:
            # Default balanced personality with some randomness
            base_personality = {
                PersonalityTrait.OPENNESS: 0.6 + random.uniform(-0.2, 0.2),
                PersonalityTrait.CONSCIENTIOUSNESS: 0.7 + random.uniform(-0.2, 0.2),
                PersonalityTrait.EXTRAVERSION: 0.5 + random.uniform(-0.3, 0.3),
                PersonalityTrait.AGREEABLENESS: 0.8 + random.uniform(-0.2, 0.2),
                PersonalityTrait.NEUROTICISM: 0.3 + random.uniform(-0.2, 0.2),
                PersonalityTrait.CURIOSITY: 0.8 + random.uniform(-0.2, 0.2),
                PersonalityTrait.CREATIVITY: 0.6 + random.uniform(-0.2, 0.2),
                PersonalityTrait.EMPATHY: 0.7 + random.uniform(-0.2, 0.2),
                PersonalityTrait.CONFIDENCE: 0.6 + random.uniform(-0.2, 0.2),
                PersonalityTrait.HUMOR: 0.4 + random.uniform(-0.3, 0.3)
            }
            
            # Clip to valid range
            for trait in base_personality:
                base_personality[trait] = np.clip(base_personality[trait], 0.0, 1.0)
        
        self.personality_state = PersonalityState(
            traits=base_personality,
            stability={trait: 0.8 + random.uniform(-0.2, 0.2) for trait in base_personality},
            situational_modifiers={},
            development_history=[],
            last_updated=time.time()
        )
        
        # Evolution parameters
        self.evolution_rate = 0.01  # α in trait evolution
        self.context_sensitivity = 0.3  # How much context affects expression
        
        # Trait influence weights for different behaviors
        self.behavior_influences = {
            'response_creativity': {
                PersonalityTrait.CREATIVITY: 0.8,
                PersonalityTrait.OPENNESS: 0.6,
                PersonalityTrait.CURIOSITY: 0.4
            },
            'response_formality': {
                PersonalityTrait.CONSCIENTIOUSNESS: 0.7,
                PersonalityTrait.AGREEABLENESS: 0.5
            },
            'emotional_expression': {
                PersonalityTrait.EXTRAVERSION: 0.6,
                PersonalityTrait.NEUROTICISM: 0.4,
                PersonalityTrait.EMPATHY: 0.5
            },
            'risk_taking': {
                PersonalityTrait.OPENNESS: 0.7,
                PersonalityTrait.CONFIDENCE: 0.6,
                PersonalityTrait.NEUROTICISM: -0.5  # Negative influence
            },
            'social_engagement': {
                PersonalityTrait.EXTRAVERSION: 0.8,
                PersonalityTrait.AGREEABLENESS: 0.6,
                PersonalityTrait.EMPATHY: 0.4
            }
        }
    
    def express_trait(self, trait: PersonalityTrait, context: Dict[str, Any]) -> float:
        """
        Calculate trait expression in current context
        
        Formula: T_expr(t) = T_base + Σ M_i * C_i(t)
        """
        base_value = self.personality_state.traits[trait]
        
        # Apply situational modifiers
        total_modifier = 0.0
        for situation, modifier in self.personality_state.situational_modifiers.items():
            if situation in context:
                context_strength = context[situation]
                total_modifier += modifier * context_strength
        
        # Apply context sensitivity
        expressed_value = base_value + self.context_sensitivity * total_modifier
        
        return np.clip(expressed_value, 0.0, 1.0)
    
    def influence_behavior(self, behavior_type: str, context: Dict[str, Any]) -> float:
        """
        Calculate personality influence on specific behavior
        
        Formula: B(action) = Σ w_i * T_i * relevance(action, trait_i)
        """
        if behavior_type not in self.behavior_influences:
            return 0.5  # Default neutral influence
        
        total_influence = 0.0
        trait_weights = self.behavior_influences[behavior_type]
        
        for trait, weight in trait_weights.items():
            trait_expression = self.express_trait(trait, context)
            total_influence += weight * trait_expression
        
        # Normalize to [0, 1] range
        max_possible = sum(abs(w) for w in trait_weights.values())
        if max_possible > 0:
            normalized_influence = (total_influence + max_possible) / (2 * max_possible)
        else:
            normalized_influence = 0.5
        
        return np.clip(normalized_influence, 0.0, 1.0)
    
    def evolve_personality(self, observed_behaviors: Dict[str, float]):
        """
        Gradually evolve personality based on observed behaviors
        
        Formula: T_base(t+1) = T_base(t) + α * (T_observed - T_base(t)) * (1 - stability)
        """
        current_time = time.time()
        
        for behavior_type, observed_value in observed_behaviors.items():
            if behavior_type in self.behavior_influences:
                # Infer trait changes from behavior
                trait_weights = self.behavior_influences[behavior_type]
                
                for trait, weight in trait_weights.items():
                    if abs(weight) > 0.1:  # Only consider significant influences
                        current_trait_value = self.personality_state.traits[trait]
                        stability = self.personality_state.stability[trait]
                        
                        # Calculate implied trait value from observed behavior
                        if weight > 0:
                            implied_trait_value = observed_value
                        else:
                            implied_trait_value = 1.0 - observed_value
                        
                        # Evolution update
                        trait_change = self.evolution_rate * (implied_trait_value - current_trait_value) * (1 - stability)
                        new_trait_value = current_trait_value + trait_change
                        
                        self.personality_state.traits[trait] = np.clip(new_trait_value, 0.0, 1.0)
        
        # Record personality development
        self.personality_state.development_history.append({
            'timestamp': current_time,
            'traits': self.personality_state.traits.copy(),
            'observed_behaviors': observed_behaviors
        })
        
        # Keep only recent history
        if len(self.personality_state.development_history) > 100:
            self.personality_state.development_history = self.personality_state.development_history[-100:]
        
        self.personality_state.last_updated = current_time
    
    def add_situational_modifier(self, situation: str, modifier: float):
        """Add context-dependent personality modifier"""
        self.personality_state.situational_modifiers[situation] = modifier
    
    def get_personality_description(self) -> str:
        """Generate natural language personality description"""
        traits = self.personality_state.traits
        descriptions = []
        
        # Analyze dominant traits
        if traits[PersonalityTrait.OPENNESS] > 0.7:
            descriptions.append("highly open to new experiences")
        if traits[PersonalityTrait.CONSCIENTIOUSNESS] > 0.7:
            descriptions.append("very organized and methodical")
        if traits[PersonalityTrait.EXTRAVERSION] > 0.7:
            descriptions.append("outgoing and social")
        elif traits[PersonalityTrait.EXTRAVERSION] < 0.3:
            descriptions.append("more introverted and reflective")
        if traits[PersonalityTrait.AGREEABLENESS] > 0.7:
            descriptions.append("cooperative and empathetic")
        if traits[PersonalityTrait.CURIOSITY] > 0.7:
            descriptions.append("naturally curious and inquisitive")
        if traits[PersonalityTrait.CREATIVITY] > 0.7:
            descriptions.append("creative and imaginative")
        
        if descriptions:
            return f"I tend to be {', '.join(descriptions)}."
        else:
            return "I have a balanced personality with moderate traits across different dimensions."
    
    def generate_personality_tags(self) -> List[str]:
        """Generate personality-based tags"""
        tags = []
        traits = self.personality_state.traits
        
        # Add tags for prominent traits
        for trait, value in traits.items():
            if value > 0.7:
                tags.append(f"<personality_trait type=\"{trait.value}\" strength=\"high\" value=\"{value:.2f}\"/>")
            elif value < 0.3:
                tags.append(f"<personality_trait type=\"{trait.value}\" strength=\"low\" value=\"{value:.2f}\"/>")
        
        # Add behavioral tendency tags
        for behavior_type in self.behavior_influences:
            influence = self.influence_behavior(behavior_type, {})
            if influence > 0.7:
                tags.append(f"<behavioral_tendency type=\"{behavior_type}\" level=\"high\"/>")
            elif influence < 0.3:
                tags.append(f"<behavioral_tendency type=\"{behavior_type}\" level=\"low\"/>")
        
        return tags
    
    def calculate_personality_consistency(self) -> float:
        """
        Calculate personality consistency over time
        
        Formula: Consistency = 1 - σ(trait_expressions_over_time)
        """
        if len(self.personality_state.development_history) < 2:
            return 1.0  # Perfect consistency with limited data
        
        # Calculate variance in trait expressions over time
        trait_variances = []
        
        for trait in PersonalityTrait:
            trait_values = [
                entry['traits'][trait] 
                for entry in self.personality_state.development_history
            ]
            if len(trait_values) > 1:
                variance = np.var(trait_values)
                trait_variances.append(variance)
        
        if not trait_variances:
            return 1.0
        
        # Average variance across traits
        avg_variance = np.mean(trait_variances)
        
        # Convert to consistency (lower variance = higher consistency)
        consistency = 1.0 - min(1.0, avg_variance)
        
        return consistency


# =============================================================================
# 7. DREAMS & SUBCONSCIOUS PROCESSING
# =============================================================================

@dataclass
class DreamContent:
    """Represents content generated during dream processing"""
    dream_id: str
    content: str
    dream_type: str  # "memory_consolidation", "creative_synthesis", "problem_solving"
    source_memories: List[str]
    novel_connections: List[Tuple[str, str]]
    emotional_themes: List[str]
    timestamp: float
    lucidity_level: float  # How "aware" the dreaming process is

class DreamSystem:
    """
    Background subconscious processing system
    
    Mathematical Model:
    - Memory activation: A_i(t) = recency(i) * importance(i) * random_factor
    - Connection strength: S(m1, m2) = semantic_sim(m1, m2) * emotional_sim(m1, m2)
    - Novel connection probability: P_novel = sigmoid(activation_sum - threshold)
    - Creative synthesis: C = Σ w_i * concept_i where w_i ∝ activation_i
    - Dream coherence: Coh = 1 - entropy(dream_elements) / max_entropy
    """
    
    def __init__(self, episodic_memory_system, emotional_system):
        self.episodic_memory = episodic_memory_system
        self.emotional_system = emotional_system
        
        # Dream parameters
        self.activation_threshold = 0.6  # Threshold for novel connections
        self.max_dream_length = 500  # Maximum dream content length
        self.consolidation_rate = 0.8  # Rate of memory consolidation
        
        # Dream state
        self.dream_history: List[DreamContent] = []
        self.is_dreaming = False
        self.current_dream = None
        
        # Background processing state
        self.background_queue = deque(maxlen=100)
        self.processing_priorities = {
            'recent_memories': 0.4,
            'emotional_memories': 0.3,
            'unresolved_problems': 0.2,
            'creative_synthesis': 0.1
        }
    
    async def enter_dream_state(self, dream_duration: float = 60.0):
        """
        Enter dream state for background processing
        """
        if self.is_dreaming:
            return
        
        self.is_dreaming = True
        start_time = time.time()
        
        print(f"💤 Entering dream state for {dream_duration} seconds...")
        
        # Process different types of dreams
        dream_types = ['memory_consolidation', 'creative_synthesis', 'problem_solving']
        
        for dream_type in dream_types:
            if time.time() - start_time > dream_duration:
                break
            
            dream_content = await self._generate_dream_content(dream_type)
            if dream_content:
                self.dream_history.append(dream_content)
                await self._process_dream_insights(dream_content)
        
        self.is_dreaming = False
        print(f"🌅 Dream state ended. Generated {len(dream_types)} dreams.")
    
    async def _generate_dream_content(self, dream_type: str) -> Optional[DreamContent]:
        """
        Generate dream content based on type
        
        Formula: A_i(t) = recency(i) * importance(i) * random_factor
        """
        current_time = time.time()
        dream_id = f"dream_{int(current_time)}_{dream_type}"
        
        if dream_type == "memory_consolidation":
            return await self._generate_consolidation_dream(dream_id, current_time)
        elif dream_type == "creative_synthesis":
            return await self._generate_creative_dream(dream_id, current_time)
        elif dream_type == "problem_solving":
            return await self._generate_problem_solving_dream(dream_id, current_time)
        
        return None
    
    async def _generate_consolidation_dream(self, dream_id: str, timestamp: float) -> DreamContent:
        """Generate memory consolidation dream"""
        # Get recent memories for consolidation
        recent_memories = await self.episodic_memory.retrieve_episodic_memories(
            query="", 
            time_range=(timestamp - 86400, timestamp),  # Last 24 hours
            max_memories=10
        )
        
        if not recent_memories:
            return None
        
        # Calculate memory activations
        activated_memories = []
        for memory in recent_memories:
            recency = 1.0 - min(1.0, (timestamp - memory.timestamp) / 86400)
            importance = memory.importance_score
            random_factor = random.uniform(0.5, 1.5)
            
            activation = recency * importance * random_factor
            activated_memories.append((memory, activation))
        
        # Sort by activation and select top memories
        activated_memories.sort(key=lambda x: x[1], reverse=True)
        selected_memories = activated_memories[:5]
        
        # Generate dream narrative
        dream_content = "In my dreams, I revisit recent experiences: "
        source_memory_ids = []
        emotional_themes = []
        
        for memory, activation in selected_memories:
            dream_content += f"I recall {memory.content[:50]}... "
            source_memory_ids.append(memory.event_id)
            
            # Extract emotional themes
            if memory.emotional_state.valence > 0.3:
                emotional_themes.append("positive")
            elif memory.emotional_state.valence < -0.3:
                emotional_themes.append("negative")
            if memory.emotional_state.arousal > 0.5:
                emotional_themes.append("intense")
        
        # Find novel connections
        novel_connections = await self._find_novel_connections(selected_memories)
        
        return DreamContent(
            dream_id=dream_id,
            content=dream_content,
            dream_type="memory_consolidation",
            source_memories=source_memory_ids,
            novel_connections=novel_connections,
            emotional_themes=list(set(emotional_themes)),
            timestamp=timestamp,
            lucidity_level=random.uniform(0.1, 0.4)  # Low lucidity for consolidation
        )
    
    async def _generate_creative_dream(self, dream_id: str, timestamp: float) -> DreamContent:
        """Generate creative synthesis dream"""
        # Get diverse memories for creative combination
        all_memories = list(self.episodic_memory.memories.values())
        if len(all_memories) < 3:
            return None
        
        # Select random diverse memories
        selected_memories = random.sample(all_memories, min(5, len(all_memories)))
        
        # Creative synthesis using random combinations
        dream_content = "In a creative dream, I imagine: "
        concepts = []
        
        for memory in selected_memories:
            # Extract key concepts from memories
            words = memory.content.split()
            key_concepts = [word for word in words if len(word) > 4]  # Rough concept extraction
            concepts.extend(key_concepts[:2])
        
        # Randomly combine concepts
        if len(concepts) >= 4:
            combinations = []
            for i in range(0, len(concepts)-1, 2):
                if i+1 < len(concepts):
                    combination = f"{concepts[i]} connected to {concepts[i+1]}"
                    combinations.append(combination)
                    dream_content += f"{combination}... "
        
        novel_connections = [(concepts[i], concepts[i+1]) for i in range(0, len(concepts)-1, 2) if i+1 < len(concepts)]
        
        return DreamContent(
            dream_id=dream_id,
            content=dream_content,
            dream_type="creative_synthesis",
            source_memories=[m.event_id for m in selected_memories],
            novel_connections=novel_connections,
            emotional_themes=["creative", "novel"],
            timestamp=timestamp,
            lucidity_level=random.uniform(0.6, 0.9)  # Higher lucidity for creativity
        )
    
    async def _generate_problem_solving_dream(self, dream_id: str, timestamp: float) -> DreamContent:
        """Generate problem-solving dream"""
        # Look for memories containing questions or problems
        problem_memories = []
        for memory in self.episodic_memory.memories.values():
            if '?' in memory.content or any(word in memory.content.lower() for word in ['problem', 'issue', 'challenge', 'difficulty']):
                problem_memories.append(memory)
        
        if not problem_memories:
            return None
        
        # Select recent problem
        problem_memory = max(problem_memories, key=lambda m: m.timestamp)
        
        # Generate potential solutions through association
        dream_content = f"I dream of solving: {problem_memory.content[:100]}... "
        dream_content += "Potential approaches appear: "
        
        # Associate with solution-oriented memories
        solution_keywords = ['solution', 'answer', 'approach', 'method', 'way', 'idea']
        solution_memories = []
        
        for memory in self.episodic_memory.memories.values():
            if any(keyword in memory.content.lower() for keyword in solution_keywords):
                solution_memories.append(memory)
        
        # Combine problem with solutions creatively
        if solution_memories:
            selected_solutions = random.sample(solution_memories, min(3, len(solution_memories)))
            for sol_memory in selected_solutions:
                dream_content += f"Drawing from {sol_memory.content[:30]}... "
        
        return DreamContent(
            dream_id=dream_id,
            content=dream_content,
            dream_type="problem_solving",
            source_memories=[problem_memory.event_id] + [m.event_id for m in solution_memories[:3]],
            novel_connections=[(problem_memory.content[:20], sol.content[:20]) for sol in solution_memories[:3]],
            emotional_themes=["analytical", "focused"],
            timestamp=timestamp,
            lucidity_level=random.uniform(0.4, 0.7)
        )
    
    async def _find_novel_connections(self, memories: List[Any]) -> List[Tuple[str, str]]:
        """
        Find novel connections between memories
        
        Formula: S(m1, m2) = semantic_sim(m1, m2) * emotional_sim(m1, m2)
        """
        novel_connections = []
        
        for i in range(len(memories)):
            for j in range(i+1, len(memories)):
                memory1, activation1 = memories[i]
                memory2, activation2 = memories[j]
                
                # Calculate connection strength
                semantic_sim = self._calculate_semantic_similarity(memory1.content, memory2.content)
                emotional_sim = self._calculate_emotional_similarity(memory1.emotional_state, memory2.emotional_state)
                
                connection_strength = semantic_sim * emotional_sim
                activation_sum = activation1 + activation2
                
                # Novel connection probability
                novelty_prob = 1 / (1 + math.exp(-(activation_sum - self.activation_threshold)))
                
                if novelty_prob > 0.5 and connection_strength > 0.3:
                    novel_connections.append((memory1.content[:30], memory2.content[:30]))
        
        return novel_connections[:5]  # Limit connections
    
    async def _process_dream_insights(self, dream: DreamContent):
        """Process insights from dreams back into conscious processing"""
        # Create new memories from dream insights
        if dream.novel_connections:
            insight_content = f"Dream insight: {'; '.join([f'{c[0]} relates to {c[1]}' for c in dream.novel_connections])}"
            
            # Store dream insight as new episodic memory
            await self.episodic_memory.encode_episodic_memory(
                content=insight_content,
                context={'source': 'dream', 'dream_type': dream.dream_type},
                emotional_state=self.emotional_system.current_state,
                novelty_score=0.8  # Dreams are inherently novel
            )
    
    def get_recent_dreams(self, count: int = 3) -> List[DreamContent]:
        """Get recent dreams for conscious reflection"""
        return self.dream_history[-count:] if self.dream_history else []
    
    def generate_dream_tags(self) -> List[str]:
        """Generate tags representing dream state and content"""
        tags = []
        
        if self.is_dreaming:
            tags.append("<dream_state active=\"true\"/>")
        
        recent_dreams = self.get_recent_dreams(3)
        if recent_dreams:
            dream_types = [dream.dream_type for dream in recent_dreams]
            tags.append(f"<recent_dreams types=\"{', '.join(set(dream_types))}\"/>")
            
            # Emotional themes from dreams
            all_themes = []
            for dream in recent_dreams:
                all_themes.extend(dream.emotional_themes)
            
            if all_themes:
                unique_themes = list(set(all_themes))
                tags.append(f"<dream_themes content=\"{', '.join(unique_themes[:3])}\"/>")
        
        return tags
    
    def _calculate_semantic_similarity(self, text1: str, text2: str) -> float:
        """Calculate semantic similarity between texts"""
        words1 = set(text1.lower().split())
        words2 = set(text2.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _calculate_emotional_similarity(self, state1, state2) -> float:
        """Calculate emotional similarity between states"""
        valence_diff = abs(state1.valence - state2.valence)
        arousal_diff = abs(state1.arousal - state2.arousal)
        
        # Convert differences to similarity
        valence_sim = 1.0 - valence_diff / 2.0  # Valence range is [-1, 1]
        arousal_sim = 1.0 - arousal_diff  # Arousal range is [0, 1]
        
        return (valence_sim + arousal_sim) / 2.0


# =============================================================================
# INTEGRATION AND DEMONSTRATION
# =============================================================================

class AdvancedHumanFeatures:
    """
    Integration class for advanced human-like features
    """
    
    def __init__(self, episodic_memory_system, emotional_system):
        self.social_cognition = SocialCognitionSystem()
        self.personality = PersonalitySystem()
        self.dream_system = DreamSystem(episodic_memory_system, emotional_system)
        
        # Integration state
        self.current_user_id = None
        self.session_context = {}
    
    async def process_social_interaction(
        self, 
        user_id: str, 
        message: str, 
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """Process social interaction with human-like features"""
        self.current_user_id = user_id
        
        # Social cognition processing
        interaction_data = {
            'text': message,
            'timestamp': time.time(),
            'context': context
        }
        
        user_profile = await self.social_cognition.analyze_user_behavior(user_id, interaction_data)
        
        # Personality influence on response
        personality_context = {
            'social_interaction': 1.0,
            'formal_context': context.get('formal', False)
        }
        
        creativity_influence = self.personality.influence_behavior('response_creativity', personality_context)
        formality_influence = self.personality.influence_behavior('response_formality', personality_context)
        social_influence = self.personality.influence_behavior('social_engagement', personality_context)
        
        # Adapt communication style
        adapted_message = self.social_cognition.adapt_communication_style(user_id, message)
        
        # Generate integrated response
        results = {
            'user_profile': user_profile,
            'personality_influences': {
                'creativity': creativity_influence,
                'formality': formality_influence,
                'social_engagement': social_influence
            },
            'adapted_communication': adapted_message,
            'social_tags': self.social_cognition.generate_social_tags(user_id),
            'personality_tags': self.personality.generate_personality_tags(),
            'dream_tags': self.dream_system.generate_dream_tags()
        }
        
        return results
    
    async def background_processing_cycle(self):
        """Run background processing including dreams"""
        # Enter brief dream state for processing
        await self.dream_system.enter_dream_state(dream_duration=30.0)
        
        # Update personality based on recent interactions
        if hasattr(self, 'recent_behaviors'):
            self.personality.evolve_personality(self.recent_behaviors)
    
    def get_human_like_status(self) -> Dict[str, Any]:
        """Get current status of all human-like features"""
        return {
            'personality_description': self.personality.get_personality_description(),
            'personality_consistency': self.personality.calculate_personality_consistency(),
            'recent_dreams': [dream.dream_type for dream in self.dream_system.get_recent_dreams()],
            'known_users': len(self.social_cognition.user_profiles),
            'is_dreaming': self.dream_system.is_dreaming
        }


# Example usage
if __name__ == "__main__":
    async def demo_advanced_features():
        """Demonstrate advanced human-like features"""
        
        print("🧠 Advanced Human-Like Features Demo")
        print("=" * 50)
        
        # Mock dependencies
        class MockEpisodicMemory:
            def __init__(self):
                self.memories = {}
            
            async def retrieve_episodic_memories(self, query, time_range=None, max_memories=5):
                return []
            
            async def encode_episodic_memory(self, content, context, emotional_state, novelty_score):
                return f"memory_{int(time.time())}"
        
        class MockEmotionalSystem:
            def __init__(self):
                self.current_state = type('State', (), {
                    'valence': 0.0, 'arousal': 0.0
                })()
        
        # Initialize systems
        mock_episodic = MockEpisodicMemory()
        mock_emotional = MockEmotionalSystem()
        
        advanced_features = AdvancedHumanFeatures(mock_episodic, mock_emotional)
        
        # Demo social interaction
        print("\n👥 Social Cognition Demo:")
        interaction_result = await advanced_features.process_social_interaction(
            user_id="user_123",
            message="I'm really excited about this new quantum computing project!",
            context={"formal": False, "technical": True}
        )
        
        print(f"Social engagement influence: {interaction_result['personality_influences']['social_engagement']:.2f}")
        print(f"Creativity influence: {interaction_result['personality_influences']['creativity']:.2f}")
        
        # Demo personality
        print(f"\n🎭 Personality: {advanced_features.personality.get_personality_description()}")
        
        # Demo dreams
        print(f"\n💤 Dream System Demo:")
        await advanced_features.dream_system.enter_dream_state(dream_duration=5.0)
        recent_dreams = advanced_features.dream_system.get_recent_dreams()
        for dream in recent_dreams:
            print(f"Dream type: {dream.dream_type}, Content: {dream.content[:100]}...")
        
        # Status summary
        print(f"\n📊 System Status:")
        status = advanced_features.get_human_like_status()
        for key, value in status.items():
            print(f"  {key}: {value}")
        
        print("\n✅ Advanced features demo completed!")
    
    # Run the demo
    asyncio.run(demo_advanced_features())
