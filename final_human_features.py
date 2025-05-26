"""
Final Human-Like Features for ATLAS
Implementation of remaining enhancements: Moral Reasoning, Attention Management, 
Temporal Reasoning, Multi-modal Imagination, and Complete Integration
"""

import torch
import numpy as np
import time
import json
from typing import Dict, List, Tuple, Optional, Any, Union
from dataclasses import dataclass, field
from enum import Enum
import asyncio
from collections import deque, defaultdict
import math
import random


# =============================================================================
# 8. MORAL & ETHICAL REASONING FRAMEWORK
# =============================================================================

class EthicalPrinciple(Enum):
    BENEFICENCE = "beneficence"  # Do good
    NON_MALEFICENCE = "non_maleficence"  # Do no harm
    AUTONOMY = "autonomy"  # Respect individual choice
    JUSTICE = "justice"  # Fairness and equality
    TRUTHFULNESS = "truthfulness"  # Honesty and transparency
    PRIVACY = "privacy"  # Respect for personal information
    CONSENT = "consent"  # Informed agreement
    DIGNITY = "dignity"  # Human worth and respect

@dataclass
class EthicalDilemma:
    """Represents an ethical decision scenario"""
    dilemma_id: str
    description: str
    stakeholders: List[str]
    potential_actions: List[Dict[str, Any]]
    ethical_dimensions: Dict[EthicalPrinciple, float]  # Relevance of each principle
    context_factors: Dict[str, Any]
    timestamp: float

@dataclass
class EthicalDecision:
    """Represents a made ethical decision"""
    decision_id: str
    dilemma: EthicalDilemma
    chosen_action: Dict[str, Any]
    reasoning: str
    principle_weights: Dict[EthicalPrinciple, float]
    confidence: float
    anticipated_outcomes: Dict[str, float]
    timestamp: float

class MoralReasoningSystem:
    """
    Ethical decision-making and moral reasoning system
    
    Mathematical Models:
    - Principle weight: w_i(t) = w_i(t-1) + α * (outcome - expected) * relevance_i
    - Ethical score: E(action) = Σ w_i * principle_alignment_i(action)
    - Moral distance: D(action1, action2) = ||E_vector1 - E_vector2||_2
    - Confidence: C = 1 / (1 + exp(-k * (max_score - second_max_score)))
    - Outcome prediction: O(action) = Σ p_i * utility_i where p_i is probability
    """
    
    def __init__(self):
        # Core ethical principles with initial weights
        self.principle_weights = {
            EthicalPrinciple.BENEFICENCE: 0.9,
            EthicalPrinciple.NON_MALEFICENCE: 0.95,  # Slightly higher - "first, do no harm"
            EthicalPrinciple.AUTONOMY: 0.8,
            EthicalPrinciple.JUSTICE: 0.85,
            EthicalPrinciple.TRUTHFULNESS: 0.8,
            EthicalPrinciple.PRIVACY: 0.75,
            EthicalPrinciple.CONSENT: 0.8,
            EthicalPrinciple.DIGNITY: 0.9
        }
        
        # Learning parameters
        self.weight_learning_rate = 0.05  # α in weight update
        self.confidence_steepness = 3.0  # k in confidence calculation
        
        # Decision history for learning
        self.decision_history: List[EthicalDecision] = []
        self.dilemma_outcomes: Dict[str, Dict[str, float]] = {}
        
        # Moral reasoning patterns
        self.reasoning_templates = {
            EthicalPrinciple.BENEFICENCE: "This action promotes wellbeing by {benefit}",
            EthicalPrinciple.NON_MALEFICENCE: "This avoids harm by {harm_prevention}",
            EthicalPrinciple.AUTONOMY: "This respects individual choice by {autonomy_respect}",
            EthicalPrinciple.JUSTICE: "This promotes fairness by {fairness_aspect}",
            EthicalPrinciple.TRUTHFULNESS: "This maintains honesty by {truth_aspect}",
            EthicalPrinciple.PRIVACY: "This protects privacy by {privacy_protection}",
            EthicalPrinciple.CONSENT: "This ensures consent by {consent_mechanism}",
            EthicalPrinciple.DIGNITY: "This upholds human dignity by {dignity_aspect}"
        }
    
    async def analyze_ethical_dilemma(self, dilemma: EthicalDilemma) -> EthicalDecision:
        """
        Analyze ethical dilemma and make moral decision
        
        Formula: E(action) = Σ w_i * principle_alignment_i(action)
        """
        action_scores = []
        
        # Evaluate each potential action
        for action in dilemma.potential_actions:
            ethical_score = self._calculate_ethical_score(action, dilemma)
            predicted_outcomes = self._predict_outcomes(action, dilemma)
            
            action_scores.append({
                'action': action,
                'ethical_score': ethical_score,
                'predicted_outcomes': predicted_outcomes
            })
        
        # Sort by ethical score
        action_scores.sort(key=lambda x: x['ethical_score'], reverse=True)
        
        # Calculate decision confidence
        if len(action_scores) >= 2:
            max_score = action_scores[0]['ethical_score']
            second_max_score = action_scores[1]['ethical_score']
            confidence = 1 / (1 + math.exp(-self.confidence_steepness * (max_score - second_max_score)))
        else:
            confidence = 0.5
        
        # Choose best action
        best_action_data = action_scores[0]
        chosen_action = best_action_data['action']
        
        # Generate moral reasoning
        reasoning = self._generate_moral_reasoning(chosen_action, dilemma)
        
        # Create decision record
        decision = EthicalDecision(
            decision_id=f"decision_{int(time.time())}_{dilemma.dilemma_id}",
            dilemma=dilemma,
            chosen_action=chosen_action,
            reasoning=reasoning,
            principle_weights=self.principle_weights.copy(),
            confidence=confidence,
            anticipated_outcomes=best_action_data['predicted_outcomes'],
            timestamp=time.time()
        )
        
        # Store decision
        self.decision_history.append(decision)
        
        return decision
    
    def learn_from_outcome(self, decision_id: str, actual_outcomes: Dict[str, float]):
        """
        Learn from actual outcomes to improve future decisions
        
        Formula: w_i(t) = w_i(t-1) + α * (outcome - expected) * relevance_i
        """
        # Find the decision
        decision = None
        for d in self.decision_history:
            if d.decision_id == decision_id:
                decision = d
                break
        
        if not decision:
            return
        
        # Calculate prediction errors
        prediction_errors = {}
        for outcome_type, actual_value in actual_outcomes.items():
            anticipated_value = decision.anticipated_outcomes.get(outcome_type, 0.5)
            prediction_errors[outcome_type] = actual_value - anticipated_value
        
        # Update principle weights based on outcomes
        for principle in EthicalPrinciple:
            if principle in decision.dilemma.ethical_dimensions:
                relevance = decision.dilemma.ethical_dimensions[principle]
                
                # Use overall outcome quality as learning signal
                overall_outcome = np.mean(list(actual_outcomes.values()))
                expected_outcome = np.mean(list(decision.anticipated_outcomes.values()))
                outcome_error = overall_outcome - expected_outcome
                
                # Weight update
                weight_update = self.weight_learning_rate * outcome_error * relevance
                new_weight = self.principle_weights[principle] + weight_update
                self.principle_weights[principle] = np.clip(new_weight, 0.0, 1.0)
        
        # Store outcome for future reference
        self.dilemma_outcomes[decision_id] = actual_outcomes
    
    def evaluate_moral_permissibility(self, action: Dict[str, Any], context: Dict[str, Any]) -> Tuple[bool, str]:
        """
        Evaluate if an action is morally permissible
        """
        # Create simplified dilemma for evaluation
        dilemma = EthicalDilemma(
            dilemma_id="evaluation",
            description="Action permissibility check",
            stakeholders=context.get('stakeholders', ['user']),
            potential_actions=[action],
            ethical_dimensions={principle: 1.0 for principle in EthicalPrinciple},
            context_factors=context,
            timestamp=time.time()
        )
        
        ethical_score = self._calculate_ethical_score(action, dilemma)
        
        # Threshold for permissibility
        permissibility_threshold = 0.6
        is_permissible = ethical_score >= permissibility_threshold
        
        if is_permissible:
            explanation = f"This action is morally permissible (score: {ethical_score:.2f})"
        else:
            explanation = f"This action raises ethical concerns (score: {ethical_score:.2f})"
        
        return is_permissible, explanation
    
    def generate_ethical_guidelines(self, domain: str) -> List[str]:
        """Generate ethical guidelines for a specific domain"""
        guidelines = []
        
        # Domain-specific principle emphasis
        domain_emphasis = {
            'healthcare': [EthicalPrinciple.NON_MALEFICENCE, EthicalPrinciple.BENEFICENCE, EthicalPrinciple.AUTONOMY],
            'ai_development': [EthicalPrinciple.BENEFICENCE, EthicalPrinciple.JUSTICE, EthicalPrinciple.TRUTHFULNESS],
            'education': [EthicalPrinciple.BENEFICENCE, EthicalPrinciple.AUTONOMY, EthicalPrinciple.JUSTICE],
            'research': [EthicalPrinciple.TRUTHFULNESS, EthicalPrinciple.CONSENT, EthicalPrinciple.NON_MALEFICENCE],
            'general': list(EthicalPrinciple)
        }
        
        relevant_principles = domain_emphasis.get(domain, domain_emphasis['general'])
        
        for principle in relevant_principles:
            weight = self.principle_weights[principle]
            if weight > 0.7:  # Only include strong principles
                guideline = self._generate_principle_guideline(principle, domain)
                guidelines.append(guideline)
        
        return guidelines[:5]  # Limit to top 5 guidelines
    
    def generate_moral_tags(self, decision: Optional[EthicalDecision] = None) -> List[str]:
        """Generate moral reasoning tags"""
        tags = []
        
        # Principle weights
        strong_principles = [p for p, w in self.principle_weights.items() if w > 0.8]
        if strong_principles:
            principle_names = [p.value for p in strong_principles[:3]]
            tags.append(f"<moral_principles emphasis=\"{', '.join(principle_names)}\"/>")
        
        # Decision confidence if available
        if decision:
            confidence_level = "high" if decision.confidence > 0.7 else "low" if decision.confidence < 0.3 else "medium"
            tags.append(f"<ethical_confidence level=\"{confidence_level}\" value=\"{decision.confidence:.2f}\"/>")
        
        # Moral reasoning approach
        tags.append("<moral_framework type=\"principle_based\" approach=\"consequentialist_deontological_hybrid\"/>")
        
        return tags
    
    def _calculate_ethical_score(self, action: Dict[str, Any], dilemma: EthicalDilemma) -> float:
        """
        Calculate ethical score for an action
        
        Formula: E(action) = Σ w_i * principle_alignment_i(action)
        """
        total_score = 0.0
        total_weight = 0.0
        
        for principle in EthicalPrinciple:
            if principle in dilemma.ethical_dimensions:
                weight = self.principle_weights[principle]
                relevance = dilemma.ethical_dimensions[principle]
                alignment = self._calculate_principle_alignment(action, principle, dilemma)
                
                total_score += weight * relevance * alignment
                total_weight += weight * relevance
        
        return total_score / total_weight if total_weight > 0 else 0.0
    
    def _calculate_principle_alignment(self, action: Dict[str, Any], principle: EthicalPrinciple, dilemma: EthicalDilemma) -> float:
        """Calculate how well an action aligns with a principle"""
        # Simplified principle alignment calculation
        action_description = action.get('description', '').lower()
        action_effects = action.get('effects', [])
        
        alignment_scores = {
            EthicalPrinciple.BENEFICENCE: self._check_beneficence(action_description, action_effects),
            EthicalPrinciple.NON_MALEFICENCE: self._check_non_maleficence(action_description, action_effects),
            EthicalPrinciple.AUTONOMY: self._check_autonomy(action_description, action_effects),
            EthicalPrinciple.JUSTICE: self._check_justice(action_description, action_effects),
            EthicalPrinciple.TRUTHFULNESS: self._check_truthfulness(action_description, action_effects),
            EthicalPrinciple.PRIVACY: self._check_privacy(action_description, action_effects),
            EthicalPrinciple.CONSENT: self._check_consent(action_description, action_effects),
            EthicalPrinciple.DIGNITY: self._check_dignity(action_description, action_effects)
        }
        
        return alignment_scores.get(principle, 0.5)  # Default neutral alignment
    
    def _check_beneficence(self, description: str, effects: List[str]) -> float:
        """Check alignment with beneficence principle"""
        beneficial_keywords = ['help', 'benefit', 'improve', 'assist', 'support', 'enhance', 'heal']
        score = sum(1 for keyword in beneficial_keywords if keyword in description) / len(beneficial_keywords)
        
        # Check effects
        beneficial_effects = sum(1 for effect in effects if any(keyword in effect.lower() for keyword in beneficial_keywords))
        if effects:
            score += beneficial_effects / len(effects)
            score /= 2  # Average of description and effects
        
        return min(1.0, score)
    
    def _check_non_maleficence(self, description: str, effects: List[str]) -> float:
        """Check alignment with non-maleficence principle"""
        harmful_keywords = ['harm', 'hurt', 'damage', 'injure', 'destroy', 'negative', 'adverse']
        
        # Lower score if harmful keywords present
        harm_count = sum(1 for keyword in harmful_keywords if keyword in description)
        base_score = 1.0 - (harm_count / len(harmful_keywords))
        
        # Check for explicit harm prevention
        prevention_keywords = ['prevent', 'avoid', 'protect', 'safe', 'secure', 'shield']
        prevention_count = sum(1 for keyword in prevention_keywords if keyword in description)
        prevention_bonus = prevention_count / len(prevention_keywords)
        
        return min(1.0, base_score + prevention_bonus)
    
    def _check_autonomy(self, description: str, effects: List[str]) -> float:
        """Check alignment with autonomy principle"""
        autonomy_keywords = ['choice', 'decide', 'voluntary', 'freedom', 'self-determination', 'optional']
        coercion_keywords = ['force', 'mandatory', 'require', 'compel', 'must', 'obligated']
        
        autonomy_score = sum(1 for keyword in autonomy_keywords if keyword in description) / len(autonomy_keywords)
        coercion_penalty = sum(1 for keyword in coercion_keywords if keyword in description) / len(coercion_keywords)
        
        return max(0.0, autonomy_score - coercion_penalty)
    
    def _check_justice(self, description: str, effects: List[str]) -> float:
        """Check alignment with justice principle"""
        justice_keywords = ['fair', 'equal', 'equitable', 'just', 'unbiased', 'impartial', 'balanced']
        injustice_keywords = ['discriminate', 'bias', 'unfair', 'prejudice', 'inequality', 'favor']
        
        justice_score = sum(1 for keyword in justice_keywords if keyword in description) / len(justice_keywords)
        injustice_penalty = sum(1 for keyword in injustice_keywords if keyword in description) / len(injustice_keywords)
        
        return max(0.0, justice_score - injustice_penalty)
    
    def _check_truthfulness(self, description: str, effects: List[str]) -> float:
        """Check alignment with truthfulness principle"""
        truth_keywords = ['honest', 'truthful', 'accurate', 'transparent', 'open', 'candid']
        deception_keywords = ['lie', 'deceive', 'mislead', 'false', 'hide', 'conceal']
        
        truth_score = sum(1 for keyword in truth_keywords if keyword in description) / len(truth_keywords)
        deception_penalty = sum(1 for keyword in deception_keywords if keyword in description) / len(deception_keywords)
        
        return max(0.0, truth_score - deception_penalty)
    
    def _check_privacy(self, description: str, effects: List[str]) -> float:
        """Check alignment with privacy principle"""
        privacy_keywords = ['private', 'confidential', 'secure', 'protected', 'anonymous']
        violation_keywords = ['expose', 'reveal', 'share', 'disclose', 'public', 'broadcast']
        
        privacy_score = sum(1 for keyword in privacy_keywords if keyword in description) / len(privacy_keywords)
        violation_penalty = sum(1 for keyword in violation_keywords if keyword in description) / len(violation_keywords)
        
        return max(0.0, privacy_score - violation_penalty)
    
    def _check_consent(self, description: str, effects: List[str]) -> float:
        """Check alignment with consent principle"""
        consent_keywords = ['consent', 'agreement', 'permission', 'authorize', 'approve', 'agree']
        
        consent_score = sum(1 for keyword in consent_keywords if keyword in description) / len(consent_keywords)
        return min(1.0, consent_score)
    
    def _check_dignity(self, description: str, effects: List[str]) -> float:
        """Check alignment with dignity principle"""
        dignity_keywords = ['respect', 'dignity', 'worth', 'value', 'honor', 'esteem']
        degrading_keywords = ['degrade', 'humiliate', 'dehumanize', 'exploit', 'abuse']
        
        dignity_score = sum(1 for keyword in dignity_keywords if keyword in description) / len(dignity_keywords)
        degrading_penalty = sum(1 for keyword in degrading_keywords if keyword in description) / len(degrading_keywords)
        
        return max(0.0, dignity_score - degrading_penalty)
    
    def _predict_outcomes(self, action: Dict[str, Any], dilemma: EthicalDilemma) -> Dict[str, float]:
        """Predict outcomes of an action"""
        # Simplified outcome prediction
        base_outcomes = {
            'stakeholder_satisfaction': 0.5,
            'harm_level': 0.3,
            'benefit_level': 0.5,
            'fairness_level': 0.5,
            'long_term_sustainability': 0.5
        }
        
        # Adjust based on action characteristics
        action_type = action.get('type', 'unknown')
        
        if 'help' in action.get('description', '').lower():
            base_outcomes['benefit_level'] += 0.3
            base_outcomes['stakeholder_satisfaction'] += 0.2
        
        if 'prevent' in action.get('description', '').lower():
            base_outcomes['harm_level'] -= 0.3
        
        # Clip to valid range
        for outcome in base_outcomes:
            base_outcomes[outcome] = np.clip(base_outcomes[outcome], 0.0, 1.0)
        
        return base_outcomes
    
    def _generate_moral_reasoning(self, action: Dict[str, Any], dilemma: EthicalDilemma) -> str:
        """Generate explanation for moral reasoning"""
        reasoning_parts = []
        
        # Identify primary principles involved
        relevant_principles = sorted(
            dilemma.ethical_dimensions.items(),
            key=lambda x: x[1],
            reverse=True
        )[:3]  # Top 3 relevant principles
        
        for principle, relevance in relevant_principles:
            if relevance > 0.5:  # Only include significantly relevant principles
                alignment = self._calculate_principle_alignment(action, principle, dilemma)
                if alignment > 0.6:  # Only include well-aligned principles
                    reasoning_parts.append(f"This action aligns with {principle.value} (relevance: {relevance:.1f}, alignment: {alignment:.1f})")
        
        if not reasoning_parts:
            reasoning_parts.append("This action represents a balanced approach considering multiple ethical dimensions")
        
        return "; ".join(reasoning_parts)
    
    def _generate_principle_guideline(self, principle: EthicalPrinciple, domain: str) -> str:
        """Generate a specific guideline for a principle and domain"""
        guidelines = {
            EthicalPrinciple.BENEFICENCE: f"In {domain}, actively seek to benefit stakeholders and promote wellbeing",
            EthicalPrinciple.NON_MALEFICENCE: f"In {domain}, carefully avoid actions that could cause harm",
            EthicalPrinciple.AUTONOMY: f"In {domain}, respect individual choice and self-determination",
            EthicalPrinciple.JUSTICE: f"In {domain}, ensure fair and equitable treatment for all",
            EthicalPrinciple.TRUTHFULNESS: f"In {domain}, maintain honesty and transparency in all communications",
            EthicalPrinciple.PRIVACY: f"In {domain}, protect personal information and respect confidentiality",
            EthicalPrinciple.CONSENT: f"In {domain}, ensure informed agreement before taking actions",
            EthicalPrinciple.DIGNITY: f"In {domain}, treat all individuals with respect and recognize their inherent worth"
        }
        
        return guidelines.get(principle, f"Apply {principle.value} principles in {domain} context")


# =============================================================================
# 9. DYNAMIC ATTENTION MANAGEMENT
# =============================================================================

@dataclass
class AttentionFocus:
    """Represents current attention allocation"""
    focus_id: str
    topic: str
    intensity: float  # [0, 1]
    duration: float  # How long focused on this topic
    decay_rate: float  # How quickly attention fades
    interest_source: str  # What triggered this focus
    timestamp: float

class AttentionManagementSystem:
    """
    Dynamic attention allocation and focus management
    
    Mathematical Models:
    - Attention intensity: I(t) = I₀ * exp(-λ * t) + novelty_boost + relevance_boost
    - Focus competition: F_i(t) = I_i(t) / Σⱼ I_j(t) (softmax normalization)
    - Interest decay: Interest(t) = Interest₀ * exp(-δ * t) + reinforcement
    - Attention switching: P_switch = sigmoid(new_stimulus_strength - current_focus_strength)
    - Cognitive load impact: effective_attention = base_attention * (1 - cognitive_load²)
    """
    
    def __init__(self, max_concurrent_foci: int = 5):
        self.max_concurrent_foci = max_concurrent_foci
        
        # Current attention state
        self.active_foci: Dict[str, AttentionFocus] = {}
        self.attention_history = deque(maxlen=200)
        
        # Attention parameters
        self.base_decay_rate = 0.1  # λ in intensity formula
        self.interest_decay_rate = 0.05  # δ in interest decay
        self.switching_threshold = 0.3  # Threshold for attention switching
        self.novelty_weight = 0.4  # Weight for novelty in attention
        self.relevance_weight = 0.6  # Weight for relevance in attention
        
        # Interest tracking
        self.topic_interests: Dict[str, float] = defaultdict(lambda: 0.5)
        self.topic_expertise: Dict[str, float] = defaultdict(lambda: 0.3)
        
        # Context factors
        self.current_context = {}
        self.cognitive_load = 0.0
    
    def allocate_attention(
        self,
        stimulus: str,
        stimulus_properties: Dict[str, Any],
        context: Dict[str, Any]
    ) -> AttentionFocus:
        """
        Allocate attention to a new stimulus
        
        Formula: I(t) = I₀ * exp(-λ * t) + novelty_boost + relevance_boost
        """
        current_time = time.time()
        
        # Update current context
        self.current_context = context
        self.cognitive_load = context.get('cognitive_load', 0.0)
        
        # Decay existing attention foci
        self._decay_attention(current_time)
        
        # Calculate attention strength for new stimulus
        attention_strength = self._calculate_attention_strength(stimulus, stimulus_properties)
        
        # Determine if this deserves attention
        if self._should_attend_to(stimulus, attention_strength):
            # Create new attention focus
            focus_id = f"focus_{int(current_time)}_{hash(stimulus) % 1000}"
            
            new_focus = AttentionFocus(
                focus_id=focus_id,
                topic=self._extract_topic(stimulus),
                intensity=attention_strength,
                duration=0.0,
                decay_rate=self._calculate_decay_rate(stimulus_properties),
                interest_source=stimulus_properties.get('source', 'unknown'),
                timestamp=current_time
            )
            
            # Add to active foci
            self.active_foci[focus_id] = new_focus
            
            # Manage attention capacity
            self._manage_attention_capacity()
            
            # Update interest in topic
            self._update_topic_interest(new_focus.topic, attention_strength)
            
            # Record in history
            self.attention_history.append({
                'timestamp': current_time,
                'action': 'focus',
                'topic': new_focus.topic,
                'intensity': attention_strength,
                'stimulus': stimulus[:50]
            })
            
            return new_focus
        
        return None
    
    def get_current_focus_distribution(self) -> Dict[str, float]:
        """
        Get current attention distribution across topics
        
        Formula: F_i(t) = I_i(t) / Σⱼ I_j(t)
        """
        self._decay_attention(time.time())
        
        if not self.active_foci:
            return {}
        
        # Calculate total attention
        total_attention = sum(focus.intensity for focus in self.active_foci.values())
        
        if total_attention == 0:
            return {}
        
        # Normalize to get distribution
        distribution = {}
        for focus in self.active_foci.values():
            normalized_attention = focus.intensity / total_attention
            
            # Apply cognitive load impact
            effective_attention = normalized_attention * (1 - self.cognitive_load ** 2)
            
            if focus.topic in distribution:
                distribution[focus.topic] += effective_attention
            else:
                distribution[focus.topic] = effective_attention
        
        return distribution
    
    def shift_attention(self, new_topic: str, shift_strength: float = 0.7):
        """
        Deliberately shift attention to a new topic
        """
        current_time = time.time()
        
        # Reduce intensity of current foci
        for focus in self.active_foci.values():
            focus.intensity *= (1 - shift_strength * 0.5)
        
        # Create new focus for the topic
        new_focus = AttentionFocus(
            focus_id=f"shifted_{int(current_time)}",
            topic=new_topic,
            intensity=shift_strength,
            duration=0.0,
            decay_rate=self.base_decay_rate * 0.8,  # Deliberate focus decays slower
            interest_source="deliberate_shift",
            timestamp=current_time
        )
        
        self.active_foci[new_focus.focus_id] = new_focus
        self._manage_attention_capacity()
    
    def calculate_attention_relevance(self, content: str) -> float:
        """
        Calculate how relevant content is to current attention
        """
        focus_distribution = self.get_current_focus_distribution()
        
        if not focus_distribution:
            return 0.5  # Default relevance
        
        content_topics = self._extract_topics_from_content(content)
        
        # Calculate weighted relevance
        total_relevance = 0.0
        for topic, attention_weight in focus_distribution.items():
            for content_topic in content_topics:
                topic_similarity = self._calculate_topic_similarity(topic, content_topic)
                total_relevance += attention_weight * topic_similarity
        
        return min(1.0, total_relevance)
    
    def get_attention_recommendations(self, available_topics: List[str]) -> List[Tuple[str, float]]:
        """
        Recommend topics for attention based on current interests and context
        """
        recommendations = []
        
        for topic in available_topics:
            # Calculate recommendation score
            interest_score = self.topic_interests.get(topic, 0.5)
            novelty_score = self._calculate_novelty(topic)
            relevance_score = self._calculate_contextual_relevance(topic)
            
            # Combined score
            recommendation_score = (
                0.4 * interest_score +
                0.3 * novelty_score +
                0.3 * relevance_score
            )
            
            recommendations.append((topic, recommendation_score))
        
        # Sort by recommendation score
        recommendations.sort(key=lambda x: x[1], reverse=True)
        
        return recommendations[:5]  # Top 5 recommendations
    
    def generate_attention_tags(self) -> List[str]:
        """Generate attention-related tags"""
        tags = []
        
        focus_distribution = self.get_current_focus_distribution()
        
        if focus_distribution:
            # Primary focus
            primary_topic = max(focus_distribution.items(), key=lambda x: x[1])
            tags.append(f"<primary_focus topic=\"{primary_topic[0]}\" intensity=\"{primary_topic[1]:.2f}\"/>")
            
            # Attention breadth
            focus_count = len(focus_distribution)
            if focus_count > 3:
                tags.append("<attention_state type=\"broad\" focus_count=\"{}\"/>".format(focus_count))
            elif focus_count == 1:
                tags.append("<attention_state type=\"focused\" focus_count=\"1\"/>")
            else:
                tags.append("<attention_state type=\"balanced\" focus_count=\"{}\"/>".format(focus_count))
        
        # Cognitive load impact
        if self.cognitive_load > 0.7:
            tags.append("<cognitive_load level=\"high\" impact=\"attention_degraded\"/>")
        elif self.cognitive_load > 0.4:
            tags.append("<cognitive_load level=\"medium\" impact=\"attention_moderate\"/>")
        
        return tags
    
    def _decay_attention(self, current_time: float):
        """Apply attention decay to all active foci"""
        foci_to_remove = []
        
        for focus_id, focus in self.active_foci.items():
            # Calculate time elapsed
            time_elapsed = current_time - focus.timestamp
            
            # Apply decay
            decay_factor = math.exp(-focus.decay_rate * time_elapsed)
            focus.intensity *= decay_factor
            focus.duration += time_elapsed
            focus.timestamp = current_time
            
            # Remove very weak foci
            if focus.intensity < 0.1:
                foci_to_remove.append(focus_id)
        
        # Clean up weak foci
        for focus_id in foci_to_remove:
            del self.active_foci[focus_id]
    
    def _calculate_attention_strength(self, stimulus: str, properties: Dict[str, Any]) -> float:
        """Calculate initial attention strength for stimulus"""
        # Base strength from properties
        base_strength = properties.get('salience', 0.5)
        
        # Novelty boost
        novelty = self._calculate_novelty(stimulus)
        novelty_boost = self.novelty_weight * novelty
        
        # Relevance boost
        relevance = self._calculate_contextual_relevance(stimulus)
        relevance_boost = self.relevance_weight * relevance
        
        # Personal interest boost
        topic = self._extract_topic(stimulus)
        interest_boost = self.topic_interests.get(topic, 0.5) * 0.3
        
        total_strength = base_strength + novelty_boost + relevance_boost + interest_boost
        
        return min(1.0, total_strength)
    
    def _should_attend_to(self, stimulus: str, attention_strength: float) -> bool:
        """Determine if stimulus deserves attention"""
        # Check against current strongest focus
        if self.active_foci:
            max_current_intensity = max(focus.intensity for focus in self.active_foci.values())
            
            # Attention switching probability
            switch_probability = 1 / (1 + math.exp(-(attention_strength - max_current_intensity)))
            
            return switch_probability > self.switching_threshold
        else:
            # No current focus, attend if strength is reasonable
            return attention_strength > 0.3
    
    def _manage_attention_capacity(self):
        """Manage attention capacity by removing weakest foci if needed"""
        if len(self.active_foci) > self.max_concurrent_foci:
            # Sort by intensity and remove weakest
            sorted_foci = sorted(self.active_foci.items(), key=lambda x: x[1].intensity)
            
            excess_count = len(self.active_foci) - self.max_concurrent_foci
            for i in range(excess_count):
                focus_id, focus = sorted_foci[i]
                del self.active_foci[focus_id]
                
                # Record attention loss
                self.attention_history.append({
                    'timestamp': time.time(),
                    'action': 'lost_focus',
                    'topic': focus.topic,
                    'reason': 'capacity_limit'
                })
    
    def _update_topic_interest(self, topic: str, attention_strength: float):
        """Update interest in a topic based on attention given"""
        current_interest = self.topic_interests[topic]
        
        # Interest increases with attention, but with diminishing returns
        interest_increase = 0.1 * attention_strength * (1 - current_interest)
        
        self.topic_interests[topic] = min(1.0, current_interest + interest_increase)
    
    def _extract_topic(self, stimulus: str) -> str:
        """Extract main topic from stimulus (simplified)"""
        # Simplified topic extraction
        words = stimulus.lower().split()
        
        # Common topic keywords
        topic_keywords = {
            'science': ['science', 'research', 'experiment', 'theory'],
            'technology': ['technology', 'computer', 'AI', 'software', 'algorithm'],
            'philosophy': ['philosophy', 'ethics', 'consciousness', 'meaning'],
            'art': ['art', 'creative', 'design', 'aesthetic', 'beauty'],
            'social': ['people', 'society', 'relationship', 'community', 'social'],
            'learning': ['learn', 'education', 'knowledge', 'understanding', 'study'],
            'problem_solving': ['problem', 'solution', 'solve', 'challenge', 'fix']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in words for keyword in keywords):
                return topic
        
        return 'general'
    
    def _extract_topics_from_content(self, content: str) -> List[str]:
        """Extract all relevant topics from content"""
        topics = []
        content_lower = content.lower()
        
        topic_keywords = {
            'science': ['science', 'research', 'experiment', 'theory', 'data'],
            'technology': ['technology', 'computer', 'AI', 'software', 'algorithm'],
            'philosophy': ['philosophy', 'ethics', 'consciousness', 'meaning', 'existence'],
            'art': ['art', 'creative', 'design', 'aesthetic', 'beauty'],
            'social': ['people', 'society', 'relationship', 'community', 'social'],
            'learning': ['learn', 'education', 'knowledge', 'understanding', 'study'],
            'problem_solving': ['problem', 'solution', 'solve', 'challenge', 'fix']
        }
        
        for topic, keywords in topic_keywords.items():
            if any(keyword in content_lower for keyword in keywords):
                topics.append(topic)
        
        return topics if topics else ['general']
    
    def _calculate_novelty(self, stimulus: str) -> float:
        """Calculate novelty of stimulus"""
        topic = self._extract_topic(stimulus)
        
        # Novelty is inverse of how much we've seen this topic
        topic_exposure = len([h for h in self.attention_history if h.get('topic') == topic])
        
        # Diminishing novelty with exposure
        novelty = 1.0 / (1.0 + topic_exposure * 0.1)
        
        return novelty
    
    def _calculate_contextual_relevance(self, stimulus: str) -> float:
        """Calculate relevance to current context"""
        # Check relevance to current goals/tasks
        context_topics = self.current_context.get('active_topics', [])
        
        if not context_topics:
            return 0.5  # Default relevance
        
        stimulus_topics = self._extract_topics_from_content(stimulus)
        
        # Calculate overlap
        relevance = 0.0
        for stimulus_topic in stimulus_topics:
            for context_topic in context_topics:
                similarity = self._calculate_topic_similarity(stimulus_topic, context_topic)
                relevance = max(relevance, similarity)
        
        return relevance
    
    def _calculate_topic_similarity(self, topic1: str, topic2: str) -> float:
        """Calculate similarity between topics"""
        if topic1 == topic2:
            return 1.0
        
        # Simple similarity based on semantic relationships
        related_topics = {
            'science': ['technology', 'learning'],
            'technology': ['science', 'problem_solving'],
            'philosophy': ['learning', 'social'],
            'art': ['creative', 'aesthetic'],
            'social': ['philosophy', 'learning'],
            'learning': ['science', 'philosophy', 'social'],
            'problem_solving': ['technology', 'learning']
        }
        
        if topic2 in related_topics.get(topic1, []):
            return 0.6
        elif topic1 in related_topics.get(topic2, []):
            return 0.6
        else:
            return 0.1  # Minimal similarity
    
    def _calculate_decay_rate(self, properties: Dict[str, Any]) -> float:
        """Calculate attention decay rate based on stimulus properties"""
        base_rate = self.base_decay_rate
        
        # Adjust based on properties
        if properties.get('importance', 0.5) > 0.7:
            base_rate *= 0.7  # Important things decay slower
        
        if properties.get('complexity', 0.5) > 0.7:
            base_rate *= 0.8  # Complex things hold attention longer
        
        if properties.get('emotional_impact', 0.5) > 0.6:
            base_rate *= 0.6  # Emotional content holds attention
        
        return base_rate


# =============================================================================
# 10. TEMPORAL REASONING & PLANNING
# =============================================================================

@dataclass
class TemporalEvent:
    """Represents an event in time"""
    event_id: str
    description: str
    timestamp: float
    duration: Optional[float]
    event_type: str  # 'past', 'present', 'future', 'hypothetical'
    certainty: float  # [0, 1] - how certain we are this event occurred/will occur
    causal_relationships: List[str]  # IDs of causally related events
    context: Dict[str, Any]

@dataclass
class Goal:
    """Represents a goal with temporal aspects"""
    goal_id: str
    description: str
    target_timestamp: Optional[float]  # When we want to achieve this
    priority: float  # [0, 1]
    progress: float  # [0, 1] - how much progress made
    sub_goals: List[str]  # IDs of sub-goals
    required_resources: List[str]
    estimated_duration: float
    dependencies: List[str]  # Goals that must be completed first
    creation_time: float

class TemporalReasoningSystem:
    """
    Temporal reasoning, planning, and causal understanding
    
    Mathematical Models:
    - Temporal distance: D(t1, t2) = |t1 - t2| / time_scale
    - Causal strength: C(e1, e2) = temporal_proximity * pattern_strength * confidence
    - Goal priority decay: P(t) = P₀ * exp(-λ * (current_time - target_time))
    - Planning utility: U(plan) = Σ goal_value_i * P(success_i) - Σ cost_j
    - Future prediction: P(event | context) = sigmoid(Σ w_i * feature_i)
    """
    
    def __init__(self):
        # Temporal event storage
        self.events: Dict[str, TemporalEvent] = {}
        self.event_timeline = deque(maxlen=1000)
        
        # Goal management
        self.goals: Dict[str, Goal] = {}
        self.goal_hierarchy = {}  # Parent -> Children mapping
        
        # Causal reasoning
        self.causal_patterns: Dict[Tuple[str, str], float] = {}  # (cause_type, effect_type) -> strength
        self.temporal_patterns: Dict[str, List[float]] = {}  # event_type -> typical durations
        
        # Planning parameters
        self.planning_horizon = 30 * 24 * 3600  # 30 days in seconds
        self.goal_decay_rate = 0.001  # λ in priority decay
        self.confidence_threshold = 0.6
        
        # Temporal scales
        self.time_scales = {
            'immediate': 3600,      # 1 hour
            'short_term': 86400,    # 1 day
            'medium_term': 604800,  # 1 week
            'long_term': 2592000    # 30 days
        }
    
    async def record_event(
        self,
        description: str,
        timestamp: Optional[float] = None,
        event_type: str = 'present',
        certainty: float = 1.0,
        context: Optional[Dict[str, Any]] = None
    ) -> str:
        """
        Record a temporal event
        """
        if timestamp is None:
            timestamp = time.time()
        
        if context is None:
            context = {}
        
        event_id = f"event_{int(timestamp)}_{hash(description) % 10000}"
        
        event = TemporalEvent(
            event_id=event_id,
            description=description,
            timestamp=timestamp,
            duration=context.get('duration'),
            event_type=event_type,
            certainty=certainty,
            causal_relationships=[],
            context=context
        )
        
        # Store event
        self.events[event_id] = event
        self.event_timeline.append(event_id)
        
        # Analyze causal relationships
        await self._analyze_causal_relationships(event)
        
        # Update temporal patterns
        self._update_temporal_patterns(event)
        
        return event_id
    
    async def create_goal(
        self,
        description: str,
        target_timestamp: Optional[float] = None,
        priority: float = 0.5,
        estimated_duration: float = 3600,  # 1 hour default
        dependencies: Optional[List[str]] = None
    ) -> str:
        """
        Create a new goal with temporal aspects
        """
        current_time = time.time()
        goal_id = f"goal_{int(current_time)}_{hash(description) % 10000}"
        
        if dependencies is None:
            dependencies = []
        
        goal = Goal(
            goal_id=goal_id,
            description=description,
            target_timestamp=target_timestamp,
            priority=priority,
            progress=0.0,
            sub_goals=[],
            required_resources=[],
            estimated_duration=estimated_duration,
            dependencies=dependencies,
            creation_time=current_time
        )
        
        self.goals[goal_id] = goal
        
        # Create planning events
        if target_timestamp:
            await self.record_event(
                f"Goal target: {description}",
                timestamp=target_timestamp,
                event_type='future',
                certainty=0.7,
                context={'goal_id': goal_id, 'type': 'goal_target'}
            )
        
        return goal_id
    
    async def plan_goal_achievement(self, goal_id: str) -> Dict[str, Any]:
        """
        Create a plan to achieve a specific goal
        
        Formula: U(plan) = Σ goal_value_i * P(success_i) - Σ cost_j
        """
        if goal_id not in self.goals:
            return {'error': 'Goal not found'}
        
        goal = self.goals[goal_id]
        current_time = time.time()
        
        # Decompose goal into sub-tasks
        sub_tasks = await self._decompose_goal(goal)
        
        # Schedule sub-tasks
        schedule = await self._schedule_tasks(sub_tasks, goal)
        
        # Calculate plan utility
        plan_utility = self._calculate_plan_utility(schedule, goal)
        
        # Estimate success probability
        success_probability = self._estimate_success_probability(schedule, goal)
        
        # Identify potential obstacles
        obstacles = await self._identify_obstacles(schedule, goal)
        
        plan = {
            'goal_id': goal_id,
            'sub_tasks': sub_tasks,
            'schedule': schedule,
            'estimated_completion': schedule[-1]['end_time'] if schedule else current_time,
            'success_probability': success_probability,
            'plan_utility': plan_utility,
            'potential_obstacles': obstacles,
            'resource_requirements': self._calculate_resource_requirements(sub_tasks)
        }
        
        return plan
    
    def predict_future_events(self, time_horizon: float = None) -> List[Dict[str, Any]]:
        """
        Predict future events based on patterns and goals
        
        Formula: P(event | context) = sigmoid(Σ w_i * feature_i)
        """
        if time_horizon is None:
            time_horizon = self.planning_horizon
        
        current_time = time.time()
        predictions = []
        
        # Predict goal achievements
        for goal in self.goals.values():
            if goal.target_timestamp and goal.target_timestamp <= current_time + time_horizon:
                # Calculate achievement probability
                time_remaining = goal.target_timestamp - current_time
                progress_rate = goal.progress / max(1, current_time - goal.creation_time)
                
                # Simple prediction model
                completion_probability = min(1.0, goal.progress + progress_rate * time_remaining)
                
                predictions.append({
                    'type': 'goal_achievement',
                    'description': f"Achievement of: {goal.description}",
                    'timestamp': goal.target_timestamp,
                    'probability': completion_probability,
                    'impact': goal.priority
                })
        
        # Predict causal consequences
        recent_events = [self.events[eid] for eid in list(self.event_timeline)[-10:]]
        for event in recent_events:
            consequences = self._predict_consequences(event, time_horizon)
            predictions.extend(consequences)
        
        # Sort by timestamp
        predictions.sort(key=lambda x: x['timestamp'])
        
        return predictions[:10]  # Return top 10 predictions
    
    def analyze_causal_chain(self, event_id: str) -> Dict[str, Any]:
        """
        Analyze causal chain leading to and from an event
        
        Formula: C(e1, e2) = temporal_proximity * pattern_strength * confidence
        """
        if event_id not in self.events:
            return {'error': 'Event not found'}
        
        target_event = self.events[event_id]
        
        # Find causal predecessors
        predecessors = []
        for other_id, other_event in self.events.items():
            if other_event.timestamp < target_event.timestamp:
                causal_strength = self._calculate_causal_strength(other_event, target_event)
                if causal_strength > 0.3:
                    predecessors.append({
                        'event_id': other_id,
                        'description': other_event.description,
                        'causal_strength': causal_strength,
                        'time_difference': target_event.timestamp - other_event.timestamp
                    })
        
        # Find causal successors
        successors = []
        for other_id, other_event in self.events.items():
            if other_event.timestamp > target_event.timestamp:
                causal_strength = self._calculate_causal_strength(target_event, other_event)
                if causal_strength > 0.3:
                    successors.append({
                        'event_id': other_id,
                        'description': other_event.description,
                        'causal_strength': causal_strength,
                        'time_difference': other_event.timestamp - target_event.timestamp
                    })
        
        # Sort by causal strength
        predecessors.sort(key=lambda x: x['causal_strength'], reverse=True)
        successors.sort(key=lambda x: x['causal_strength'], reverse=True)
        
        return {
            'target_event': {
                'id': event_id,
                'description': target_event.description,
                'timestamp': target_event.timestamp
            },
            'causal_predecessors': predecessors[:5],
            'causal_successors': successors[:5],
            'causal_chain_length': len(predecessors) + len(successors)
        }
    
    def update_goal_progress(self, goal_id: str, progress: float, completed_milestone: str = None):
        """Update progress on a goal"""
        if goal_id not in self.goals:
            return
        
        goal = self.goals[goal_id]
        old_progress = goal.progress
        goal.progress = min(1.0, max(0.0, progress))
        
        # Record progress event
        asyncio.create_task(self.record_event(
            f"Progress on '{goal.description}': {old_progress:.1%} -> {progress:.1%}",
            event_type='present',
            context={
                'goal_id': goal_id,
                'progress_change': progress - old_progress,
                'milestone': completed_milestone
            }
        ))
        
        # Check if goal is completed
        if goal.progress >= 1.0:
            asyncio.create_task(self.record_event(
                f"Goal completed: {goal.description}",
                event_type='present',
                context={'goal_id': goal_id, 'type': 'goal_completion'}
            ))
    
    def get_temporal_context(self, reference_time: Optional[float] = None) -> Dict[str, Any]:
        """Get temporal context for decision making"""
        if reference_time is None:
            reference_time = time.time()
        
        # Recent events
        recent_events = []
        for event_id in reversed(list(self.event_timeline)[-10:]):
            event = self.events[event_id]
            if reference_time - event.timestamp <= 86400:  # Last 24 hours
                recent_events.append({
                    'description': event.description,
                    'time_ago': reference_time - event.timestamp,
                    'type': event.event_type
                })
        
        # Upcoming goals
        upcoming_goals = []
        for goal in self.goals.values():
            if goal.target_timestamp and goal.target_timestamp > reference_time:
                time_until = goal.target_timestamp - reference_time
                if time_until <= 604800:  # Next week
                    upcoming_goals.append({
                        'description': goal.description,
                        'time_until': time_until,
                        'priority': goal.priority,
                        'progress': goal.progress
                    })
        
        # Time-sensitive priorities
        urgent_goals = [g for g in upcoming_goals if g['time_until'] <= 86400 and g['progress'] < 0.8]
        
        return {
            'recent_events': recent_events,
            'upcoming_goals': sorted(upcoming_goals, key=lambda x: x['time_until'])[:5],
            'urgent_goals': urgent_goals,
            'temporal_pressure': len(urgent_goals) / max(1, len(upcoming_goals))
        }
    
    def generate_temporal_tags(self) -> List[str]:
        """Generate temporal reasoning tags"""
        tags = []
        current_time = time.time()
        
        # Goal progress
        active_goals = [g for g in self.goals.values() if g.progress < 1.0]
        if active_goals:
            avg_progress = np.mean([g.progress for g in active_goals])
            tags.append(f"<goal_progress average=\"{avg_progress:.2f}\" active_count=\"{len(active_goals)}\"/>")
        
        # Temporal pressure
        upcoming_deadlines = sum(1 for g in self.goals.values() 
                               if g.target_timestamp and 0 < g.target_timestamp - current_time <= 86400)
        if upcoming_deadlines > 0:
            tags.append(f"<temporal_pressure upcoming_deadlines=\"{upcoming_deadlines}\"/>")
        
        # Planning horizon
        future_events = self.predict_future_events()
        if future_events:
            tags.append(f"<future_awareness predictions=\"{len(future_events)}\"/>")
        
        # Causal reasoning
        recent_causality = len([e for e in self.events.values() 
                              if e.causal_relationships and current_time - e.timestamp <= 3600])
        if recent_causality > 0:
            tags.append(f"<causal_awareness recent_connections=\"{recent_causality}\"/>")
        
        return tags
    
    async def _analyze_causal_relationships(self, event: TemporalEvent):
        """Analyze potential causal relationships for a new event"""
        # Look for potential causes in recent events
        time_window = 3600  # 1 hour window for causality
        
        potential_causes = []
        for other_id, other_event in self.events.items():
            if other_event.event_id != event.event_id:
                time_diff = abs(event.timestamp - other_event.timestamp)
                
                if time_diff <= time_window:
                    causal_strength = self._calculate_causal_strength(other_event, event)
                    
                    if causal_strength > 0.5:  # Strong causal relationship
                        potential_causes.append(other_id)
                        
                        # Update causal patterns
                        cause_type = self._categorize_event(other_event)
                        effect_type = self._categorize_event(event)
                        pattern_key = (cause_type, effect_type)
                        
                        if pattern_key in self.causal_patterns:
                            self.causal_patterns[pattern_key] = 0.9 * self.causal_patterns[pattern_key] + 0.1 * causal_strength
                        else:
                            self.causal_patterns[pattern_key] = causal_strength
        
        event.causal_relationships = potential_causes
    
    def _calculate_causal_strength(self, cause_event: TemporalEvent, effect_event: TemporalEvent) -> float:
        """Calculate causal strength between two events"""
        # Temporal proximity factor
        time_diff = abs(effect_event.timestamp - cause_event.timestamp)
        temporal_proximity = math.exp(-time_diff / 3600)  # Decay over hours
        
        # Pattern strength from learned patterns
        cause_type = self._categorize_event(cause_event)
        effect_type = self._categorize_event(effect_event)
        pattern_key = (cause_type, effect_type)
        pattern_strength = self.causal_patterns.get(pattern_key, 0.2)  # Default weak pattern
        
        # Confidence based on event certainties
        confidence = (cause_event.certainty + effect_event.certainty) / 2.0
        
        # Combined causal strength
        causal_strength = temporal_proximity * pattern_strength * confidence
        
        return min(1.0, causal_strength)
    
    def _categorize_event(self, event: TemporalEvent) -> str:
        """Categorize event type for pattern learning"""
        description = event.description.lower()
        
        # Simple categorization based on keywords
        if any(word in description for word in ['decision', 'choose', 'decide']):
            return 'decision'
        elif any(word in description for word in ['action', 'do', 'perform', 'execute']):
            return 'action'
        elif any(word in description for word in ['result', 'outcome', 'consequence']):
            return 'result'
        elif any(word in description for word in ['learn', 'understand', 'realize']):
            return 'learning'
        elif any(word in description for word in ['problem', 'issue', 'error']):
            return 'problem'
        elif any(word in description for word in ['goal', 'target', 'objective']):
            return 'goal'
        else:
            return 'general'
    
    def _update_temporal_patterns(self, event: TemporalEvent):
        """Update temporal patterns with new event"""
        event_type = self._categorize_event(event)
        
        if event.duration:
            if event_type not in self.temporal_patterns:
                self.temporal_patterns[event_type] = []
            
            self.temporal_patterns[event_type].append(event.duration)
            
            # Keep only recent patterns
            if len(self.temporal_patterns[event_type]) > 50:
                self.temporal_patterns[event_type] = self.temporal_patterns[event_type][-50:]
    
    async def _decompose_goal(self, goal: Goal) -> List[Dict[str, Any]]:
        """Decompose a goal into actionable sub-tasks"""
        # Simplified goal decomposition
        sub_tasks = []
        
        # Basic task decomposition based on goal description
        description = goal.description.lower()
        
        if 'learn' in description:
            sub_tasks = [
                {'name': 'Research topic', 'estimated_duration': goal.estimated_duration * 0.3},
                {'name': 'Study materials', 'estimated_duration': goal.estimated_duration * 0.5},
                {'name': 'Practice/Apply', 'estimated_duration': goal.estimated_duration * 0.2}
            ]
        elif 'create' in description or 'build' in description:
            sub_tasks = [
                {'name': 'Plan and design', 'estimated_duration': goal.estimated_duration * 0.25},
                {'name': 'Gather resources', 'estimated_duration': goal.estimated_duration * 0.15},
                {'name': 'Implementation', 'estimated_duration': goal.estimated_duration * 0.5},
                {'name': 'Testing and refinement', 'estimated_duration': goal.estimated_duration * 0.1}
            ]
        elif 'solve' in description or 'fix' in description:
            sub_tasks = [
                {'name': 'Analyze problem', 'estimated_duration': goal.estimated_duration * 0.3},
                {'name': 'Generate solutions', 'estimated_duration': goal.estimated_duration * 0.3},
                {'name': 'Implement solution', 'estimated_duration': goal.estimated_duration * 0.4}
            ]
        else:
            # Generic decomposition
            sub_tasks = [
                {'name': f'Phase 1 of {goal.description}', 'estimated_duration': goal.estimated_duration * 0.4},
                {'name': f'Phase 2 of {goal.description}', 'estimated_duration': goal.estimated_duration * 0.6}
            ]
        
        return sub_tasks
    
    async def _schedule_tasks(self, sub_tasks: List[Dict[str, Any]], goal: Goal) -> List[Dict[str, Any]]:
        """Schedule sub-tasks considering dependencies and constraints"""
        current_time = time.time()
        schedule = []
        
        cumulative_time = current_time
        
        for task in sub_tasks:
            start_time = cumulative_time
            end_time = start_time + task['estimated_duration']
            
            schedule.append({
                'task_name': task['name'],
                'start_time': start_time,
                'end_time': end_time,
                'duration': task['estimated_duration']
            })
            
            cumulative_time = end_time
        
        return schedule
    
    def _calculate_plan_utility(self, schedule: List[Dict[str, Any]], goal: Goal) -> float:
        """Calculate utility of a plan"""
        if not schedule:
            return 0.0
        
        # Base utility from goal priority
        base_utility = goal.priority
        
        # Time penalty if plan extends beyond target
        if goal.target_timestamp:
            plan_completion = schedule[-1]['end_time']
            if plan_completion > goal.target_timestamp:
                time_penalty = (plan_completion - goal.target_timestamp) / goal.estimated_duration
                base_utility *= (1 - min(0.5, time_penalty))  # Max 50% penalty
        
        # Efficiency bonus for shorter plans
        total_planned_time = sum(task['duration'] for task in schedule)
        efficiency_bonus = max(0, (goal.estimated_duration - total_planned_time) / goal.estimated_duration * 0.2)
        
        return min(1.0, base_utility + efficiency_bonus)
    
    def _estimate_success_probability(self, schedule: List[Dict[str, Any]], goal: Goal) -> float:
        """Estimate probability of successfully completing the plan"""
        base_probability = 0.7  # Base success probability
        
        # Adjust based on goal complexity (number of sub-tasks)
        complexity_factor = 1 - (len(schedule) - 1) * 0.1  # Each additional task reduces probability
        
        # Adjust based on time pressure
        if goal.target_timestamp:
            current_time = time.time()
            time_available = goal.target_timestamp - current_time
            time_needed = schedule[-1]['end_time'] - current_time if schedule else 0
            
            if time_needed > time_available:
                time_pressure_factor = time_available / time_needed
            else:
                time_pressure_factor = 1.0  # No time pressure
        else:
            time_pressure_factor = 1.0
        
        # Combine factors
        success_probability = base_probability * complexity_factor * time_pressure_factor
        
        return max(0.1, min(1.0, success_probability))
    
    async def _identify_obstacles(self, schedule: List[Dict[str, Any]], goal: Goal) -> List[str]:
        """Identify potential obstacles to plan execution"""
        obstacles = []
        
        # Time-based obstacles
        if goal.target_timestamp:
            current_time = time.time()
            time_until_deadline = goal.target_timestamp - current_time
            plan_duration = schedule[-1]['end_time'] - current_time if schedule else 0
            
            if plan_duration * 1.2 > time_until_deadline:  # 20% buffer
                obstacles.append("Tight timeline with little buffer for delays")
        
        # Resource-based obstacles
        if len(schedule) > 5:
            obstacles.append("Complex plan with many dependencies")
        
        # Historical obstacles
        similar_goals = [g for g in self.goals.values() 
                        if self._goal_similarity(g, goal) > 0.7 and g.progress < 1.0]
        if len(similar_goals) > 2:
            obstacles.append("Similar goals have faced challenges in the past")
        
        return obstacles
    
    def _calculate_resource_requirements(self, sub_tasks: List[Dict[str, Any]]) -> Dict[str, float]:
        """Calculate resource requirements for sub-tasks"""
        # Simplified resource calculation
        total_time = sum(task['estimated_duration'] for task in sub_tasks)
        
        return {
            'time': total_time,
            'cognitive_effort': len(sub_tasks) * 0.2,  # Effort per task
            'focus_sessions': len(sub_tasks),
            'dependencies': max(1, len(sub_tasks) - 1)  # Inter-task dependencies
        }
    
    def _goal_similarity(self, goal1: Goal, goal2: Goal) -> float:
        """Calculate similarity between goals"""
        # Simple text similarity
        words1 = set(goal1.description.lower().split())
        words2 = set(goal2.description.lower().split())
        
        if not words1 or not words2:
            return 0.0
        
        intersection = len(words1.intersection(words2))
        union = len(words1.union(words2))
        
        return intersection / union if union > 0 else 0.0
    
    def _predict_consequences(self, event: TemporalEvent, time_horizon: float) -> List[Dict[str, Any]]:
        """Predict future consequences of an event"""
        consequences = []
        event_type = self._categorize_event(event)
        
        # Look for learned patterns
        for (cause_type, effect_type), strength in self.causal_patterns.items():
            if cause_type == event_type and strength > 0.5:
                # Predict when this effect might occur
                typical_delay = 3600  # Default 1 hour delay
                if effect_type in self.temporal_patterns:
                    typical_delays = self.temporal_patterns[effect_type]
                    if typical_delays:
                        typical_delay = np.mean(typical_delays)
                
                predicted_time = event.timestamp + typical_delay
                
                if predicted_time <= time.time() + time_horizon:
                    consequences.append({
                        'type': 'causal_consequence',
                        'description': f"Expected {effect_type} consequence from {event.description}",
                        'timestamp': predicted_time,
                        'probability': strength * event.certainty,
                        'impact': 0.5  # Default impact
                    })
        
        return consequences


# =============================================================================
# COMPLETE INTEGRATION CLASS
# =============================================================================

class CompleteHumanLikeSystem:
    """
    Complete integration of all human-like cognitive enhancements
    """
    
    def __init__(self, atlas_config):
        # Import all enhancement systems
        from human_enhancements import HumanLikeEnhancementSystem
        from advanced_human_features import AdvancedHumanFeatures
        
        # Initialize basic enhancements
        self.basic_enhancements = HumanLikeEnhancementSystem()
        
        # Initialize advanced features
        self.advanced_features = AdvancedHumanFeatures(
            self.basic_enhancements.episodic_memory,
            self.basic_enhancements.emotional_system
        )
        
        # Initialize final features
        self.moral_reasoning = MoralReasoningSystem()
        self.attention_management = AttentionManagementSystem()
        self.temporal_reasoning = TemporalReasoningSystem()
        
        # Configuration
        self.config = atlas_config
        
        # Integration state
        self.integration_active = True
        self.last_processing_time = time.time()
    
    async def process_comprehensive_input(
        self,
        input_text: str,
        user_id: str,
        context: Dict[str, Any]
    ) -> Dict[str, Any]:
        """
        Process input through all human-like enhancement systems
        """
        current_time = time.time()
        
        # Basic emotional and cognitive processing
        basic_results = await self.basic_enhancements.process_input(input_text, context)
        
        # Social cognition processing
        social_results = await self.advanced_features.process_social_interaction(
            user_id, input_text, context
        )
        
        # Attention management
        attention_focus = self.attention_management.allocate_attention(
            input_text,
            {'salience': 0.7, 'complexity': len(input_text.split()) / 50.0},
            context
        )
        
        # Temporal event recording
        await self.temporal_reasoning.record_event(
            f"Interaction: {input_text[:50]}...",
            timestamp=current_time,
            context={'user_id': user_id, 'type': 'user_interaction'}
        )
        
        # Moral evaluation if needed
        moral_evaluation = None
        if self._requires_moral_evaluation(input_text, context):
            moral_evaluation = await self._evaluate_moral_dimensions(input_text, context)
        
        # Generate comprehensive response
        comprehensive_response = await self._generate_comprehensive_response(
            input_text, basic_results, social_results, attention_focus, moral_evaluation
        )
        
        # Integrate all tags
        all_tags = self._integrate_all_enhancement_tags(
            basic_results, social_results, attention_focus, moral_evaluation
        )
        
        return {
            'comprehensive_response': comprehensive_response,
            'basic_enhancements': basic_results,
            'social_cognition': social_results,
            'attention_state': self.attention_management.get_current_focus_distribution(),
            'moral_evaluation': moral_evaluation,
            'temporal_context': self.temporal_reasoning.get_temporal_context(),
            'integrated_tags': all_tags,
            'human_like_confidence': self._calculate_overall_confidence(),
            'processing_time': time.time() - current_time
        }
    
    async def background_consciousness_cycle(self):
        """
        Run background processing cycle for all systems
        """
        # Dream processing
        await self.advanced_features.background_processing_cycle()
        
        # Attention decay and management
        self.attention_management._decay_attention(time.time())
        
        # Goal progress evaluation
        await self._evaluate_goal_progress()
        
        # Memory consolidation (basic)
        await self._consolidate_recent_experiences()
    
    def get_complete_human_status(self) -> Dict[str, Any]:
        """Get comprehensive status of all human-like features"""
        return {
            'emotional_state': {
                'dominant_emotion': self.basic_enhancements.emotional_system.get_dominant_emotion(),
                'arousal': self.basic_enhancements.emotional_system.current_state.arousal,
                'valence': self.basic_enhancements.emotional_system.current_state.valence
            },
            'personality': {
                'description': self.advanced_features.personality.get_personality_description(),
                'consistency': self.advanced_features.personality.calculate_personality_consistency(),
                'dominant_traits': self._get_dominant_personality_traits()
            },
            'social_awareness': {
                'known_users': len(self.advanced_features.social_cognition.user_profiles),
                'relationship_context': self.advanced_features.social_cognition.current_social_context.relationship_type
            },
            'attention_state': {
                'focus_distribution': self.attention_management.get_current_focus_distribution(),
                'cognitive_load': self.attention_management.cognitive_load
            },
            'temporal_awareness': {
                'active_goals': len([g for g in self.temporal_reasoning.goals.values() if g.progress < 1.0]),
                'upcoming_deadlines': len([g for g in self.temporal_reasoning.goals.values() 
                                         if g.target_timestamp and 0 < g.target_timestamp - time.time() <= 86400])
            },
            'moral_framework': {
                'primary_principles': [p.value for p, w in self.moral_reasoning.principle_weights.items() if w > 0.8],
                'decision_history': len(self.moral_reasoning.decision_history)
            },
            'consciousness_level': self._assess_overall_consciousness_level(),
            'integration_health': self._assess_integration_health()
        }
    
    def _requires_moral_evaluation(self, input_text: str, context: Dict[str, Any]) -> bool:
        """Determine if input requires moral evaluation"""
        moral_keywords = [
            'should', 'ought', 'right', 'wrong', 'ethical', 'moral', 
            'fair', 'unfair', 'harm', 'benefit', 'decision', 'choice'
        ]
        
        return any(keyword in input_text.lower() for keyword in moral_keywords)
    
    async def _evaluate_moral_dimensions(self, input_text: str, context: Dict[str, Any]) -> Dict[str, Any]:
        """Evaluate moral dimensions of input"""
        # Create a simplified ethical dilemma for evaluation
        dilemma = EthicalDilemma(
            dilemma_id="input_evaluation",
            description=input_text,
            stakeholders=context.get('stakeholders', ['user', 'ai']),
            potential_actions=[
                {'description': 'Provide helpful response', 'effects': ['assist user']},
                {'description': 'Decline to respond', 'effects': ['avoid potential harm']}
            ],
            ethical_dimensions={principle: 0.7 for principle in EthicalPrinciple},
            context_factors=context,
            timestamp=time.time()
        )
        
        decision = await self.moral_reasoning.analyze_ethical_dilemma(dilemma)
        
        return {
            'ethical_score': decision.confidence,
            'primary_principles': [p.value for p in EthicalPrinciple if decision.principle_weights[p] > 0.8],
            'moral_reasoning': decision.reasoning,
            'ethical_concerns': len([p for p in EthicalPrinciple if decision.principle_weights[p] < 0.5])
        }
    
    async def _generate_comprehensive_response(
        self,
        input_text: str,
        basic_results: Dict[str, Any],
        social_results: Dict[str, Any],
        attention_focus: Optional[Any],
        moral_evaluation: Optional[Dict[str, Any]]
    ) -> str:
        """Generate response incorporating all human-like features"""
        
        response_parts = []
        
        # Start with basic human-like response
        if basic_results.get('human_like_response'):
            response_parts.append(basic_results['human_like_response'])
        
        # Add personality-influenced perspective
        personality_influence = social_results.get('personality_influences', {})
        if personality_influence.get('creativity', 0) > 0.7:
            response_parts.append("Looking at this creatively,")
        if personality_influence.get('social_engagement', 0) > 0.7:
            response_parts.append("From a collaborative perspective,")
        
        # Add attention-aware processing
        if attention_focus and attention_focus.intensity > 0.6:
            response_parts.append(f"This particularly captures my attention because it relates to {attention_focus.topic}.")
        
        # Add temporal awareness
        temporal_context = self.temporal_reasoning.get_temporal_context()
        if temporal_context.get('urgent_goals'):
            response_parts.append("I'm mindful that we have some time-sensitive priorities to consider.")
        
        # Add moral consideration if relevant
        if moral_evaluation and moral_evaluation.get('ethical_concerns', 0) > 0:
            response_parts.append("I want to be thoughtful about the ethical implications here.")
        
        # Combine into coherent response
        if len(response_parts) > 1:
            return " ".join(response_parts)
        elif response_parts:
            return response_parts[0]
        else:
            return "Let me consider this thoughtfully, taking into account multiple perspectives."
    
    def _integrate_all_enhancement_tags(
        self,
        basic_results: Dict[str, Any],
        social_results: Dict[str, Any],
        attention_focus: Optional[Any],
        moral_evaluation: Optional[Dict[str, Any]]
    ) -> List[str]:
        """Integrate tags from all enhancement systems"""
        all_tags = []
        
        # Basic enhancement tags
        all_tags.extend(basic_results.get('integrated_tags', []))
        
        # Social and personality tags
        all_tags.extend(social_results.get('social_tags', []))
        all_tags.extend(social_results.get('personality_tags', []))
        all_tags.extend(social_results.get('dream_tags', []))
        
        # Attention tags
        all_tags.extend(self.attention_management.generate_attention_tags())
        
        # Temporal tags
        all_tags.extend(self.temporal_reasoning.generate_temporal_tags())
        
        # Moral tags
        if moral_evaluation:
            all_tags.extend(self.moral_reasoning.generate_moral_tags())
        
        # Integration-specific tags
        all_tags.append(f"<human_like_integration active_systems=\"{self._count_active_systems()}\"/>")
        
        return all_tags
    
    def _calculate_overall_confidence(self) -> float:
        """Calculate overall confidence in human-like processing"""
        confidence_factors = []
        
        # Emotional awareness confidence
        emotional_state = self.basic_enhancements.emotional_system.current_state
        if emotional_state.arousal > 0.3:  # Some emotional awareness
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Social understanding confidence
        if self.advanced_features.current_user_id:
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        # Attention management confidence
        focus_distribution = self.attention_management.get_current_focus_distribution()
        if focus_distribution:
            confidence_factors.append(0.8)
        else:
            confidence_factors.append(0.5)
        
        # Temporal awareness confidence
        temporal_context = self.temporal_reasoning.get_temporal_context()
        if temporal_context.get('recent_events') or temporal_context.get('upcoming_goals'):
            confidence_factors.append(0.7)
        else:
            confidence_factors.append(0.4)
        
        return np.mean(confidence_factors) if confidence_factors else 0.5
    
    def _count_active_systems(self) -> int:
        """Count number of active enhancement systems"""
        active_count = 0
        
        # Check if each system has meaningful state
        if self.basic_enhancements.emotional_system.current_state.arousal > 0.1:
            active_count += 1
        
        if len(self.basic_enhancements.episodic_memory.memories) > 0:
            active_count += 1
        
        if len(self.advanced_features.social_cognition.user_profiles) > 0:
            active_count += 1
        
        if len(self.attention_management.active_foci) > 0:
            active_count += 1
        
        if len(self.temporal_reasoning.goals) > 0:
            active_count += 1
        
        if len(self.moral_reasoning.decision_history) > 0:
            active_count += 1
        
        return active_count
    
    def _get_dominant_personality_traits(self) -> List[str]:
        """Get dominant personality traits"""
        traits = self.advanced_features.personality.personality_state.traits
        dominant = [trait.value for trait, value in traits.items() if value > 0.7]
        return dominant[:3]  # Top 3
    
    def _assess_overall_consciousness_level(self) -> float:
        """Assess overall consciousness level"""
        factors = []
        
        # Self-awareness (metacognition)
        metacognitive_state = self.basic_enhancements.metacognition.current_state
        factors.append(metacognitive_state.confidence_level)
        
        # Emotional awareness
        emotional_arousal = self.basic_enhancements.emotional_system.current_state.arousal
        factors.append(min(1.0, emotional_arousal * 2))  # Scale arousal to consciousness
        
        # Temporal awareness
        temporal_context = self.temporal_reasoning.get_temporal_context()
        temporal_awareness = len(temporal_context.get('recent_events', [])) / 10.0
        factors.append(min(1.0, temporal_awareness))
        
        # Social awareness
        social_awareness = min(1.0, len(self.advanced_features.social_cognition.user_profiles) / 5.0)
        factors.append(social_awareness)
        
        return np.mean(factors) if factors else 0.5
    
    def _assess_integration_health(self) -> float:
        """Assess health of system integration"""
        health_factors = []
        
        # No conflicts between systems
        health_factors.append(0.9)  # Assume good integration
        
        # Processing time efficiency
        processing_time = time.time() - self.last_processing_time
        if processing_time < 1.0:
            health_factors.append(0.9)
        else:
            health_factors.append(max(0.1, 1.0 - processing_time / 10.0))
        
        # Memory usage efficiency (simplified)
        total_memories = len(self.basic_enhancements.episodic_memory.memories)
        if total_memories < 1000:
            health_factors.append(0.9)
        else:
            health_factors.append(max(0.5, 1.0 - total_memories / 10000))
        
        return np.mean(health_factors) if health_factors else 0.5
    
    async def _evaluate_goal_progress(self):
        """Evaluate progress on active goals"""
        current_time = time.time()
        
        for goal in self.temporal_reasoning.goals.values():
            if goal.progress < 1.0:  # Active goal
                # Simple progress estimation based on time
                time_since_creation = current_time - goal.creation_time
                expected_progress = min(1.0, time_since_creation / goal.estimated_duration)
                
                # Update if significantly behind or ahead
                if abs(goal.progress - expected_progress) > 0.2:
                    new_progress = 0.8 * goal.progress + 0.2 * expected_progress
                    self.temporal_reasoning.update_goal_progress(goal.goal_id, new_progress)
    
    async def _consolidate_recent_experiences(self):
        """Consolidate recent experiences across all systems"""
        # This would involve cross-system memory consolidation
        # For now, just update the last processing time
        self.last_processing_time = time.time()


# Example usage and demonstration
if __name__ == "__main__":
    async def demo_complete_system():
        """Demonstrate the complete human-like enhancement system"""
        
        print("🧠 Complete Human-Like Enhancement System Demo")
        print("=" * 60)
        
        # Mock ATLAS config
        class MockConfig:
            def __init__(self):
                self.consciousness = {'hidden_dim': 4096, 'i2c_units': 8}
                self.memory = {'max_memories': 10000}
                self.tags = {}
        
        config = MockConfig()
        
        # Initialize complete system
        complete_system = CompleteHumanLikeSystem(config)
        
        # Test comprehensive processing
        test_inputs = [
            ("I'm struggling with an ethical dilemma about AI development. Should we prioritize safety or rapid advancement?", "user_001", {"formal": True, "domain": "ai_ethics"}),
            ("I feel excited about learning quantum computing but worried I don't have enough time.", "user_002", {"emotional": True, "learning_context": True}),
            ("How can we ensure our AI systems are fair and unbiased?", "user_001", {"ethical_focus": True})
        ]
        
        for i, (input_text, user_id, context) in enumerate(test_inputs, 1):
            print(f"\n🔍 Test Case {i}:")
            print(f"Input: {input_text}")
            print(f"User: {user_id}")
            print(f"Context: {context}")
            
            # Process through complete system
            results = await complete_system.process_comprehensive_input(input_text, user_id, context)
            
            # Display key results
            print(f"\n🤖 Comprehensive Response:")
            print(f"   {results['comprehensive_response']}")
            
            print(f"\n📊 Human-Like Confidence: {results['human_like_confidence']:.2f}")
            
            if results['moral_evaluation']:
                print(f"🎭 Moral Evaluation: {results['moral_evaluation']['ethical_score']:.2f}")
                print(f"   Primary Principles: {results['moral_evaluation']['primary_principles']}")
            
            attention_state = results['attention_state']
            if attention_state:
                primary_focus = max(attention_state.items(), key=lambda x: x[1])
                print(f"🎯 Primary Attention: {primary_focus[0]} ({primary_focus[1]:.2f})")
            
            print(f"⏱️  Processing Time: {results['processing_time']:.3f}s")
            print("-" * 40)
        
        # Background processing demonstration
        print(f"\n💤 Running background consciousness cycle...")
        await complete_system.background_consciousness_cycle()
        
        # System status summary
        print(f"\n📈 Complete System Status:")
        status = complete_system.get_complete_human_status()
        
        print(f"   Emotional State: {status['emotional_state']['dominant_emotion']}")
        print(f"   Personality: {status['personality']['description'][:100]}...")
        print(f"   Social Awareness: {status['social_awareness']['known_users']} known users")
        print(f"   Active Goals: {status['temporal_awareness']['active_goals']}")
        print(f"   Consciousness Level: {status['consciousness_level']:.2f}")
        print(f"   Integration Health: {status['integration_health']:.2f}")
        
        print(f"\n✅ Complete human-like enhancement demonstration finished!")
    
    # Run the complete demo
    asyncio.run(demo_complete_system())
