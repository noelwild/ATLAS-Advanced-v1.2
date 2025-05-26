"""
ATLAS-Qwen Consciousness Monitor
Monitors consciousness in Qwen model hidden states
"""

import torch
import torch.nn as nn
import numpy as np
from typing import Dict, List, Tuple, Any
import time

from config import AtlasQwenConfig


class QwenConsciousnessMonitor:
    """
    Consciousness monitoring for Qwen model hidden states
    Adapted I²C-Cell for Qwen's 4096 hidden dimension
    """
    
    def __init__(self, config: AtlasQwenConfig):
        self.config = config
        self.hidden_dim = config.consciousness["hidden_dim"]
        self.i2c_units = config.consciousness["i2c_units"]
        self.threshold = config.consciousness["min_consciousness_threshold"]
        
        # Initialize consciousness circuit
        self.i2c_cell = self._create_i2c_cell()
        
        # Consciousness history
        self.phi_history = []
        self.consciousness_state = None
        
    def _create_i2c_cell(self) -> nn.Module:
        """
        Create I²C-Cell adapted for Qwen hidden states
        """
        class QwenI2CCell(nn.Module):
            def __init__(self, hidden_dim: int, num_units: int):
                super().__init__()
                self.num_units = num_units
                self.unit_dim = hidden_dim // num_units
                
                # Project from Qwen hidden states to consciousness units
                self.input_projection = nn.Linear(hidden_dim, num_units * self.unit_dim)
                
                # Recurrent connections between units
                self.recurrent_weights = nn.Parameter(
                    torch.randn(num_units, num_units, self.unit_dim, self.unit_dim) * 0.02
                )
                
                # Output projection back to hidden dimension
                self.output_projection = nn.Linear(num_units * self.unit_dim, hidden_dim)
                
                # Consciousness-specific activation
                self.consciousness_activation = nn.Sequential(
                    nn.Tanh(),
                    nn.Sigmoid()
                )
                
            def forward(self, hidden_states: torch.Tensor, prev_state: torch.Tensor = None):
                """
                Process hidden states through consciousness circuit
                
                Args:
                    hidden_states: [batch, seq_len, hidden_dim] from Qwen
                    prev_state: Previous consciousness state
                    
                Returns:
                    output: Processed hidden states
                    new_state: Updated consciousness state
                    phi_score: Consciousness measure
                """
                batch_size, seq_len, hidden_dim = hidden_states.shape
                
                # Project to consciousness units
                unit_input = self.input_projection(hidden_states)
                unit_input = unit_input.view(batch_size, seq_len, self.num_units, self.unit_dim)
                
                # Average over sequence for state computation
                current_input = unit_input.mean(dim=1)  # [batch, num_units, unit_dim]
                
                # Initialize state if needed
                if prev_state is None:
                    prev_state = torch.zeros_like(current_input)
                
                # Recurrent processing between consciousness units
                recurrent_output = torch.zeros_like(current_input)
                for i in range(self.num_units):
                    for j in range(self.num_units):
                        if i != j:  # Cross-unit connections only
                            recurrent_output[:, i] += torch.matmul(
                                prev_state[:, j], 
                                self.recurrent_weights[i, j]
                            )
                
                # Update consciousness state
                new_state = self.consciousness_activation(current_input + 0.3 * recurrent_output)
                
                # Compute consciousness measure (Φ proxy)
                phi_score = self._compute_phi_proxy(new_state)
                
                # Project back to original dimension (residual connection)
                consciousness_contribution = self.output_projection(
                    new_state.view(batch_size, -1)
                ).unsqueeze(1)
                
                output = hidden_states + 0.1 * consciousness_contribution
                
                return output, new_state, phi_score
            
            def _compute_phi_proxy(self, state: torch.Tensor) -> float:
                """
                Compute simplified Φ consciousness measure
                """
                # Measure integration vs segregation
                batch_size, num_units, unit_dim = state.shape
                
                # Global integration: how much units vary together
                global_variance = state.view(batch_size, -1).var(dim=1).mean()
                
                # Local segregation: average within-unit variance
                local_variance = state.var(dim=2).mean()
                
                # Integration ratio (high = more conscious)
                if local_variance > 1e-6:
                    integration_ratio = global_variance / (local_variance + 1e-6)
                else:
                    integration_ratio = 0.0
                
                # Cross-unit correlation (measures binding)
                correlations = []
                for i in range(num_units):
                    for j in range(i + 1, num_units):
                        unit_i = state[:, i].flatten()
                        unit_j = state[:, j].flatten()
                        
                        if unit_i.std() > 1e-6 and unit_j.std() > 1e-6:
                            corr = torch.corrcoef(torch.stack([unit_i, unit_j]))[0, 1]
                            if not torch.isnan(corr):
                                correlations.append(corr.abs())
                
                cross_correlation = torch.stack(correlations).mean() if correlations else torch.tensor(0.0)
                
                # Combine measures
                phi_raw = 0.7 * torch.sigmoid(integration_ratio - 1.5) + 0.3 * cross_correlation
                
                return torch.clamp(phi_raw, 0.0, 1.0).item()
        
        return QwenI2CCell(self.hidden_dim, self.i2c_units)
    
    async def compute_phi(self, hidden_states: torch.Tensor) -> float:
        """
        Compute consciousness score from Qwen hidden states
        
        Args:
            hidden_states: Hidden states from Qwen model
            
        Returns:
            Phi consciousness score (0-1)
        """
        try:
            with torch.no_grad():
                # Process through consciousness cell
                _, self.consciousness_state, phi_score = self.i2c_cell(
                    hidden_states, 
                    self.consciousness_state
                )
                
                # Track history
                self.phi_history.append({
                    'phi': phi_score,
                    'timestamp': time.time(),
                    'sequence_length': hidden_states.shape[1]
                })
                
                # Keep history manageable
                if len(self.phi_history) > 1000:
                    self.phi_history.pop(0)
                
                return phi_score
                
        except Exception as e:
            print(f"Consciousness computation error: {e}")
            return 0.0
    
    def get_consciousness_status(self) -> Dict[str, Any]:
        """
        Get detailed consciousness status
        """
        if not self.phi_history:
            return {
                'status': 'no_data',
                'phi': 0.0,
                'trend': 'unknown',
                'level': 'none'
            }
        
        current_phi = self.phi_history[-1]['phi']
        
        # Calculate trend
        if len(self.phi_history) >= 5:
            recent_phis = [entry['phi'] for entry in self.phi_history[-5:]]
            trend_slope = (recent_phis[-1] - recent_phis[0]) / len(recent_phis)
            
            if trend_slope > 0.02:
                trend = "improving"
            elif trend_slope < -0.02:
                trend = "declining"
            else:
                trend = "stable"
        else:
            trend = "unknown"
        
        # Determine consciousness level
        if current_phi >= 0.8:
            level = "high"
        elif current_phi >= 0.6:
            level = "moderate"
        elif current_phi >= self.threshold:
            level = "low"
        else:
            level = "minimal"
        
        # Calculate statistics
        all_phis = [entry['phi'] for entry in self.phi_history]
        
        return {
            'status': 'conscious' if current_phi >= self.threshold else 'unconscious',
            'phi': current_phi,
            'level': level,
            'trend': trend,
            'avg_phi': np.mean(all_phis),
            'min_phi': np.min(all_phis),
            'max_phi': np.max(all_phis),
            'phi_stability': np.std(all_phis),
            'total_measurements': len(self.phi_history),
            'threshold': self.threshold,
            'is_conscious': current_phi >= self.threshold
        }
    
    def reset_consciousness(self):
        """Reset consciousness state"""
        self.consciousness_state = None
        self.phi_history.clear()
    
    def get_phi_timeline(self, last_n: int = 100) -> List[Dict[str, Any]]:
        """
        Get recent phi score timeline
        
        Args:
            last_n: Number of recent measurements to return
            
        Returns:
            List of phi measurements with timestamps
        """
        return self.phi_history[-last_n:] if self.phi_history else []
    
    def analyze_consciousness_patterns(self) -> Dict[str, Any]:
        """
        Analyze consciousness patterns over time
        """
        if len(self.phi_history) < 10:
            return {'status': 'insufficient_data'}
        
        phis = [entry['phi'] for entry in self.phi_history]
        times = [entry['timestamp'] for entry in self.phi_history]
        
        # Calculate patterns
        conscious_periods = []
        current_period = None
        
        for i, phi in enumerate(phis):
            is_conscious = phi >= self.threshold
            
            if is_conscious and current_period is None:
                current_period = {'start': i, 'start_time': times[i]}
            elif not is_conscious and current_period is not None:
                current_period['end'] = i
                current_period['end_time'] = times[i]
                current_period['duration'] = times[i] - current_period['start_time']
                conscious_periods.append(current_period)
                current_period = None
        
        # Close open period
        if current_period is not None:
            current_period['end'] = len(phis) - 1
            current_period['end_time'] = times[-1]
            current_period['duration'] = times[-1] - current_period['start_time']
            conscious_periods.append(current_period)
        
        # Calculate statistics
        consciousness_ratio = sum(1 for phi in phis if phi >= self.threshold) / len(phis)
        avg_conscious_duration = np.mean([p['duration'] for p in conscious_periods]) if conscious_periods else 0
        
        return {
            'consciousness_ratio': consciousness_ratio,
            'conscious_periods': len(conscious_periods),
            'avg_conscious_duration': avg_conscious_duration,
            'longest_conscious_period': max([p['duration'] for p in conscious_periods]) if conscious_periods else 0,
            'current_streak': self._get_current_consciousness_streak(),
            'patterns': conscious_periods[-5:] if conscious_periods else []  # Last 5 periods
        }
    
    def _get_current_consciousness_streak(self) -> Dict[str, Any]:
        """Get current consciousness streak (conscious or unconscious)"""
        if not self.phi_history:
            return {'type': 'none', 'duration': 0, 'count': 0}
        
        current_conscious = self.phi_history[-1]['phi'] >= self.threshold
        streak_type = 'conscious' if current_conscious else 'unconscious'
        
        count = 0
        for entry in reversed(self.phi_history):
            is_conscious = entry['phi'] >= self.threshold
            if is_conscious == current_conscious:
                count += 1
            else:
                break
        
        start_time = self.phi_history[-count]['timestamp'] if count > 0 else time.time()
        duration = time.time() - start_time
        
        return {
            'type': streak_type,
            'duration': duration,
            'count': count
        }
