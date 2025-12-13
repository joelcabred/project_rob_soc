from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random
import math

class GaussianPolicyNetwork(nn.Module):
    """
    Stochastic policy network for continuous actions
    Outputs mean and log_std for Gaussian distribution πθ(a|s)
    """
    def __init__(self, state_size, action_size, hidden_size=256, log_std_min=-20, log_std_max=2):
        super(GaussianPolicyNetwork, self).__init__()
        
        self.fc1 = nn.Linear(state_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.mean = nn.Linear(hidden_size, action_size)
        self.log_std = nn.Linear(hidden_size, action_size)
        
        self.log_std_min = log_std_min
        self.log_std_max = log_std_max
        
    def forward(self, state):
        """
        state: [batch, state_size]
        returns: mean [batch, action_size], log_std [batch, action_size]
        """
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        
        mean = torch.tanh(self.mean(x))  # Actions in [-1, 1]
        log_std = self.log_std(x)
        log_std = torch.clamp(log_std, self.log_std_min, self.log_std_max)
        
        return mean, log_std
    
    def sample(self, state):
        """
        Sample action from policy
        Returns: action, log_prob
        """
        action_mean, log_std = self.forward(state)
        std = log_std.exp()  # amount of noise / uncertainty --> for exploration 
        
        # Sample from Gaussian
        normal = torch.distributions.Normal(action_mean, std)
        action = normal.rsample()  # Reparameterization trick
        action = torch.tanh(action)  # Squash to [-1, 1]
        
        # Compute log probability
        log_prob = normal.log_prob(action)
        log_prob = log_prob.sum(dim=-1, keepdim=True)
        
        return action, log_prob
    
    def log_proba(self, state, action):     # To distinguish from normal.log_prob(action)
        """
        Compute log probability of action under current policy
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        normal = torch.distributions.Normal(mean, std)
        log_proba = normal.log_prob(action)
        log_proba = log_proba.sum(dim=-1, keepdim=True)
        
        return log_proba
    
    def entropy(self, state):
        """
        Compute entropy H(πθ(·|s)) for entropy regularization
        For Gaussian: H = 0.5 * log(2πe * σ²)
        """
        mean, log_std = self.forward(state)
        std = log_std.exp()
        
        # Entropy for multivariate Gaussian
        entropy = 0.5 * (1.0 + math.log(2 * math.pi)) + log_std
        entropy = entropy.sum(dim=-1, keepdim=True)
        
        return entropy


class DeepCOACHModule:
    """
    Deep COACH module implementing Algorithm DeepCOACH from the paper
    Uses eligibility traces, importance sampling, and window-based feedback
    """
    def __init__(self, state_size, action_size, 
                 window_size=10,                # L in paper
                 eligibility_decay=0.35,        # λ in paper
                 human_delay=1,                 # d in paper
                 learning_rate=0.00025,         # α in paper
                 entropy_coef=1.5,              # β in paper
                 minibatch_size=6,              # m in paper
                 buffer_size=1000):
        
        self.state_size = state_size
        self.action_size = action_size
        self.window_size = window_size
        self.eligibility_decay = eligibility_decay
        self.human_delay = human_delay
        self.learning_rate = learning_rate
        self.entropy_coef = entropy_coef
        self.minibatch_size = minibatch_size
        
        # Policy network πθ(a|s)
        self.policy = GaussianPolicyNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.policy.parameters(), lr=learning_rate)
        
        # Eligibility replay buffer E
        self.replay_buffer = deque(maxlen=buffer_size)
        
        # Current window w
        self.current_window = []
        
        # Statistics
        self.feedback_count = 0
        self.total_samples = 0
        self.num_windows = 0

    def has_valid_model(self):
        """Check if we have enough windows to train"""
        return len(self.replay_buffer) > 5
        
    def observe_step(self, state, action, action_prob):
        """
        Observe a step: record st-d, at-d, pt-d
        
        Args:
            state: current state (will be stored with delay)
            action: action taken
            action_prob: probability πθ(at|st) when action was taken
        """
        self.current_window.append({
            'state': state.copy(),
            'action': action.copy(),
            'prob': action_prob,
            'feedback': 0.0  # Will be updated when feedback arrives
        })
        self.total_samples += 1
    
    def give_feedback(self, feedback_value):
        """
        Human gives feedback ft (can be continuous, not just -1, 0, +1)
        
        Args:
            feedback_value: scalar feedback from human
        """
        if len(self.current_window) == 0:
            print("Warning: No trajectory to assign feedback")
            return
        
        # Apply human delay: feedback at time t affects state at time t-d
        delay_idx = max(0, len(self.current_window) - 1 - self.human_delay)
        
        # Store feedback at delayed timestep
        if delay_idx < len(self.current_window):
            self.current_window[delay_idx]['feedback'] = feedback_value
        
        # If feedback is non-zero, truncate window and store in replay buffer
        if feedback_value != 0:
            self.feedback_count += 1
            
            # Truncate to L most recent entries
            window_to_store = self.current_window[-self.window_size:]
            
            # Store in eligibility replay buffer
            self.replay_buffer.append({
                'window': window_to_store,
                'final_feedback': feedback_value,
                'feedback_id': self.feedback_count
            })
            self.num_windows += 1
            
            # Reset window
            self.current_window = []
            
            print(f"Feedback {feedback_value:+.2f} → Window with {len(window_to_store)} steps stored "
                  f"(buffer: {len(self.replay_buffer)} windows)")
    
    def train_batch(self):
        """
        Train using eligibility traces and importance sampling (Algorithm DeepCOACH)
        
        Returns:
            loss: scalar loss value (None if insufficient data)
        """
        if len(self.replay_buffer) < self.minibatch_size:
            return None
        
        # Sample minibatch W of m windows from E
        minibatch = random.sample(self.replay_buffer, min(self.minibatch_size, len(self.replay_buffer)))
        
        # Accumulator for eligibility traces
        total_eligibility = None
        
        for window_data in minibatch:
            window = window_data['window']
            F = window_data['final_feedback']  # Final feedback signal
            
            # Initialize eligibility trace eλ ← 0
            eligibility_trace = None
            
            for step in window:
                state = torch.FloatTensor(step['state']).unsqueeze(0)
                action = torch.FloatTensor(step['action']).unsqueeze(0)
                old_prob = step['prob']  # πθ_old(a|s)
                
                # Compute current log probability: log πθ(a|s)
                log_prob_current = self.policy.log_proba(state, action)
                
                # Compute importance sampling ratio: πθ(a|s) / p
                # In log space: exp(log πθ(a|s) - log p)
                importance_ratio = torch.exp(log_prob_current - math.log(old_prob + 1e-10))
                
                # Compute gradient: ∇θ log πθ(a|s)
                self.optimizer.zero_grad()
                log_prob_current.backward(retain_graph=True)
                
                # Get gradients
                grad_log_prob = []
                for param in self.policy.parameters():
                    if param.grad is not None:
                        grad_log_prob.append(param.grad.clone().flatten())
                grad_log_prob = torch.cat(grad_log_prob)
                
                # Update eligibility trace: eλ ← λ*eλ + πθ(a|s)/p * ∇θ log πθ(a|s)
                if eligibility_trace is None:
                    eligibility_trace = importance_ratio.item() * grad_log_prob
                else:
                    eligibility_trace = self.eligibility_decay * eligibility_trace + importance_ratio.item() * grad_log_prob
            
            # Accumulate: ēλ ← ēλ + F*eλ
            if eligibility_trace is not None:
                if total_eligibility is None:
                    total_eligibility = F * eligibility_trace
                else:
                    total_eligibility += F * eligibility_trace
        
        # Average over minibatch: ēλ ← (1/m) * ēλ
        if total_eligibility is not None:
            total_eligibility = total_eligibility / len(minibatch)
            
            # Add entropy regularization: ēλ ← ēλ + β*∇θ H(πθ(·|st))
            # Sample a recent state for entropy
            if self.current_window:
                recent_state = torch.FloatTensor(self.current_window[-1]['state']).unsqueeze(0)
            else:
                # Use a state from buffer
                recent_state = torch.FloatTensor(self.replay_buffer[-1]['window'][-1]['state']).unsqueeze(0)
            
            entropy = self.policy.entropy(recent_state)
            self.optimizer.zero_grad()
            (-entropy).backward()  # Negative because we want to maximize entropy
            
            grad_entropy = []
            for param in self.policy.parameters():
                if param.grad is not None:
                    grad_entropy.append(param.grad.clone().flatten())
            grad_entropy = torch.cat(grad_entropy)
            
            total_eligibility += self.entropy_coef * grad_entropy
            
            # Update parameters: θ ← θ + α*ēλ
            idx = 0
            for param in self.policy.parameters():
                num_params = param.numel()

                # Update automatically self.policy parameters
                param.data += self.learning_rate * total_eligibility[idx:idx+num_params].reshape(param.shape)  # θ ← θ + α*ēλ
                idx += num_params
            
            return total_eligibility.norm().item()
        
        return None
    
    def predict_action(self, state, deterministic=False):
        """
        Predict action: at = argmax πθ(a|st) or sample
        
        Returns:
            action: numpy array
            prob: probability/density of the action
        """
        self.policy.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            
            if deterministic:
                # Use mean of the policy (argmax)
                mean, _ = self.policy.forward(state_t)
                action = mean
                # For deterministic, compute probability at mean
                log_prob = self.policy.log_proba(state_t, action)
                prob = torch.exp(log_prob).item()
            else:
                # Sample from policy
                action, log_prob = self.policy.sample(state_t)
                prob = torch.exp(log_prob).item()
            
        self.policy.train()
        return action.detach().numpy()[0], prob
    
    def reset_trajectory(self):
        """
        Clear current window (call at episode start)
        """
        self.current_window = []
    
    def get_stats(self):
        """Returns useful statistics"""
        return {
            'feedback_count': self.feedback_count,
            'windows_stored': len(self.replay_buffer),
            'current_window_length': len(self.current_window),
            'total_steps_observed': self.total_samples
        }
