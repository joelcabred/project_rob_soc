from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class HumanRewardModel(nn.Module):
    """
    Neural network that learns to predict human prosodic feedback (-1, 0, +1)
    based on state-action pairs
    """
    def __init__(self, state_size, action_size, hidden_size=256):
        super(HumanRewardModel, self).__init__()
        
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Predicts scalar reward
        
    def forward(self, state, action):
        """
        state: [batch, state_size]
        action: [batch, action_size]
        returns: [batch, 1] human reward prediction
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # No activation, continuous output
    
class TAMERModule:
    """
    TAMER module that handles prosodic human feedback over trajectories
    """
    def __init__(self, state_size, action_size, 
                 credit_window=15, 
                 decay_factor=0.9,
                 learning_rate=0.001):
        
        self.state_size = state_size
        self.action_size = action_size
        self.credit_window = credit_window
        self.decay_factor = decay_factor
        
        # Network that predicts human reward
        self.reward_model = HumanRewardModel(state_size, action_size)
        self.optimizer = optim.Adam(self.reward_model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Small deque for recent trajectory (credit window)
        self.recent_trajectory = deque(maxlen=credit_window)
        
        # Large buffer for training examples
        self.training_buffer = deque(maxlen=10000)
        
        # Statistics
        self.feedback_count = 0
        self.total_samples = 0

        self.num_feedback_samples = 0

    def has_valid_model(self):
        return self.num_feedback_samples > 10
        
    def observe_step(self, state, action):
        """
        Records each step of the trajectory
        Call this on each episode step
        """
        self.recent_trajectory.append({
            'state': state.copy(),
            'action': action.copy()
        })
        self.total_samples += 1
    
    def give_prosody_feedback(self, feedback_value):
        """
        Human gives prosodic feedback on recent trajectory
        
        Args:
            feedback_value: -1 (negative prosody), 0 (neutral), +1 (positive prosody)
        """
        assert feedback_value in [-1, 0, 1], "Feedback must be -1, 0, or +1"
        
        if len(self.recent_trajectory) == 0:
            print("Warning: No trajectory to assign feedback")
            return
        
        self.feedback_count += 1
        samples_added = 0
        
        # Assigns credit to all steps in the recent window
        # with temporal decay (more recent steps receive more credit)
        for i, step in enumerate(reversed(self.recent_trajectory)):
            # i=0 is the most recent step, i=N-1 is the oldest
            credit = feedback_value * (self.decay_factor ** i)
            
            self.training_buffer.append({
                'state': step['state'],
                'action': step['action'],
                'human_reward': credit,
                'feedback_id': self.feedback_count
            })
            samples_added += 1
        
        print(f"Feedback {feedback_value:+d} → {samples_added} steps credited "
              f"(buffer: {len(self.training_buffer)} samples)")
    
    def train_batch(self, batch_size=64):
        """
        Trains the human reward model with a batch
        Call this periodically during training
        
        Returns:
            loss: batch loss (None if insufficient data)
        """
        if len(self.training_buffer) < batch_size:
            return None
        
        # Random batch sample
        batch = random.sample(self.training_buffer, batch_size)
        
        states = torch.FloatTensor(np.array([x['state'] for x in batch]))
        actions = torch.FloatTensor(np.array([x['action'] for x in batch]))
        rewards = torch.FloatTensor(np.array([x['human_reward'] for x in batch])).unsqueeze(1)
        
        # Forward pass
        predicted_rewards = self.reward_model(states, actions)
        loss = self.criterion(predicted_rewards, rewards)
        
        # Backward pass
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        return loss.item()
    
    def predict_human_reward(self, state, action):
        """
        Predicts what prosodic feedback the human would give for this state-action
        
        Returns:
            reward: predicted scalar value (approximately between -1 and +1)
        """
        self.reward_model.eval()
        with torch.no_grad():
            state_t = torch.FloatTensor(state).unsqueeze(0)
            action_t = torch.FloatTensor(action).unsqueeze(0)
            reward = self.reward_model(state_t, action_t)
        self.reward_model.train()
        return reward.item()
    
    def reset_trajectory(self):
        """
        Clears the recent trajectory
        Call at the start of each new episode
        """
        self.recent_trajectory.clear()
    
    def get_stats(self):
        """Returns useful statistics"""
        return {
            'feedback_count': self.feedback_count,
            'training_samples': len(self.training_buffer),
            'trajectory_length': len(self.recent_trajectory),
            'total_steps_observed': self.total_samples
        }