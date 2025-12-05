import torch
import torch.nn as nn

from collections import deque
import random
import torch.optim as optim

import numpy as np


class DQN(nn.Module):
    def __init__(self, input_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, output_dim)
        
    def forward(self, x):
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))
        return self.fc3(x)
    
class DQNAgent:
    def __init__(self, 
                 env,
                 input_dim: int, 
                 output_dim: int, 
                 learning_rate: float = 1e-3,
                 gamma: float = 0.99,
                 memory_size: int = 10000, 
                 batch_size: int = 64,
                 criterion: torch.optim = nn.MSELoss,
                 optimizer: torch.optim = optim.Adam,
                 epsilon_decay: float = 0.995,
                 epsilon_min: float = 0.1):
        
        self.policy_net = DQN(input_dim, output_dim)
        self.target_net = DQN(input_dim, output_dim)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()
        
        self.optimizer = optimizer(self.policy_net.parameters(), lr=learning_rate)
        self.memory = deque(maxlen=memory_size)
        self.gamma = gamma
        self.epsilon = 1.0
        self.batch_size = batch_size

        self.criterion = criterion()

        self.env = env
        
        self.epsilon_decay = epsilon_decay
        self.epsilon_min = epsilon_min

        
    def select_action(self, state):
        """Epsilon-greedy action selection"""
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        
        state = torch.FloatTensor(state).unsqueeze(0)
        q_values = self.policy_net(state)
        return torch.argmax(q_values).item()
    
    def store_transition(self, state, action, reward, next_state, done):
        """Store experience in replay buffer"""
        self.memory.append((state, action, reward, next_state, done))
    
    def optimize_model(self):
        """Train the policy network"""
        if len(self.memory) < self.batch_size: #not enough memory
            return
        
        batch = random.sample(self.memory, self.batch_size)

        state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)

        state_batch = torch.FloatTensor(np.array(state_batch))
        action_batch = torch.LongTensor(np.array(action_batch)).unsqueeze(1)
        reward_batch = torch.FloatTensor(np.array(reward_batch))
        next_state_batch = torch.FloatTensor(np.array(next_state_batch))
        done_batch = torch.FloatTensor(np.array(done_batch))

        # Compute q of actions (actual Q)
        q_values = self.policy_net(state_batch).gather(1, action_batch).squeeze()

        # Compute Target Q-Values
        with torch.no_grad():
            max_next_q_values = self.target_net(next_state_batch).max(1)[0]

            #Bellman equation: Target = Recompensa + Gamma * max(Q(s', a')) * (1 - done)
            target_q_values = reward_batch + self.gamma * max_next_q_values * (1 - done_batch)

        loss = self.criterion(q_values, target_q_values)

        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
    
    def update_target_network(self):
        """Copy weights from policy to target network"""
        self.target_net.load_state_dict(self.policy_net.state_dict())
    
    def decay_epsilon(self):
        """Reduce exploration rate"""
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)
