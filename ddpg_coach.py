from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from deep_coach import *

class Actor(nn.Module):
    def __init__(self, state_size, action_size):
        super(Actor, self).__init__()
        
        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400, 300)
        self.fc3 = nn.Linear(300, action_size)
        
        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)
        nn.init.uniform_(self.fc3.bias, -0.003, 0.003) 

    def forward(self, state):
        x = F.relu(self.fc1(state))
        x = F.relu(self.fc2(x))
        return torch.tanh(self.fc3(x))
    
class Critic(nn.Module):
    def __init__(self, state_size, action_size):
        super(Critic, self).__init__()

        self.fc1 = nn.Linear(state_size, 400)
        self.fc2 = nn.Linear(400 + action_size, 300)
        self.fc3 = nn.Linear(300, 1)

        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)
        nn.init.uniform_(self.fc3.bias, -0.003, 0.003) 

    def forward(self, state, action):
        x = F.relu(self.fc1(state))
        x = torch.cat([x, action], dim=1)
        x = F.relu(self.fc2(x))
        return self.fc3(x)

class OUNoise:
    """EXPLORATION: Ornstein-Uhlenbeck process"""
    def __init__(self, size, mu=0., theta=0.15, sigma=0.6):
        self.mu = mu * np.ones(size)
        self.theta = theta
        self.sigma = sigma
        self.reset()

    def reset(self):
        self.state = self.mu.copy()

    def sample(self):
        dx = self.theta * (self.mu - self.state) + self.sigma * np.random.randn(len(self.state))
        self.state += dx
        return self.state

class DDPGAgent:
    def __init__(self, state_size, action_size, use_coach=False, 
                 alpha=0.7, beta=0.3):
        """
        Args:
            state_size: dimension of state
            action_size: dimension of action
            use_coach: whether to use DeepCOACH
            alpha: weight for DDPG action (autonomous learning)
            beta: weight for COACH action (human corrections)
        """
        self.state_size = state_size
        self.action_size = action_size
        self.memory = deque(maxlen=100000)
        self.gamma = 0.99
        self.tau = 0.001
        self.actor_lr = 0.001
        self.critic_lr = 0.001

        self.actor = Actor(state_size, action_size)
        self.critic = Critic(state_size, action_size)
        self.target_actor = Actor(state_size, action_size)
        self.target_critic = Critic(state_size, action_size)

        self.target_actor.load_state_dict(self.actor.state_dict())
        self.target_critic.load_state_dict(self.critic.state_dict())

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=self.actor_lr)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=self.critic_lr)

        self.critic_criterion = nn.MSELoss()

        self.noise = OUNoise(action_size)

        self.use_coach = use_coach
        self.alpha = alpha  # weight for DDPG action
        self.beta = beta    # weight for COACH action
        
        if self.use_coach:
            self.coach = DeepCOACHModule(
                state_size, 
                action_size,
                window_size=15,
                eligibility_decay=0.9,
                human_delay=0,
                learning_rate=0.001,
                entropy_coef=0.01,
                minibatch_size=4
            )
            print(f"Using DeepCOACH (Algorithm 1): alpha={alpha}, beta={beta}")
            print(f"Action blending: action = {alpha}*ddpg + {beta}*coach")
        else:
            self.coach = None
        
        self.last_coach_prob = 1.0

    def remember(self, state, action, action_prob, reward, next_state, done):
        """
        Store experience in replay buffer
        Note: 'action' here is the blended action (ddpg + coach)
        
        Args:
            action_prob: probability πθ(action|state) for importance sampling
        """
        if self.use_coach:
            # Observe the step for eligibility traces (st-d, at-d, pt-d)
            self.coach.observe_step(state, action, action_prob)

        self.memory.append((state, action, reward, next_state, done))

    def act(self, state, add_noise=True):
        """
        Select action using blend of DDPG and COACH policies
        """
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        # Get DDPG action
        self.actor.eval() 
        with torch.no_grad():
            ddpg_action = self.actor(state_tensor)
        self.actor.train() 
        
        ddpg_action = ddpg_action.detach().numpy()[0]

        # Add exploration noise to DDPG action
        if add_noise:
            ddpg_action += self.noise.sample()
            ddpg_action = np.clip(ddpg_action, -1, 1)
        
        # Blend with COACH action if available and trained
        if self.use_coach and self.coach.has_valid_model():
            coach_action, coach_prob = self.coach.predict_action(state, deterministic=False)
            
            # Store action probability for importance sampling
            self.last_coach_prob = coach_prob
            
            # Blend actions: weighted combination
            blended_action = self.alpha * ddpg_action + self.beta * coach_action
            blended_action = np.clip(blended_action, -1, 1)
            
            return blended_action, coach_prob
        else:
            # No COACH or not enough training data yet
            return ddpg_action, 1.0

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)

        states = torch.FloatTensor(np.array([i[0] for i in minibatch]))
        actions = torch.FloatTensor(np.array([i[1] for i in minibatch]))
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).unsqueeze(1)
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch]))
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).unsqueeze(1)

        # Train Critic
        with torch.no_grad():
            next_actions = self.target_actor(next_states)
            target_q_values = self.target_critic(next_states, next_actions)
            target_values = rewards + (self.gamma * target_q_values * (1 - dones))
        
        current_q_values = self.critic(states, actions)
        critic_loss = self.critic_criterion(current_q_values, target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # Train Actor
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        # Update target networks
        self.update_target_networks()

        # Train COACH if being used
        losses = {
            'actor_loss': actor_loss.item(), 
            'critic_loss': critic_loss.item()
        }
        
        if self.use_coach:
            eligibility_norm = self.coach.train_batch()  # Magnitude of eligibility trace update
            losses['coach_eligibility_norm'] = eligibility_norm
        
        return losses

    def update_target_networks(self):
        # Soft update Actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        # Soft update Critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
    def give_feedback(self, feedback_value):
        """
        Provide human feedback (scalar value)
        
        Args:
            feedback_value: scalar feedback (-1, 0, +1 or continuous)
        """
        if not self.use_coach:
            print("Warning: not using DeepCOACH")
            return
        
        self.coach.give_feedback(feedback_value)

    def reset_episode(self):
        self.noise.reset()
        if self.use_coach:
            self.coach.reset_trajectory()

    def get_coach_stats(self):
        if self.use_coach:
            return self.coach.get_stats()
        return None
    
    def adjust_blend_weights(self, alpha, beta):
        """
        Adjust the blending weights between DDPG and COACH
        Useful for progressive autonomy (reduce beta over time)
        """
        assert abs(alpha + beta - 1.0) < 0.01, "alpha + beta should sum to ~1.0"
        self.alpha = alpha
        self.beta = beta
        print(f"Updated blend weights: alpha={alpha}, beta={beta}")
