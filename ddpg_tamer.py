from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

from deep_tamer import *

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
    def __init__(self, state_size,action_size):
        super(Critic,self).__init__()

        self.fc1 = nn.Linear(state_size,400)
        self.fc2 = nn.Linear(400+action_size,300)
        self.fc3 = nn.Linear(300,1)

        nn.init.uniform_(self.fc3.weight, -0.003, 0.003)
        nn.init.uniform_(self.fc3.bias, -0.003, 0.003) 


    def forward(self,state,action):
        x = F.relu(self.fc1(state))
        x = torch.cat([x,action],dim=1)
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
    def __init__(self, state_size, action_size, use_tamer=False, 
                 alpha=0.5, beta=0.5):
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

        self.use_tamer = use_tamer
        self.alpha = alpha  #weight intrisec reward
        self.beta = beta    #weight human feedback
        
        if self.use_tamer:
            self.tamer = TAMERModule(state_size, action_size)
            print(f"Using TAMER: alpha={alpha}, beta={beta}")
        else:
            self.tamer = None

    def remember(self,state,action,reward,next_state,done):
        if self.use_tamer:
            # predict human feedback
            human_reward = self.tamer.predict_human_reward(state, action)
            
            combined_reward = self.alpha * reward + self.beta * human_reward
            

            self.tamer.observe_step(state, action)
        else:
            combined_reward = reward

        self.memory.append((state,action,combined_reward,next_state,done))

    def act(self,state,add_noise=True):
        state_tensor = torch.FloatTensor(state).unsqueeze(0)

        self.actor.eval() 
        with torch.no_grad():
            action = self.actor(state_tensor)
        self.actor.train() 
        
        action = action.detach().numpy()[0]

        if add_noise:
            action += self.noise.sample()
            action = np.clip(action, -1, 1)
        
        return action
    

    def train(self, batch_size):
        if len(self.memory) < batch_size:
            return
        
        minibatch = random.sample(self.memory, batch_size)

        states = torch.FloatTensor(np.array([i[0] for i in minibatch]))
        actions = torch.FloatTensor(np.array([i[1] for i in minibatch]))
        rewards = torch.FloatTensor(np.array([i[2] for i in minibatch])).unsqueeze(1) # Shape: [batch, 1]
        next_states = torch.FloatTensor(np.array([i[3] for i in minibatch]))
        dones = torch.FloatTensor(np.array([i[4] for i in minibatch])).unsqueeze(1) # Shape: [batch, 1]

        with torch.no_grad():
            #Get next action from network Target actor
            next_actions = self.target_actor(next_states)
            #Get Q from Critic
            target_q_values = self.target_critic(next_states, next_actions)
            #Bellman Target: r + gamma * Q_target * (1 - done)
            target_values = rewards + (self.gamma * target_q_values * (1 - dones))
        
        current_q_values = self.critic(states, actions)

        critic_loss = self.critic_criterion(current_q_values, target_values)

        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()

        # we want to max the Q-value so we minimize the neg
        predicted_actions = self.actor(states)
        actor_loss = -self.critic(states, predicted_actions).mean()

        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()

        self.update_target_networks()

        if self.use_tamer:
            tamer_loss = self.tamer.train_batch(batch_size=64)
            return {'actor_loss': actor_loss.item(), 
                    'critic_loss': critic_loss.item(),
                    'tamer_loss': tamer_loss}
        
        return {'actor_loss': actor_loss.item(), 
                'critic_loss': critic_loss.item()}


    def update_target_networks(self):
        # Soft update Actor
        for target_param, param in zip(self.target_actor.parameters(), self.actor.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
        # Soft update Critic
        for target_param, param in zip(self.target_critic.parameters(), self.critic.parameters()):
            target_param.data.copy_(self.tau * param.data + (1.0 - self.tau) * target_param.data)
        
    def give_human_feedback(self, feedback_value):
        """
        Args:
            feedback_value: -1 (neg), 0 (neutral), +1 (pos)
        """
        if not self.use_tamer:
            print("Warning: not using TAMER")
            return
        
        self.tamer.give_prosody_feedback(feedback_value)

    def reset_episode(self):
        self.noise.reset()
        if self.use_tamer:
            self.tamer.reset_trajectory()

    def get_tamer_stats(self):
        if self.use_tamer:
            return self.tamer.get_stats()
        return None