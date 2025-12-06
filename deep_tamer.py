from collections import deque
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import numpy as np
import random

class HumanRewardModel(nn.Module):
    """
    Red neuronal que aprende a predecir el feedback prosódico humano (-1, 0, +1)
    basándose en pares estado-acción
    """
    def __init__(self, state_size, action_size, hidden_size=256):
        super(HumanRewardModel, self).__init__()
        
        self.fc1 = nn.Linear(state_size + action_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, 1)  # Predice recompensa escalar
        
    def forward(self, state, action):
        """
        state: [batch, state_size]
        action: [batch, action_size]
        returns: [batch, 1] predicción de recompensa humana
        """
        x = torch.cat([state, action], dim=1)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)  # Sin activación, salida continua
    
class TAMERModule:
    """
    Módulo TAMER que maneja feedback humano prosódico sobre trayectorias
    """
    def __init__(self, state_size, action_size, 
                 credit_window=15, 
                 decay_factor=0.9,
                 learning_rate=0.001):
        
        self.state_size = state_size
        self.action_size = action_size
        self.credit_window = credit_window
        self.decay_factor = decay_factor
        
        # Red que predice recompensa humana
        self.reward_model = HumanRewardModel(state_size, action_size)
        self.optimizer = optim.Adam(self.reward_model.parameters(), lr=learning_rate)
        self.criterion = nn.MSELoss()
        
        # Deque pequeño para trayectoria reciente (ventana de crédito)
        self.recent_trajectory = deque(maxlen=credit_window)
        
        # Buffer grande para ejemplos de entrenamiento
        self.training_buffer = deque(maxlen=10000)
        
        # Estadísticas
        self.feedback_count = 0
        self.total_samples = 0

        self.num_feedback_samples = 0

    def has_valid_model(self):
        return self.num_feedback_samples > 10
        
    def observe_step(self, state, action):
        """
        Registra cada paso de la trayectoria
        Llamar esto en cada step del episodio
        """
        self.recent_trajectory.append({
            'state': state.copy(),
            'action': action.copy()
        })
        self.total_samples += 1
    
    def give_prosody_feedback(self, feedback_value):
        """
        Humano da feedback prosódico sobre la trayectoria reciente
        
        Args:
            feedback_value: -1 (prosody negativa), 0 (neutral), +1 (prosody positiva)
        """
        assert feedback_value in [-1, 0, 1], "Feedback debe ser -1, 0, o +1"
        
        if len(self.recent_trajectory) == 0:
            print("Warning: No hay trayectoria para asignar feedback")
            return
        
        self.feedback_count += 1
        samples_added = 0
        
        # Asigna crédito a todos los pasos en la ventana reciente
        # con decaimiento temporal (pasos más recientes reciben más crédito)
        for i, step in enumerate(reversed(self.recent_trajectory)):
            # i=0 es el paso más reciente, i=N-1 es el más antiguo
            credit = feedback_value * (self.decay_factor ** i)
            
            self.training_buffer.append({
                'state': step['state'],
                'action': step['action'],
                'human_reward': credit,
                'feedback_id': self.feedback_count
            })
            samples_added += 1
        
        print(f"Feedback {feedback_value:+d} → {samples_added} pasos creditados "
              f"(buffer: {len(self.training_buffer)} samples)")
    
    def train_batch(self, batch_size=64):
        """
        Entrena el modelo de recompensa humana con un batch
        Llamar esto periódicamente durante el entrenamiento
        
        Returns:
            loss: pérdida del batch (None si no hay suficientes datos)
        """
        if len(self.training_buffer) < batch_size:
            return None
        
        # Sample batch aleatorio
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
        Predice qué feedback prosódico daría el humano para este estado-acción
        
        Returns:
            reward: valor escalar predicho (aproximadamente entre -1 y +1)
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
        Limpia la trayectoria reciente
        Llamar al inicio de cada nuevo episodio
        """
        self.recent_trajectory.clear()
    
    def get_stats(self):
        """Retorna estadísticas útiles"""
        return {
            'feedback_count': self.feedback_count,
            'training_samples': len(self.training_buffer),
            'trajectory_length': len(self.recent_trajectory),
            'total_steps_observed': self.total_samples
        }