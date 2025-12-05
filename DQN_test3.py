import gymnasium as gym
import torch
import torch.nn as nn
import torch.optim as optim
import random
import numpy as np
from collections import deque

import keyboard
import time # pour ralentir l'execussion afin qu'on puisse y mettre le feedback

from DQN import *

print("Script lanzado")
# --- ENV ---
# Creamos el entorno CartPole. El objetivo es mantener el péndulo equilibrado.
env = gym.make("CartPole-v1", render_mode = "human")
print("Env creada")


# --- HYPER ---
learning_rate = 0.001
gamma = 0.99 # Factor de descuento para recompensas futuras
epsilon = 1.0  # Tasa de exploración (explorar vs. explotar)
epsilon_min = 0.01
epsilon_decay = 0.995 # Epsilon disminuye con cada episodio
batch_size = 64
target_update_freq = 1000
memory_size = 10000
episodes = 100

w_feedback = 3.0 # <---- peso del feedback humano (la señal se multiplica por esto)

# --- FEEDBACK FUNCTION ---
# Variables de estado para detectar la pulsación inicial de 'p' o 'n'
prev_n, prev_p = False, False

def feedback_prosody():
    """
    Detecta si se ha pulsado 'p' (positivo) o 'n' (negativo) justo ahora.
    Devuelve 1 (positivo), -1 (negativo) o 0 (nada).
    """
    global prev_p, prev_n
    # Mapeo simple: p = +1, n = -1

    p = keyboard.is_pressed('p')
    n = keyboard.is_pressed('n')

    fb = 0
    # Chequea si 'p' acaba de ser presionado (frente ascendente)
    if p and not prev_p:# positivo
        fb = 1
    # Chequea si 'n' acaba de ser presionado (frente ascendente)
    elif n and not prev_n: # negativo
        fb = -1

    # Actualiza el estado de las teclas para el siguiente paso
    prev_p = p
    prev_n = n
    return fb

# --- Q NETWORKS ---
input_dim = env.observation_space.shape[0]
output_dim = env.action_space.n
# policy_net: La red que se entrena y se usa para seleccionar acciones
policy_net = DQN(input_dim, output_dim)     
# target_net: La red de destino, usada para calcular los valores Q futuros (objetivos)
target_net = DQN(input_dim, output_dim)     
target_net.load_state_dict(policy_net.state_dict())       # Inicializa con los pesos de policy_net
target_net.eval() # Pone la red de destino en modo evaluación (no se entrena)

optimizer = optim.Adam(policy_net.parameters(), lr=learning_rate)
memory = deque(maxlen=memory_size) # El buffer de repetición de experiencias

def select_action(state, epsilon):
    """Selecciona una acción usando la estrategia Epsilon-Greedy."""
    if random.random() < epsilon:
        return env.action_space.sample() # Exploración: acción aleatoria
    
    # Explotación: elige la mejor acción según la red
    state = torch.FloatTensor(state).unsqueeze(0)
    q_values = policy_net(state)
    return torch.argmax(q_values).item()

def optimize_model():
    """Realiza un paso de optimización de la red de política."""
    if len(memory) < batch_size:      # No hay suficientes datos para un lote completo
        return
        
    batch = random.sample(memory, batch_size)
    # Desempaqueta el lote en 5 listas separadas
    state_batch, action_batch, reward_batch, next_state_batch, done_batch = zip(*batch)
        
    # Convierte a tensores de PyTorch
    state_batch = torch.FloatTensor(np.array(state_batch))
    action_batch = torch.LongTensor(np.array(action_batch)).unsqueeze(1)
    reward_batch = torch.FloatTensor(np.array(reward_batch))
    next_state_batch = torch.FloatTensor(np.array(next_state_batch))
    done_batch = torch.FloatTensor(np.array(done_batch))

    # 1. Calcular los valores Q de las acciones tomadas (valor Q actual)
    q_values = policy_net(state_batch).gather(1, action_batch).squeeze()

    # 2. Calcular los valores Q objetivo (Target Q-Values)
    with torch.no_grad():
        # Usamos la red de destino (target_net) para la estabilidad
        max_next_q_values = target_net(next_state_batch).max(1)[0]
        # Ecuación de Bellman: Target = Recompensa + Gamma * max(Q(s', a')) * (1 - done)
        target_q_values = reward_batch + gamma * max_next_q_values * (1 - done_batch)

    # 3. Calcular la pérdida (Error Cuadrático Medio)
    loss = nn.MSELoss()(q_values, target_q_values)
    
    # 4. Propagación hacia atrás y actualización de pesos
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

print("Redes creadas, inicio del entrenamiento")
# --- TRAINING LOOP ---
steps_done = 0
for episode in range(episodes):
    state, _ = env.reset()
    done = False

    print(f"\nEpisode {episode+1}/{episodes}, epsilon={epsilon:.3f}")

    while not done:
        # Hacemos una pequeña pausa para que el humano pueda reaccionar
        time.sleep(0.1)

        action = select_action(state, epsilon)
        next_state, reward, terminated, truncated, info = env.step(action)
        done = terminated or truncated

        # --- FEEDBACK: SIMPLIFICADO ---
        # 1. Chequea si hay señal de feedback ('p' o 'n')
        fb_signal = feedback_prosody()

        # 2. Calcula la recompensa extra del humano
        feedback = 0
        if fb_signal != 0:
            feedback = w_feedback * fb_signal
            print(f"[Feedback Humano Aplicado] {feedback:.1f}")

        # 3. La recompensa total es la suma de la recompensa del entorno y la recompensa humana
        reward_total = reward + feedback

        # Almacena la transición en la memoria de repetición
        memory.append((state, action, reward_total, next_state, done))
        state = next_state

        optimize_model()

        # Actualiza la red de destino periódicamente
        if steps_done > 0 and steps_done % target_update_freq == 0:
            target_net.load_state_dict(policy_net.state_dict())

        steps_done += 1

    # Decaimiento de Epsilon al final del episodio
    epsilon = max(epsilon_min, epsilon * epsilon_decay)
    print(f"Episode {episode+1}/{episodes} completado.")