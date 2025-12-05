"""
Test básico de Spinning Up con DDPG en Pendulum
"""
import gymnasium as gym
import torch
from spinup import ddpg_pytorch as ddpg

def main():
    # Crear environment simple
    env = gym.make('Pendulum-v1')
    
    print("=" * 50)
    print("Testing Spinning Up DDPG")
    print("=" * 50)
    print(f"Environment: Pendulum-v1")
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print("=" * 50)
    
    # Entrenar con DDPG
    ddpg(
        env_fn=lambda: gym.make('Pendulum-v1'),
        ac_kwargs=dict(hidden_sizes=[256, 256]),  # Tamaño de las redes
        seed=0,
        steps_per_epoch=4000,  # Pasos por epoch
        epochs=10,  # Número de epochs (10 epochs = 40k steps)
        replay_size=int(1e6),  # Tamaño del replay buffer
        gamma=0.99,  # Discount factor
        polyak=0.995,  # Para target network updates
        pi_lr=1e-3,  # Learning rate del actor
        q_lr=1e-3,  # Learning rate del critic
        batch_size=100,
        start_steps=10000,  # Pasos de exploración random al inicio
        update_after=1000,  # Cuando empezar a entrenar
        update_every=50,  # Cada cuántos pasos actualizar
        act_noise=0.1,  # Ruido de exploración
        num_test_episodes=10,  # Episodios para evaluar
        max_ep_len=200,  # Máxima longitud de episodio
        logger_kwargs=dict(output_dir='./spinup_logs', exp_name='ddpg_pendulum'),
        save_freq=1  # Guardar modelo cada epoch
    )
    
    print("\n✅ Training completed!")
    print("Logs saved in: ./spinup_logs/ddpg_pendulum")
    print("Model saved as: ./spinup_logs/ddpg_pendulum/pyt_save/model.pt")

if __name__ == '__main__':
    main()