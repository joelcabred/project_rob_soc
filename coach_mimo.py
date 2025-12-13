"""
Training script for DDPG + DeepCOACH on self-touch task
"""
import numpy as np
import os
os.environ["KMP_DUPLICATE_LIB_OK"]="TRUE"
import gymnasium as gym
import time
import argparse
import yaml
import matplotlib.pyplot as plt 

import torch
import torch.nn as nn
import torch.optim as optim

from collections import deque

import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
import mimoEnv.utils as env_utils

import random
import cv2

import sys
sys.path.append(".")
sys.path.append("..")
import babybench.utils as bb_utils

from ddpg_coach import DDPGAgent
from rewards import * 
from TactileObsWrapper import TactileObsWrapper

episodes = 15
batch_size = 64

use_coach = True

from feedback_capture_coach import HumanFeedbackCaptureCoach


def scale_action(raw_action, env):
    """Scale action from [-1, 1] to environment action space"""
    low  = env.action_space.low
    high = env.action_space.high
    return low + 0.5 * (raw_action + 1.0) * (high - low)


def capture_action_correction(steps, min_interval=30):
    """
    Capture human action correction via keyboard
    
    Returns:
        corrected_action: numpy array or None
    """
    # Simple placeholder - in real implementation, you'd capture
    # human demonstrations through teleoperation or other interface
    # For now, this is a stub that returns None
    
    # TODO: Implement actual action correction capture
    # Options:
    # 1. Record human teleoperation
    # 2. Use predefined good actions for certain states
    # 3. Interactive GUI for specifying corrections
    
    return None


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_selftouch.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--train_for', default=1000, type=int,
                        help='Total timesteps of training')
    parser.add_argument('--alpha', default=0.7, type=float,
                        help='Weight for DDPG action (autonomous learning)')
    parser.add_argument('--beta', default=0.3, type=float,
                        help='Weight for COACH action (human corrections)')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = bb_utils.make_env(config)
    env.reset()
    env = TactileObsWrapper(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(
        env, 
        lambda obs: np.clip(obs, -10, 10)
    )
    
    print("Action low:", env.action_space.low)
    print("Action high:", env.action_space.high)
    print("Action shape:", env.action_space.shape)
    print('-'*50)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]

    agent = DDPGAgent(input_dim, output_dim, use_coach=use_coach,
                      alpha=args.alpha, beta=args.beta)
    
    feedback_capture = HumanFeedbackCaptureCoach() if use_coach else None

    scores = []
    avg_scores = []
    coach_losses = []

    smax = args.train_for
    base = env.unwrapped
    renderer = env.rendered

    print("\n" + "="*70)
    print("DDPG + DeepCOACH Training (Algorithm 1)")
    print("="*70)
    print(f"Action blending: {args.alpha}*DDPG + {args.beta}*COACH")
    print("Press 'g' for GOOD feedback (+1.0)")
    print("Press 'b' for BAD feedback (-1.0)")
    print("Press 'n' for NEUTRAL feedback (0.0)")
    print("="*70 + "\n")

    for episode in range(episodes):
        steps = 0
        state, _ = env.reset()
        if use_coach and feedback_capture:
            feedback_capture.reset()

        agent.reset_episode()

        total_reward = 0
        total_intrinsic_reward = 0

        episode_done = False

        prev_obs = None
        prev_region = set()

        env.last_action = np.zeros(env.action_space.shape[0])

        while steps < smax and not episode_done:
            steps += 1

            raw_action, action_prob = agent.act(state)  # Blended action in [-1, 1] and probability
            action = scale_action(raw_action, env)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_state, _, terminated, truncated, _ = env.step(action)
            env.last_action = action.copy()
            done = terminated or truncated

            # Compute intrinsic reward
            intrinsic_reward, prev_region = compute_reward_full(
                next_state,
                prev_obs,
                env,
                prev_region
            )
            lim_pen = joint_limit_penalty(action, env)
            intrinsic_reward -= 0.8 * lim_pen

            # Render
            renderer.update_scene(base.data)
            img = renderer.render()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Renderer", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Renderer", 720, 720)   
            cv2.imshow("Renderer", img_bgr)
            
            # Capture human feedback using feedback_capture module
            if use_coach and feedback_capture:
                feedback = feedback_capture.capture_feedback(
                    steps, 
                    min_interval=30
                )
                
                if feedback is not None:
                    agent.give_feedback(feedback)

            # Store experience and train (with action probability)
            agent.remember(state, raw_action, action_prob, intrinsic_reward, next_state, done)
            losses = agent.train(batch_size)

            if use_coach and losses and 'coach_eligibility_norm' in losses:
                if losses['coach_eligibility_norm'] is not None:
                    coach_losses.append(losses['coach_eligibility_norm'])

            prev_obs = next_state.copy()
            state = next_state
            total_intrinsic_reward += intrinsic_reward
            total_reward += intrinsic_reward

            episode_done = done
        
        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        print(f"\nEpisode {episode + 1}/{episodes}")
        print(f"  Score: {total_reward:.2f}")
        print(f"  Avg Score (last 100): {avg_score:.2f}")
        
        if use_coach:
            stats = agent.get_coach_stats()
            if stats:
                print(f"  COACH Stats:")
                print(f"    - Feedback count: {stats['feedback_count']}")
                print(f"    - Windows stored: {stats['windows_stored']}")
                print(f"    - Has valid model: {agent.coach.has_valid_model()}")

    env.close()

    # Display final feedback statistics
    if use_coach and feedback_capture:
        print("\n" + "="*70)
        print("FINAL FEEDBACK STATISTICS")
        print("="*70)
        feedback_stats = feedback_capture.get_stats()
        print(f"Total feedbacks: {feedback_stats['total']}")
        print(f"  - Positive: {feedback_stats['positive']}")
        print(f"  - Negative: {feedback_stats['negative']}")
        print(f"  - Neutral: {feedback_stats['neutral']}")
        print("="*70 + "\n")

    # Save models
    torch.save(agent.actor.state_dict(), "actor_coach_selftouch.pt")
    torch.save(agent.critic.state_dict(), "critic_coach_selftouch.pt")
    if use_coach:
        torch.save(agent.coach.policy.state_dict(), "coach_policy_selftouch.pt")

    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(scores)
    plt.title('Scores per Episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(avg_scores)
    plt.title('Moving Average (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.grid(True)
    
    if coach_losses:
        plt.subplot(1, 3, 3)
        plt.plot(coach_losses)
        plt.title('DeepCOACH Eligibility Trace Magnitude')
        plt.xlabel('Training Step')
        plt.ylabel('||Eligibility||')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ddpg_coach_training.png')
    print("\nSaved training plots as 'ddpg_coach_training.png'")
    print("Saved models:")
    print("  - actor_coach_selftouch.pt")
    print("  - critic_coach_selftouch.pt")
    if use_coach:
        print("  - coach_policy_selftouch.pt")

if __name__ == '__main__':
    main()
