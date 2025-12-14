"""
Training script for Pure DeepCOACH on self-touch task
No DDPG blending - only COACH policy with human feedback
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

from deep_coach import DeepCOACHModule
from rewards import * 
from TactileObsWrapper import TactileObsWrapper
from feedback_capture_coach import HumanFeedbackCaptureCoach

episodes = 15
batch_size = 64


def scale_action(raw_action, env):
    """Scale action from [-1, 1] to environment action space"""
    low  = env.action_space.low
    high = env.action_space.high
    return low + 0.5 * (raw_action + 1.0) * (high - low)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='config_selftouch.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--train_for', default=1000, type=int,
                        help='Total timesteps of training')
    parser.add_argument('--window_size', default=10, type=int,
                        help='Window size L for DeepCOACH')
    parser.add_argument('--eligibility_decay', default=0.35, type=float,
                        help='Eligibility decay λ')
    parser.add_argument('--learning_rate', default=0.00025, type=float,
                        help='Learning rate α')
    parser.add_argument('--entropy_coef', default=1.5, type=float,
                        help='Entropy coefficient β')
    parser.add_argument('--minibatch_size', default=6, type=int,
                        help='Minibatch size m')
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

    # Pure COACH agent (no DDPG)
    coach = DeepCOACHModule(
        input_dim, 
        output_dim,
        window_size=args.window_size,
        eligibility_decay=args.eligibility_decay,
        human_delay=0,
        learning_rate=args.learning_rate,
        entropy_coef=args.entropy_coef,
        minibatch_size=args.minibatch_size
    )
    
    feedback_capture = HumanFeedbackCaptureCoach()

    scores = []
    avg_scores = []
    eligibility_norms = []

    smax = args.train_for
    base = env.unwrapped
    renderer = env.rendered

    print("\n" + "="*70)
    print("Pure DeepCOACH Training (Algorithm 1 - No DDPG)")
    print("="*70)
    print(f"Window size L: {args.window_size}")
    print(f"Eligibility decay λ: {args.eligibility_decay}")
    print(f"Learning rate α: {args.learning_rate}")
    print(f"Entropy coefficient β: {args.entropy_coef}")
    print(f"Minibatch size m: {args.minibatch_size}")
    print("\nPress 'g' for GOOD feedback (+1.0)")
    print("Press 'b' for BAD feedback (-1.0)")
    print("Press 'n' for NEUTRAL feedback (0.0)")
    print("="*70 + "\n")

    for episode in range(episodes):
        steps = 0
        state, _ = env.reset()
        feedback_capture.reset()
        coach.reset_trajectory()

        total_reward = 0

        episode_done = False

        prev_obs = None
        prev_region = set()

        env.last_action = np.zeros(env.action_space.shape[0])

        while steps < smax and not episode_done:
            steps += 1

            # Get action from COACH policy only
            raw_action, action_prob = coach.predict_action(state, deterministic=False)
            action = scale_action(raw_action, env)
            action = np.clip(action, env.action_space.low, env.action_space.high)

            next_state, _, terminated, truncated, _ = env.step(action)
            env.last_action = action.copy()
            done = terminated or truncated

            # Compute intrinsic reward (optional - can be used for monitoring)
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
            
            # Capture human feedback
            feedback = feedback_capture.capture_feedback(steps, min_interval=30)
            
            if feedback is not None:
                coach.give_feedback(feedback)

            # Observe step for eligibility traces (COACH only)
            coach.observe_step(state, raw_action, action_prob)
            
            # Train COACH
            if coach.has_valid_model():
                eligibility_norm = coach.train_batch()
                if eligibility_norm is not None:
                    eligibility_norms.append(eligibility_norm)

            prev_obs = next_state.copy()
            state = next_state
            total_reward += intrinsic_reward

            episode_done = done
        
        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        print(f"\nEpisode {episode + 1}/{episodes}")
        print(f"  Score: {total_reward:.2f}")
        print(f"  Avg Score (last 100): {avg_score:.2f}")
        
        stats = coach.get_stats()
        if stats:
            print(f"  COACH Stats:")
            print(f"    - Feedback count: {stats['feedback_count']}")
            print(f"    - Windows stored: {stats['windows_stored']}")
            print(f"    - Has valid model: {coach.has_valid_model()}")

    env.close()

    # Display final feedback statistics
    print("\n" + "="*70)
    print("FINAL FEEDBACK STATISTICS")
    print("="*70)
    feedback_stats = feedback_capture.get_stats()
    print(f"Total feedbacks: {feedback_stats['total']}")
    print(f"  - Positive: {feedback_stats['positive']}")
    print(f"  - Negative: {feedback_stats['negative']}")
    print(f"  - Neutral: {feedback_stats['neutral']}")
    print("="*70 + "\n")

    # Save model
    torch.save(coach.policy.state_dict(), "coach_pure_selftouch.pt")

    # Plot results
    plt.figure(figsize=(15, 5))
    
    plt.subplot(1, 3, 1)
    plt.plot(scores)
    plt.title('Scores per Episode (Pure COACH)')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    
    plt.subplot(1, 3, 2)
    plt.plot(avg_scores)
    plt.title('Moving Average (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')
    plt.grid(True)
    
    if eligibility_norms:
        plt.subplot(1, 3, 3)
        plt.plot(eligibility_norms)
        plt.title('DeepCOACH Eligibility Trace Magnitude')
        plt.xlabel('Training Step')
        plt.ylabel('||Eligibility||')
        plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('pure_coach_training.png')
    print("\nSaved training plots as 'pure_coach_training.png'")
    print("Saved model: coach_pure_selftouch.pt")

if __name__ == '__main__':
    main()
