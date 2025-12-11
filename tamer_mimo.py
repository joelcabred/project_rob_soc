"""
Training script for DDPG agent with TAMER integration for learning self-touch behavior in MIMo.
Combines traditional RL rewards with real-time human feedback captured via keyboard input during
training episodes. Supports interactive training where humans can provide positive (+), negative (-),
or neutral (0) feedback to shape the agent's behavior beyond the programmed reward function.
"""

import numpy as np
import os
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

#from ddpg import DDPGAgent
from ddpg_tamer import DDPGAgent

from rewards import * 

from TactileObsWrapper import TactileObsWrapper

NUMBER_SENSORS = {}
BODY_IDS = set()

'''
0 world
1 mimo_location
2 hip
3 lower_body
4 upper_body
5 head
6 left_eye
7 right_eye
8 right_upper_arm
9 right_lower_arm
10 right_hand
11 right_fingers
12 left_upper_arm
13 left_lower_arm
14 left_hand
15 left_fingers
16 right_upper_leg
17 right_lower_leg
18 right_foot
19 right_toes
20 left_upper_leg
21 left_lower_leg
22 left_foot
23 left_toes
'''

'''
HAND: 1944
FINGERS: 172
'''

episodes = 15
batch_size = 64

use_tamer = True

from feedback_capture import *


def scale_action(raw_action, env):
    low  = env.action_space.low
    high = env.action_space.high
    return low + 0.5 * (raw_action + 1.0) * (high - low)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='examples/config_selftouch.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--train_for', default=1000, type=int,
                        help='Total timesteps of training')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = bb_utils.make_env(config)
    env.reset()
    env = TactileObsWrapper(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation( #to gain stability
                                            env, 
                                            lambda obs: np.clip(obs, -10, 10)
                                        )
    
    print("Action low:", env.action_space.low)
    print("Action high:", env.action_space.high)
    print("Action shape:", env.action_space.shape)
    print('-'*50)

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]

    agent = DDPGAgent(input_dim,output_dim,use_tamer=use_tamer,
                      alpha=1.0,beta=0.1)
    feedback_capture = HumanFeedbackCapture() if use_tamer else None

    scores = []
    avg_scores = []
    tamer_losses = []


    smax = args.train_for
    base = env.unwrapped
    renderer = env.rendered

    for episode in range(episodes):
        steps = 0
        state, _ = env.reset()

        agent.reset_episode()
        if use_tamer and feedback_capture:
            feedback_capture.reset()


        total_reward = 0
        total_intrinsic_reward = 0
        total_human_reward = 0

        episode_done = False

        prev_obs = None
        prev_region = set()

        env.last_action = np.zeros(env.action_space.shape[0])

        while steps < smax and not episode_done:
            steps += 1

            raw_action = agent.act(state)          # [-1,1]
            action = scale_action(raw_action, env) # [low_i, high_i] per joint
            action = np.clip(action, env.action_space.low, env.action_space.high)


            next_state, _, terminated, truncated, _ = env.step(action)
            env.last_action = action.copy()
            done = terminated or truncated

            intrinsic_reward, prev_region = compute_reward_full(
                next_state,
                prev_obs,
                env,
                prev_region
            )
            lim_pen = joint_limit_penalty(action, env)
            intrinsic_reward -= 0.8 * lim_pen
            
            

            renderer.update_scene(base.data)
            img = renderer.render()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Renderer", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Renderer", 720, 720)   
            cv2.imshow("Renderer", img_bgr)
            cv2.waitKey(1) 

            human_feedback = None
            if use_tamer and feedback_capture:
                human_feedback = feedback_capture.capture_feedback(
                    steps, 
                    min_interval=30
                )

                if human_feedback is not None:
                    agent.give_human_feedback(human_feedback)
                
            print(intrinsic_reward)

            agent.remember(state, action, intrinsic_reward, next_state, done)
            losses = agent.train(batch_size)

            if use_tamer and losses and 'tamer_loss' in losses:
                if losses['tamer_loss'] is not None:
                    tamer_losses.append(losses['tamer_loss'])

            prev_obs = next_state.copy()
            state = next_state
            total_intrinsic_reward  += intrinsic_reward
            total_reward += intrinsic_reward

            episode_done = done
        
        scores.append(total_reward)
        avg_score = np.mean(scores[-100:])
        avg_scores.append(avg_score)

        if (episode + 1) % 10 == 0:
            print(f"Episode {episode + 1}/{episodes} | Score: {total_reward:.2f} | Avg Score (100): {avg_score:.2f}")
    


    env.close()

    torch.save(agent.actor.state_dict(), "actor_selftouch.pt")
    torch.save(agent.critic.state_dict(), "critic_selftouch.pt")


    plt.figure(figsize=(12, 5))
    
    plt.subplot(1, 2, 1)
    plt.plot(scores)
    plt.title('Scores per episode')
    plt.xlabel('Episode')
    plt.ylabel('Score')
    plt.grid(True)
    
    plt.subplot(1, 2, 2)
    plt.plot(avg_scores)
    plt.title('Moving Average (100 episodes)')
    plt.xlabel('Episode')
    plt.ylabel('Average Score')

    plt.grid(True)
    
    plt.tight_layout()
    plt.savefig('ddpg_training.png')
    print("Saved as 'ddpg_training.png'")

if __name__ == '__main__':
    main()
