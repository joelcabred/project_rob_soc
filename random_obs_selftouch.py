"""
Example: Random policy for hand-regard baseline
"""
import numpy as np
import os
import gymnasium as gym
import time
import argparse
import mujoco
import yaml
import matplotlib.pyplot as plt 

import mimoEnv
from mimoEnv.envs.mimo_env import MIMoEnv
import mimoEnv.utils as env_utils

import cv2

import sys
sys.path.append(".")
sys.path.append("..")
import babybench.utils as bb_utils

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


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument('--config', default='examples/config_selftouch.yml', type=str,
                        help='The configuration file to set up environment variables')
    parser.add_argument('--train_for', default=10000, type=int,
                        help='Total timesteps of training')
    args = parser.parse_args()
    
    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = bb_utils.make_env(config)
    env.reset()

    env_new = TactileObsWrapper(env)
    
    steps = 0
    obs, _ = env_new.reset()

    smax=args.train_for
    base = env_new.unwrapped

    renderer = env_new.rendered
    
    while steps < smax:
        steps += 1
        action = env.action_space.sample()  # Random action
        obs, reward, terminated, truncated, info = env_new.step(action)
        
        print(obs)

        renderer.update_scene(base.data)
        img = renderer.render()
        img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
        
        cv2.namedWindow("Renderer", cv2.WINDOW_NORMAL)
        cv2.resizeWindow("Renderer", 720, 720)   
        cv2.imshow("Renderer", img_bgr)
        cv2.waitKey(1) 

        done = terminated or truncated

        if done:
            obs,_ = env.reset()

    env.close()

if __name__ == '__main__':
    main()
