"""
Example: Q-learning with human feedback for self-touch in BabyBench.

Discrete actions (primitives):
    a0: raise right arm
    a1: lower right arm
    a2: move right hand closer to body
    a3: move right hand away from body
    a4: move left hand closer to body
    a5: move left hand away from body

Reward comes ONLY from human (p/n/Enter),
as in the paper "Learning to Refine Behavior Using Prosodic Feedback".
"""

import argparse
import sys
import os

import numpy as np
import gymnasium as gym
from gymnasium.spaces import Discrete
import mujoco
import matplotlib.pyplot as plt
import yaml

sys.path.append(".")
sys.path.append("..")
import babybench.utils as bb_utils


# ===================== JOINT INDICES / PRIMITIVES =====================

# Right shoulder
RIGHT_SHOULDER_HORIZONTAL = 6   # forward/backward
RIGHT_SHOULDER_ABDUCTION  = 7   # lateral (raise/lower)
RIGHT_ELBOW               = 9

# Left shoulder
LEFT_SHOULDER_HORIZONTAL  = 10
LEFT_SHOULDER_ABDUCTION   = 11
LEFT_ELBOW                = 13

# Hand / fingers (in case you use them later)
RIGHT_HAND_JOINTS = [8, 14, 15, 16]
LEFT_HAND_JOINTS  = [12, 17, 18, 19]
RIGHT_FINGERS     = 20
LEFT_FINGERS      = 21

NB_ACTIONS = 12

ACTION_NAMES = {
    0 : 'Raise Right Arm',
    1 : 'Lower Right Arm',
    2 : 'Move Right Hand Closer',
    3 : 'Move Right Hand Away',
    4 : 'Raise Left Arm',
    5 : 'Lower Left Arm',
    6 : 'Move Left Hand Closer',
    7 : 'Move Left Hand Away',
    8 : 'Right Elbow +',
    9 : 'Right Elbow -',
    10: 'Left Elbow +',
    11: 'Left Elbow -'
}

def primitive_action(a_idx, low, high, delta=0.3):
    """
    Maps discrete action (0..NB_ACTIONS) to a continuous vector (30,)
    """
    action = np.zeros_like(low, dtype=np.float32)

    if a_idx == 0:
        # a0: raise right arm (use right abduction)
        j = RIGHT_SHOULDER_ABDUCTION
        action[j] = np.clip(action[j] + delta, low[j], high[j])

    elif a_idx == 1:
        # a1: lower right arm
        j = RIGHT_SHOULDER_ABDUCTION
        action[j] = np.clip(action[j] - delta, low[j], high[j])

    elif a_idx == 2:
        # a2: move right hand closer to body (x axis / horizontal)
        j = RIGHT_SHOULDER_HORIZONTAL
        action[j] = np.clip(action[j] + delta, low[j], high[j])

    elif a_idx == 3:
        # a3: move right hand away from body
        j = RIGHT_SHOULDER_HORIZONTAL
        action[j] = np.clip(action[j] - delta, low[j], high[j])
    elif a_idx == 4:
        # a4: raise left arm
        j = LEFT_SHOULDER_ABDUCTION
        action[j] = np.clip(action[j] + delta, low[j], high[j])
    elif a_idx == 5:
        # a5: lower left arm
        j = LEFT_SHOULDER_ABDUCTION
        action[j] = np.clip(action[j] - delta, low[j], high[j])
        
    elif a_idx == 6:
        # a6: move left hand closer to body
        j = LEFT_SHOULDER_HORIZONTAL
        action[j] = np.clip(action[j] + delta, low[j], high[j])

    elif a_idx == 7:
        # a7: move left hand away from body
        j = LEFT_SHOULDER_HORIZONTAL
        action[j] = np.clip(action[j] - delta, low[j], high[j])
    
    elif a_idx == 8:
        # a8: right elbow +
        j = RIGHT_ELBOW
        action[j] = np.clip(action[j] + delta, low[j], high[j])

    elif a_idx == 9:
        # a9: right elbow -
        j = RIGHT_ELBOW
        action[j] = np.clip(action[j] - delta, low[j], high[j])

    elif a_idx == 10:
        # a10: left elbow +
        j = LEFT_ELBOW
        action[j] = np.clip(action[j] + delta, low[j], high[j])
    
    elif a_idx == 11:
        # a11: left elbow -
        j = LEFT_ELBOW
        action[j] = np.clip(action[j] - delta, low[j], high[j])

    return action


def discretize_state(obs):
    """
    Converts observation (dict with 'touch') to a discrete state 0..3,
    based on the number of active contact sensors.
    TODO: This doesn't work, since touch also counts the touch with the ground
    """
    touch = obs["touch"]
    n_touch = np.sum(touch > 0)
    print(n_touch)

    if n_touch == 0:
        return 0        # no contact
    elif n_touch == 1:
        return 1        # single contact
    elif n_touch <= 3:
        return 2        # 2 or 3 contacts
    else:
        return 3        # 4 or more contacts
    
    


# ===================== WRAPPERS =====================

class DiscretePrimitivesWrapper(gym.Env):
    """
    Env wrapper that exposes a discrete action_space {0..NB_ACTIONS} and
    translates each discrete action to a continuous primitive.
    """
    def __init__(self, env, delta=0.8):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = Discrete(NB_ACTIONS)  # a0..a9
        self.low = env.action_space.low
        self.high = env.action_space.high
        self.delta = delta

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, a_idx):
        # Translate discrete action to continuous vector
        action_vec = primitive_action(a_idx, self.low, self.high, self.delta)
        obs, reward, terminated, truncated, info = self.env.step(action_vec)
        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render()

    def close(self):
        return self.env.close()


class HumanFeedbackWrapper(gym.Env):
    """
    Env wrapper that requests human feedback via keyboard:
    - 'p' -> +1
    - 'n' -> -1
    - Enter / other -> 0
    """
    def __init__(self, env, use_env_reward=False):
        super().__init__()
        self.env = env
        self.observation_space = env.observation_space
        self.action_space = env.action_space
        self.use_env_reward = use_env_reward

    def get_human_feedback(self):
        if np.random.random() < 0.3:
            fb = input("[p] positive, [n] negative, [Enter] nothing: ").strip().lower()
        else:
            return 0
        if fb == "p":
            return 1.0
        elif fb == "n":
            return -1.0
        else:
            return 0.0

    def reset(self, **kwargs):
        return self.env.reset(**kwargs)

    def step(self, action):
        print("action =", action, f'({ACTION_NAMES[action]})')
        obs, env_reward, terminated, truncated, info = self.env.step(action)

        r_h = self.get_human_feedback()

        if self.use_env_reward:
            reward = env_reward + r_h
        else:
            reward = r_h   # ONLY human reward

        info = dict(info)
        info["env_reward"] = env_reward
        info["human_reward"] = r_h

        return obs, reward, terminated, truncated, info

    def render(self):
        return self.env.render() if hasattr(self.env, "render") else None

    def close(self):
        return self.env.close()


# ===================== Q-LEARNING =====================

def choose_action(Q, state, epsilon, n_actions):
    """
    Epsilon-greedy policy over the Q-table.
    """
    if np.random.rand() < epsilon:
        return np.random.randint(n_actions)   # explore
    return int(np.argmax(Q[state]))           # exploit


def run_episode(env, Q, alpha, gamma, epsilon, base, renderer,train_for):
    obs, info = env.reset()
    s = discretize_state(obs)
    done = False

    steps = 0
    while steps < train_for and not done:
        steps+=1
        a = choose_action(Q, s, epsilon, Q.shape[1])

        next_obs, r_h, terminated, truncated, info = env.step(a)

        #render
        renderer.update_scene(base.data)
        img = renderer.render()
        plt.imshow(img)
        plt.axis("off")
        plt.pause(0.01)
        

        done = terminated or truncated
        s_next = discretize_state(next_obs)

        best_next = np.max(Q[s_next])
        td_target = r_h + gamma * best_next
        Q[s, a] = (1 - alpha) * Q[s, a] + alpha * td_target

        s = s_next

        if done:
            obs,_ = env.reset()



# ===================== MAIN =====================

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--config',
        default='examples/config_selftouch.yml',
        type=str,
        help='Configuration file for BabyBench/MIMo.'
    )
    parser.add_argument(
        '--episodes',
        default=2,
        type=int,
        help='Number of Q-learning episodes (human in the loop).'
    )
    parser.add_argument(
        '--alpha',
        default=0.1,
        type=float,
        help='Learning rate for Q-learning.'
    )
    parser.add_argument(
        '--gamma',
        default=0.9,
        type=float,
        help='Discount factor for Q-learning.'
    )
    parser.add_argument(
        '--epsilon',
        default=0.2,
        type=float,
        help='Exploration probability (epsilon-greedy).'
    )
    parser.add_argument('--train_for', 
                        default=150, 
                        type=int,
                        help='Total timesteps of training')
    args = parser.parse_args()


    with open(args.config) as f:
        config = yaml.safe_load(f)

    #1) normal babybench env 
    env_cont = bb_utils.make_env(config)

    # 2) discrete primitives wrapper
    env_disc = DiscretePrimitivesWrapper(env_cont, delta=0.8)

    # 3) human feedback wrapper (only human reward)
    env_human = HumanFeedbackWrapper(env_disc, use_env_reward=False)


    # 4) Q-table: 4 states x NB_ACTIONS actions
    n_states = 4
    n_actions = NB_ACTIONS
    Q = np.zeros((n_states, n_actions), dtype=float)

    print("Starting training with human feedback...")
    print("Actions: 0..10 (primitives). Reward: p=+1, n=-1, Enter=0.\n")
    base = env_cont.unwrapped
    model = base.model
    data = base.data
    renderer = mujoco.Renderer(model)

    for ep in range(args.episodes):
        print(f"\n=== Episode {ep + 1}/{args.episodes} ===")
        run_episode(env_human, Q, args.alpha, args.gamma, args.epsilon, base, renderer, args.train_for)
        print("Current Q-table:")
        print(Q)

    print("\nTraining finished. Final Q:")
    print(Q)
    np.save(os.path.join(config["save_dir_Q"], "Q.npy"), Q)



if __name__ == '__main__':
    main()