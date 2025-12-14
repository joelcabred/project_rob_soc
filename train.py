# train.py
"""
Training script for DDPG agent on BabyBench self-touch task.
Supports training with or without human evaluative feedback (DeepTAMER).
Real-time rendering is always enabled.
"""

import numpy as np
import gymnasium as gym
import yaml
import argparse
import torch
import cv2
import sys
sys.path.append("BabyBench2025_Starter_Kit")

import  BabyBench2025_Starter_Kit.babybench.utils as bb_utils
from ddpg_tamer import DDPGAgent
from TactileObsWrapper import TactileObsWrapper
from rewards import compute_reward_full

from feedback.human_feedback import HumanFeedbackCapture


import warnings
warnings.filterwarnings(
    "ignore",
    message="pkg_resources is deprecated as an API",
    category=UserWarning,
)

from utils import scale_action

# ----------------- main -----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_selftouch.yml")
    parser.add_argument("--episodes", type=int, default=20)
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--use_tamer", action="store_true")
    parser.add_argument("--feedback",type=str,default="keyboard",choices=["keyboard", "prosody", "speech"],help="Type of human feedback backend")
    args = parser.parse_args()

    # -------- env --------
    with open(args.config) as f:
        config = yaml.safe_load(f)

    env = bb_utils.make_env(config)
    env.reset()
    env = TactileObsWrapper(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(
        env, lambda obs: np.clip(obs, -10, 10)
    )

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]

    # -------- agent --------
    agent = DDPGAgent(
        input_dim,
        output_dim,
        use_tamer=args.use_tamer,
        alpha=1.0,
        beta=0.1
    )
    
    feedback = None
    if args.use_tamer:
        
        if args.feedback == "keyboard":
            from feedback.keyboard import KeyboardFeedback
            backend = KeyboardFeedback()

        elif args.feedback == "prosody":
            from feedback.audio_prosody import AudioProsodyFeedback
            backend = AudioProsodyFeedback(
                calib_path="valence_calib_posneg.joblib",
                print_debug=True,
            )

        elif args.feedback == "speech":
            from feedback.audio_speech import SpeechFeedback
            backend = SpeechFeedback(
                print_debug=True,
            )

        else:
            raise ValueError(f"Unknown feedback type: {args.feedback}")

        feedback = HumanFeedbackCapture(
            backend,
            min_interval=5
        )

    # -------- render --------
    base = env.unwrapped
    renderer = env.rendered

    # -------- logs --------
    episode_rewards = []
    episode_intrinsic_rewards = []

    # -------- training loop --------
    for ep in range(args.episodes):
        state, _ = env.reset()
        agent.reset_episode()
        if feedback:
            feedback.reset()

        prev_obs = None
        prev_region = set()
        env.last_action = np.zeros(output_dim)

        total_reward = 0.0
        total_intrinsic = 0.0

        for step in range(args.steps):
            # --- action ---
            raw_action = agent.act(state)  # [-1,1]
            action = scale_action(raw_action, env)
            action = np.clip(
                action,
                env.action_space.low,
                env.action_space.high
            )

            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated
            env.last_action = action.copy()

            # --- intrinsic reward ---
            intrinsic_reward, prev_region = compute_reward_full(
                next_state,
                prev_obs,
                env,
                prev_region
            )

            # --- human feedback ---
            if args.use_tamer and feedback:
                h = feedback.capture_feedback(step)
                if h is not None:
                    agent.give_human_feedback(h)

            # --- learning ---
            agent.remember(
                state,
                action,
                intrinsic_reward,
                next_state,
                done
            )
            agent.train(batch_size=64)

            # --- render (ALWAYS ON) ---
            
            renderer.update_scene(base.data)
            img = renderer.render()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            cv2.namedWindow("Training", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Training", 720, 720)
            cv2.imshow("Training", img_bgr)
            cv2.waitKey(1)
            

            # --- bookkeeping ---
            total_reward += intrinsic_reward
            total_intrinsic += intrinsic_reward
            prev_obs = next_state.copy()
            state = next_state

            if done:
                break

        episode_rewards.append(total_reward)
        episode_intrinsic_rewards.append(total_intrinsic)

        print(
            f"[Episode {ep+1}/{args.episodes}] "
            f"Total reward: {total_reward:.2f}"
        )

    # -------- save --------
    #suffix = "tamer" if args.use_tamer else "baseline"
    if not args.use_tamer:
        suffix = 'ddpg'
    else:
        if args.feedback == "keyboard":
            suffix = 'dtk'
        elif args.feedback == "speech":
            suffix = 'dtw'
        elif args.feedback == "prosody":
            suffix = "dtp"

    torch.save(agent.actor.state_dict(), f"models/actor_{suffix}.pt")
    torch.save(agent.critic.state_dict(), f"models/critic_{suffix}.pt")

    np.save(f"results/train_rewards/train_rewards_{suffix}.npy", np.array(episode_rewards))

    env.close()
    cv2.destroyAllWindows()
    print("Training finished.")


if __name__ == "__main__":
    main()
