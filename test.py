# test.py
"""
Evaluation script for a trained DDPG agent on the BabyBench self-touch task.
Computes behavioral metrics with real-time rendering enabled.
"""

import numpy as np
import gymnasium as gym
import yaml
import argparse
import torch
import cv2
import sys
sys.path.append("BabyBench2025_Starter_Kit")

import BabyBench2025_Starter_Kit.babybench.utils as bb_utils
from ddpg_tamer import DDPGAgent
from TactileObsWrapper import TactileObsWrapper
from utils import *

# ----------------- main -----------------

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default="config_selftouch.yml")
    parser.add_argument("--model", required=True, help="Path to actor weights (.pt)")
    parser.add_argument("--episodes", type=int, default=10)
    parser.add_argument("--steps", type=int, default=1000)
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
    agent = DDPGAgent(input_dim, output_dim, use_tamer=False)
    agent.actor.load_state_dict(torch.load(args.model, map_location="cpu"))
    agent.actor.eval()

    # -------- render --------
    base = env.unwrapped
    renderer = env.rendered

    # -------- metrics --------
    all_metrics = []

    for ep in range(args.episodes):
        state, _ = env.reset()

        first_touch_step = None
        contact_steps = 0
        freeze_steps = 0
        regions = set()

        for t in range(args.steps):
            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                raw_action = agent.actor(state_t).numpy()[0]

            action = scale_action(raw_action, env)
            action = np.clip(action,
                             env.action_space.low,
                             env.action_space.high)

            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # ---- METRICS ----
            regions_touched = detect_regions(env)

            # self-touch detection (NO floor contact)
            if len(regions_touched) > 0:
                contact_steps += 1
                if first_touch_step is None:
                    first_touch_step = t

            # diversity
            regions |= regions_touched

            # freeze
            if extract_hand_speed(next_state) < 1.0:
                freeze_steps += 1


            # ---- render (ALWAYS ON) ----
            
            renderer.update_scene(base.data)
            
            img = renderer.render()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)
            
            cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Test", 720, 720)
            cv2.imshow("Test", img_bgr)
            cv2.waitKey(1)
            

            state = next_state
            if done:
                break

        metrics = {
            "episode": ep + 1,
            "first_touch_step": first_touch_step,
            "contact_rate": contact_steps / max(1, args.steps),
            "unique_regions": len(regions),
            "freeze_ratio": freeze_steps / max(1, args.steps)
        }
        all_metrics.append(metrics)

        print("\n[EPISODE METRICS]")
        for k, v in metrics.items():
            print(f"{k}: {v}")

    # -------- summary --------
    print("\n========== SUMMARY ==========")
    keys = ["first_touch_step", "contact_rate", "unique_regions", "freeze_ratio"]
    for k in keys:
        vals = [m[k] for m in all_metrics if m[k] is not None]
        if len(vals) > 0:
            print(f"{k}: {np.mean(vals):.3f}")

    env.close()
    cv2.destroyAllWindows()

    # ---------- COLLECT METRICS ----------
    first_touch_vals = [m["first_touch_step"] for m in all_metrics
                        if m["first_touch_step"] is not None]
    contact_rates = [m["contact_rate"] for m in all_metrics]
    unique_regions_vals = [m["unique_regions"] for m in all_metrics]
    freeze_ratios = [m["freeze_ratio"] for m in all_metrics]


    # ---------- PRINT SUMMARY ----------
    print("\n========== SUMMARY (mean over episodes) ==========")
    print(f"first_touch_step: {np.mean(first_touch_vals):.3f}")
    print(f"contact_rate: {np.mean(contact_rates):.3f}")
    print(f"unique_regions: {np.mean(unique_regions_vals):.3f}")
    print(f"freeze_ratio: {np.mean(freeze_ratios):.3f}")

    suffix = ''

    if "ddpg" in args.model:
        suffix = "ddpg"
    elif "dtk" in args.model:
        suffix = "dtk"
    elif "dtp" in args.model:
        suffix = "dtp"
    elif "dtw" in args.model:
        suffix = "dtw"


    np.save(f"results/all_metrics/all_metrics_{suffix}.npy", np.array(all_metrics))



if __name__ == "__main__":
    main()
