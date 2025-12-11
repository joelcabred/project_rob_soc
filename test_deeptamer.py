"""
Evaluation script for testing a TAMER-enhanced DDPG agent's self-touch behavior in MIMo.
Loads a pre-trained actor network from a human-feedback-trained agent and evaluates its performance
using metrics like time to first touch, body region coverage, and movement patterns. Demonstrates
the behavioral outcomes of combining reinforcement learning with human guidance.
"""
import numpy as np
import gymnasium as gym
import yaml
import torch
import cv2
import sys
from collections import defaultdict

sys.path.append(".")
sys.path.append("..")

import babybench.utils as bb_utils
from ddpg_tamer import DDPGAgent
from TactileObsWrapper import TactileObsWrapper


def scale_action(raw_action, env):
    low  = env.action_space.low
    high = env.action_space.high
    return low + 0.5 * (raw_action + 1.0) * (high - low)


def detect_region(model, touch, left_ids, right_ids, body_ids):
    contacts = touch.get_contacts()
    contacts_clean = []

    left_contact_ids = set()
    right_contact_ids = set()

    for c_id, b_id, f in contacts:
        if b_id in left_ids:
            left_contact_ids.add(c_id)
        elif b_id in right_ids:
            right_contact_ids.add(c_id)
        else:
            contacts_clean.append([c_id,b_id,f])

    touched = set()

    for c_id, b_id, _ in contacts_clean:
        if b_id in body_ids:
            #9 right_lower_arm
            #13 left_lower_arm
            if c_id in left_contact_ids and b_id != 13:
                touched.add(b_id)

            if c_id in right_contact_ids and b_id != 9:
                touched.add(b_id)



    return touched


# ---------- HAND SPEED (freeze) ----------
def extract_hand_speed(obs):
    qvel = obs[64:64+12]   # QVEL_START : QVEL_START+N_Q
    return np.linalg.norm(qvel)


def main():
    with open("examples/config_selftouch.yml") as f:
        config = yaml.safe_load(f)

    env = bb_utils.make_env(config)
    env.reset()
    env = TactileObsWrapper(env)
    env = gym.wrappers.NormalizeObservation(env)
    env = gym.wrappers.TransformObservation(env,
                                             lambda obs: np.clip(obs, -10, 10))

    input_dim = env.observation_space.shape[0]
    output_dim = env.action_space.shape[0]

    agent = DDPGAgent(input_dim, 
                      output_dim,
                      use_tamer = True,
                      alpha = 0.3,
                      beta = 0.7)
    
    agent.actor.load_state_dict(torch.load("actor_selftouch.pt", map_location="cpu"))
    agent.actor.eval()

    base = env.unwrapped
    renderer = env.rendered

    episodes = 10
    max_steps = 1000

    all_metrics = []

    left_ids = [env.LEFT_HAND_ID,env.LEFT_FINGERS_ID]
    right_ids = [env.RIGHT_HAND_ID,env.RIGHT_FINGERS_ID]

    for episode in range(episodes):
        state, _ = env.reset()
        done = False
        step = 0

        regions = []

        first_touch_step = None
        freeze_steps = 0

        regions = set()


        while not done and step < max_steps:
            step += 1

            with torch.no_grad():
                state_t = torch.FloatTensor(state).unsqueeze(0)
                raw_action = agent.actor(state_t).numpy()[0]

            action = scale_action(raw_action, env)
            action = np.clip(action,
                             env.action_space.low,
                             env.action_space.high)

            next_state, _, terminated, truncated, _ = env.step(action)
            done = terminated or truncated

            # -------- REGION ----------
            region = detect_region(env.unwrapped.model,
                                   env.unwrapped.touch,
                                   left_ids,right_ids,env.BODY_IDS)

            regions = regions.union(region)

            if len(regions) > 0:
                if first_touch_step is None:
                    first_touch_step = step


            # -------- FREEZE ----------
            hand_speed = extract_hand_speed(next_state)
            print(hand_speed)
            if hand_speed < 1.0:
                freeze_steps += 1

            state = next_state

            # -------- RENDER ----------
            renderer.update_scene(base.data)
            img = renderer.render()
            img_bgr = cv2.cvtColor(img, cv2.COLOR_RGB2BGR)

            cv2.namedWindow("Test", cv2.WINDOW_NORMAL)
            cv2.resizeWindow("Test", 720, 720)
            cv2.imshow("Test", img_bgr)
            cv2.waitKey(1)

        unique_regions = len(set(regions))
        freeze_ratio = freeze_steps / step

        metrics = {
            "episode": episode + 1,
            "first_touch_step": first_touch_step,
            "unique_regions": unique_regions,
            "freeze_ratio": freeze_ratio
        }

        all_metrics.append(metrics)

        print("\n[TEST METRICS]")
        for k, v in metrics.items():
            print(f"{k}: {v}")

    env.close()
    cv2.destroyAllWindows()

    # ---------- METRICS SUMMARY ----------
    print("\n========== SUMMARY ==========")
    for key in all_metrics[0].keys():
        if key == "episode":
            continue
        vals = [m[key] for m in all_metrics if m[key] is not None]
        if len(vals) > 0:
            print(f"{key}: {np.mean(vals):.3f}")


if __name__ == "__main__":
    main()
