import numpy as np

def split_obs(env, obs, k=10):
    """
    Splits obs into:
      left, right, qpos, qvel, torque
    using the convention of TactileObsWrapper.get_obs_vector().
    """
    left_dim = right_dim = 2 * (3 + k)
    n_q = len(env.ids_joints_q)
    n_tau = len(env.ids_joints_torques)

    left   = obs[0:left_dim]
    right  = obs[left_dim:left_dim + right_dim]

    qpos_start = left_dim + right_dim
    qvel_start = qpos_start + n_q
    torque_start = qvel_start + n_q

    qpos   = obs[qpos_start:qpos_start + n_q]
    qvel   = obs[qvel_start:qvel_start + n_q]
    torque = obs[torque_start:torque_start + n_tau]

    return left, right, qpos, qvel, torque


def extract_forces_from_arm_block(arm_vec, k=10):
    """
    arm_vec: 'left' or 'right' block = (3 + k) hand + (3 + k) fingers
    """
    hand_forces    = arm_vec[3:3 + k]
    fingers_forces = arm_vec[3 + k + 3 : 3 + k + 3 + k]
    return np.concatenate([hand_forces, fingers_forces])


def extract_touch_features(env, obs, k=10):
    left, right, _, _, _ = split_obs(env, obs, k=k)
    left_forces  = extract_forces_from_arm_block(left, k)
    right_forces = extract_forces_from_arm_block(right, k)
    return np.concatenate([left_forces, right_forces])


def extract_torques(env, obs, k=10):
    _, _, _, _, torque = split_obs(env, obs, k=k)
    return torque


def extract_hand_speed(env, obs, k=10):
    """
    Average velocity of the 'arms/hands', using ALL the qvel values of the arm joints.
    """
    _, _, qpos, qvel, _ = split_obs(env, obs, k=k)
    names = env.NAMES_JOINTS_Q   # orden de construcción de qpos/qvel

    right_idxs = [i for i, name in enumerate(names) if "right_" in name]
    left_idxs  = [i for i, name in enumerate(names) if "left_"  in name]

    right_speed = np.linalg.norm(qvel[right_idxs]) if right_idxs else 0.0
    left_speed  = np.linalg.norm(qvel[left_idxs])  if left_idxs  else 0.0

    if right_idxs and left_idxs:
        return 0.5 * (right_speed + left_speed)
    elif right_idxs:
        return right_speed
    elif left_idxs:
        return left_speed
    else:
        return 0.0


def detect_region(touch, left_ids, right_ids, body_ids):
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


def joint_limit_penalty(action, env):
    low  = env.action_space.low
    high = env.action_space.high

    mid = 0.5 * (low + high)
    span = 0.5 * (high - low) + 1e-6

    norm_dist = np.abs((action - mid) / span)  
    penalty = np.sum(np.clip(norm_dist - 0.8, 0.0, 1.0)) 

    return penalty


def compute_reward_full(obs, prev_obs, env, prev_region, k=10):
    touch = env.unwrapped.touch
    model = env.unwrapped.model

    # 1. Contact reward
    touch_curr = extract_touch_features(env, obs, k=k)
    touch_raw = np.sum(np.abs(touch_curr))
    touch_reward = np.tanh(0.05 * touch_raw)   # [0,1]

    # 2. Novelty reward (touch far from last touch)
    novelty = 0.0
    if prev_obs is not None:
        touch_prev = extract_touch_features(env, prev_obs, k=k)
        novelty = np.linalg.norm(touch_curr - touch_prev)

    # 3. Region (touch different parts of the body)
    left_ids = [env.LEFT_HAND_ID,env.LEFT_FINGERS_ID]
    right_ids = [env.RIGHT_HAND_ID,env.RIGHT_FINGERS_ID]
    region = detect_region(touch, left_ids,right_ids,env.BODY_IDS)

    same_region_penalty = -1.0 if region <= prev_region else +0.3


    # 5. Penalty for not moving the hand
    hand_speed = extract_hand_speed(env, obs, k=k)
    #print('--',hand_speed)
    still_penalty = -0.5 if hand_speed < 1.0 else 0.0

    # 6. Penalty energy
    torque = extract_torques(env, obs, k=k)
    energy_penalty = 0.001 * np.sum(torque**2)

    # 7. articular limits
    lim_pen = joint_limit_penalty(env.last_action, env)

    # 8. Speed penalty
    speed_penalty = -0.5 if hand_speed > 3.0 else 0.0


    reward = (
        4.0 * touch_reward
      + 1.0 * novelty
      + same_region_penalty
      + still_penalty
      - energy_penalty
      - 0.8 * lim_pen
      + speed_penalty
    )

    #reward = np.clip(reward, -5.0, 5.0)

    return float(reward), region



