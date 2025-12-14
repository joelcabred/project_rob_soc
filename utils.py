import numpy as np

def scale_action(raw_action, env):
    low, high = env.action_space.low, env.action_space.high
    return low + 0.5 * (raw_action + 1.0) * (high - low)


def extract_hand_speed(obs):
    qvel = obs[64:64 + 12]
    return np.linalg.norm(qvel)


def detect_regions(env):
    """
    Returns a set of body IDs touched at the current step.
    """
    touch = env.unwrapped.touch
    model = env.unwrapped.model

    left_ids = [env.LEFT_HAND_ID, env.LEFT_FINGERS_ID]
    right_ids = [env.RIGHT_HAND_ID, env.RIGHT_FINGERS_ID]
    body_ids = env.BODY_IDS

    contacts = touch.get_contacts()
    contacts_clean = []

    left_contact_ids = set()
    right_contact_ids = set()

    for c_id, b_id, _ in contacts:
        if b_id in left_ids:
            left_contact_ids.add(c_id)
        elif b_id in right_ids:
            right_contact_ids.add(c_id)
        else:
            contacts_clean.append((c_id, b_id))

    touched = set()
    for c_id, b_id in contacts_clean:
        if b_id in body_ids:
            # avoid counting self-arm contacts
            if c_id in left_contact_ids and b_id != 13:
                touched.add(b_id)
            if c_id in right_contact_ids and b_id != 9:
                touched.add(b_id)

    return touched

