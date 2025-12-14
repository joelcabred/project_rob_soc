import numpy as np
from audio_feedback_valence import feedback_from_mic, TAU

def get_reward():
    res = feedback_from_mic()
    if res is None:
        return None
    fb, v, conf, _ = res
    r = float(np.tanh(v / (TAU + 1e-9)))
    return r

def collect(label, n=10):
    rewards = []
    print(f"\n=== {label} x{n} ===")
    i = 0
    while i < n:
        input("Entrée pour enregistrer > ")
        r = get_reward()
        if r is None:
            continue
        rewards.append(r)
        print(f"reward={r:+.3f}")
        i += 1
    return np.array(rewards, dtype=np.float32)

def main():
    pos = collect("POS", n=10)
    neg = collect("NEG", n=10)

    # signe correct ?
    acc_pos = float((pos > 0).mean())
    acc_neg = float((neg < 0).mean())
    acc = float(((pos > 0).sum() + (neg < 0).sum()) / (len(pos) + len(neg)))

    print("\n==================== RESULTS ====================")
    print(f"TAU = {TAU:.6f}")
    print(f"POS: mean={pos.mean():+.3f} std={pos.std():.3f}  sign_acc={acc_pos:.2f}")
    print(f"NEG: mean={neg.mean():+.3f} std={neg.std():.3f}  sign_acc={acc_neg:.2f}")
    print(f"Overall sign accuracy: {acc:.2f}")
    print("=================================================\n")

if __name__ == "__main__":
    main()
