import numpy as np
import sounddevice as sd
import librosa
import joblib
from transformers import pipeline

# =========================
# CONFIG
# =========================
SR = 16000
DURATION = 3.5          # secondes d'enregistrement
TOP_DB = 25             # pour trim silence
MODEL_ID = "superb/wav2vec2-base-superb-er"

N_PER_CLASS = 15        # nombre d'exemples POS et NEG à collecter
MIN_SPEECH_SEC = 0.6    # durée minimale de parole (après trim)
OUT_PATH = "valence_calib_posneg.joblib"

clf = pipeline("audio-classification", model=MODEL_ID)

# Labels réels vus chez toi: ang, hap, neu, sad
CLASS_ORDER = ["ang", "hap", "neu", "sad"]


def record():
    print(f"🎙️ Parle ({DURATION:.1f}s)...")
    a = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    y = a[:, 0]

    y, _ = librosa.effects.trim(y, top_db=TOP_DB)

    if len(y) < int(MIN_SPEECH_SEC * SR):
        print("Trop court / trop de silence. Recommence plus fort/longtemps.")
        return None

    y = y / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)


def probs_from_audio(y):
    out = clf({"array": y, "sampling_rate": SR}, top_k=None)
    d = {o["label"].lower(): float(o["score"]) for o in out}
    p = np.array([d.get(lbl, 0.0) for lbl in CLASS_ORDER], dtype=np.float32)
    return p  # [ang, hap, neu, sad]


def valence_from_probs(p):
    # valence = happy - angry
    return float(p[1] - p[0])  # hap - ang


def collect():
    samples = []  # list of dicts: {"label": "POS"/"NEG", "valence": v, "probs": p}
    for target in ["POS", "NEG"]:
        print(f"\n=== {target} ({N_PER_CLASS} exemples) ===")
        i = 0
        while i < N_PER_CLASS:
            input("Entrée pour enregistrer > ")
            y = record()
            if y is None:
                continue
            p = probs_from_audio(y)
            v = valence_from_probs(p)
            print(f"probs [ang,hap,neu,sad]={p} | valence(hap-ang)={v:+.3f}")
            samples.append({"label": target, "valence": v, "probs": p})
            i += 1
    return samples


def choose_threshold(samples):
    """
    Choix robuste de tau pour définir une "zone de confiance" :
    - plus |v| est grand, plus on est confiant.
    Ici on prend tau = médiane des |v| (tu peux ajuster).
    """
    abs_v = np.array([abs(s["valence"]) for s in samples], dtype=np.float32)
    tau = float(np.median(abs_v))
    return tau


def evaluate(samples):
    """
    Évalue un classifieur ultra-simple:
      pred = POS si v>=0 sinon NEG
    et retourne accuracy.
    """
    y_true = np.array([s["label"] for s in samples])
    v = np.array([s["valence"] for s in samples], dtype=np.float32)
    y_pred = np.where(v >= 0.0, "POS", "NEG")
    acc = float((y_pred == y_true).mean())
    return acc


def main():
    samples = collect()

    tau = choose_threshold(samples)
    acc = evaluate(samples)

    print("\n==============================")
    print("Calibration terminée")
    print(f"Accuracy (sur tes samples, règle v>=0): {acc:.3f}")
    print(f"Tau (médiane |valence|) = {tau:.3f}")
    print("==============================\n")

    bundle = {
        "model_id": MODEL_ID,
        "sr": SR,
        "duration": DURATION,
        "top_db": TOP_DB,
        "min_speech_sec": MIN_SPEECH_SEC,
        "class_order": CLASS_ORDER,
        "tau": tau,
        "note": "valence = p(hap) - p(ang), pred = sign(valence)",
    }
    joblib.dump(bundle, OUT_PATH)
    print(f"[OK] Sauvegardé: {OUT_PATH}")


if __name__ == "__main__":
    main()
