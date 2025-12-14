import numpy as np
import sounddevice as sd
import librosa
import joblib
from transformers import pipeline

CALIB_PATH = "valence_calib_posneg.joblib"

calib = joblib.load(CALIB_PATH)

MODEL_ID = calib["model_id"]
SR = calib["sr"]
DURATION = calib["duration"]
TOP_DB = calib["top_db"]
MIN_SPEECH_SEC = calib["min_speech_sec"]
CLASS_ORDER = calib["class_order"]
TAU = calib["tau"]

clf = pipeline("audio-classification", model=MODEL_ID)

def record():
    a = sd.rec(int(DURATION * SR), samplerate=SR, channels=1, dtype="float32")
    sd.wait()
    y = a[:, 0]
    y, _ = librosa.effects.trim(y, top_db=TOP_DB)
    if len(y) < int(MIN_SPEECH_SEC * SR):
        # trop court / silence
        return None
    y = y / (np.max(np.abs(y)) + 1e-9)
    return y.astype(np.float32)

def probs_from_audio(y):
    out = clf({"array": y, "sampling_rate": SR}, top_k=None)
    d = {o["label"].lower(): float(o["score"]) for o in out}
    p = np.array([d.get(lbl, 0.0) for lbl in CLASS_ORDER], dtype=np.float32)
    return p  # [ang, hap, neu, sad]

def feedback_from_mic(TAU=TAU):
    """
    Retourne:
      feedback: "POS" ou "NEG"
      valence: v = p(hap) - p(ang)
      confidence: score [0..1] basé sur |v| et tau
      probs: vecteur [ang,hap,neu,sad]
    """
    print(f"SPEAK ({DURATION:.1f}s)...")
    y = record()
    if y is None:
        print("⚠️ Trop court/silence. Recommence.")
        return None

    p = probs_from_audio(y)
    v = float(p[1] - p[0])  # hap - ang

    fb = "POS" if v >= 0.0 else "NEG"

    # confiance: 0 si |v|=0, ~1 si |v| >> tau
    conf = float(np.clip(abs(v) / (TAU + 1e-9), 0.0, 1.0))

    return fb, v, conf, p

if __name__ == "__main__":
    while True:
        s = input("Entrée=test / q=quit > ").strip().lower()
        if s == "q":
            break
        res = feedback_from_mic()
        if res is None:
            continue
        fb, v, conf, p = res
        print(f"➡️ {fb} | valence={v:+.3f} | conf={conf:.3f} | [ang,hap,neu,sad]={p}")
        print("-" * 60)
