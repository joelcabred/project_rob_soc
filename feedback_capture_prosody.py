"""
HumanFeedbackCapture (audio/prosody)

Drop-in replacement for keyboard-based feedback_capture.py.

- Micro toujours actif (InputStream)
- VAD RMS simple
- SUPERB wav2vec2 emotion model
- Valence = hap - ang
- Feedback discret (+1 / -1) ou reward continu
"""

import time
import queue
import threading
import numpy as np
import sounddevice as sd
import librosa
import joblib
import torch
from transformers import AutoFeatureExtractor, AutoModelForAudioClassification


class HumanFeedbackCapture:
    def __init__(
        self,
        calib_path="valence_calib_posneg.joblib",
        device=None,
        channels=1,
        chunk_sec=0.25,
        vad_rms_threshold=0.012,
        silence_hang_sec=0.35,
        min_speech_sec=0.8,
        max_speech_sec=3.5,
        trim_top_db=25,
        use_continuous_reward=True,
        discrete_deadzone=0.25,
        print_debug=True,
    ):
        # feedback bookkeeping (API compatible)
        self.current_feedback = None
        self.feedback_history = []
        self.last_feedback_step = -999

        # ---- load calibration ----
        calib = joblib.load(calib_path)
        self.MODEL_ID = calib["model_id"]
        self.SR = int(calib["sr"])
        self.TAU = float(calib["tau"])
        self.CLASS_ORDER = calib["class_order"]  # ["ang","hap","neu","sad"]

        # ---- audio params ----
        self.device_name = device
        self.channels = channels
        self.chunk_size = int(self.SR * chunk_sec)
        self.vad_rms_threshold = vad_rms_threshold
        self.silence_hang_sec = silence_hang_sec
        self.min_speech_sec = min_speech_sec
        self.max_speech_sec = max_speech_sec
        self.trim_top_db = trim_top_db

        # ---- feedback shaping ----
        self.use_continuous_reward = use_continuous_reward
        self.discrete_deadzone = discrete_deadzone
        self.print_debug = print_debug

        # ---- queues / thread ----
        self._audio_q = queue.Queue()
        self._event_q = queue.Queue()
        self._stop = threading.Event()

        # ---- load HF model (NO pipeline, NO torchvision) ----
        self.torch_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.extractor = AutoFeatureExtractor.from_pretrained(self.MODEL_ID)
        self.model = AutoModelForAudioClassification.from_pretrained(self.MODEL_ID)
        self.model.to(self.torch_device)
        self.model.eval()

        if self.print_debug:
            print(f"[AUDIO] Loaded {self.MODEL_ID} on {self.torch_device}")

        # ---- start microphone ----
        self._stream = sd.InputStream(
            samplerate=self.SR,
            channels=self.channels,
            blocksize=self.chunk_size,
            dtype="float32",
            device=self.device_name,
            callback=self._audio_callback,
        )
        self._stream.start()

        self._thread = threading.Thread(target=self._worker_loop, daemon=True)
        self._thread.start()

    # ======================================================
    # Public API 
    # ======================================================
    def reset(self):
        self.last_feedback_step = -999

    def capture_feedback(self, current_step, min_interval=15):
        if current_step - self.last_feedback_step < min_interval:
            return None

        latest = None
        try:
            while True:
                latest = self._event_q.get_nowait()
        except queue.Empty:
            pass

        if latest is None:
            return None

        fb = latest["feedback_int"]
        if fb is not None:
            self.last_feedback_step = current_step
            self.current_feedback = fb
            self.feedback_history.append(
                {"step": current_step, "feedback": fb}
            )

            if self.print_debug:
                print(
                    f"[AUDIO FEEDBACK] step={current_step} "
                    f"fb={fb:+d} reward={latest['reward_cont']:+.3f} "
                    f"valence={latest['valence']:+.4f} conf={latest['conf']:.3f}"
                )

        return fb

    def get_stats(self):
        if not self.feedback_history:
            return {'total': 0, 'positive': 0, 'negative': 0, 'neutral': 0}

        positive = sum(f['feedback'] == 1 for f in self.feedback_history)
        negative = sum(f['feedback'] == -1 for f in self.feedback_history)
        neutral  = sum(f['feedback'] == 0 for f in self.feedback_history)

        return {
            'total': len(self.feedback_history),
            'positive': positive,
            'negative': negative,
            'neutral': neutral
        }

    def close(self):
        self._stop.set()
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    # ======================================================
    # Internals
    # ======================================================
    def _audio_callback(self, indata, frames, time_info, status):
        if status and self.print_debug:
            print(status)
        self._audio_q.put(indata[:, 0].copy())

    @staticmethod
    def _rms(x):
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def _probs_from_audio(self, y):
        inputs = self.extractor(
            y,
            sampling_rate=self.SR,
            return_tensors="pt",
            padding=True
        )
        inputs = {k: v.to(self.torch_device) for k, v in inputs.items()}

        with torch.no_grad():
            logits = self.model(**inputs).logits
            probs = torch.softmax(logits, dim=-1).cpu().numpy()[0]

        id2label = self.model.config.id2label
        d = {id2label[i].lower(): float(probs[i]) for i in range(len(probs))}

        return np.array(
            [d.get(lbl, 0.0) for lbl in self.CLASS_ORDER],
            dtype=np.float32
        )

    def _valence(self, p):
        idx_ang = self.CLASS_ORDER.index("ang")
        idx_hap = self.CLASS_ORDER.index("hap")
        return float(p[idx_hap] - p[idx_ang])

    def _reward(self, v):
        return float(np.tanh(v / (self.TAU + 1e-9)))

    def _to_discrete_feedback(self, r):
        if abs(r) < self.discrete_deadzone:
            return None
        return +1 if r >= 0 else -1

    def _worker_loop(self):
        in_speech = False
        speech_buf = []
        last_voice_t = None

        while not self._stop.is_set():
            try:
                block = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            e = self._rms(block)
            now = time.time()
            voice = e >= self.vad_rms_threshold

            if voice:
                if not in_speech:
                    in_speech = True
                    speech_buf = []
                speech_buf.append(block)
                last_voice_t = now
                continue

            if in_speech and (now - last_voice_t) > self.silence_hang_sec:
                y = np.concatenate(speech_buf)
                in_speech = False
                speech_buf = []

                if y.size < int(self.SR * self.min_speech_sec):
                    continue

                y = y[: int(self.SR * self.max_speech_sec)]
                y, _ = librosa.effects.trim(y, top_db=self.trim_top_db)
                y = y / (np.max(np.abs(y)) + 1e-9)

                p = self._probs_from_audio(y.astype(np.float32))
                v = self._valence(p)
                r = self._reward(v)
                conf = float(np.clip(abs(v) / (self.TAU + 1e-9), 0.0, 1.0))
                fb = self._to_discrete_feedback(r)

                self._event_q.put({
                    "feedback_int": fb,
                    "reward_cont": r,
                    "valence": v,
                    "conf": conf,
                    "ts": now
                })
