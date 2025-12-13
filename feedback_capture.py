import sounddevice as sd
import numpy as np
import whisper
from transformers import pipeline
import threading
import time

# SETTINGS
SAMPLE_RATE = 16000
RECORD_SECONDS = 3
CHUNK_DURATION = 0.5
RMS_THRESHOLD = 0.02
COOLDOWN_SECONDS = 2

# MODELS
whisper_model = whisper.load_model("base")
sentiment_analyzer = pipeline("sentiment-analysis")

# HELPERS
def record_audio(duration=RECORD_SECONDS):
    audio = sd.rec(
        int(duration * SAMPLE_RATE),
        samplerate=SAMPLE_RATE,
        channels=1,
        dtype="float32"
    )
    sd.wait()
    return audio.flatten()


def compute_rms(audio):
    return np.sqrt(np.mean(np.square(audio)))


def analyze_sentiment(text):
    if not text:
        return 0

    text_l = text.lower()
    words = text_l.replace(",", "").replace(".", "").split()

    positive = {"yes", "yeah", "yep", "yup"}
    negative = {"no", "nope", "nah"}

    if any(w in positive for w in words):
        return 1
    if any(w in negative for w in words):
        return -1

    filler = {"hmm", "mm", "uh", "ah", "hmmm"}
    if words and all(w in filler for w in words):
        return 0

    res = sentiment_analyzer(text)[0]["label"]
    return 1 if res == "POSITIVE" else -1



class HumanFeedbackCapture:
    def __init__(self):
        self.latest_feedback = 0.0
        self.lock = threading.Lock()
        self.running = True

    def start(self):
        threading.Thread(target=self._listen_loop, daemon=True).start()

    def _listen_loop(self):
        last_trigger = 0
        print(" Voice feedback listener started")

        while self.running:
            chunk = sd.rec(
                int(CHUNK_DURATION * SAMPLE_RATE),
                samplerate=SAMPLE_RATE,
                channels=1,
                dtype="float32"
            )
            sd.wait()
            chunk = chunk.flatten()

            if compute_rms(chunk) > RMS_THRESHOLD:
                now = time.time()
                if now - last_trigger < COOLDOWN_SECONDS:
                    continue

                last_trigger = now
                audio = record_audio()
                text = whisper_model.transcribe(audio, fp16=False)["text"]
                reward = analyze_sentiment(text)

                with self.lock:
                    self.latest_feedback = reward

                

    def get_feedback(self):
        with self.lock:
            r = self.latest_feedback
            self.latest_feedback = 0.0
        return r

    def stop(self):
        self.running = False
