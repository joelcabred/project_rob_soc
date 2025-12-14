# feedback/audio_speech.py
from .base import FeedbackBackend

import time
import threading
import queue
import numpy as np
import sounddevice as sd
from scipy.signal import resample_poly

import whisper
from transformers import pipeline


class SpeechFeedback(FeedbackBackend):
    def __init__(self,
        device=5,
        input_sr=44100,
        whisper_sr=16000,
        chunk_sec=0.5,
        record_sec=3.0,
        rms_threshold=0.02,
        cooldown_sec=2.0,
        print_debug=False,
    ):
        self.print_debug = print_debug

        # Audio params
        self.input_sr = input_sr
        self.whisper_sr = whisper_sr
        self.chunk_size = int(input_sr * chunk_sec)
        self.record_sec = record_sec
        self.rms_threshold = rms_threshold
        self.cooldown_sec = cooldown_sec

        # Threading / state
        self._audio_q = queue.Queue()
        self._event_q = queue.Queue()
        self._stop = threading.Event()
        self._last_trigger_t = 0.0

        self.whisper_model = whisper.load_model("base")
        self.sentiment = pipeline("sentiment-analysis")

        if self.print_debug:
            print("[SPEECH] Whisper + sentiment loaded")

        sd.default.samplerate = self.input_sr
        sd.default.channels = 1
        sd.default.dtype = "float32"

        self._stream = sd.InputStream(
            blocksize=self.chunk_size,
            device=device,
            callback=self._audio_callback,
        )
        self._stream.start()

        self._thread = threading.Thread(
            target=self._worker_loop, daemon=True
        )
        self._thread.start()

        print('USING SPEECH FEEDBACK')

    def poll(self):
        latest = None
        try:
            while True:
                latest = self._event_q.get_nowait()
        except queue.Empty:
            pass

        if latest is None:
            return None

        if self.print_debug:
            print(f"[SPEECH] fb={latest['feedback']} text='{latest['text']}'")

        return latest["feedback"]

    def reset(self):
        pass

    def close(self):
        self._stop.set()
        try:
            self._stream.stop()
            self._stream.close()
        except Exception:
            pass

    def _audio_callback(self, indata, frames, time_info, status):
        if status and self.print_debug:
            print(status)
        self._audio_q.put(indata[:, 0].copy())

    @staticmethod
    def _rms(x):
        return float(np.sqrt(np.mean(x * x) + 1e-12))

    def _analyze_text(self, text):
        """
        Simple heuristic + sentiment.
        """
        if not text:
            return 0

        words = (
            text.lower()
            .replace(",", "")
            .replace(".", "")
            .split()
        )

        positive = {"yes", "yeah", "yep", "yup", "good", "nice"}
        negative = {"no", "nope", "nah", "bad", "stop"}

        if any(w in positive for w in words):
            return +1
        if any(w in negative for w in words):
            return -1

        filler = {"hmm", "mm", "uh", "ah"}
        if words and all(w in filler for w in words):
            return 0

        res = self.sentiment(text)[0]["label"]
        return +1 if res == "POSITIVE" else -1

    def _worker_loop(self):
        buffer = []
        last_emit = 0.0

        while not self._stop.is_set():
            try:
                block = self._audio_q.get(timeout=0.1)
            except queue.Empty:
                continue

            buffer.append(block)

            audio = np.concatenate(buffer)
            rms = self._rms(audio)

            if rms < self.rms_threshold:
                continue

            now = time.time()
            if now - last_emit < self.cooldown_sec:
                continue

            last_emit = now
            buffer.clear()

            # take last N seconds
            max_len = int(self.input_sr * self.record_sec)
            audio = audio[-max_len:]

            # resample for Whisper
            audio_16k = resample_poly(
                audio, self.whisper_sr, self.input_sr
            )

            result = self.whisper_model.transcribe(
                audio_16k,
                fp16=False,
                language="en",
            )

            text = result["text"].strip()
            fb = self._analyze_text(text)

            self._event_q.put({
                "feedback": fb,
                "text": text,
                "ts": now,
            })
