import sounddevice as sd
import whisper
from transformers import pipeline
import threading
import tkinter as tk
from tkinter import ttk

SAMPLE_RATE = 16000
RECORD_SECONDS = 3

whisper_model = whisper.load_model("base")
sentiment_analyzer = pipeline("sentiment-analysis")

class HumanFeedbackManager:
    def __init__(self, agent=None):
        self.agent = agent

    def apply_sentiment(self, sentiment):
        if sentiment == "POSITIVE": value = +1
        elif sentiment == "NEGATIVE": value = -1
        else: value = 0
        if self.agent: self.agent.apply_human_feedback(value)
        return value

class HumanFeedbackCapture:
    def __init__(self, agent=None):
        self.manager = HumanFeedbackManager(agent)
        self.last_feedback_value = 0

        self.root = tk.Tk()
        self.root.title("Live Feedback App")
        self.root.geometry("450x250")
        style = ttk.Style()
        style.configure("TButton", font=("Helvetica", 20, "bold"), padding=20)

        self.btn = ttk.Button(self.root, text="🎤 Give Feedback 🎤",
                              command=lambda: threading.Thread(target=self._on_button_click).start())
        self.btn.pack(expand=True, padx=50, pady=50)

    def _on_button_click(self):
        audio = sd.rec(int(RECORD_SECONDS*SAMPLE_RATE), samplerate=SAMPLE_RATE, channels=1, dtype='float32')
        sd.wait()
        text = whisper_model.transcribe(audio, fp16=False).get("text","").strip()
        sentiment = self._analyze_sentiment(text)
        self.last_feedback_value = self.manager.apply_sentiment(sentiment)

    def _analyze_sentiment(self, text):
        if not text: return "NEUTRAL"
        filler_words = ["hmm","mm","uh","uh-huh","huh","ah","okay","hmmm"]
        if all(w in filler_words for w in text.lower().split()): return "NEUTRAL"
        return sentiment_analyzer(text)[0]["label"].upper()

    def get_feedback(self): return self.last_feedback_value
    def start_gui(self): self.root.mainloop()
