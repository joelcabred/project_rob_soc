import sounddevice as sd
import numpy as np
import torch
import whisper
from transformers import pipeline
import threading
import tkinter as tk
from tkinter import ttk, messagebox, font
from datetime import datetime

# Settings
SAMPLE_RATE = 16000
RECORD_SECONDS = 3
LOG_FILE = "feedback_log.txt"  # File to store logs

# Load Models
whisper_model = whisper.load_model("base")
sentiment_analyzer = pipeline("sentiment-analysis")

# Helper Functions 
def record_audio(duration=RECORD_SECONDS, sr=SAMPLE_RATE):
    """Record audio from the microphone and return as a 1D numpy array."""
    print("Recording...")
    audio = sd.rec(int(duration * sr), samplerate=sr, channels=1, dtype='float32')
    sd.wait()
    return audio.flatten()

def transcribe_audio(audio):
    """Transcribe audio to text using Whisper."""
    print("Predicting...")
    result = whisper_model.transcribe(audio, fp16=False)
    return result.get("text", "").strip()

def analyze_sentiment(text):
    """Analyze text sentiment, filtering out filler words."""
    if not text:
        return "NEUTRAL"

    filler_words = ["hmm", "mm", "uh", "uh-huh", "huh", "ah", "okay", "hmmm"]
    words = text.lower().replace(",", "").replace(".", "").strip().split()

    if all(w in filler_words for w in words):
        return "NEUTRAL"

    res = sentiment_analyzer(text)[0]
    return res["label"].upper()  # POSITIVE or NEGATIVE

def log_feedback(text, sentiment):
    """Append transcription and sentiment to the log file with timestamp."""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    with open(LOG_FILE, "a", encoding="utf-8") as f:
        f.write(f"{timestamp} | Sentiment: {sentiment} ")

def on_button_click():
    """Record audio, transcribe it, analyze sentiment, log it, and show result."""
    audio = record_audio()
    text = transcribe_audio(audio)
    sentiment = analyze_sentiment(text)
    log_feedback(text, sentiment)
    messagebox.showinfo("Detected Sentiment", f"Sentiment: {sentiment}")

    # send feedback to RL agent
    """to be integrated with 
      RL agent"""

# --- Build GUI ---
root = tk.Tk()
root.title("Live Feedback App")
root.geometry("450x250")
root.resizable(False, False)

# Custom style for a bigger, nicer button
style = ttk.Style()
style.configure("TButton", font=("Helvetica", 20, "bold"), padding=20)

btn = ttk.Button(
    root,
    text="🎤 Give Feedback 🎤",
    command=lambda: threading.Thread(target=on_button_click).start()
)
btn.pack(expand=True, padx=50, pady=50)

root.mainloop()



class HumanFeedbackManager:
    def __init__(self, agent, window=5):
        self.agent = agent
        self.window = window  # last N actions to assign reward to

    def apply_sentiment(self, sentiment):
        if sentiment == "POSITIVE":
            r = +1.0
        elif sentiment == "NEGATIVE":
            r = -1.0
        else:  # NEUTRAL
            r = 0.0

        # send reward to agent (DeepCoach style)
        self.agent.apply_human_feedback(r)
