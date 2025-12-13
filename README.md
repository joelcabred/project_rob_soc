Human Feedback Capture System
Overview

This project captures real-time audio, transcribes speech to text using OpenAI's Whisper model, and analyzes the sentiment of the resulting transcript using the DistilBERT sentiment analysis model from HuggingFace. The system uses Voice Activity Detection (VAD) to ensure only relevant audio is processed and provides feedback based on the analyzed sentiment.

Requirements

Python 3.8+

Dependencies:

pip install sounddevice numpy openai-whisper transformers

Setup and Usage

Install dependencies:
Make sure to install all required libraries by running:

pip install -r requirements.txt


Running the feedback capture:
To start the feedback capture process, simply run the script:

python feedback_capture.py


Stopping the listener:
To stop the real-time feedback listener, press CTRL+C or call feedback_listener.stop().

Adjusting settings:
You can configure the following parameters in the script:

SAMPLE_RATE: The sample rate for audio recording.

RECORD_SECONDS: Duration of each audio capture.

CHUNK_DURATION: Duration of each audio chunk processed.

RMS_THRESHOLD: The threshold to trigger voice detection.

COOLDOWN_SECONDS: Time to wait before accepting new feedback.

How It Works

Audio Recording:
The system captures audio in small chunks, ensuring that only relevant speech is processed.

Voice Activity Detection:
A simple RMS threshold is used to detect whether speech is present in the audio chunk. If speech is detected, the system proceeds to transcribe the audio.

Speech Transcription:
The Whisper model is used to transcribe the captured audio into text. This model is robust to various accents, languages, and noisy environments.

Sentiment Analysis:
The resulting text is analyzed for sentiment using the DistilBERT model, which determines if the sentiment is positive or negative based on the content of the speech.

Real-Time Feedback:
The system provides real-time feedback based on the sentiment analysis, which can be accessed using the get_feedback() method.
