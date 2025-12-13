# Human Feedback Capture System

## Overview

This project is designed to capture real-time audio, transcribe it to text using OpenAI's **Whisper** model, and analyze the sentiment of the resulting text using the **DistilBERT** sentiment analysis model from HuggingFace. It incorporates **Voice Activity Detection (VAD)** to only process relevant audio and provide real-time feedback based on sentiment analysis.

This system is ideal for applications that require quick human feedback, such as virtual assistants, interactive AI systems, or feedback-driven applications.

## Features

- **Real-time Audio Capture**: Continuously records audio and processes it in chunks.
- **Speech-to-Text Transcription**: Converts audio to text using the Whisper ASR model.
- **Sentiment Analysis**: Analyzes the sentiment of the transcribed text using the DistilBERT model.
- **Voice Activity Detection (VAD)**: Detects speech and reduces unnecessary processing during silent or non-relevant audio.
- **Real-time Feedback**: Provides feedback based on sentiment, categorizing it as positive or negative.

## Depedencies
```bash
pip install sounddevice numpy openai-whisper transformers
```

## Adjusting Settings

You can modify the following parameters in the script to customize behavior:

- `SAMPLE_RATE`: The sample rate of the audio capture.
- `RECORD_SECONDS`: The duration for each audio capture.
- `CHUNK_DURATION`: The duration for each audio chunk.
- `RMS_THRESHOLD`: The RMS threshold for detecting speech.
- `COOLDOWN_SECONDS`: The cooldown time before accepting new feedback.

## How It Works

### Audio Capture and VAD
The system continuously records audio in small chunks. **Voice Activity Detection (VAD)** is used to filter out silent or irrelevant audio by measuring the RMS (Root Mean Square) value of each chunk.

### Speech-to-Text
Once speech is detected, the recorded audio is passed through the **Whisper** model to transcribe it into text. Whisper is robust to background noise, accents, and different languages, making it ideal for diverse environments.

### Sentiment Analysis
The transcribed text is then passed to **DistilBERT**, a lightweight version of the BERT model, for sentiment classification. It determines whether the text has a positive or negative sentiment.

### Real-Time Feedback
The feedback is generated based on the sentiment analysis, with positive feedback leading to a different action compared to negative feedback. This feedback can be used in applications such as conversational AI or emotion-aware systems.
