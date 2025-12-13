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

## Requirements

Make sure you have **Python 3.8+** installed. The following dependencies are required:

```bash
pip install sounddevice numpy openai-whisper transformers
