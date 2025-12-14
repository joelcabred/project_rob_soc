## ProTAMER: Prosody-Based Human Feedback for Learning Self-Touch Behaviors in Social Robots

This project trains and evaluates DDPG agents on the BabyBench self-touch task,
with optional human feedback (DeepTAMER).

GIT: https://github.com/joelcabred/project_rob_soc
---

## 1. Environment setup

The code was developed and tested using Conda (Python 3.10).

Create and activate the environment:

    conda env create -f environment.yml
    conda activate babybench-gpu

All the following steps must be executed **inside this environment**.

---

## 2. Install BabyBench

Clone the BabyBench starter kit:

    git clone https://github.com/babybench/BabyBench2025_Starter_Kit.git
    cd BabyBench2025_Starter_Kit

Install BabyBench requirements:

    pip install -r requirements.txt

Install MIMo in editable mode:

    pip install -e MIMo

At this point, the BabyBench environment is fully installed.

---

## OpenCV note (GUI support)

This project uses OpenCV for real-time rendering via `cv2.imshow`.

Make sure that the GUI-enabled version of OpenCV is installed:

    pip uninstall -y opencv-python-headless
    pip install opencv-python

The headless version (`opencv-python-headless`) is not supported.


## Note on Whisper

Whisper is installed directly from source:

https://github.com/openai/whisper

This is handled automatically when creating the Conda environment.

---

## 4. Usage

### Train

    python train.py

### Test

    python test.py --model models/actor_ddpg.pt

### GUI

    python gui.py

---


## Prosody-based Human Feedback for Reinforcement Learning (Deep TAMER)

This project supports human evaluative feedback using **voice prosody** to guide the training of a reinforcement learning agent following the **Deep TAMER** paradigm, as an alternative to keyboard-based feedback.

The system is designed for the **MIMo / BabyBench** environment and relies on a pretrained **SUPERB wav2vec2 emotion recognition model**, calibrated to the user’s own voice.

Human feedback is inferred continuously from speech prosody and converted into a scalar reward signal used during policy learning.

---

## File Description

### 1. `model_calibration_perso.py`

Personal voice calibration and model setup.

This script:
- Records voice samples from the user in **positive** and **negative** emotional states
- Uses the pretrained `superb/wav2vec2-base-superb-er` emotion recognition model
- Extracts emotion probabilities
- Computes a valence score:

    valence = P(happy) − P(angry)

- Estimates a normalization factor `TAU`
- Saves a personalized calibration file

Output file:

    valence_calib_posneg.joblib

**This script must be executed once per user and microphone setup.**

---

### 2. `audio_feedback_valence.py`

Audio feedback inference and configuration module.

This module:
- Loads the personalized calibration file (`valence_calib_posneg.joblib`)
- Records short speech segments from the microphone
- Computes:
  - valence
  - a continuous reward in the interval [-1, 1] using:

        reward = tanh(valence / TAU)

- Returns interpretable feedback values suitable for reinforcement learning

---

### 3. `functional_model_test.py`

Standalone functional test of the prosody-based feedback model.

This script verifies:
- Microphone input
- Emotion model loading
- Calibration correctness
- Stability of valence and reward outputs

**This file is intended for validation only and is not used during RL training.**

---

## Recommended Workflow

1. Run personal voice calibration:

    python model_calibration_perso.py

2. Test the prosody-based feedback model:

    python functional_model_test.py

3. Train the agent using Deep TAMER with prosody feedback (see Training section)


## Speech-based Human Feedback (Deep TAMER)

This project also supports **speech-based human evaluative feedback** as an alternative to keyboard input.

Spoken feedback is captured in real time, transcribed using **OpenAI Whisper**, and analyzed with a **DistilBERT** sentiment model. The resulting sentiment is converted into an evaluative signal and integrated into the reinforcement learning loop following the **Deep TAMER** paradigm.

The system is fully compatible with the **MIMo / BabyBench** environment and supports hands-free interaction during training.

---

### Main components

- Real-time audio capture with RMS-based voice activity detection (VAD)
- Speech-to-text transcription using Whisper
- Sentiment analysis using DistilBERT
- Conversion of sentiment into evaluative feedback compatible with Deep TAMER

---

### `feedback/audio_speech.py`

Speech-based feedback module for reinforcement learning.

This module:
- Continuously records microphone input
- Detects speech segments using VAD
- Transcribes speech using Whisper
- Applies sentiment analysis to the transcribed text
- Outputs evaluative feedback signals during training

It acts as a drop-in replacement for keyboard-based feedback and preserves the original TAMER API.

---

### Usage notes

- Whisper is installed directly from source as part of the environment setup
- Rendering should be enabled to visualize agent behavior during training
- This feedback mode is best suited for short spoken evaluative cues (e.g., approval or disapproval)
