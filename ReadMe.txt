1) Prosody-based Human Feedback for Reinforcement Learning (Deep TAMER)

This repository implements human evaluative feedback using voice prosody to guide the training of a reinforcement learning agent (Deep TAMER), instead of keyboard-based feedback.

The system is designed to work with the MIMo / BabyBench environment and uses a pretrained SUPERB wav2vec2 emotion recognition model, calibrated to the user’s own voice.

2) File Description
1. model_calibration_perso.py

Personal model creation and voice calibration
Records voice samples from the user (POS / NEG)
Uses superb/wav2vec2-base-superb-er
Extracts emotion probabilities
Computes a valence score:

	valence = P(happy) − P(angry)

Estimates a normalization factor TAU
Saves a personalized calibration file

Output file:
	==> valence_calib_posneg.joblib

** This script must be executed once per user / microphone setup.


2. audio_feedback_valence.py

Audio feedback configuration and inference module
Loads the calibrated model (valence_calib_posneg.joblib)
Records short speech segments
Computes:
	- valence
	- continuous reward in the interval [-1, 1] using: 
	      reward = tanh(valence / TAU)
Returns interpretable feedback values


3. functional_model_test.py

Standalone functional test of the prosody model

Tests : 
- microphone input
- emotion model loading
- calibration correctness
- stability of valence and reward outputs

** This file is only used to verify that the model works correctly before integration with reinforcement learning.

4. feedback_capture_prosody.py

Prosody-based feedback module for RL training (Deep TAMER)
Drop-in replacement for feedback_capture.py
Fully compatible with MIMo / BabyBench
Microphone is constantly active
Automatic voice activity detection (VAD)
Converts voice prosody into:
	- valence
	- continuous feedback reward
Preserves the original TAMER API

** Usage notes:

Replace feedback_capture.py with this file
Enable rendering to visualize the agent during training
Use TAMER mode to train with vocal feedback


3) Recommended Workflow

Run personal calibration: python model_calibration_perso.py
Test the prosody model: python functional_model_test.py
