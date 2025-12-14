## BabyBench – Self-Touch Learning

This project trains and evaluates DDPG agents on the BabyBench self-touch task,
with optional human feedback (DeepTAMER).

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

## Notes

- All commands must be run from the project root directory.
- Real-time rendering is enabled during both training and testing.
- GPU support requires a compatible NVIDIA GPU and CUDA 12.1.
