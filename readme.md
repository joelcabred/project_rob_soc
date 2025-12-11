# TAMER-Enhanced Self-Touch Learning for MIMo Robot

This project implements a Deep Deterministic Policy Gradient (DDPG) agent enhanced with TAMER (Training an Agent Manually via Evaluative Reinforcement) to learn self-touch behaviors in the MIMo robotic environment. The agent learns from both environmental rewards and real-time human feedback.

## Prerequisites

- BabyBench environment installed
- All files must be placed in the **root directory** of BabyBench

## Project Files

### Core Components

#### `feedback_capture.py`
Human feedback capture system that processes keyboard inputs during training. Allows trainers to provide real-time evaluative feedback (+1 for positive, -1 for negative, 0 for neutral) with configurable intervals to prevent spam. Maintains feedback history and statistics.

#### `deep_tamer.py`
TAMER module implementation featuring a neural network that learns to predict human rewards from state-action pairs. Includes temporal credit assignment with exponential decay to distribute feedback across recent trajectory steps. Manages training buffers and provides reward predictions for the RL agent.

#### `ddpg.py`
Standard DDPG agent implementation with actor-critic architecture, experience replay, target networks, and Ornstein-Uhlenbeck noise for exploration. Serves as the baseline RL agent without human feedback integration.

#### `ddpg_tamer.py`
TAMER-enhanced DDPG agent that combines traditional RL rewards with learned human feedback signals. Features configurable weighting (alpha for intrinsic rewards, beta for human feedback) to balance both learning signals. Integrates the TAMER module for human-in-the-loop training.

### Environment and Rewards

#### `TactileObsWrapper.py`
Gymnasium wrapper that transforms MIMo observations into tactile-focused feature vectors. Extracts k-nearest sensor readings from contact points on hands and fingers, combined with joint positions, velocities, and torques to create a comprehensive observation space.

#### `rewards.py`
Multi-objective reward shaping functions including:
- Tactile contact detection and intensity
- Movement novelty rewards
- Body region exploration bonuses
- Motion constraint penalties (joint limits, speed)
- Energy efficiency penalties
- Stillness penalties

### Training Scripts

#### `ddpg_mimo.py`
Training script for baseline DDPG agent without human feedback. Trains the agent using only the shaped environmental rewards for self-touch behavior learning.

#### `tamer_mimo.py` ⭐ **[PRIMARY TRAINING SCRIPT]**
**Main training script** for TAMER-enhanced agent. Supports interactive training where humans provide keyboard feedback during episodes:
- Press `+` or `=` for positive feedback
- Press `-` or `_` for negative feedback  
- Press `0` for neutral feedback

Combines intrinsic rewards with human feedback signals and saves trained models.

### Testing Scripts

#### `test_ddpg.py`
Evaluation script for baseline DDPG agents. Loads pre-trained models and measures performance metrics:
- Time to first touch
- Number of unique body regions contacted
- Movement freeze ratios

#### `test_deeptamer.py` ⭐ **[PRIMARY TESTING SCRIPT]**
**Main evaluation script** for TAMER-trained agents. Tests the behavioral outcomes of human-feedback-enhanced training with the same metrics as the baseline test script.

## Usage

### Training with Human Feedback
```bash
python tamer_mimo.py --config examples/config_selftouch.yml --train_for 1000
```

During training:
- Watch the robot's behavior in the visualization window
- Provide feedback using keyboard:
  - `+` or `=`: Positive feedback (good behavior)
  - `-` or `_`: Negative feedback (bad behavior)
  - `0`: Neutral feedback

### Testing Trained Agent
```bash
python test_deeptamer.py
```

This will load the saved model (`actor_selftouch.pt`) and evaluate performance across 10 test episodes.

## Architecture Overview
```
Human Feedback (keyboard) → HumanFeedbackCapture
                                    ↓
                            TAMERModule (learns reward model)
                                    ↓
Environmental Rewards  →  DDPGAgent (combines both signals)
                                    ↓
                            MIMo Robot (self-touch behavior)
```

## Key Parameters

- **alpha**: Weight for intrinsic (environmental) rewards (default: 1.0)
- **beta**: Weight for human feedback rewards (default: 0.1)
- **credit_window**: Number of recent steps to assign feedback credit (default: 15)
- **decay_factor**: Exponential decay for temporal credit assignment (default: 0.9)

## Output

- `actor_selftouch.pt`: Trained actor network
- `critic_selftouch.pt`: Trained critic network
- `ddpg_training.png`: Training curves visualization
- Console output: Training metrics, feedback statistics, and loss values

## Citation

This implementation is based on the TAMER framework for human-in-the-loop reinforcement learning, applied to tactile self-exploration in robotic systems.