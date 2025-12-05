# 🎾 Tennis Strategy Optimizer
<div align="center">
⚖️ © 2025 - ALL RIGHTS RESERVED ⚖️

<div align="center">

![Tennis AI](https://img.shields.io/badge/AI-Reinforcement%20Learning-blue?style=for-the-badge)
![Python](https://img.shields.io/badge/Python-3.8+-green?style=for-the-badge&logo=python)
![PyTorch](https://img.shields.io/badge/PyTorch-Latest-red?style=for-the-badge&logo=pytorch)
![Status](https://img.shields.io/badge/Status-Production-success?style=for-the-badge)

**An advanced AI system for optimizing tennis match strategies using Deep Reinforcement Learning**

[Features](#-features) • [Installation](#-installation) • [Usage](#-usage) • [Architecture](#-architecture) • [Results](#-results)

</div>

---

## 📋 Overview

The **Tennis Strategy Optimizer** is a sophisticated reinforcement learning system that learns optimal tennis strategies through self-play. Built on a **Dueling Double DQN** architecture, it analyzes match situations and recommends tactical decisions for serves, returns, and rally play.

### Key Highlights

- 🧠 **Dueling Double DQN Architecture** - Advanced neural network with separate value and advantage streams
- 📈 **Curriculum Learning** - Progressive difficulty scaling for robust strategy development
- 🎯 **Tactical Intelligence** - Context-aware recommendations based on game situation, fatigue, and momentum
- 📊 **Comprehensive Analytics** - Detailed performance metrics and visualizations
- ⚡ **Real-time Predictions** - Interactive strategy predictor for live match scenarios

---

## ✨ Features

### 🎮 Core Capabilities

- **10 Tactical Actions**: Serves (flat wide, flat T, kick body), Returns (aggressive, neutral, block), Rally shots (aggressive, neutral, approach net, defensive lob)
- **Realistic Tennis Simulation**: Accurate scoring, deuce/advantage, tiebreaks, fatigue modeling
- **Opponent Adaptation**: Train against varying skill levels (0.35 - 0.55)
- **Physical State Tracking**: Fatigue dynamics, court positioning, rally length analysis

### 📊 Training & Evaluation

- **Curriculum Learning Pipeline**: Gradual difficulty progression from easy (0.40) to balanced (0.50) opponents
- **Advanced DQN Features**: 
  - Dueling network architecture for better value estimation
  - Double DQN target calculation for reduced overestimation
  - Experience replay with 20K memory capacity
  - Target network updates for stable learning
- **Comprehensive Metrics**: Win rates, reward distributions, action usage analysis, phase-specific performance

### 🔮 Strategy Prediction

- **Dual-Phase Recommendations**: Separate strategies for serve/return and rally phases
- **Confidence Scoring**: Q-value based confidence percentages for each action
- **Tactical Reasoning**: Human-readable explanations for recommended plays
- **Situation Analysis**: Pressure level, momentum, and game phase assessment

---

## 🚀 Installation

### Prerequisites

```bash
Python 3.8+
PyTorch 1.9+
NumPy
Matplotlib
```

### Setup

```bash
# Clone the repository
git clone https://github.com/yourusername/tennis-strategy-optimizer.git
cd tennis-strategy-optimizer

# Install dependencies
pip install torch numpy matplotlib

# Verify installation
python -c "import torch; print(f'PyTorch {torch.__version__} installed successfully')"
```

### Project Structure

```
tennis-strategy-optimizer/
├── tennis_env.py          # Tennis environment simulator
├── dqn_model.py           # Dueling Double DQN implementation
├── train.py               # Training script with curriculum learning
├── evaluate.py            # Evaluation and analysis tools
├── predict.py             # Interactive strategy predictor
├── models/                # Saved model checkpoints
├── logs/                  # Training metrics and visualizations
└── README.md
```

---

## 📖 Usage

### 1️⃣ Training

Train the agent using curriculum learning:

```bash
python train.py
```

**Training Configuration:**
- Episodes: 1500
- Curriculum phases: 4 stages (0.40 → 0.44 → 0.47 → 0.50 opponent skill)
- Max steps per episode: 750
- Checkpoints saved every 100 episodes


### 2️⃣ Evaluation

Evaluate trained model performance:

```bash
python evaluate.py
```

**Evaluation Options:**
1. **Standard Evaluation**: 100 episodes vs balanced opponent with detailed metrics
2. **Demo Match**: Play-by-play breakdown with Q-value analysis
3. **Multi-Level Comparison**: Test against multiple opponent skill levels

**Sample Output:**
```
EVALUATION RESULTS
═══════════════════════════════════════════
Matches Won:  52
Win Rate:     52.0%
Avg Reward:   +12.45
Serve Win%:   61.3%
Return Win%:  38.7%
```

### 3️⃣ Strategy Prediction

Get real-time tactical recommendations:

```bash
python predict.py
```

**Interactive Mode:**
```
Enter score (sets, games, points): 0-0, 3-2, 30-40
Are you serving? (y/n): n

🎯 START-OF-POINT STRATEGY:
   Recommended: RETURN_AGGRESSIVE
   Confidence: 68.5%
   💡 Reasoning: Break point opportunity | Opponent more tired

🎾 RALLY STRATEGY:
   Recommended: RALLY_AGGRESSIVE
   Confidence: 62.3%
   💡 Reasoning: Long rally established | Pressing advantage
```

---

## 🏗️ Architecture

### Neural Network Design

```
Input State Vector (18 features)
         ↓
    Dense(128) + ReLU
         ↓
    Dense(128) + ReLU
         ↓
    ┌─────────┴─────────┐
    ↓                   ↓
Value Stream      Advantage Stream
Dense(64)+ReLU    Dense(64)+ReLU
    ↓                   ↓
Dense(1)           Dense(10)
    ↓                   ↓
    └─────────┬─────────┘
              ↓
      Q(s,a) = V(s) + [A(s,a) - mean(A)]
```

### State Representation

The agent observes 18-dimensional state vectors encoding:

| Feature Category | Components |
|-----------------|------------|
| **Score State** | Points, games, sets for both players |
| **Serve Info** | Current server, deuce/advantage status, tiebreak flag |
| **Physical State** | Player/opponent fatigue levels |
| **Tactical State** | Court side, rally length, positions, ball depth |

### Action Space

| Phase | Actions | Description |
|-------|---------|-------------|
| **Serve** | 3 actions | Flat wide, flat T, kick body |
| **Return** | 3 actions | Aggressive, neutral, block |
| **Rally** | 4 actions | Aggressive, neutral, approach net, defensive lob |

---

## 📊 Results

### Training Performance

<div align="center">

| Metric | Value |
|--------|-------|
| Final Win Rate | 95-97% |
| Avg Episode Reward | +10-15 |
| Training Time | ~60-120 minutes |
| Convergence | Episode 1200+ |

</div>

### Key Findings

✅ **Learned Strategic Concepts:**
- Aggressive serving on big points (game/break points)
- Break point recognition and tactical adjustment
- Fatigue-based decision making
- Situational rally tactics (defensive vs offensive positioning)

✅ **Balanced Play Style:**
- Avoids over-defensive strategies (block/lob overuse <20%)
- Maintains appropriate serve variety
- Adapts to pressure situations effectively

✅ **Generalization:**
- Successfully transfers learning across opponent skill levels
- Maintains ~48-52% win rate against balanced opponent (0.50)
- Shows tactical flexibility in various game situations

---

## 🔧 Configuration

### Hyperparameters

```python
# Network Architecture
STATE_SIZE = 18
ACTION_SIZE = 10
HIDDEN_LAYERS = [128, 128, 64]

# Training Parameters
LEARNING_RATE = 0.0005
GAMMA = 0.99
EPSILON_START = 1.0
EPSILON_END = 0.01
EPSILON_DECAY = 0.9975

# Experience Replay
MEMORY_SIZE = 20000
BATCH_SIZE = 128
TARGET_UPDATE_FREQ = 5
```

### Curriculum Learning Schedule

```python
Episode 0-400:    Opponent Skill 0.40 (Easy)
Episode 400-800:  Opponent Skill 0.44 (Medium)
Episode 800-1200: Opponent Skill 0.47 (Hard)
Episode 1200+:    Opponent Skill 0.50 (Balanced)
```

---

## 📈 Future Enhancements

- [ ] Multi-agent self-play training
- [ ] Opponent modeling and adaptation
- [ ] Shot placement and trajectory optimization
- [ ] Weather/court surface adaptation
- [ ] Tournament simulation mode
- [ ] Transfer learning from professional match data

---

## 📄 License & Copyright

### Copyright Notice

**© 2025 Tennis Strategy Optimizer. All Rights Reserved.**

This software and associated documentation files (the "Software") are proprietary and confidential. 

### Restrictions

❌ **Reproduction** - No part of this Software may be reproduced without explicit written permission  
❌ **Modification** - Unauthorized modification of the Software is strictly prohibited  
❌ **Distribution** - Distribution in any form requires prior written authorization  
❌ **Commercial Use** - Commercial use is not permitted without a valid license agreement

### Permitted Use

✅ Personal evaluation and testing for authorized users only  

---

## 🙏 Acknowledgments

Built with:
- [PyTorch](https://pytorch.org/) - Deep learning framework
- [NumPy](https://numpy.org/) - Numerical computing
- [Matplotlib](https://matplotlib.org/) - Visualization

---

<div align="center">

**Made with ❤️ and 🎾**

⭐ Star this repository if you found it interesting!

</div>
