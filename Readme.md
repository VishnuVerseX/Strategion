# 🎾 Tennis Strategy Optimizer using Reinforcement Learning

A complete Deep Q-Learning system that learns optimal tennis strategies based on game state (score, serve, sets, fatigue, etc.).

## 📋 System Overview

This project uses **Deep Q-Network (DQN)** reinforcement learning to:
- Simulate realistic tennis matches
- Learn optimal strategies for different game situations
- Recommend actions based on score, serving position, fatigue, and more
- Generate training data automatically (no dataset needed!)

## 🎯 Features

- **8 Strategic Actions**: aggressive serve, safe serve, aggressive return, defensive return, net approach, baseline rally, drop shot, lob
- **Complete Game State Tracking**: Points, games, sets, serve, court side, fatigue, deuce/advantage
- **No External Dataset Required**: Generates training data through self-play simulation
- **Real-time Strategy Prediction**: Get recommendations for any game situation
- **Performance Analytics**: Visualize training progress and evaluate agent performance

## 📁 Project Structure

```
tennis-strategy-optimizer/
│
├── tennis_env.py          # Tennis environment simulator
├── dqn_model.py          # Deep Q-Network model & agent
├── train.py              # Training script
├── evaluate.py           # Model evaluation & testing
├── predict.py            # Strategy prediction interface
├── requirements.txt      # Python dependencies
├── README.md            # This file
│
├── models/              # Saved model checkpoints (created during training)
│   └── tennis_dqn_final.pth
│
└── logs/                # Training logs and visualizations (created during training)
    ├── training_results.png
    └── training_metrics_episode_X.json
```

## 🚀 Installation & Setup

### Step 1: Clone or Download Files

Create a new directory and save all the Python files there:

```bash
mkdir tennis-strategy-optimizer
cd tennis-strategy-optimizer
```

Save these files:
- `tennis_env.py`
- `dqn_model.py`
- `train.py`
- `evaluate.py`
- `predict.py`
- `requirements.txt`

### Step 2: Create Virtual Environment (Recommended)

```bash
# Create virtual environment
python -m venv venv

# Activate it
# On Windows:
venv\Scripts\activate
# On Mac/Linux:
source venv/bin/activate
```

### Step 3: Install Dependencies

```bash
pip install -r requirements.txt
```

This will install:
- PyTorch (Deep Learning framework)
- NumPy (Numerical computing)
- Matplotlib (Visualization)

## 🎮 How to Use

### 1️⃣ Train the Model (First Time)

This generates training data and trains the agent:

```bash
python train.py
```

**What happens:**
- Creates `models/` and `logs/` directories
- Simulates 1000 tennis matches (configurable)
- Trains DQN agent using experience replay
- Saves model checkpoints every 100 episodes
- Generates training visualization graphs

**Training Time:** ~10-30 minutes depending on your machine

**Output:**
- Progress updates every 10 episodes
- Model saved to `models/tennis_dqn_final.pth`
- Training graphs saved to `logs/training_results.png`

### 2️⃣ Evaluate the Model

Test the trained agent's performance:

```bash
python evaluate.py
```

**What happens:**
- Loads trained model
- Runs 100 test matches
- Shows win rate, average rewards, action distribution
- Generates evaluation visualizations
- Optional: Run detailed demo match

**Output:**
- Performance statistics
- Evaluation graphs saved to `logs/evaluation_results.png`

### 3️⃣ Get Strategy Predictions

#### Interactive Mode (Recommended)

```bash
python predict.py
```

Then enter game states:
```
Enter score: 0-0, 2-1, 30-40
Are you serving? (y/n): n

🎯 RECOMMENDED STRATEGY: AGGRESSIVE_RETURN
   Confidence Score: 2.345
```

Or type `quick` for random scenarios!

#### Batch Analysis Mode

Analyze common scenarios:

```bash
python predict.py batch
```

This shows optimal strategies for:
- Break point opportunities
- Serving for the match
- Defending match point
- And more!

## 📊 Understanding the State

The system tracks 15 state variables:

| Variable | Description | Range |
|----------|-------------|-------|
| player_points | Points in current game | 0-4 |
| opponent_points | Opponent points | 0-4 |
| player_games | Games won in set | 0-7 |
| opponent_games | Opponent games | 0-7 |
| player_sets | Sets won | 0-2 |
| opponent_sets | Opponent sets | 0-2 |
| is_player_serving | Serving status | 0/1 |
| is_deuce | Deuce situation | 0/1 |
| player_advantage | Player has advantage | 0/1 |
| opponent_advantage | Opponent has advantage | 0/1 |
| is_tiebreak | Tiebreak active | 0/1 |
| player_fatigue | Player fatigue level | 0-1 |
| opponent_fatigue | Opponent fatigue | 0-1 |
| court_side | Deuce or Ad side | 0/1 |
| rally_length | Current rally length | 0+ |

## 🎯 Actions

The agent chooses from 8 strategic actions:

1. **aggressive_serve** - High-risk serve (65% base success)
2. **safe_serve** - Conservative serve (55% base success)
3. **aggressive_return** - Attack the return (45% base success)
4. **defensive_return** - Safe return (50% base success)
5. **net_approach** - Come to net (55% base success)
6. **baseline_rally** - Consistent groundstrokes (50% base success)
7. **drop_shot** - Risky drop shot (40% base success)
8. **lob** - Defensive lob (45% base success)

Success probabilities are adjusted based on:
- Game situation (serving vs. returning)
- Player fatigue
- Score pressure
- Tactical context

## ⚙️ Configuration

### Modify Training Parameters

Edit `train.py`:

```python
# Training configuration
NUM_EPISODES = 1000        # Number of matches to simulate
MAX_STEPS = 500           # Max steps per match
SAVE_INTERVAL = 100       # Save model every N episodes
```

### Adjust Learning Parameters

Edit `dqn_model.py` `DQNAgent.__init__()`:

```python
learning_rate = 0.001     # Learning rate
gamma = 0.95              # Discount factor
epsilon_start = 1.0       # Initial exploration
epsilon_end = 0.01        # Final exploration
epsilon_decay = 0.995     # Exploration decay
memory_size = 10000       # Replay buffer size
batch_size = 64           # Training batch size
```

## 📈 Monitoring Training

### Watch Real-time Progress

During training, you'll see:

```
Episode 100/1000 | Avg Reward: 2.34 | Avg Length: 143 | Win Rate: 52.00% | Epsilon: 0.605
Episode 200/1000 | Avg Reward: 3.12 | Avg Length: 156 | Win Rate: 58.00% | Epsilon: 0.366
```

### Analyze Training Graphs

After training, check `logs/training_results.png` for:
- Episode rewards over time
- Episode lengths
- Win rate progression
- Training loss

## 🔧 Troubleshooting

### "Model not found" Error

**Problem:** Running `evaluate.py` or `predict.py` before training

**Solution:** Run `python train.py` first

### Out of Memory Error

**Problem:** Not enough RAM/GPU memory

**Solution:** Reduce batch size in `dqn_model.py`:
```python
batch_size = 32  # Instead of 64
```

### Low Win Rate

**Problem:** Agent not learning well

**Solution:** 
- Train for more episodes (2000+)
- Adjust learning rate
- Check training graphs for convergence

### Import Errors

**Problem:** Missing dependencies

**Solution:**
```bash
pip install --upgrade torch numpy matplotlib
```

## 🎓 How It Works

### 1. Environment Simulation (`tennis_env.py`)

- Simulates realistic tennis scoring rules
- Tracks complete game state
- Calculates success probabilities based on:
  - Action type
  - Game situation
  - Player fatigue
  - Score pressure

### 2. Deep Q-Network (`dqn_model.py`)

- Neural network with 4 layers (128→128→64→8 neurons)
- Takes 15-dimensional state vector as input
- Outputs Q-values for each of 8 actions
- Uses experience replay for stable learning
- Target network for training stability

### 3. Training Loop (`train.py`)

```
For each episode:
  1. Reset environment to start of match
  2. While match not over:
     a. Agent selects action (ε-greedy)
     b. Environment simulates point outcome
     c. Store experience in replay memory
     d. Sample batch and train neural network
     e. Update state
  3. Decay exploration rate
  4. Update target network periodically
  5. Save checkpoint
```

### 4. Prediction (`predict.py`)

- Loads trained model
- Converts game state to vector
- Feeds through neural network
- Returns action with highest Q-value

## 📚 Advanced Usage

### Resume Training from Checkpoint

```python
# In train.py, before training loop:
agent.load_model('models/tennis_dqn_episode_500.pth')
# Then continue training
```

### Custom Scenarios

```python
from predict import TennisStrategyPredictor

predictor = TennisStrategyPredictor()

result = predictor.predict_from_state(
    player_points=3,      # 40
    opponent_points=3,    # 40 (Deuce)
    player_games=5,
    opponent_games=4,
    player_sets=1,
    opponent_sets=1,
    is_player_serving=True,
    player_fatigue=0.6,
    opponent_fatigue=0.4
)

print(result['recommended_action'])
```

### Export Predictions to CSV

Add to `predict.py`:

```python
import csv

# After getting predictions
with open('predictions.csv', 'w', newline='') as f:
    writer = csv.writer(f)
    writer.writerow(['State', 'Action', 'Q-Value'])
    for action in result['all_actions']:
        writer.writerow([
            result['game_state'],
            action['action'],
            action['q_value']
        ])
```

## 🔬 Performance Benchmarks

Expected results after 1000 training episodes:

- **Win Rate:** 55-65%
- **Average Reward:** 2-4 per episode
- **Convergence:** ~500-700 episodes
- **Training Time:** 15-30 minutes (CPU)

## 🤝 Contributing

Ideas for enhancement:
- Add more actions (serve placement, spin variations)
- Incorporate opponent modeling
- Add surface type (clay, grass, hard court)
- Multi-agent training
- Real match data integration

## 📄 License

This is an educational project. Feel free to use and modify!

## 🙋 FAQ

**Q: Do I need a GPU?**
A: No, but it speeds up training. CPU works fine for this project.

**Q: Can I use real match data?**
A: Yes! You can modify `tennis_env.py` to load real match data instead of simulation.

**Q: How accurate are the predictions?**
A: The agent learns strategies that work in simulation. Real tennis has more variables, but core strategic principles transfer.

**Q: Can I change the scoring system?**
A: Yes, modify `_update_score()` in `tennis_env.py` for different formats (e.g., no-ad scoring).

**Q: Why Deep Q-Learning?**
A: DQN handles continuous states well and learns without game tree search. Perfect for complex domains like tennis strategy.

## 🎉 Quick Start Summary

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Train the model
python train.py

# 3. Evaluate performance
python evaluate.py

# 4. Get predictions
python predict.py
```

That's it! You now have a working tennis strategy optimizer! 🎾🏆