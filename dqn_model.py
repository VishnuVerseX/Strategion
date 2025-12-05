"""
Dueling Double DQN Model for Tennis Strategy
Upgrades:
- Dueling network architecture (value + advantage streams)
- Double-DQN target calculation for more stable learning
"""

import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from collections import deque
import random


class DuelingDQN(nn.Module):
    def __init__(self, state_size: int, action_size: int):
        super(DuelingDQN, self).__init__()

        self.fc1 = nn.Linear(state_size, 128)
        self.fc2 = nn.Linear(128, 128)

        # Value stream: V(s)
        self.value_fc = nn.Linear(128, 64)
        self.value_out = nn.Linear(64, 1)

        # Advantage stream: A(s, a)
        self.adv_fc = nn.Linear(128, 64)
        self.adv_out = nn.Linear(64, action_size)

        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu(self.fc2(x))

        # Value branch
        v = self.relu(self.value_fc(x))
        v = self.value_out(v)  # shape: (batch, 1)

        # Advantage branch
        a = self.relu(self.adv_fc(x))
        a = self.adv_out(a)     # shape: (batch, num_actions)

        # Combine: Q(s,a) = V(s) + A(s,a) - mean_a A(s,a)
        a_mean = a.mean(dim=1, keepdim=True)
        q = v + (a - a_mean)
        return q


class ReplayMemory:
    def __init__(self, capacity: int):
        self.memory = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        """Save a transition."""
        self.memory.append((state, action, reward, next_state, done))

    def sample(self, batch_size: int):
        """Sample a batch of transitions."""
        return random.sample(self.memory, batch_size)

    def __len__(self):
        return len(self.memory)


class DQNAgent:
    def __init__(
        self,
        state_size: int,
        action_size: int,
        learning_rate: float = 0.001,
        gamma: float = 0.99,          # slightly higher for long-horizon tennis
        epsilon_start: float = 1.0,
        epsilon_end: float = 0.01,
        epsilon_decay: float = 0.995,
        memory_size: int = 10000,
        batch_size: int = 64,
        target_update: int = 10
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_update = target_update

        # Device configuration
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        print(f"Using device: {self.device}")

        # Dueling Q-Network and Target Network
        self.policy_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net = DuelingDQN(state_size, action_size).to(self.device)
        self.target_net.load_state_dict(self.policy_net.state_dict())
        self.target_net.eval()

        # Optimizer and loss
        self.optimizer = optim.Adam(self.policy_net.parameters(), lr=learning_rate)
        self.criterion = nn.SmoothL1Loss()

        # Replay memory
        self.memory = ReplayMemory(memory_size)

        # Training metrics
        self.steps_done = 0
        self.episode_rewards = []
        self.losses = []

    def select_action(
        self,
        state: np.ndarray,
        valid_actions=None,
        training: bool = True
    ) -> int:
        """
        Epsilon-greedy action selection over ONLY valid_actions.

        - training=True  -> epsilon-greedy (explore + exploit)
        - training=False -> pure greedy (no random exploration)
        """
        if valid_actions is None:
            valid_actions = list(range(self.action_size))

        # Exploration
        if training and random.random() < self.epsilon:
            return random.choice(valid_actions)

        # Exploitation
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor).cpu().numpy()[0]

        # Mask invalid actions
        masked_q = np.full_like(q_values, -1e9)
        masked_q[valid_actions] = q_values[valid_actions]
        return int(np.argmax(masked_q))

    def store_transition(self, state, action, reward, next_state, done):
        """Store transition in replay memory."""
        self.memory.push(state, action, reward, next_state, done)

    def train_step(self):
        """Perform one Double-DQN training step."""
        if len(self.memory) < self.batch_size:
            return None

        # Sample batch
        transitions = self.memory.sample(self.batch_size)
        batch_state, batch_action, batch_reward, batch_next_state, batch_done = zip(*transitions)

        # Convert to tensors
        state_batch = torch.FloatTensor(np.array(batch_state)).to(self.device)
        action_batch = torch.LongTensor(batch_action).unsqueeze(1).to(self.device)
        reward_batch = torch.FloatTensor(batch_reward).unsqueeze(1).to(self.device)
        next_state_batch = torch.FloatTensor(np.array(batch_next_state)).to(self.device)
        done_batch = torch.FloatTensor(batch_done).unsqueeze(1).to(self.device)

        # Q(s,a) for current states
        q_values = self.policy_net(state_batch)
        state_action_values = q_values.gather(1, action_batch)

        # Double-DQN target:
        with torch.no_grad():
            # 1) Use policy net to choose best actions in next states
            next_q_policy = self.policy_net(next_state_batch)
            next_actions = next_q_policy.argmax(dim=1, keepdim=True)  # shape (batch,1)

            # 2) Evaluate those actions using target net
            next_q_target = self.target_net(next_state_batch)
            next_state_values = next_q_target.gather(1, next_actions)

            # 3) Bellman target
            expected_state_action_values = reward_batch + \
                (self.gamma * next_state_values * (1 - done_batch))

        # Loss
        loss = self.criterion(state_action_values, expected_state_action_values)

        # Optimize
        self.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(self.policy_net.parameters(), 1.0)
        self.optimizer.step()

        self.losses.append(loss.item())
        return loss.item()

    def update_target_network(self):
        """Update target network with policy network weights."""
        self.target_net.load_state_dict(self.policy_net.state_dict())

    def decay_epsilon(self):
        """Decay exploration rate."""
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)

    def save_model(self, filepath: str):
        """Save model weights and training state."""
        torch.save(
            {
                "policy_net_state_dict": self.policy_net.state_dict(),
                "target_net_state_dict": self.target_net.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "epsilon": self.epsilon,
                "episode_rewards": self.episode_rewards,
                "losses": self.losses,
            },
            filepath,
        )
        print(f"Model saved to {filepath}")

    def load_model(self, filepath: str):
        """Load model weights and training state."""
        checkpoint = torch.load(filepath, map_location=self.device)
        self.policy_net.load_state_dict(checkpoint["policy_net_state_dict"])
        self.target_net.load_state_dict(checkpoint["target_net_state_dict"])
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])
        self.epsilon = checkpoint["epsilon"]
        self.episode_rewards = checkpoint["episode_rewards"]
        self.losses = checkpoint["losses"]
        print(f"Model loaded from {filepath}")

    def get_q_values(self, state: np.ndarray) -> np.ndarray:
        """Get Q-values for all actions given a state."""
        with torch.no_grad():
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
            q_values = self.policy_net(state_tensor)
            return q_values.cpu().numpy()[0]