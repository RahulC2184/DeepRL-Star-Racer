import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import random
from collections import deque

class QNetwork(nn.Module):
    """
    Dueling DQN Network Architecture.
    
    This network separates the Q-value calculation into two streams:
    1. Value Stream: Estimates the value of the state (V(s)).
    2. Advantage Stream: Estimates the advantage of each action (A(s, a)).
    
    It combines them using: Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
    """
    def __init__(self, input_size, output_size):
        super(QNetwork, self).__init__()
        
        # Shared feature-learning base
        self.fc1 = nn.Linear(input_size, 128)
        self.fc2 = nn.Linear(128, 128)

        # --- Dueling Architecture Streams ---

        # 1. The Value Stream (estimates V(s))
        # This stream estimates the overall value of being in a state.
        self.value_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, 1)  # Outputs a single scalar value for the state
        )

        # 2. The Advantage Stream (estimates A(s, a))
        # This stream estimates the advantage of taking each action in that state.
        self.advantage_stream = nn.Sequential(
            nn.Linear(128, 128),
            nn.ReLU(),
            nn.Linear(128, output_size) # Outputs one value per action
        )

    def forward(self, x):
        # Pass input through the shared base
        x = torch.relu(self.fc1(x))
        x = torch.relu(self.fc2(x))

        # --- Dueling Combination ---
        
        # Calculate V(s) and A(s, a)
        value = self.value_stream(x)
        advantages = self.advantage_stream(x)
        
        # Combine V(s) and A(s, a) to get the final Q-values
        # Q(s, a) = V(s) + (A(s, a) - mean(A(s, a)))
        q_values = value + (advantages - advantages.mean(dim=1, keepdim=True))
        
        return q_values

class ReplayBuffer:
    """A standard Experience Replay Buffer."""
    def __init__(self, capacity):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        state = np.array(state, dtype=np.float32)
        next_state = np.array(next_state, dtype=np.float32)
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        state, action, reward, next_state, done = zip(*random.sample(self.buffer, batch_size))
        return np.stack(state), np.array(action), np.array(reward), np.stack(next_state), np.array(done)

    def __len__(self):
        return len(self.buffer)

class DQNAgent:
    """
    The main DQN Agent.
    """
    def __init__(self, state_size, action_size, learning_rate=1e-3, gamma=0.99, epsilon_start=1.0, epsilon_end=0.01, epsilon_decay=0.995, buffer_capacity=50000, batch_size=64):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_end = epsilon_end
        
        # --- FIX: Slower exploration decay ---
        # The default 0.995 decays way too fast (per step).
        # This new value decays per-step, but much more slowly.
        self.epsilon_decay = 0.99995 
        # --- END FIX ---
        
        self.batch_size = batch_size

        self.q_network = QNetwork(state_size, action_size)
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=learning_rate)
        self.replay_buffer = ReplayBuffer(buffer_capacity)
        
    def choose_action(self, state):
        """Chooses an action using an epsilon-greedy policy."""
        if random.random() < self.epsilon:
            return random.randrange(self.action_size)
        
        state_tensor = torch.FloatTensor(state).unsqueeze(0)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def update(self):
        """Samples from the buffer and performs a gradient descent step."""
        if len(self.replay_buffer) < self.batch_size:
            return
            
        state_batch, action_batch, reward_batch, next_state_batch, done_batch = self.replay_buffer.sample(self.batch_size)
        
        state_batch = torch.FloatTensor(state_batch)
        action_batch = torch.LongTensor(action_batch)
        reward_batch = torch.FloatTensor(reward_batch)
        next_state_batch = torch.FloatTensor(next_state_batch)
        done_batch = torch.FloatTensor(done_batch)
        
        q_values = self.q_network(state_batch).gather(1, action_batch.unsqueeze(1)).squeeze(1)
        
        next_q_values = self.q_network(next_state_batch).max(1)[0]
        
        expected_q_values = reward_batch + self.gamma * next_q_values * (1 - done_batch)
        
        loss = nn.MSELoss()(q_values, expected_q_values.detach())
        
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()
        
        self.epsilon = max(self.epsilon_end, self.epsilon * self.epsilon_decay)