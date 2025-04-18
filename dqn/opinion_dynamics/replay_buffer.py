import torch
from collections import deque
import numpy as np
import random
import pickle


class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim=1, n_step=0, gamma=0.9):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_step = n_step
        self.buffer = deque(maxlen=self.max_size)
        self.gamma = gamma

    def __len__(self):
        
        return len(self.buffer)

    def append(self, state, action, reward, next_state, done):
        """
        Append a transition to the replay buffer.
        Normalize the state and next_state tensors before storing.
        """
        # Ensure consistent dimensionality for states and next_states
        state = np.array(state).reshape(-1)  # Flatten to 1D
        next_state = np.array(next_state).reshape(-1)  # Flatten to 1D
        beta_idx, w = action
        beta_idx = np.array(beta_idx, dtype=np.int64)
        w = np.array(w).reshape(-1)
        self.buffer.append((state, (beta_idx, w), reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.
        Ensure that the states and next_states are stacked into tensors.
        """
        if batch_size > len(self):
            raise ValueError("Not enough transitions to sample")

        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        # Split action into beta indices and w vectors
        beta_indices, ws = zip(*actions)

        states = torch.tensor(np.stack(states), dtype=torch.float32)
        next_states = torch.tensor(np.stack(next_states), dtype=torch.float32)
        beta_indices = torch.tensor(beta_indices, dtype=torch.long)
        ws = torch.tensor(np.stack(ws), dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, (beta_indices, ws), rewards, next_states, dones

    def sample_n_step(self, batch_size, stride=1):
        """
        Sample N-step transitions for bootstrapped updates.
        """
        if batch_size > len(self):
            raise ValueError("Not enough transitions to sample")

        transitions = []
        for _ in range(batch_size):
            start_idx = random.randint(0, len(self) - self.n_step * stride - 1)
            end_idx = start_idx + self.n_step * stride
            samples = self.buffer[start_idx:end_idx:stride]

            state, _, _, _, _ = samples[0]
            _, _, _, next_state, done = samples[-1]

            reward = 0
            for i in range(self.n_step):
                _, action, r, _, _ = samples[i * stride]
                reward += r * pow(self.gamma, i)

            transitions.append((state, action, reward, next_state, done))

        states, actions, rewards, next_states, dones = zip(*transitions)

        # Normalize and stack outputs
        states = torch.cat([s for s in states], dim=0)
        next_states = torch.cat([s for s in next_states], dim=0)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, actions, rewards, next_states, dones

    def save(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, file_name):
        with open(file_name, "rb") as f:
            self.buffer = pickle.load(f)
