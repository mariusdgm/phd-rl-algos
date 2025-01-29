import torch
from collections import deque
import numpy as np
import random
import pickle


class ReplayBuffer:
    def __init__(self, max_size, state_dim, action_dim, n_step):
        self.max_size = max_size
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.n_step = n_step
        self.buffer = deque(maxlen=self.max_size)

    def __len__(self):
        return len(self.buffer)

    def _normalize_state(self, state):
        """
        Ensure that states are 2D tensors with shape (1, state_dim).
        If the state is 1D, unsqueeze it to make it 2D.
        """
        if isinstance(state, torch.Tensor):
            state = state.float()
        else:
            state = torch.tensor(state, dtype=torch.float32)

        if state.ndim == 1:  # If 1D, add a batch dimension
            state = state.unsqueeze(0)
        return state

    def append(self, state, action, reward, next_state, done):
        """
        Append a transition to the replay buffer.
        Normalize the state and next_state tensors before storing.
        """
        # Ensure consistent dimensionality for states and next_states
        state = np.array(state).reshape(-1)  # Flatten to 1D
        next_state = np.array(next_state).reshape(-1)  # Flatten to 1D
        action = np.array(action).reshape(-1)  # Flatten to 1D
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.
        Ensure that the states and next_states are stacked into tensors.
        """
        if batch_size > len(self):
            raise ValueError("Not enough transitions to sample")

        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        states = torch.tensor(np.array(states), dtype=torch.float32)
        next_states = torch.tensor(np.array(next_states), dtype=torch.float32)
        actions = torch.tensor(actions, dtype=torch.float32)
        rewards = torch.tensor(rewards, dtype=torch.float32)
        dones = torch.tensor(dones, dtype=torch.float32)

        return states, actions, rewards, next_states, dones

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
                reward += r * pow(0.99, i)

            transitions.append((state, action, reward, next_state, done))

        states, actions, rewards, next_states, dones = zip(*transitions)

        # Normalize and stack outputs
        states = torch.cat([self._normalize_state(s) for s in states], dim=0)
        next_states = torch.cat([self._normalize_state(s) for s in next_states], dim=0)
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
