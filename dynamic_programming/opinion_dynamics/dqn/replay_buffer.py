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

    def append(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        if batch_size > len(self):
            raise ValueError("Not enough transitions to sample")

        samples = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*samples)

        return states, actions, rewards, next_states, dones

    def sample_n_step(self, batch_size, stride=1):
        if batch_size > len(self):
            raise ValueError("Not enough transitions to sample")

        transitions = []
        for _ in range(batch_size):
            start_idx = random.randint(0, len(self) - self.n_step*stride - 1)
            end_idx = start_idx + self.n_step*stride
            samples = self.buffer[start_idx:end_idx:stride]

            state, _, _, _, _ = samples[0]
            _, _, _, next_state, done = samples[-1]

            reward = 0
            for i in range(self.n_step):
                _, action, r, _, _ = samples[i*stride]
                reward += r * pow(0.99, i)

            transitions.append((state, action, reward, next_state, done))

        states, actions, rewards, next_states, dones = zip(*transitions)

        return states, actions, rewards, next_states, dones

    def save(self, file_name):
        with open(file_name, "wb") as f:
            pickle.dump(self.buffer, f)

    def load(self, file_name):
        with open(file_name, "rb") as f:
            self.buffer = pickle.load(f)
