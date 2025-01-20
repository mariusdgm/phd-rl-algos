import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


class OpinionNet(nn.Module):
    def __init__(self, nr_agents, nr_betas=2, lin_hidden_out_size=64):
        super(OpinionNet, self).__init__()

        self.nr_agents = nr_agents
        self.nr_betas = nr_betas  # Number of \(\beta\) grid points
        self.lin_hidden_out_size = lin_hidden_out_size

        # Fully connected layers for state features
        self.fc = nn.Sequential(
            nn.Linear(self.nr_agents, self.lin_hidden_out_size),
            nn.ReLU(),
            nn.Linear(self.lin_hidden_out_size, self.lin_hidden_out_size),
            nn.ReLU(),
        )

        # Predict \( q(x, \beta; \theta), A, b \) for all \(\beta\) grid points
        self.predict_A_b_q = nn.Linear(
            self.lin_hidden_out_size, self.nr_betas * (2 * self.nr_agents + 1)
        )

    def forward(self, x):
        """
        Forward pass for the network.

        Args:
            x (torch.Tensor): Input state features.

        Returns:
            q_vals (torch.Tensor): Q-values for all \(\beta\) grid points.
            A_diag (torch.Tensor): Predicted positive definite diagonal elements of \( A \) for each \(\beta\).
            b (torch.Tensor): Predicted bias vectors \( b \) for each \(\beta\).
        """
        # Process input through fully connected layers
        features = self.fc(x)

        # Predict \( q, A, b \)
        A_b_q = self.predict_A_b_q(features)  # Shape: (batch_size, nr_betas * (2 * nr_agents + 1))
        A_b_q = A_b_q.view(-1, self.nr_betas, 2 * self.nr_agents + 1)

        # Extract Q-values, A, and b
        q_vals = A_b_q[:, :, 0]  # Q-values for all \(\beta\)
        A_diag = torch.exp(A_b_q[:, :, 1 : self.nr_agents + 1])  # Positive definite diagonal
        b = A_b_q[:, :, self.nr_agents + 1 :]  # Bias vectors

        return q_vals, A_diag, b

