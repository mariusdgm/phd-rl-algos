import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


class OpinionNet(nn.Module):
    def __init__(
        self,
        nr_agents,
        nr_budgets=2,  # Number of discrete \(\beta\) values
        lin_hidden_out_size=64,
    ):
        super(OpinionNet, self).__init__()

        self.nr_agents = nr_agents
        self.nr_budgets = nr_budgets
        self.lin_hidden_out_size = lin_hidden_out_size

        # Fully connected layers for state features
        self.fc = nn.Sequential(
            OrderedDict(
                [
                    ("lin1", nn.Linear(self.nr_agents, self.lin_hidden_out_size)),
                    ("relu1", nn.ReLU()),
                    (
                        "lin2",
                        nn.Linear(self.lin_hidden_out_size, self.lin_hidden_out_size),
                    ),
                    ("relu2", nn.ReLU()),
                ]
            )
        )

        # Additional embedding for \(\beta\)
        self.beta_embedding = nn.Embedding(self.nr_budgets, self.lin_hidden_out_size)

        # Predict \( q, A, b \)
        self.predict_A_b_q = nn.Linear(self.lin_hidden_out_size, 2 * self.nr_agents + 1)

    def forward(self, x, beta_idx):
        """
        Forward pass for the network.

        Args:
            x (torch.Tensor): Input state features (current opinions of nodes).
            beta_idx (torch.Tensor): Indices representing discrete \(\beta_j\).

        Returns:
            w_star (torch.Tensor): Optimal weights \( w^* \).
            q_vals (torch.Tensor): Q-values for all \(\beta\).
            A_diag (torch.Tensor): Predicted positive definite diagonal elements of \( A \).
            b (torch.Tensor): Predicted bias vector \( b \).
        """
        # Process input through fully connected layers
        features = self.fc(x)

        # Embed \(\beta\) and combine with features
        beta_features = self.beta_embedding(
            beta_idx
        )  # Shape: (batch_size, lin_hidden_out_size)
        combined_features = features + beta_features

        # Predict \( q, A, b \)
        A_b_q = self.predict_A_b_q(
            combined_features
        )  # Shape: (batch_size, 2 * nr_agents + 1)
        q_vals = A_b_q[:, 0]  # Q-value
        A_diag = torch.exp(
            A_b_q[:, 1 : self.nr_agents + 1]
        )  # Ensure positive definiteness
        b = A_b_q[:, self.nr_agents + 1 :]  # Bias vector

        # Compute \( w^* \)
        A_inv = 1.0 / A_diag
        w_star = -A_inv * b  # Element-wise multiplication

        # Full Q-value
        q_val_full = (
            q_vals
            + 0.5 * torch.sum(w_star**2 * A_diag, dim=1)
            + torch.sum(b * w_star, dim=1)
        )

        return w_star, q_val_full, A_diag, b
