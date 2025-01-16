import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np


class OpinionNet(nn.Module):
    def __init__(
        self,
        nr_agents,
        nr_budgets=2,
        lin_hidden_out_size=64,
    ):
        super(OpinionNet, self).__init__()

        self.nr_agents = nr_agents  # Input size (number of nodes in network)
        self.nr_budgets = nr_budgets
        self.lin_hidden_out_size = lin_hidden_out_size

        # Fully connected layers
        self.fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "lin1",
                        nn.Linear(self.nr_agents, self.lin_hidden_out_size),
                    ),
                    ("relu1", nn.ReLU()),
                    (
                        "lin2",
                        nn.Linear(self.lin_hidden_out_size, self.lin_hidden_out_size),
                    ),
                    ("relu2", nn.ReLU()),
                ]
            )
        )

        # Predict \( A_j \) and \( b_j \) for each discrete \( \beta \)
        # \( A \): Ensure positive definiteness by outputting positive diagonal elements
        self.predict_A_b = nn.Sequential(
            OrderedDict(
                [
                    (
                        "lin3",
                        nn.Linear(
                            self.lin_hidden_out_size, nr_budgets + 2 * self.nr_agents
                        ),
                    ),  # Predict both A_j (diagonal) and b_j
                ]
            )
        )

    def forward(self, x):
        """
        Forward pass for the network.

        Args:
            x (torch.Tensor): Input state features (current opinions of nodes).

        Returns:
            w_star (torch.Tensor): Optimal weights \( w^* \).
            max_q (torch.Tensor): Maximum Q-value for the given \( x \) and \( \beta \).
            A_diag (torch.Tensor): Predicted positive definite diagonal elements of \( A \).
            b (torch.Tensor): Predicted bias vector \( b \).
        """
        # Process input through fully connected layers
        features = self.fc(x)

        # Predict \( A_j \) and \( b_j \)
        A_b_J = self.predict_A_b(features)  # Shape: (batch_size, 2 * num_actions)

        # Split predictions into A (diagonal elements) and b
        A_diag = torch.exp(
            A_b_J[:, : self.nr_agents - self.nr_budgets]
        )  # Ensure positive diagonal
        b = A_b_J[
            :, self.nr_agents : -self.nr_budgets
        ]  # Shape: (batch_size, num_actions)
        Q = A_b_J[-self.nr_budgets :]

        # Compute \( w^* = - A^{-1} b \)
        # A is diagonal, so \( A^{-1} \) is simply the reciprocal of the diagonal
        A_inv = 1.0 / A_diag
        w_star = -A_inv * b  # Element-wise multiplication

        q_vals = (
            Q
            + 1 / 2 * np.transpose(w_star) * A_diag * w_star
            + np.transpose(b) * w_star
        )

        return w_star, q_vals, A_diag, b
