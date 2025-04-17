import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F


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
        self.predict_A_b_c = nn.Linear(
            self.lin_hidden_out_size, self.nr_betas * (2 * self.nr_agents + 1)
        )
        
        with torch.no_grad():
            full_bias = self.predict_A_b_c.bias  # shape: (nr_betas * (2*n + 1),)
            block_size = 2 * self.nr_agents + 1
            for j in range(self.nr_betas):
                full_bias[j * block_size] = 0.0  # Initialize c_j to 0.

    def forward(self, x, w=None):
        """
        Forward pass for the network.

        Args:
            x (torch.Tensor): Input state features, shape (B, N)
            w (torch.Tensor, optional): Optional action vector for each sample, shape (B, N)

        Returns:
            dict with:
                - A_diag (torch.Tensor): shape (B, J, N), positive definite diagonals of A
                - b (torch.Tensor): shape (B, J, N), linear coefficients
                - c (torch.Tensor): shape (B, J), bias term
                - q (torch.Tensor): shape (B, J), Q-values (only if w is provided)
        """
        features = self.fc(x)

        A_b_c_net = self.predict_A_b_c(features)
        A_b_c_net = A_b_c_net.view(-1, self.nr_betas, 2 * self.nr_agents + 1)

        A_diag = F.softplus(A_b_c_net[:, :, 1 : self.nr_agents + 1]) + 1e-6  # (B, J, N)
        b = A_b_c_net[:, :, self.nr_agents + 1 :]                            # (B, J, N)
        c = A_b_c_net[:, :, 0]                                              # (B, J)

        output = {
            "A_diag": A_diag,
            "b": b,
            "c": c
        }

        if w is not None:
            w = w.unsqueeze(1)  # (B, 1, N)
            q = -0.5 * torch.sum(A_diag * (w ** 2), dim=2) + torch.sum(b * w, dim=2) + c  # (B, J)
            output["q"] = q

        return output



