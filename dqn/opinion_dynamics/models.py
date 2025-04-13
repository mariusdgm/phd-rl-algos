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
        # TODO: Can we apply the exp here somehow?
        
        # TODO: need layer for Q values

        with torch.no_grad():
            full_bias = self.predict_A_b_c.bias  # shape: (nr_betas * (2*n + 1),)
            block_size = 2 * self.nr_agents + 1
            for j in range(self.nr_betas):
                full_bias[j * block_size] = 0.0  # Initialize c_j to 0.

    def forward(self, x):
        """
        Forward pass for the network.

        Args:
            x (torch.Tensor): Input state features.

        Returns:
            A_diag (torch.Tensor): Predicted positive definite diagonal elements of \( A \) for each \(\beta\).
            b (torch.Tensor): Predicted bias vectors \( b \) for each \(\beta\).
            c
        """
        # Process input through fully connected layers
        features = self.fc(x)

        # Predict \( free\_term, A, b \)
        A_b_c_net = self.predict_A_b_c(features)  # Shape: (batch_size, nr_betas * (2 * nr_agents + 1))
        A_b_c_net = A_b_c_net.view(-1, self.nr_betas, 2 * self.nr_agents + 1)

        A_diag = F.softplus(A_b_c_net[:, :, 1 : self.nr_agents + 1]) + 1e-6  # Positive definite diagonal
        b = A_b_c_net[:, :, self.nr_agents + 1 :]  # Bias vectors
        c = A_b_c_net[:, :, 0]

        # TODO: add w as input, compute Q as output in forward too
        
        # q = Net(x, w)
        # a, b, c = Net.get_a_b_c()

        return A_diag, b, c



