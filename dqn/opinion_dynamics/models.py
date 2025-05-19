import torch
import torch.nn as nn
from collections import OrderedDict
import numpy as np
import torch.nn.functional as F


class OpinionNet(nn.Module):
    def __init__(self, nr_agents, nr_betas=2, lin_hidden_size=64):
        super(OpinionNet, self).__init__()

        self.nr_agents = nr_agents
        self.nr_betas = nr_betas  # Number of \(\beta\) grid points
        self.lin_hidden_size = lin_hidden_size

        # Fully connected layers for state features
        self.fc = nn.Sequential(
            nn.Linear(self.nr_agents, self.lin_hidden_size),
            nn.ReLU(),
            nn.Linear(self.lin_hidden_size, self.lin_hidden_size),
            nn.ReLU(),
        )

        # Predict \( q(x, \beta; \theta), A, b \) for all \(\beta\) grid points
        self.predict_A_b_c = nn.Linear(
            self.lin_hidden_size, self.nr_betas * (2 * self.nr_agents + 1)
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
        b = A_b_c_net[:, :, self.nr_agents + 1 :]  # (B, J, N)
        c = A_b_c_net[:, :, 0]  # (B, J)

        output = {"A_diag": A_diag, "b": b, "c": c}

        if w is not None:
            w = w.unsqueeze(1)  # (B, 1, N)
            q = self.compute_q_values(w, A_diag, b, c)  # (B, J)
            output["q"] = q

        return output

    @staticmethod
    def compute_w_star(A_diag, b):
        """
        Compute the optimal weight allocation \( w^* \) given A_diag and b.

        Args:
            A_diag (torch.Tensor): Diagonal elements of A, shape (batch_size, nr_betas, nr_agents).
            b (torch.Tensor): Bias term, shape (batch_size, nr_betas, nr_agents).

        Returns:
            torch.Tensor: Optimal weight allocation \( w^* \), shape (batch_size, nr_betas, nr_agents).
        """
        A_inv = 1.0 / A_diag  # Inverse of diagonal A
        w_star = A_inv * b  # Compute raw w*

        return w_star

    @staticmethod
    def compute_q_values(w_star, A_diag, b, c):
        """
        Compute Q-values using the optimal weight allocation w*.

        Args:
            w_star (torch.Tensor): Optimal weight allocation, shape (batch_size, nr_betas, nr_agents).
            A_diag (torch.Tensor): Diagonal elements of A, shape (batch_size, nr_betas, nr_agents).
            b (torch.Tensor): Bias term, shape (batch_size, nr_betas, nr_agents).
            c (torch.Tensor): Free term, shape (batch_size, nr_betas).

        Returns:
            torch.Tensor: Computed Q-values, shape (batch_size, nr_betas).
        """
        assert (
            w_star.shape == A_diag.shape == b.shape
        ), f"Shape mismatch: w_star={w_star.shape}, A_diag={A_diag.shape}, b={b.shape}"

        quadratic_term = (w_star * (A_diag * w_star)) / 2
        linear_term = w_star * b
        q_values = c.unsqueeze(2) - quadratic_term + linear_term  # (B, J, N)

        q_values = q_values.sum(dim=2)
        
        return q_values

    @staticmethod
    def apply_action_noise(w, noise_amplitude):
        """
        Add noise to the weight vector w.

        Args:
            w (torch.Tensor): The optimal weight vector computed from the network.
            noise_amplitude (float): The scale of the Gaussian noise.

        Returns:
            torch.Tensor: The noisy, normalized weight vector.
        """
        noise = torch.randn_like(w) * noise_amplitude
        noisy_w = w + noise

        return noisy_w

    @staticmethod
    def compute_action_from_w(w: torch.Tensor, beta: torch.Tensor):
        """
        Compute the action u from allocation weights w and beta values.

        Args:
            w (torch.Tensor): Allocation weights, shape (batch_size, num_agents)
            beta (torch.Tensor): Per-agent beta values, shape (batch_size, num_agents)
            max_u (float): Maximum action value

        Returns:
            torch.Tensor: Actions u, shape (batch_size, num_agents), capped at max_u per agent
        """
        # Softmax also normalizes
        w = F.softmax(w, dim=-1)
        u = w * beta 
        return u
