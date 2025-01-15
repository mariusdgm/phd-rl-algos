import torch
import torch.nn as nn
from collections import OrderedDict

class OpinionNet(nn.Module):
    def __init__(
        self,
        in_features,
        num_actions,
        lin_hidden_out_size=64,
    ):
        super(OpinionNet, self).__init__()

        self.in_features = in_features  # Input size (number of opinions)
        self.num_actions = num_actions  # Number of agents/actions
        self.lin_hidden_out_size = lin_hidden_out_size

        # Fully connected layers to process the input vector
        self.fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "lin1",
                        nn.Linear(self.in_features, self.lin_hidden_out_size),
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
                        nn.Linear(self.lin_hidden_out_size, 2 * self.num_actions),
                    ),  # Predict both A_j (diagonal) and b_j
                ]
            )
        )

    def forward(self, x, beta):
        """
        Forward pass for the network.

        Args:
            x (torch.Tensor): Input state features (current opinions of nodes).
            beta (torch.Tensor): Discrete input representing \( \beta \).

        Returns:
            w_star (torch.Tensor): Optimal weights \( w^* \).
            max_q (torch.Tensor): Maximum Q-value for the given \( x \) and \( \beta \).
            A_diag (torch.Tensor): Predicted positive definite diagonal elements of \( A \).
            b (torch.Tensor): Predicted bias vector \( b \).
        """
        # Process input through fully connected layers
        features = self.fc(x)

        # Predict \( A_j \) and \( b_j \)
        A_b = self.predict_A_b(features)  # Shape: (batch_size, 2 * num_actions)

        # Split predictions into A (diagonal elements) and b
        A_diag = torch.exp(A_b[:, : self.num_actions])  # Ensure positive diagonal
        b = A_b[:, self.num_actions :]  # Shape: (batch_size, num_actions)

        # Compute \( w^* = - A^{-1} b \)
        # A is diagonal, so \( A^{-1} \) is simply the reciprocal of the diagonal
        A_inv = 1.0 / A_diag
        w_star = -A_inv * b  # Element-wise multiplication

        # Compute \( Q(x, \beta, w^*) \)
        quadratic_term = 0.5 * (w_star**2 * A_diag).sum(dim=1, keepdim=True)  # w^T A w
        linear_term = (b * w_star).sum(dim=1, keepdim=True)  # b^T w
        max_q = quadratic_term + linear_term

        return w_star, max_q, A_diag, b
