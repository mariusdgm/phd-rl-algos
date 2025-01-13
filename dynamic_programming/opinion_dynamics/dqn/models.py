import torch.autograd as autograd
import torch.nn as nn
import torch

from collections import OrderedDict


class OpinionNet(nn.Module):
    def __init__(
        self,
        in_features,
        in_channels,
        num_actions,
        conv_hidden_out_size=16,
        lin_hidden_out_size=64,
    ):
        super(OpinionNet, self).__init__()

        self.in_features = in_features
        self.in_channels = in_channels
        self.num_actions = num_actions  # Number of agents/actions

        self.conv_hidden_out_size = conv_hidden_out_size
        self.lin_hidden_out_size = lin_hidden_out_size

        # Convolutional feature extractor
        self.features = nn.Sequential(
            OrderedDict(
                [
                    (
                        "conv1",
                        nn.Conv2d(
                            self.in_channels,
                            self.conv_hidden_out_size,
                            kernel_size=3,
                            stride=1,
                        ),
                    ),
                    ("relu1", nn.ReLU()),
                    (
                        "conv2",
                        nn.Conv2d(
                            self.conv_hidden_out_size,
                            self.conv_hidden_out_size,
                            kernel_size=3,
                            stride=1,
                        ),
                    ),
                    ("relu2", nn.ReLU()),
                ]
            )
        )

        # Fully connected layers to process flattened features
        self.fc = nn.Sequential(
            OrderedDict(
                [
                    (
                        "lin1",
                        nn.Linear(self.size_linear_unit(), self.lin_hidden_out_size),
                    ),
                    ("relu3", nn.ReLU()),
                ]
            )
        )

        # Predict \( A_j \) and \( b_j \) for each discrete \( \beta_j \)
        # \( A \): Ensure positive definiteness by outputting positive diagonal elements
        self.predict_A_b = nn.Sequential(
            OrderedDict(
                [
                    (
                        "lin2",
                        nn.Linear(self.lin_hidden_out_size, 2 * self.num_actions),
                    ),  # Predict both A_j (diagonal) and b_j
                ]
            )
        )

    def size_linear_unit(self):
        # Get the flattened size of the convolutional features
        return self.features(torch.zeros(1, *self.in_features)).view(1, -1).size(1)

    def forward(self, x, beta):
        """
        Forward pass for the network.

        Args:
            x (torch.Tensor): Input state features.
            beta (torch.Tensor): Discrete input representing \( \beta \).

        Returns:
            w_star (torch.Tensor): Optimal weights \( w^* \).
            A (torch.Tensor): Predicted positive definite matrix \( A \).
            b (torch.Tensor): Predicted bias vector \( b \).
        """
        # Extract convolutional features
        features = self.features(x)
        features = features.view(features.size(0), -1)  # Flatten
        features = self.fc(features)  # Pass through fully connected layers

        # Predict \( A_j \) and \( b_j \) for each \( \beta \)
        A_b = self.predict_A_b(features)  # Shape: (batch_size, 2 * num_actions)

        # Split predictions into A (diagonal elements) and b
        A_diag = torch.exp(A_b[:, : self.num_actions])  # Ensure positive diagonal
        b = A_b[:, self.num_actions :]  # Shape: (batch_size, num_actions)

        # Compute \( w^* = - A^{-1} b \)
        # A is diagonal, so \( A^{-1} \) is simply the reciprocal of the diagonal
        A_inv = 1.0 / A_diag
        w_star = -A_inv * b  # Element-wise multiplication

        return w_star, A_diag, b
