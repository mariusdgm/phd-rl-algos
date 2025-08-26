import torch
import torch.nn as nn
import torch.nn.functional as F


class OpinionNetFixedW(nn.Module):
    def __init__(self, nr_agents, nr_betas=2, lin_hidden_size=64):
        super(OpinionNetFixedW, self).__init__()

        self.nr_agents = nr_agents
        self.nr_betas = nr_betas  
        self.lin_hidden_size = lin_hidden_size

        self.fc = nn.Sequential(
            nn.Linear(self.nr_agents, self.lin_hidden_size),
            nn.ReLU(),
            nn.Linear(self.lin_hidden_size, self.lin_hidden_size),
            nn.ReLU(),
        )

        self.predict_c = nn.Linear(self.lin_hidden_size, self.nr_betas)

        with torch.no_grad():
            self.predict_c.bias.zero_()

    def forward(self, x):
        """
        Args:
            x (torch.Tensor): Input state, shape (B, N)
        Returns:
            dict with:
                - c (torch.Tensor): shape (B, J)
        """
        features = self.fc(x)
        c = self.predict_c(features)  # (B, J)
        return {"c": c}
    
class OpinionNetCommonAB(nn.Module):
    def __init__(self, nr_agents, nr_betas=2, lin_hidden_size=64):
        super(OpinionNetCommonAB, self).__init__()

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

        self.predict_shared_A_b = nn.Linear(self.lin_hidden_size, 2 * self.nr_agents)
        self.predict_c = nn.Linear(self.lin_hidden_size, self.nr_betas)

        # Optional: initialize c_j biases to zero
        with torch.no_grad():
            self.predict_c.bias.zero_()

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

        # Shared A and b
        A_b_shared = self.predict_shared_A_b(features)  # (B, 2N)
        A_diag = F.softplus(A_b_shared[:, :self.nr_agents]) + 1e-6  # (B, N)
        b = A_b_shared[:, self.nr_agents:]  # (B, N)

        # Repeat across betas
        A_diag = A_diag.unsqueeze(1).repeat(1, self.nr_betas, 1)  # (B, J, N)
        b = b.unsqueeze(1).repeat(1, self.nr_betas, 1)            # (B, J, N)

        # Independent c values for each beta
        c = self.predict_c(features)  # (B, J)

        # print(f"A_diag shape: {A_diag.shape}, b shape: {b.shape}, c shape: {c.shape}")
        
        output = {"A_diag": A_diag, "b": b, "c": c}

        # We are braodcasting w over all J levels here
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
        w = A_inv * b  # Compute raw w*

        return w

    @staticmethod
    def compute_q_values(w, A_diag, b, c):
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
            w.shape == A_diag.shape == b.shape
        ), f"Shape mismatch: w_star={w.shape}, A_diag={A_diag.shape}, b={b.shape}"

        # Quadratic term: w^T A w = sum_i A_i * w_i^2
        quadratic_term = 0.5 * (A_diag * w.pow(2)).sum(dim=2)  # shape (B, J)

        # Linear term: b^T w
        linear_term = (b * w).sum(dim=2)  # shape (B, J)

        # Total Q-value
        q_values = c - quadratic_term + linear_term  # shape (B, J)
        
        assert q_values.shape == (w.shape[0], w.shape[1]), \
            f"Expected shape (B, J), got {q_values.shape}"
        
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
        A_b_c_net = A_b_c_net.reshape(-1, self.nr_betas, 2 * self.nr_agents + 1)

        A_diag = F.softplus(A_b_c_net[:, :, 1 : self.nr_agents + 1]) + 1e-6  # (B, J, N)
        b = A_b_c_net[:, :, self.nr_agents + 1 :]  # (B, J, N)
        c = A_b_c_net[:, :, 0]  # (B, J)

        # print(f"A_diag shape: {A_diag.shape}, b shape: {b.shape}, c shape: {c.shape}")
        
        output = {"A_diag": A_diag, "b": b, "c": c}

        # We are braodcasting w over all J levels here
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
        w = A_inv * b  # Compute raw w*

        return w

    @staticmethod
    def compute_q_values(w, A_diag, b, c):
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
            w.shape == A_diag.shape == b.shape
        ), f"Shape mismatch: w_star={w.shape}, A_diag={A_diag.shape}, b={b.shape}"

        # Quadratic term: w^T A w = sum_i A_i * w_i^2
        quadratic_term = 0.5 * (A_diag * w.pow(2)).sum(dim=2)  # shape (B, J)

        # Linear term: b^T w
        linear_term = (b * w).sum(dim=2)  # shape (B, J)

        # Total Q-value
        q_values = c - quadratic_term + linear_term  # shape (B, J)
        
        assert q_values.shape == (w.shape[0], w.shape[1]), \
            f"Expected shape (B, J), got {q_values.shape}"
        
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
