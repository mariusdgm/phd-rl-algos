import torch
import torch.nn as nn
import torch.nn.functional as F

from typing import Optional, Dict

  
class OpinionNetCommonAB(nn.Module):
    def __init__(
        self,
        nr_agents: int,
        nr_betas: int = 2,
        lin_hidden_size: int = 64,
        c_tanh_scale: Optional[float] = None,  
        softplus_beta: float = 1.0,            # softness for A
        wstar_eps: float = 1e-6,               # safety in w* division
        return_w_star_default: bool = False,   # default for forward()
    ):
        super().__init__()
        self.nr_agents = nr_agents
        self.nr_betas = nr_betas
        self.lin_hidden_size = lin_hidden_size
        self.c_tanh_scale = c_tanh_scale
        self.softplus_beta = softplus_beta
        self.wstar_eps = wstar_eps
        self.return_w_star_default = return_w_star_default

        # Trunk
        self.fc = nn.Sequential(
            nn.Linear(self.nr_agents, self.lin_hidden_size),
            nn.ReLU(),
            nn.Linear(self.lin_hidden_size, self.lin_hidden_size),
            nn.ReLU(),
        )

        # Shared A and b across betas; independent c per beta
        self.predict_shared_A_b = nn.Linear(self.lin_hidden_size, 2 * self.nr_agents)
        self.predict_c = nn.Linear(self.lin_hidden_size, self.nr_betas)

        # ---- init for stability ----
        with torch.no_grad():
            # Kaiming for trunk
            for m in self.fc:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity='relu')
                    nn.init.zeros_(m.bias)

            # Kaiming for heads
            nn.init.kaiming_uniform_(self.predict_shared_A_b.weight, a=0.0, nonlinearity='linear')
            nn.init.kaiming_uniform_(self.predict_c.weight, a=0.0, nonlinearity='linear')

            # Biases: A ~ 1.0 after softplus, b = 0, c = 0
            def inv_softplus(y: float, beta: float = 1.0):
                import math
                return math.log(math.expm1(beta * y)) / beta

            a_bias = inv_softplus(1.0, beta=self.softplus_beta)
            # Layout: [A(1..N) | b(1..N)]
            ab_bias = torch.zeros(2 * self.nr_agents)
            ab_bias[: self.nr_agents] = a_bias
            self.predict_shared_A_b.bias.copy_(ab_bias)

            self.predict_c.bias.zero_()

    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None,
                return_w_star: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, N) state features
            w: optional actions (B, N); if given, returns q for each beta
            return_w_star: include w_star in outputs (defaults to class setting)

        Returns:
            dict with:
                A_diag: (B, J, N) > 0
                b:      (B, J, N)
                c:      (B, J)
                q:      (B, J)      if w is provided
                w_star: (B, J, N)   if requested
        """
        if return_w_star is None:
            return_w_star = self.return_w_star_default

        B = x.shape[0]
        features = self.fc(x)  # (B, H)

        # Shared A,b (B, 2N) -> split
        A_b_shared = self.predict_shared_A_b(features)
        A_raw = A_b_shared[:, : self.nr_agents]                 # (B, N)
        b_shared = A_b_shared[:, self.nr_agents :]              # (B, N)

        # Positive A diagonal + tiny eps
        A_diag_single = F.softplus(A_raw, beta=self.softplus_beta) + 1e-6  # (B, N)

        # Repeat across betas
        A_diag = A_diag_single.unsqueeze(1).expand(B, self.nr_betas, self.nr_agents)  # (B, J, N)
        b = b_shared.unsqueeze(1).expand(B, self.nr_betas, self.nr_agents)            # (B, J, N)

        # Independent c per beta
        c = self.predict_c(features)  # (B, J)
        if self.c_tanh_scale is not None:
            c = torch.tanh(c) * self.c_tanh_scale  # optional clamp

        out = {"A_diag": A_diag, "b": b, "c": c}

        # Optionally compute w*
        if return_w_star or (w is not None):
            w_star = self.compute_w_star(A_diag, b, eps=self.wstar_eps)
            if return_w_star:
                out["w_star"] = w_star

        # Optionally compute q(w)
        if w is not None:
            wJ = w.unsqueeze(1).expand(B, self.nr_betas, self.nr_agents)  # (B, J, N)
            q = self.compute_q_values(wJ, A_diag, b, c)
            out["q"] = q

        return out

    @staticmethod
    def compute_w_star(A_diag: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        w* = b / (A_diag + eps)
        Shapes: A_diag, b: (B, J, N) -> (B, J, N)
        """
        return b / (A_diag + eps)

    @staticmethod
    def compute_q_values(w: torch.Tensor, A_diag: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for given w.
        Args:
            w:      (B, J, N)
            A_diag: (B, J, N)
            b:      (B, J, N)
            c:      (B, J)
        Returns:
            q:      (B, J)
        """
        assert w.shape == A_diag.shape == b.shape, \
            f"Shape mismatch: w={w.shape}, A_diag={A_diag.shape}, b={b.shape}"
        quad = 0.5 * (A_diag * w.pow(2)).sum(dim=2)  # (B, J)
        lin  = (b * w).sum(dim=2)                    # (B, J)
        return c - quad + lin

    @staticmethod
    def apply_action_noise(w: torch.Tensor, noise_amplitude: float) -> torch.Tensor:
        """Add Gaussian noise to w."""
        return w + torch.randn_like(w) * noise_amplitude

    @staticmethod
    def compute_action_from_w(w: torch.Tensor, beta: torch.Tensor, max_u: Optional[float] = None) -> torch.Tensor:
        """
        u_i = softmax(w)_i * beta_i; optional per-element cap via max_u.
        Args:
            w:    (B, N)
            beta: (B, N) or (1, N)
        """
        w_norm = F.softmax(w, dim=-1)
        u = w_norm * beta
        if max_u is not None:
            u = torch.clamp(u, max=max_u)
        return u
    
class OpinionNet(nn.Module):
    def __init__(
        self, 
        nr_agents: int,
        nr_betas: int = 2,
        lin_hidden_size: int = 64,
        c_tanh_scale: Optional[float] = None,  # keep None to match your winning setup
        softplus_beta: float = 1.0,            # softness for A; leave default
        wstar_eps: float = 1e-6,               # numerical safety for w* division
        return_w_star_default: bool = False,   # default behavior for forward()
        ):
        super().__init__()
        self.nr_agents = nr_agents
        self.nr_betas = nr_betas
        self.lin_hidden_size = lin_hidden_size
        self.c_tanh_scale = c_tanh_scale
        self.softplus_beta = softplus_beta
        self.wstar_eps = wstar_eps
        self.return_w_star_default = return_w_star_default

        # Feature trunk
        self.fc = nn.Sequential(
            nn.Linear(self.nr_agents, self.lin_hidden_size),
            nn.ReLU(),
            nn.Linear(self.lin_hidden_size, self.lin_hidden_size),
            nn.ReLU(),
        )

        # Predict [c | A_diag(n) | b(n)] for each beta head
        self.predict_A_b_c = nn.Linear(
            self.lin_hidden_size, self.nr_betas * (2 * self.nr_agents + 1)
        )

        # ---- Initialization for stability (keeps behavior similar but safer) ----
        with torch.no_grad():
            # Kaiming for weights
            for m in self.fc:
                if isinstance(m, nn.Linear):
                    nn.init.kaiming_uniform_(m.weight, a=0.0, nonlinearity='relu')
                    nn.init.zeros_(m.bias)

            nn.init.kaiming_uniform_(self.predict_A_b_c.weight, a=0.0, nonlinearity='linear')

            # Bias layout per beta: [c, A(1..N), b(1..N)]
            full_bias = self.predict_A_b_c.bias  # (nr_betas * (2N+1),)
            block = 2 * self.nr_agents + 1
            # Softplus^{-1}(1.0) for A so softplus(bias) ~ 1 initially
            def inv_softplus(y: float, beta: float = 1.0):
                # inverse of softplus: x = log(exp(beta*y) - 1)/beta
                import math
                return math.log(math.expm1(beta * y)) / beta
            a_bias = inv_softplus(1.0, beta=self.softplus_beta)

            for j in range(self.nr_betas):
                off = j * block
                full_bias[off + 0] = 0.0                         # c_j = 0
                full_bias[off + 1 : off + 1 + self.nr_agents] = a_bias  # A_diag ~ 1.0
                full_bias[off + 1 + self.nr_agents : off + block] = 0.0  # b = 0

    def forward(self, x: torch.Tensor, w: Optional[torch.Tensor] = None,
                return_w_star: Optional[bool] = None) -> Dict[str, torch.Tensor]:
        """
        Args:
            x: (B, N) state features
            w: optional actions (B, N); if given, returns q for each beta
            return_w_star: if True, include w_star in output (defaults to class default)

        Returns dict with:
            A_diag: (B, J, N)  > 0
            b:      (B, J, N)
            c:      (B, J)
            q:      (B, J)     only if w is provided
            w_star: (B, J, N)  only if requested
        """
        if return_w_star is None:
            return_w_star = self.return_w_star_default

        B = x.shape[0]
        features = self.fc(x)  # (B, H)

        # Raw head: (B, J*(2N+1)) -> (B, J, 2N+1)
        A_b_c_net = self.predict_A_b_c(features).reshape(
            -1, self.nr_betas, 2 * self.nr_agents + 1
        )

        # Parse and activate
        c = A_b_c_net[:, :, 0]  # (B, J)
        A_raw = A_b_c_net[:, :, 1 : self.nr_agents + 1]  # (B, J, N)
        b = A_b_c_net[:, :, self.nr_agents + 1 :]        # (B, J, N)

        # Positive diagonal via softplus
        # softplus(x; beta) = 1/beta * log(1 + exp(beta*x))
        A_diag = F.softplus(A_raw, beta=self.softplus_beta) + 1e-6  # (B, J, N)

        # Optional stability clamp on c (OFF by default to keep your current behavior)
        if self.c_tanh_scale is not None:
            c = torch.tanh(c) * self.c_tanh_scale

        output = {"A_diag": A_diag, "b": b, "c": c}

        # Optionally emit q(w) and/or w*
        if return_w_star or (w is not None):
            w_star = self.compute_w_star(A_diag, b, eps=self.wstar_eps)
            if return_w_star:
                output["w_star"] = w_star

        if w is not None:
            # Broadcast w -> (B, J, N)
            wJ = w.unsqueeze(1).expand(B, self.nr_betas, self.nr_agents)
            q = self.compute_q_values(wJ, A_diag, b, c)
            output["q"] = q

        return output

    @staticmethod
    def compute_w_star(A_diag: torch.Tensor, b: torch.Tensor, eps: float = 1e-6) -> torch.Tensor:
        """
        w* = b / (A_diag + eps)
        Shapes:
            A_diag: (B, J, N), b: (B, J, N) -> w*: (B, J, N)
        """
        return b / (A_diag + eps)

    @staticmethod
    def compute_q_values(w: torch.Tensor, A_diag: torch.Tensor, b: torch.Tensor, c: torch.Tensor) -> torch.Tensor:
        """
        Compute Q-values for given w.
        Args:
            w:      (B, J, N)
            A_diag: (B, J, N)
            b:      (B, J, N)
            c:      (B, J)
        Returns:
            q:      (B, J)
        """
        assert w.shape == A_diag.shape == b.shape, \
            f"Shape mismatch: w={w.shape}, A_diag={A_diag.shape}, b={b.shape}"
        quad = 0.5 * (A_diag * w.pow(2)).sum(dim=2)  # (B, J)
        lin  = (b * w).sum(dim=2)                    # (B, J)
        q = c - quad + lin                           # (B, J)
        return q

    @staticmethod
    def apply_action_noise(w: torch.Tensor, noise_amplitude: float) -> torch.Tensor:
        """
        Add Gaussian noise to w (no normalization here).
        """
        noise = torch.randn_like(w) * noise_amplitude
        return w + noise

    @staticmethod
    def compute_action_from_w(w: torch.Tensor, beta: torch.Tensor, max_u: Optional[float] = None) -> torch.Tensor:
        """
        Convert allocation weights to actions:
            u_i = softmax(w)_i * beta_i
        Optionally cap each u by max_u if provided.
        Args:
            w:    (B, N)
            beta: (B, N) or (1, N)
        Returns:
            u:    (B, N)
        """
        w_norm = F.softmax(w, dim=-1)
        u = w_norm * beta
        if max_u is not None:
            u = torch.clamp(u, max=max_u)
        return u
