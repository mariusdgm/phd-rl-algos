import os, sys


def get_dir_n_levels_up(path, n):
    # Go up n levels from the given path
    for _ in range(n):
        path = os.path.dirname(path)
    return path


proj_root = get_dir_n_levels_up(os.path.abspath(__file__), 2)
sys.path.append(proj_root)


import datetime
import torch
import numpy as np
from collections import defaultdict

from pathlib import Path
from typing import Dict


import torch.optim as optim
import torch.nn.functional as F


from opinion_dynamics.replay_buffer import ReplayBuffer
from opinion_dynamics.models import OpinionNet
from opinion_dynamics.experiments.algos import centrality_based_continuous_control
from opinion_dynamics.utils.my_logging import setup_logger
from opinion_dynamics.utils.generic import replace_keys, seed_everything
from opinion_dynamics.utils.env_setup import EnvironmentFactory


# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"

def initialize_network_to_match_policy(opinion_net, env, available_budget, beta_idx=0):
    """
    One-shot initialization: modify the final linear layer weights
    so that w* computed by the network matches the centrality-based policy.
    Only for beta_idx=0.
    """
    opinion_net.eval()

    # Get the state and the desired action (weights)
    state, _ = env.reset(randomize_opinions=False)
    action, _ = centrality_based_continuous_control(env, available_budget)
    w_target = action / available_budget  # Normalize

    # Forward pass up to the feature layer
    with torch.no_grad():
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)  # (1, N)
        features = opinion_net.fc(state_tensor).squeeze(0)  # shape: (hidden_size,)

        # Fix A_diag = 1.0 → b = -w_target
        A_diag_val = torch.ones(env.num_agents)
        
        # TODO: need to also invert the softmax + normalize
        b_val = -torch.tensor(w_target, dtype=torch.float32)

        # Build the output vector for predict_A_b_c: [c, A_diag, b]
        block = torch.cat([
            torch.tensor([0.0]),        # c
            A_diag_val,                # A_diag
            b_val                      # b
        ])  # shape: (2N + 1,)

        # Inject this block into the layer's weight projection manually
        out_proj = opinion_net.predict_A_b_c
        with torch.no_grad():
            # Zero the weights (simplifies control)
            out_proj.weight.zero_()
            out_proj.bias.zero_()

            # Map this fixed feature to our desired block for beta_idx = 0
            block_size = 2 * env.num_agents + 1
            start = beta_idx * block_size
            end = start + block_size

            out_proj.bias[start:end] = block

    print("OpinionNet initialized to match centrality-based policy for current state.")
    return opinion_net

class AgentDQN:
    def __init__(
        self,
        experiment_output_folder=None,
        experiment_name=None,
        resume_training_path=None,
        save_checkpoints=True,
        logger=None,
        config={},
    ) -> None:
        """A DQN agent implementation.

        Args:

            experiment_output_folder (str, optional): Path to the folder where the training outputs will be saved.
                                                         Defaults to None.
            experiment_name (str, optional): A string describing the experiment being run. Defaults to None.
            resume_training_path (str, optional): Path to the folder where the outputs of a previous training
                                                    session can be found. Defaults to None.
            save_checkpoints (bool, optional): Whether to save the outputs of the training. Defaults to True.
            logger (logger, optional): Necessary Logger instance. Defaults to None.
            config (Dict, optional): Settings of the agent relevant to the models and training.
                                    If none is provided in the input, the agent will automatically build the default settings.
                                    Defaults to {}.
            enable_tensorboard_logging (bool, optional): Specifies if logs should also be made using tensorboard.
                                                        Defaults to True.
        """

        # assign environments
        self.save_checkpoints = save_checkpoints
        self.logger = logger or setup_logger("dqn")

        # set up path names
        self.experiment_output_folder = experiment_output_folder
        self.experiment_name = experiment_name

        self.model_file_folder = (
            "model_checkpoints"  # models will be saved at each epoch
        )
        self.model_checkpoint_file_basename = "mck"

        if self.experiment_output_folder and self.experiment_name:
            self.replay_buffer_file = os.path.join(
                self.experiment_output_folder, f"{self.experiment_name}_replay_buffer"
            )
            self.train_stats_file = os.path.join(
                self.experiment_output_folder, f"{self.experiment_name}_train_stats"
            )

        self.config = config
        if self.config:
            self.config = replace_keys(self.config, "args_", "args")

        self._load_config_settings(self.config)
        self._init_models(self.config)  # init policy, target and optim

        # Set initial values related to training and monitoring
        self.t = 0  # frame nr
        self.episodes = 0  # episode nr
        self.policy_model_update_counter = 0
        
        self.log_stride = 20_000  # change to taste (e.g., 5_000 if debugging)
        
        # Metrics to track training progress
        self._last_entropy = None
        self._last_frac_over_cap = None
        self._last_rel_target_drift = None
        
        self.reset_training_episode_tracker()

        self.training_stats = []
        self.validation_stats = []

        # check that all paths were provided and that the files can be found
        if resume_training_path:
            self.load_training_state(resume_training_path)

    def _should_log(self):
        return (self.t % self.log_stride) == 0 and self.t > 0

    def _make_model_checkpoint_file_path(self, experiment_output_folder, epoch_cnt=0):
        """Dynamically build the path where to save the model checkpoint."""
        return os.path.join(
            experiment_output_folder,
            self.model_file_folder,
            f"{self.model_checkpoint_file_basename}_{epoch_cnt}",
        )

    def load_training_state(self, resume_training_path: str):
        """In order to resume training the following files are needed:
        - ReplayBuffer file
        - Training stats file
        - Model weights file (found as the last checkpoint in the models subfolder)
        Args:
            resume_training_path (str): path to where the files needed to resume training can be found

        Raises:
            FileNotFoundError: Raised if a required file was not found.
        """

        ### build the file paths
        resume_files = {}

        resume_files["replay_buffer_file"] = os.path.join(
            resume_training_path, f"{self.experiment_name}_replay_buffer"
        )
        resume_files["train_stats_file"] = os.path.join(
            resume_training_path, f"{self.experiment_name}_train_stats"
        )

        # check that the file paths exist
        for file in resume_files:
            if not os.path.exists(resume_files[file]):
                self.logger.info(
                    f"Could not find the file {resume_files[file]} for {file} either because a wrong path was given, or because no training was done for this experiment."
                )
                return False

        # read through the stats file to find what was the epoch for the last recorded state
        self.load_training_stats(resume_files["train_stats_file"])
        self.replay_buffer.load(resume_files["replay_buffer_file"])

        epoch_cnt = len(self.training_stats)

        resume_files["checkpoint_model_file"] = self._make_model_checkpoint_file_path(
            resume_training_path, epoch_cnt
        )
        if not os.path.exists(resume_files["checkpoint_model_file"]):
            raise FileNotFoundError(
                f"Could not find the file {resume_files['checkpoint_model_file']} for 'checkpoint_model_file'."
            )

        self.load_models(resume_files["checkpoint_model_file"])

        self.logger.info(
            f"Loaded previous training status from the following files: {str(resume_files)}"
        )

    def _load_config_settings(self, config={}):
        """
        Load the settings from config.
        If config was not provided, then default values are used.
        """
        agent_params = config.get("agent_params", {}).get("args", {})

        # setup training configuration
        self.train_step_cnt = agent_params.get("train_step_cnt", 200_000)
        self.validation_enabled = agent_params.get("validation_enabled", True)
        self.validation_step_cnt = agent_params.get("validation_step_cnt", 100_000)
        self.validation_epsilon = agent_params.get("validation_epsilon", 0.0)

        self.replay_start_size = agent_params.get("replay_start_size", 5_000)

        self.batch_size = agent_params.get("batch_size", 32)
        self.training_freq = agent_params.get("training_freq", 4)
        self.target_model_update_freq = agent_params.get(
            "target_model_update_freq", 100
        )
        self.tau = agent_params.get("target_soft_tau", 0.005)
        self.gamma = agent_params.get("gamma", 0.99)
        # self.loss_function = agent_params.get("loss_fcn", "mse_loss")

        self.action_w_noise_amplitude = agent_params.get(
            "action_w_noise_amplitude", 0.3
        )
        self.betas = agent_params.get("betas", [0, 0.5, 1])

        eps_settings = agent_params.get(
            "epsilon", {"start": 1.0, "end": 0.01, "decay": 250_000}
        )
        self.epsilon_by_frame = self._get_linear_decay_function(
            start=eps_settings["start"],
            end=eps_settings["end"],
            decay=eps_settings["decay"],
            eps_decay_start=self.replay_start_size,
        )

        self._read_and_init_envs()  # sets up in_features etc...

        buffer_settings = config.get(
            "replay_buffer", {"max_size": 100_000, "n_step": 0}
        )
        self.replay_buffer = ReplayBuffer(
            max_size=buffer_settings.get("max_size", 100_000),
            state_dim=self.in_features,
            action_dim=self.train_env.action_space.shape[0],
            n_step=buffer_settings.get("n_step", 0),
            betas=self.betas,
        )

        self.logger.info("Loaded configuration settings.")

    def _get_exp_decay_function(self, start: float, end: float, decay: float):
        return lambda x: end + (start - end) * np.exp(-1.0 * x / decay)

    def _get_linear_decay_function(
        self, start: float, end: float, decay: float, eps_decay_start: float = None
    ):
        """Return a function that enables getting the value of epsilon at step x.

        Args:
            start (float): start value of the epsilon function (x=0)
            end (float): end value of the epsilon function (x=decay)
            decay (float): how many steps to reach the end value
            eps_decay_start(float, optional): after how many frames to actually start decaying,
                                            uses self.replay_start_size by default

        Returns:
            function: function to compute the epsillon based on current frame counter
        """
        if not eps_decay_start:
            eps_decay_start = self.replay_start_size

        return lambda x: max(
            end, min(start, start - (start - end) * ((x - eps_decay_start) / decay))
        )

    def _init_models(self, config):
        """Instantiate the policy and target networks.

        Args:
            config (Dict): Settings with parameters for the models

        Raises:
            ValueError: The configuration contains an estimator name that the agent does not
                        know to instantiate.
        """
        estimator_settings = config.get("estimator", {"model": "Conv_QNET", "args": {}})

        if estimator_settings["model"] == "OpinionNet":
            self.policy_model = OpinionNet(
                self.in_features,
                nr_betas=len(self.betas),
                **estimator_settings["args"],
            )
            self.target_model = OpinionNet(
                self.in_features,
                nr_betas=len(self.betas),
                **estimator_settings["args"],
            )
            self.policy_model.train()
            self.target_model.eval()
            # initialize_network_to_match_policy(self.policy_model, self.train_env, available_budget=2.0)
            # initialize_network_to_match_policy(self.target_model, self.validation_env, available_budget=2.0)
            
        else:
            estimator_name = estimator_settings["model"]
            raise ValueError(f"Could not setup estimator. Tried with: {estimator_name}")

        optimizer_settings = config.get("optim", {"name": "Adam", "args": {}})
        self.optimizer = optim.Adam(
            self.policy_model.parameters(), **optimizer_settings["args"]
        )
        
        self.logger.info("Initialized networks and optimizer.")

    @torch.no_grad()
    def _soft_update(self, tau: float = 0.005):
        # measure drift BEFORE update (occasionally)
        if self._should_log():
            try:
                num, den = 0.0, 1e-12
                for tp, sp in zip(self.target_model.parameters(), self.policy_model.parameters()):
                    num += (tp.data - sp.data).pow(2).sum().item()
                    den += sp.data.pow(2).sum().item()
                rel_dist = (num ** 0.5) / (den ** 0.5)
                self._last_rel_target_drift = rel_dist
                self.logger.info(f"target_drift@t={self.t} | ||t-s||/||s||={rel_dist:.3e} | tau={tau}")
            except Exception as e:
                self.logger.debug(f"[drift-log-skip] {e}")
                
        # EMA for parameters
        for tp, sp in zip(self.target_model.parameters(), self.policy_model.parameters()):
            tp.data.lerp_(sp.data, tau)
        for tb, sb in zip(self.target_model.buffers(), self.policy_model.buffers()):
            tb.copy_(sb)
        
    def _make_train_env(self):
        return self.env_factory.get_randomized_env()
    
    def _make_validation_env(self):
        total_versions = len(self.env_factory.validation_versions)

        # Determine how many times each version has been used
        usage_counts = [
            (version, self.validation_env_counters.get(version, 0))
            for version in range(total_versions)
        ]

        # Find the version(s) with the minimum usage
        min_usage = min(count for _, count in usage_counts)
        least_used_versions = [v for v, count in usage_counts if count == min_usage]

        # Break ties by choosing the smallest version index
        chosen_version = min(least_used_versions)

        env = self.env_factory.get_validation_env(version=chosen_version)
        self.validation_env_counters[chosen_version] += 1
        return env
    
    def _read_and_init_envs(self):
        """Read dimensions of the input and output of the simulation environment"""
        self.env_factory = EnvironmentFactory()
        self.validation_env_counters = defaultdict(int)

        self.train_env = self._make_train_env()
        self.validation_env = self._make_validation_env()
        
        self.train_env_s, _ = self.train_env.reset(randomize_opinions=True) 
        self.val_env_s, _ = self.validation_env.reset()
        
        self.in_features = self.train_env.observation_space.shape[0]
        self.num_actions = self.train_env.action_space.shape[0]

        

    def load_models(self, models_load_file):
        checkpoint = torch.load(models_load_file, weights_only=False)
        self.policy_model.load_state_dict(checkpoint["policy_model_state_dict"])
        self.policy_model.train()
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.target_model.eval()
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def load_training_stats(self, training_stats_file):
        checkpoint = torch.load(training_stats_file, weights_only=False)

        self.t = checkpoint["frame"]
        self.episodes = checkpoint["episode"]
        self.policy_model_update_counter = checkpoint["policy_model_update_counter"]

        self.training_stats = checkpoint["training_stats"]
        self.validation_stats = checkpoint["validation_stats"]

    def save_checkpoint(self):
        self.logger.info(f"Saving checkpoint at t = {self.t} ...")
        self.save_model()
        self.save_training_status()
        self.replay_buffer.save(self.replay_buffer_file)
        self.logger.info(f"Checkpoint saved at t = {self.t}")

    def save_model(self):
        model_file = self._make_model_checkpoint_file_path(
            self.experiment_output_folder, len(self.training_stats)
        )
        Path(os.path.dirname(model_file)).mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "policy_model_state_dict": self.policy_model.state_dict(),
                "target_model_state_dict": self.target_model.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
            },
            model_file,
        )
        self.logger.debug(f"Models saved at t = {self.t}")

    def save_training_status(self):
        status_dict = {
            "frame": self.t,
            "episode": self.episodes,
            "policy_model_update_counter": self.policy_model_update_counter,
            "training_stats": self.training_stats,
            "validation_stats": self.validation_stats,
        }

        torch.save(
            status_dict,
            self.train_stats_file,
        )

        self.logger.debug(f"Training status saved at t = {self.t}")


    # def select_action(self, state: torch.Tensor, epsilon: float = None, random_action: bool = False):
    #     if state.dim() == 1:
    #         state = state.unsqueeze(0)
    #     B = state.shape[0]  # Always 1
    #     N = self.num_actions
    #     J = len(self.betas)
    #     uniform_w = torch.ones(N, dtype=torch.float32) / N  # (N,)
    #     all_uniform_w = uniform_w.unsqueeze(0).repeat(J, 1)  # (J, N)
    #     w_full = all_uniform_w.unsqueeze(0).repeat(B, 1, 1)  # (1, J, N)

    #     with torch.no_grad():
    #         c = self.policy_model(state)["c"]  # (1, J)
    #         if random_action or (epsilon is not None and np.random.rand() < epsilon):
    #             beta_idx = torch.randint(low=0, high=J, size=(1,), dtype=torch.long)
    #             action_type = "random"
    #         else:
    #             _, beta_idx = c.max(dim=1)  # (1,)
    #             action_type = "greedy"

    #         betas_tensor = torch.tensor([self.betas[int(i)] for i in beta_idx], dtype=torch.float32)
    #         action = betas_tensor.unsqueeze(1) * uniform_w.unsqueeze(0)  # (1, N)
    #         w_full.zero_()
    #         w_full[0, beta_idx[0], :] = uniform_w
    #         avg_q = c[0, beta_idx[0]].item()

    #     # if self.t % 100000 == 0:
    #     #     self.logger.info(f"select_action: state shape: {state.shape}")
    #     #     self.logger.info(f"select_action: uniform_w shape: {uniform_w.shape}")
    #     #     self.logger.info(f"select_action: all_uniform_w shape: {all_uniform_w.shape}")
    #     #     self.logger.info(f"select_action: w_full shape: {w_full.shape}")
    #     #     self.logger.info(f"select_action: q_values (c) shape: {c.shape}")
    #     #     self.logger.info(f"select_action: ({action_type}) beta_idx: {beta_idx}")
    #     #     self.logger.info(f"select_action: betas_tensor shape: {betas_tensor.shape}, value: {betas_tensor}")
    #     #     self.logger.info(f"select_action: action shape: {action.shape}")
    #     #     self.logger.info(f"select_action: w_full after scatter shape: {w_full.shape}")
    #     #     self.logger.info(f"select_action: avg_q: {avg_q}")

    #     return (
    #         action.cpu().numpy(),             # (1, N)
    #         beta_idx.cpu().numpy().astype(np.int64),  # (1,)
    #         w_full.cpu().numpy(),             # (1, J, N)
    #         avg_q,
    #     )
    
    # def model_learn(self, sample, debug=True):
    #     """Compute the loss with TD learning for the FixedW (discrete β) model."""
    #     # Unpack; we ignore ws for FixedW
    #     states, (beta_indices, _ws), rewards, next_states, dones = sample

    #     device = next(self.policy_model.parameters()).device
    #     B = len(states)

    #     # ---- Tensors ----
    #     states      = torch.as_tensor(np.stack(states),      dtype=torch.float32, device=device)
    #     next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device)

    #     # Make beta_indices a flat (B,) Long tensor
    #     # If your buffer stores shape (B,1) arrays, this still works.
    #     beta_indices = torch.as_tensor(beta_indices, dtype=torch.long, device=device).view(-1)

    #     rewards = torch.as_tensor(rewards, dtype=torch.float32, device=device).view(B, 1)
    #     dones   = torch.as_tensor(dones,   dtype=torch.float32, device=device).view(B, 1)

    #     # ---- Q(s, β) from policy model ----
    #     self.policy_model.train()
    #     q = self.policy_model(states)["c"]               # (B, J)
    #     q_sa = q.gather(1, beta_indices.unsqueeze(1))    # (B, 1)

    #     # ---- Double DQN target: use online net to select, target net to evaluate ----
    #     with torch.no_grad():
    #         q_next_online = self.policy_model(next_states)["c"]           # (B, J)
    #         next_beta_idx = q_next_online.argmax(dim=1, keepdim=True)     # (B, 1)

    #         q_next_target = self.target_model(next_states)["c"]           # (B, J)
    #         max_next_q    = q_next_target.gather(1, next_beta_idx)        # (B, 1)

    #         target = rewards + self.gamma * (1.0 - dones) * max_next_q    # (B, 1)

    #     # ---- Loss & step ----
    #     # Huber is typically more stable than MSE for Q-learning
    #     loss = F.smooth_l1_loss(q_sa, target)

    #     self.optimizer.zero_grad()
    #     loss.backward()
    #     torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 10.0)
    #     self.optimizer.step()

    #     return float(loss.item())

    # Original select action
    def select_action(self, state: torch.Tensor, epsilon: float = None, random_action: bool = False):
        """
        Select an action by evaluating the Q-function over the β grid.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, Optional[float]]:
                - u: continuous action (B, N)
                - beta_idx: index of selected beta (B,)
                - w: all candidate weight vectors (B, J, N)
                - Q-value (scalar)
        """
        if state.dim() == 1:
            state = state.unsqueeze(0)

        with torch.no_grad():
            # === Forward pass ===
            abc_model = self.policy_model(state)
            A_diag, b, c = abc_model["A_diag"], abc_model["b"], abc_model["c"]
            B, J, N = A_diag.shape

            assert b.shape == (B, J, N), f"b shape mismatch: {b.shape}"
            assert c.shape == (B, J), f"c shape mismatch: {c.shape}"

            w_star = self.policy_model.compute_w_star(A_diag, b)  # (B, J, N)
            q_values = self.policy_model.compute_q_values(w_star, A_diag, b, c)  # (B, J)

            assert w_star.shape == (B, J, N), f"w_star shape mismatch: {w_star.shape}"
            assert q_values.shape == (B, J), f"q_values shape mismatch: {q_values.shape}"

            noise_amplitude = self.action_w_noise_amplitude * epsilon

            # === RANDOM ACTION BRANCH ===
            if random_action or (epsilon is not None and np.random.rand() < epsilon):
                rand_idx = torch.randint(low=0, high=J, size=(B,), dtype=torch.long)

                w_rand = w_star.gather(1, rand_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, N)).squeeze(1)  # (B, N)
                assert w_rand.shape == (B, N), f"w_rand shape: {w_rand.shape}"

                w_rand_noisy = self.policy_model.apply_action_noise(w_rand, noise_amplitude)

                rand_beta_values = torch.tensor(self.betas)[rand_idx].unsqueeze(1).expand(-1, N)  # (B, N)

                u = self.policy_model.compute_action_from_w(w_rand_noisy, rand_beta_values)

                w_full = torch.zeros_like(w_star)
                w_full.scatter_(1, rand_idx.reshape(-1, 1, 1).expand(-1, 1, N), w_rand_noisy.unsqueeze(1))

                q_rand = q_values.gather(1, rand_idx.unsqueeze(1)).squeeze(1)  # (B,)
                q_rand_mean = q_rand.mean(dim=0)  # scalar

                return (
                    u.cpu().numpy(),
                    rand_idx.cpu().numpy().astype(np.int64),
                    w_full.cpu().numpy(),
                    q_rand_mean.item(),
                )

            # === GREEDY ACTION BRANCH ===
            max_q, beta_idx = q_values.max(dim=1)  # (B,)
            beta_idx_exp = beta_idx.unsqueeze(1).unsqueeze(2).expand(-1, 1, N)

            assert beta_idx.shape == (B,), f"beta_idx shape: {beta_idx.shape}"
            assert beta_idx_exp.shape == (B, 1, N), f"beta_idx_exp shape: {beta_idx_exp.shape}"

            optimal_w = w_star.gather(1, beta_idx_exp).squeeze(1)  # (B, N)

            optimal_w_noisy = self.policy_model.apply_action_noise(optimal_w, noise_amplitude)

            beta_values = torch.tensor(
                [self.betas[int(i)] for i in beta_idx], dtype=torch.float32
            ).unsqueeze(1).expand(-1, N)  # (B, N)

            u = self.policy_model.compute_action_from_w(optimal_w_noisy, beta_values)

            w_full = torch.zeros_like(w_star)
            w_full.scatter_(1, beta_idx_exp, optimal_w_noisy.unsqueeze(1))

            assert u.shape == (B, N), f"u shape: {u.shape}"

            return (
                u.cpu().numpy(),
                beta_idx.cpu().numpy().astype(np.int64),
                w_full.cpu().numpy(),
                max_q.mean().item(),
            )
               
    def model_learn(self, sample, debug=True):
        """TD learning with Double DQN targets, Huber loss, grad clipping, and soft target updates."""
        # Unpack
        states, (beta_indices, ws), rewards, next_states, dones = sample

        device = next(self.policy_model.parameters()).device
        B = len(states)

        # Tensors
        states      = torch.as_tensor(np.stack(states),      dtype=torch.float32, device=device)
        next_states = torch.as_tensor(np.stack(next_states), dtype=torch.float32, device=device)
        beta_indices = torch.as_tensor(beta_indices, dtype=torch.long,  device=device).view(-1)   # (B,)
        ws          = torch.as_tensor(np.stack(ws), dtype=torch.float32, device=device)           # (B, J, N)
        rewards     = torch.as_tensor(rewards, dtype=torch.float32, device=device).view(B, 1)
        dones       = torch.as_tensor(dones,   dtype=torch.float32, device=device).view(B, 1)

        # ---- Q(s, β) using the *online* network ----
        self.policy_model.train()
        abc = self.policy_model(states)
        A_diag, b, c = abc["A_diag"], abc["b"], abc["c"]           # A_diag/b: (B, J, N), c: (B, J)

        # Q(s,β) evaluated at the stored ws (shape must match A_diag/b)
        assert ws.shape == A_diag.shape, f"ws: {ws.shape}, A_diag: {A_diag.shape}"
        q_values = self.policy_model.compute_q_values(ws, A_diag, b, c)  # (B, J)

        # Gather Q(s, β_taken)
        q_sa = q_values.gather(1, beta_indices.unsqueeze(1))  # (B, 1)

        # ---- Double DQN target ----
        with torch.no_grad():
            # Online net selects β' on next state
            abc_next_online = self.policy_model(next_states)
            A_do, b_o, c_o  = abc_next_online["A_diag"], abc_next_online["b"], abc_next_online["c"]
            w_star_next_online = self.policy_model.compute_w_star(A_do, b_o)
            q_next_online = self.policy_model.compute_q_values(w_star_next_online, A_do, b_o, c_o)  # (B, J)
            next_beta_idx = q_next_online.argmax(dim=1, keepdim=True)                               # (B, 1)

            # Target net evaluates that choice
            abc_next_tgt = self.target_model(next_states)
            A_dt, b_t, c_t = abc_next_tgt["A_diag"], abc_next_tgt["b"], abc_next_tgt["c"]
            w_star_next_tgt = self.target_model.compute_w_star(A_dt, b_t)
            q_next_target = self.target_model.compute_q_values(w_star_next_tgt, A_dt, b_t, c_t)     # (B, J)
            max_next_q = q_next_target.gather(1, next_beta_idx)                                     # (B, 1)

            target = rewards + self.gamma * (1.0 - dones) * max_next_q                              # (B, 1)
            
            # Clamp targets
            with torch.no_grad():
                target = target.clamp(-200.0, 200.0)
                
        if self._should_log():
            try:
                with torch.no_grad():
                    td      = (target - q_sa)                    # (B,1)
                    td_abs  = td.abs()
                    qsa_m   = q_sa.mean().item()
                    tgt_m   = target.mean().item()
                    td_m    = td.mean().item()
                    # p95 robust stat
                    k = max(1, int(0.95 * td_abs.numel()))
                    td_p95  = td_abs.kthvalue(k)[0].item()
                    A_min   = A_diag.min().item()
                    A_max   = A_diag.max().item()
                    b_mean  = b.abs().mean().item()
                    c_mean  = c.abs().mean().item()
                    clamp_p = (target.abs() >= 200.0).float().mean().item()
                self.logger.info(
                    f"learn@t={self.t} | q_sa={qsa_m:.3g} | tgt={tgt_m:.3g} "
                    f"| td={td_m:.3g} | |td|_p95={td_p95:.3g} "
                    f"| A[min,max]=[{A_min:.3g},{A_max:.3g}] | |b|_mean={b_mean:.3g} | |c|_mean={c_mean:.3g} "
                    f"| clamp%={clamp_p*100:.2f}"
                )
            except Exception as e:
                self.logger.debug(f"[learn-log-skip] {e}")
                
        # ---- Loss & optimization ----
        loss = F.smooth_l1_loss(q_sa, target)           # Huber
        self.optimizer.zero_grad()
        loss.backward()
        
        grad_norm_pre = None
        nonfinite_grads = 0
        if self._should_log():
            try:
                total_sq = 0.0
                for p in self.policy_model.parameters():
                    if p.grad is not None:
                        g = p.grad.detach()
                        total_sq += g.pow(2).sum().item()
                        if not torch.isfinite(g).all():
                            nonfinite_grads += 1
                grad_norm_pre = total_sq ** 0.5
            except Exception:
                pass
        
        torch.nn.utils.clip_grad_norm_(self.policy_model.parameters(), 10.0)
        
        grad_norm_post = None
        if self._should_log():
            try:
                total_sq = 0.0
                for p in self.policy_model.parameters():
                    if p.grad is not None:
                        total_sq += p.grad.detach().pow(2).sum().item()
                grad_norm_post = total_sq ** 0.5
            except Exception:
                pass

        # Current LR (first param group)
        lr_val = None
        if self._should_log():
            try:
                lr_val = self.optimizer.param_groups[0].get("lr", None)
            except Exception:
                pass

        # Param L2 norm (for scale context)
        param_norm = None
        if self._should_log():
            try:
                with torch.no_grad():
                    s = 0.0
                    for p in self.policy_model.parameters():
                        s += p.data.pow(2).sum().item()
                    param_norm = s ** 0.5
            except Exception:
                pass

        # Log the grad/param stats together (once per stride)
        if self._should_log():
            try:
                self.logger.info(
                    "optim@t=%d | loss=%.4g | grad||pre=%.3g | grad||post=%.3g | "
                    "nonfinite_grads=%d | param||=%.3g | lr=%s" %
                    (
                        self.t,
                        float(loss.item()),
                        (grad_norm_pre if grad_norm_pre is not None else float('nan')),
                        (grad_norm_post if grad_norm_post is not None else float('nan')),
                        nonfinite_grads,
                        (param_norm if param_norm is not None else float('nan')),
                        (f"{lr_val:.3g}" if isinstance(lr_val, float) else str(lr_val)),
                    )
                )
            except Exception:
                pass
        
        self.optimizer.step()

        # ---- Gentler target updates (Polyak) ----
        self._soft_update(tau=0.005)

        # Post-step sanity check for non-finite params (rare, but helpful)
        if self._should_log():
            try:
                for name, p in self.policy_model.named_parameters():
                    if torch.isnan(p).any() or torch.isinf(p).any():
                        self.logger.info(f"[WARN] non-finite param in {name} @ t={self.t}")
                        break
            except Exception:
                pass
    
        return float(loss.item())
    
    

    def train(self, train_epochs: int) -> True:
        """The main call for the training loop of the DQN Agent.

        Args:
            train_epochs (int): Represent the number of epochs we want to train for.
                            Note: if the training is resumed, then the number of training epochs that will be done is
                            as many as is needed to reach the train_epochs number.
        """
        if not self.training_stats:
            self.logger.info(f"Starting training session at: {self.t}")
        else:
            epochs_left_to_train = train_epochs - len(self.training_stats)
            self.logger.info(
                f"Resuming training session at: {self.t} ({epochs_left_to_train} epochs left)"
            )
            train_epochs = epochs_left_to_train

        for epoch in range(train_epochs):
            start_time = datetime.datetime.now()

            ep_train_stats = self.train_epoch()
            self.display_training_epoch_info(ep_train_stats)
            self.training_stats.append(ep_train_stats)

            if self.validation_enabled:
                ep_validation_stats = self.validate_epoch()
                self.display_validation_epoch_info(ep_validation_stats)
                self.validation_stats.append(ep_validation_stats)

            if self.save_checkpoints:
                self.save_checkpoint()

            end_time = datetime.datetime.now()
            epoch_time = end_time - start_time

            self.logger.info(f"Epoch {epoch} completed in {epoch_time}")
            self.logger.info("\n")

        self.logger.info(
            f"Ended training session after {train_epochs} epochs at t = {self.t}"
        )

        return True

    def train_epoch(self) -> Dict:
        """Do a single training epoch.

        Returns:
            Dict: dictionary containing the statistics of the training epoch.
        """
        self.logger.info(f"Starting training epoch at t = {self.t}")
        epoch_t = 0
        policy_trained_times = 0
        target_trained_times = 0

        epoch_episode_rewards = []
        epoch_episode_discounted_rewards = []
        epoch_episode_nr_frames = []
        epoch_losses = []
        epoch_max_qs = []

        start_time = datetime.datetime.now()
        while epoch_t < self.train_step_cnt:
            (
                is_terminated,
                truncated,
                epoch_t,
                current_episode_reward,
                current_episode_discounted_reward,
                ep_frames,
                ep_policy_trained_times,
                ep_target_trained_times,
                ep_losses,
                ep_max_qs,
            ) = self.train_episode(epoch_t, self.train_step_cnt)

            policy_trained_times += ep_policy_trained_times
            target_trained_times += ep_target_trained_times

            if is_terminated or truncated:
                # we only want to append these stats if the episode was completed,
                # otherwise it means it was stopped due to the agent nr of frames criterion
                epoch_episode_rewards.append(current_episode_reward)
                epoch_episode_discounted_rewards.append(
                    current_episode_discounted_reward
                )
                epoch_episode_nr_frames.append(ep_frames)
                epoch_losses.extend(ep_losses)
                epoch_max_qs.extend(ep_max_qs)

                self.episodes += 1
                self.reset_training_episode_tracker()

        end_time = datetime.datetime.now()
        epoch_time = end_time - start_time

        epoch_stats = self.compute_training_epoch_stats(
            epoch_episode_rewards,
            epoch_episode_discounted_rewards,
            epoch_episode_nr_frames,
            policy_trained_times,
            target_trained_times,
            epoch_losses,
            epoch_max_qs,
            epoch_time,
        )

        return epoch_stats

    def train_episode(self, epoch_t: int, train_frames: int):
        """Do a single training episode.

        Args:
            epoch_t (int): The total number of frames seen in this epoch, relevant for early stopping of
                            the training episode.
            train_frames (int): How many frames we want to limit the training epoch to

        Returns:
            Tuple[bool, int, float, int, int, int, list, list]: Information relevant to this training episode. Some variables are stored in
                                                            the class so that the training episode can resume in the following epoch.
        """
        policy_trained_times = 0
        target_trained_times = 0

        is_terminated = False
        truncated = False
        while (not is_terminated) and (not truncated) and (epoch_t < train_frames):
            self.logger.debug(f"State (s) shape before step: {self.train_env_s.shape}")

            action, beta_idx, w, max_q = self.select_action(
                torch.tensor(self.train_env_s, dtype=torch.float32),
                epsilon=self.epsilon_by_frame(self.t),
            )
            action = np.squeeze(action)
            
            if self._should_log():
                try:
                    # action is (N,)
                    act = torch.tensor(action, dtype=torch.float32)
                    frac_over_cap = (act > 0.4).float().mean().item()
                    topk_vals, _ = torch.topk(act, k=min(3, act.numel()))
                    # Pull chosen w "logits" for entropy proxy
                    bidx = int(np.asarray(beta_idx).reshape(-1)[0])
                    w_tensor = torch.tensor(w, dtype=torch.float32)  # (1, J, N)
                    w_chosen = w_tensor[0, bidx, :]                  # (N,)
                    with torch.no_grad():
                        p = torch.softmax(w_chosen, dim=-1)
                        H = -(p * p.clamp_min(1e-8).log()).sum().item()
                    self.logger.info(
                        f"t={self.t} | eps={self.epsilon_by_frame(self.t):.4f} | beta_idx={bidx} "
                        f"| action_entropy={H:.3f} | frac_u>0.4={frac_over_cap:.3f} "
                        f"| top3_u={topk_vals.tolist()} | max_q={max_q:.3g}"
                    )
                    self._last_entropy = H
                    self._last_frac_over_cap = frac_over_cap
                    
                except Exception as e:
                    self.logger.debug(f"[act-log-skip] {e}")

            self.logger.debug(f"State shape: {self.train_env_s.shape}")
            self.logger.debug(f"Action shape: {action.shape}")
            self.logger.debug(f"Beta index: {beta_idx}")

            if self._should_log():
                if not np.isfinite(action).all():
                    self.logger.info(f"[WARN] non-finite action at t={self.t}: {action}")
        
            action = np.squeeze(action)
            s_prime, reward, is_terminated, truncated, info = self.train_env.step(
                action
            )

            self.logger.debug(f"State (s') shape after step: {s_prime.shape}")
        
            done_flag = bool(is_terminated or truncated)
            self.replay_buffer.append(
                self.train_env_s, (beta_idx, w), reward, s_prime, done_flag
            )
            self.max_qs.append(max_q)

            if self._should_log():
                try:
                    self.logger.info(
                        f"buffer@t={self.t} | size={len(self.replay_buffer)} | batch={self.batch_size} "
                        f"| train_freq={self.training_freq} | gamma={self.gamma}"
                    )
                except Exception as e:
                    self.logger.debug(f"[buffer-log-skip] {e}")
        
            # Train policy model
            if (
                self.t > self.replay_start_size
                and len(self.replay_buffer) >= self.batch_size
            ):
                if self.t % self.training_freq == 0:
                    sample = self.replay_buffer.sample(self.batch_size)
                    loss_val = self.model_learn(sample)
                    self.losses.append(loss_val)
                    self.policy_model_update_counter += 1
                    policy_trained_times += 1

                # Disabled target model update because we do soft updates
                # if (
                #     self.policy_model_update_counter > 0
                #     and self.policy_model_update_counter % self.target_model_update_freq
                #     == 0
                # ):
                #     self.target_model.load_state_dict(self.policy_model.state_dict())
                #     target_trained_times += 1

            self.current_episode_reward += reward
            self.current_episode_discounted_reward += self.discount_factor * reward
            self.discount_factor *= self.gamma
            self.t += 1
            epoch_t += 1
            self.ep_frames += 1

            self.train_env_s = s_prime

        return (
            is_terminated,
            truncated,
            epoch_t,
            self.current_episode_reward,
            self.current_episode_discounted_reward,
            self.ep_frames,
            policy_trained_times,
            target_trained_times,
            self.losses,
            self.max_qs,
        )

    def reset_training_episode_tracker(self):
        """Resets the environment and the variables that keep track of the training episode."""
        self.current_episode_reward = 0.0
        self.current_episode_discounted_reward = 0.0
        self.discount_factor = 1.0

        self.ep_frames = 0
        self.losses = []
        self.max_qs = []

        self.train_env_s, _ = self.train_env.reset(randomize_opinions=True)

    def display_training_epoch_info(self, stats):
        extra = (
            f" | Entropy(last)={None if self._last_entropy is None else round(self._last_entropy,3)}"
            f" | frac_u>0.4(last)={None if self._last_frac_over_cap is None else round(self._last_frac_over_cap,3)}"
            f" | tgt_drift(last)={None if self._last_rel_target_drift is None else f'{self._last_rel_target_drift:.2e}'}"
        )
        
        self.logger.info(
            "TRAINING STATS"
            + " | Frames seen: "
            + str(self.t)
            + " | Episode: "
            + str(self.episodes)
            + " | Max reward: "
            + str(stats["episode_rewards"]["max"])
            + " | Avg reward: "
            + str(stats["episode_rewards"]["mean"])
            + " | Avg frames (episode): "
            + str(stats["episode_frames"]["mean"])
            + " | Avg max Q: "
            + str(stats["episode_max_qs"]["mean"])
            + " | Epsilon: "
            + str(self.epsilon_by_frame(self.t))
            + " | Train epoch time: "
            + str(stats["epoch_time"])
            + extra
        )

    def compute_training_epoch_stats(
        self,
        episode_rewards,
        episode_discounted_rewards,
        episode_nr_frames,
        policy_trained_times,
        target_trained_times,
        ep_losses,
        ep_max_qs,
        epoch_time,
    ) -> Dict:
        """Computes the statistics of the current training epoch.

        Args:
            episode_rewards (List): list contraining the final reward of each episode in the current epoch.
            episode_discounted_rewards (List): list contraining the final discounted reward of each episode in the current epoch.
            episode_nr_frames (List): list contraining the final number of frames of each episode in the current epoch.
            policy_trained_times (int): Number representing how many times the policy network was updated.
            target_trained_times (int): Number representing how many times the target network was updated.
            ep_losses (List): list contraining losses from the current epoch.
            ep_max_qs (List): list contraining maximum Q values from the current epoch.
            epoch_time (float): How much time the epoch took to compute in seconds.

        Returns:
            Dict: Dictionary with the relevant statistics
        """
        stats = {}

        stats["frame_stamp"] = self.t
        stats["greedy_epsilon"] = self.epsilon_by_frame(self.t)

        stats["episode_rewards"] = self.get_vector_stats(episode_rewards)
        stats["episode_discounted_rewards"] = self.get_vector_stats(
            episode_discounted_rewards
        )
        stats["episode_frames"] = self.get_vector_stats(episode_nr_frames)
        stats["episode_losses"] = self.get_vector_stats(ep_losses)
        stats["episode_max_qs"] = self.get_vector_stats(ep_max_qs)

        stats["policy_trained_times"] = policy_trained_times
        stats["target_trained_times"] = target_trained_times
        stats["epoch_time"] = epoch_time

        return stats

    def get_vector_stats(self, vector):
        """Compute statistics for a list of values, handling None gracefully."""
        stats = {}

        # Filter out None values
        clean_vector = [v for v in vector if v is not None]

        if len(clean_vector) > 0:
            stats["min"] = np.nanmin(clean_vector)
            stats["max"] = np.nanmax(clean_vector)
            stats["mean"] = np.nanmean(clean_vector)
            stats["median"] = np.nanmedian(clean_vector)
            stats["std"] = np.nanstd(clean_vector)
        else:
            stats["min"] = None
            stats["max"] = None
            stats["mean"] = None
            stats["median"] = None
            stats["std"] = None

        return stats

    def validate_epoch(self):
        self.logger.info(f"Starting validation epoch at t = {self.t}")

        epoch_episode_rewards = []
        epoch_episode_discounted_rewards = []
        epoch_episode_nr_frames = []
        epoch_max_qs = []
        valiation_t = 0

        start_time = datetime.datetime.now()

        while valiation_t < self.validation_step_cnt:
            (
                current_episode_reward,
                current_episode_discounted_reward,
                ep_frames,
                ep_max_qs,
            ) = self.validate_episode()

            valiation_t += ep_frames

            epoch_episode_rewards.append(current_episode_reward)
            epoch_episode_discounted_rewards.append(current_episode_discounted_reward)
            epoch_episode_nr_frames.append(ep_frames)
            epoch_max_qs.extend(ep_max_qs)

        end_time = datetime.datetime.now()
        epoch_time = end_time - start_time

        epoch_stats = self.compute_validation_epoch_stats(
            epoch_episode_rewards,
            epoch_episode_discounted_rewards,
            epoch_episode_nr_frames,
            epoch_max_qs,
            epoch_time,
        )
        return epoch_stats

    def compute_validation_epoch_stats(
        self,
        episode_rewards,
        episode_discounted_rewards,
        episode_nr_frames,
        ep_max_qs,
        epoch_time,
    ) -> Dict:
        """Computes the statistics of the current validation epoch.

        Args:
            episode_rewards (List): list contraining the final reward of each episode in the current epoch.
            episode_discounted_rewards (List): list contraining the final discounted reward of each episode in the current epoch.
            episode_nr_frames (List): list contraining the final number of frames of each episode in the current epoch.
            ep_max_qs (List): list contraining maximum Q values from the current epoch.
            epoch_time (float): How much time the epoch took to compute in seconds.

        Returns:
            Dict: Dictionary with the relevant statistics
        """
        stats = {}

        stats["frame_stamp"] = self.t

        stats["episode_rewards"] = self.get_vector_stats(episode_rewards)
        stats["episode_discounted_rewards"] = self.get_vector_stats(
            episode_discounted_rewards
        )
        stats["episode_frames"] = self.get_vector_stats(episode_nr_frames)
        stats["episode_max_qs"] = self.get_vector_stats(ep_max_qs)
        stats["epoch_time"] = epoch_time

        return stats

    def validate_episode(self):
        """Do a single validation episode.

        Returns:
            Tuple[int, int, List, Dict]: Tuple parameters relevant to the validation episode.
                                    The first element is the cumulative reward of the episode.
                                    The second element is the number of frames that were part of the episode.
                                    The third element is a list of the maximum Q values seen.
                                    The fourth element is a dictionary containing the number of times each reward was seen.
        """
        current_episode_reward = 0.0
        current_episode_discounted_reward = 0.0
        discount_factor = 1.0
        ep_frames = 0
        max_qs = []

        # Remake the env because we cycle through setups
        self.validation_env = self._make_validation_env()
        s, info = self.validation_env.reset()
        s = torch.tensor(s, device=device).float()

        is_terminated = False
        truncated = False
        while (
            (not is_terminated)
            and (not truncated)
            and (ep_frames < self.validation_step_cnt)
        ):
            action, betas, w, max_q = self.select_action(
                torch.tensor(s, dtype=torch.float32), epsilon=self.validation_epsilon
            )
            action = np.squeeze(action)
            s_prime, reward, is_terminated, truncated, info = self.validation_env.step(
                action
            )
            s_prime = torch.tensor(s_prime, device=device).float()

            max_qs.append(max_q)
            current_episode_reward += reward
            current_episode_discounted_reward += discount_factor * reward
            discount_factor *= self.gamma
            ep_frames += 1
            s = s_prime

        return (
            current_episode_reward,
            current_episode_discounted_reward,
            ep_frames,
            max_qs,
        )

    def display_validation_epoch_info(self, stats):
        self.logger.info(
            "VALIDATION STATS"
            + " | Max reward: "
            + str(stats["episode_rewards"]["max"])
            + " | Avg reward: "
            + str(stats["episode_rewards"]["mean"])
            + " | Avg frames (episode): "
            + str(stats["episode_frames"]["mean"])
            + " | Avg max Q: "
            + str(stats["episode_max_qs"]["mean"])
            + " | Validation epoch time: "
            + str(stats["epoch_time"])
        )

    


def main():
    pass


if __name__ == "__main__":
    seed_everything(0)
    main()
    # play_game_visual("breakout")
