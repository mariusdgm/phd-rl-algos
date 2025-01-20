import os, sys


def get_dir_n_levels_up(path, n):
    # Go up n levels from the given path
    for _ in range(n):
        path = os.path.dirname(path)
    return path


proj_root = get_dir_n_levels_up(os.path.abspath(__file__), 2)
sys.path.append(proj_root)

import time
import datetime
import torch
import random
import numpy as np

from pathlib import Path
from typing import List, Dict


import torch.optim as optim
import torch.nn.functional as F

import gym

from opinion_dynamics.replay_buffer import ReplayBuffer
from opinion_dynamics.utils.experiment import seed_everything
from opinion_dynamics.utils.my_logging import setup_logger
from opinion_dynamics.utils.generic import merge_dictionaries, replace_keys
from opinion_dynamics.models import OpinionNet

# device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
device = "cpu"


class AgentDQN:
    def __init__(
        self,
        train_env,
        validation_env,
        experiment_output_folder=None,
        experiment_name=None,
        resume_training_path=None,
        save_checkpoints=True,
        logger=None,
        config={},
    ) -> None:
        """A DQN agent implementation.

        Args:
            train_env (gym.env): An instantiated gym Environment
            validation_env (gym.env): An instantiated gym Environment that was created with the same
                                    parameters as train_env. Used to be able to do validation epochs and
                                    return to the same training point.
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
        self.train_env = train_env
        self.validation_env = validation_env

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

        self.reset_training_episode_tracker()

        self.training_stats = []
        self.validation_stats = []

        # check that all paths were provided and that the files can be found
        if resume_training_path:
            self.load_training_state(resume_training_path)

        self.betas = [0, 1]
        self.num_betas = len(self.betas)

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
        self.validation_epsilon = agent_params.get("validation_epsilon", 0.001)

        self.replay_start_size = agent_params.get("replay_start_size", 5_000)

        self.batch_size = agent_params.get("batch_size", 32)
        self.training_freq = agent_params.get("training_freq", 4)
        self.target_model_update_freq = agent_params.get(
            "target_model_update_freq", 100
        )
        self.gamma = agent_params.get("gamma", 0.99)
        self.loss_function = agent_params.get("loss_fcn", "mse_loss")

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
            "replay_buffer", {"max_size": 100_000, "action_dim": 1, "n_step": 0}
        )
        self.replay_buffer = ReplayBuffer(
            max_size=buffer_settings.get("max_size", 100_000),
            state_dim=self.in_features,
            action_dim=buffer_settings.get("action_dim", 1),
            n_step=buffer_settings.get("n_step", 0),
        )

        self.reward_perception = None
        reward_perception_config = config.get("reward_perception", None)

        if reward_perception_config:
            reward_perception_mapping = reward_perception_config.get("parameters", None)
            if reward_perception_mapping:
                self.logger.info("Setup reward mapping.")
                self.reward_perception = reward_perception_mapping

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
                self.num_actions,
                **estimator_settings["args"],
            )
            self.target_model = OpinionNet(
                self.in_features,
                self.num_actions,
                **estimator_settings["args"],
            )
        else:
            estimator_name = estimator_settings["model"]
            raise ValueError(f"Could not setup estimator. Tried with: {estimator_name}")

        optimizer_settings = config.get("optim", {"name": "Adam", "args": {}})
        self.optimizer = optim.Adam(
            self.policy_model.parameters(), **optimizer_settings["args"]
        )

        self.logger.info("Initialized newtworks and optimizer.")

    def _read_and_init_envs(self):
        """Read dimensions of the input and output of the simulation environment"""
        self.in_features = self.train_env.observation_space.shape[0]
        self.num_actions = self.train_env.action_space.shape[0]

        self.train_s, _ = self.train_env.reset()
        self.env_s, _ = self.validation_env.reset()

    def load_models(self, models_load_file):
        checkpoint = torch.load(models_load_file)
        self.policy_model.load_state_dict(checkpoint["policy_model_state_dict"])
        self.policy_model.train()
        self.target_model.load_state_dict(checkpoint["target_model_state_dict"])
        self.target_model.train()
        self.optimizer.load_state_dict(checkpoint["optimizer_state_dict"])

    def load_training_stats(self, training_stats_file):
        checkpoint = torch.load(training_stats_file)

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

    def select_action(
        self, state: torch.Tensor, epsilon: float = None, random_action: bool = False
    ):
        """
        Select an action by evaluating the Q-function over the \(\beta\) grid.

        Args:
            state (torch.Tensor): Current state.
            epsilon (float, optional): Epsilon for exploration. Defaults to None.
            random_action (bool, optional): Whether to select a random action. Defaults to False.

        Returns:
            Tuple[torch.Tensor, float]: The optimal weights and the maximum Q-value.
        """
        max_q = np.nan

        if random_action or (epsilon is not None and np.random.rand() < epsilon):
            # Random action: Uniformly select weights within the valid range
            action = self.train_env.action_space.sample()
            return action, max_q

        # Exploit: Use the policy model
        with torch.no_grad():
            q_vals, A_diag, b = self.policy_model(state.unsqueeze(0))

            # Compute \( w^* = -A^{-1} b \) for each \(\beta\)
            A_inv = 1.0 / A_diag  # Inverse of diagonal A
            w_star = -A_inv * b  # Shape: (batch_size, nr_betas, nr_agents)

            # Normalize \( w^* \) to satisfy \(\sum w_i = 1\)
            w_star = torch.clamp(w_star, 0, 1)
            w_star = w_star / w_star.sum(dim=2, keepdim=True)

            # Scale \( w^* \) to the valid range [0, env.u_max]
            w_star_scaled = w_star * self.train_env.u_max

            # Explicit computation of \( Q(x, \beta, w^*) = q(x, \beta) - \frac{1}{2} b^T A^{-1} b \)
            b_T = b.transpose(
                1, 2
            )  # Transpose b to have shape (batch_size, nr_betas, 1)
            A_inv_b = A_inv * b  # Element-wise multiplication for diagonal A
            b_A_inv_b = (
                torch.bmm(b_T, A_inv_b.unsqueeze(2)).squeeze(2).squeeze(1)
            )  # Quadratic form
            q_full = q_vals - 0.5 * b_A_inv_b

            # Select the best \(\beta\)
            max_q, beta_idx = q_full.max(dim=1)
            best_w = w_star_scaled[0, beta_idx]  # Scale the best \( w^* \)

        return best_w.cpu().numpy(), max_q.item()

    def get_max_q_val_for_state(self, state):
        with torch.inference_mode():
            return self.policy_model(state).max(1)[0].item()

    def get_q_val_for_action(self, state, action):
        with torch.inference_mode():
            return torch.index_select(
                self.policy_model(state), 1, action.squeeze(0)
            ).item()

    def get_action_from_model(self, state):
        with torch.inference_mode():
            return self.policy_model(state).max(1)[1].view(1, 1)

    # def get_max_q_and_action(self, state):
    #     with torch.inference_mode():
    #         maxq_and_action = self.policy_model(state).max(1)
    #         q_val = maxq_and_action[0].item()
    #         action = maxq_and_action[1].view(1, 1)
    #         return action, q_val

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
        epoch_episode_nr_frames = []
        epoch_losses = []
        epoch_max_qs = []

        start_time = datetime.datetime.now()
        while epoch_t < self.train_step_cnt:
            (
                is_terminated,
                epoch_t,
                current_episode_reward,
                ep_frames,
                ep_policy_trained_times,
                ep_target_trained_times,
                ep_losses,
                ep_max_qs,
            ) = self.train_episode(epoch_t, self.train_step_cnt)

            policy_trained_times += ep_policy_trained_times
            target_trained_times += ep_target_trained_times

            if is_terminated:
                # we only want to append these stats if the episode was completed,
                # otherwise it means it was stopped due to the nr of frames criterion
                epoch_episode_rewards.append(current_episode_reward)
                epoch_episode_nr_frames.append(ep_frames)
                epoch_losses.extend(ep_losses)
                epoch_max_qs.extend(ep_max_qs)

                self.episodes += 1
                self.reset_training_episode_tracker()

        end_time = datetime.datetime.now()
        epoch_time = end_time - start_time

        epoch_stats = self.compute_training_epoch_stats(
            epoch_episode_rewards,
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
        while (not is_terminated) and (
            epoch_t < train_frames
        ):  # can early stop episode if the frame limit was reached
            action, max_q = self.select_action(self.train_s, self.t, self.num_actions)
            s_prime, reward, is_terminated, truncated, info = self.train_env.step(
                action
            )
            s_prime = torch.tensor(s_prime, device=device).float()

            self.replay_buffer.append(
                self.train_s, action, reward, s_prime, is_terminated
            )

            self.max_qs.append(max_q)

            # Start learning when there's enough data and when we can sample a batch of size BATCH_SIZE
            if (
                self.t > self.replay_start_size
                and len(self.replay_buffer) >= self.batch_size
            ):
                # Train every training_freq number of frames
                if self.t % self.training_freq == 0:
                    sample = self.replay_buffer.sample(self.batch_size)
                    loss_val = self.model_learn(sample)

                    self.losses.append(loss_val)
                    self.policy_model_update_counter += 1
                    policy_trained_times += 1

                # Update the target network only after some number of policy network updates
                if (
                    self.policy_model_update_counter > 0
                    and self.policy_model_update_counter % self.target_model_update_freq
                    == 0
                ):
                    self.target_model.load_state_dict(self.policy_model.state_dict())
                    target_trained_times += 1

            self.current_episode_reward += reward

            self.t += 1
            epoch_t += 1
            self.ep_frames += 1

            # Continue the process
            self.train_s = s_prime

        # end of episode, return episode statistics:
        return (
            is_terminated,
            epoch_t,
            self.current_episode_reward,
            self.ep_frames,
            policy_trained_times,
            target_trained_times,
            self.losses,
            self.max_qs,
        )

    def reset_training_episode_tracker(self):
        """Resets the environment and the variables that keep track of the training episode."""
        self.current_episode_reward = 0.0
        self.ep_frames = 0
        self.losses = []
        self.max_qs = []

        self.train_s, _ = self.train_env.reset()
        self.train_s = torch.tensor(self.train_s, device=device).float()

    def display_training_epoch_info(self, stats):
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
        )

    def compute_training_epoch_stats(
        self,
        episode_rewards,
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

        stats["episode_rewards"] = self.get_vector_stats(episode_rewards)
        stats["episode_frames"] = self.get_vector_stats(episode_nr_frames)
        stats["episode_losses"] = self.get_vector_stats(ep_losses)
        stats["episode_max_qs"] = self.get_vector_stats(ep_max_qs)

        stats["policy_trained_times"] = policy_trained_times
        stats["target_trained_times"] = target_trained_times
        stats["epoch_time"] = epoch_time

        return stats

    def get_vector_stats(self, vector):
        """Do a single validation epoch.

        Returns:
            Dict: dictionary containing the statistics of the training epoch.
        """
        stats = {}

        if len(vector) > 0:
            stats["min"] = np.nanmin(vector)
            stats["max"] = np.nanmax(vector)
            stats["mean"] = np.nanmean(vector)
            stats["median"] = np.nanmedian(vector)
            stats["std"] = np.nanstd(vector)

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
        epoch_episode_nr_frames = []
        epoch_max_qs = []
        valiation_t = 0

        start_time = datetime.datetime.now()

        while valiation_t < self.validation_step_cnt:
            (
                current_episode_reward,
                ep_frames,
                ep_max_qs,
            ) = self.validate_episode()

            valiation_t += ep_frames

            epoch_episode_rewards.append(current_episode_reward)
            epoch_episode_nr_frames.append(ep_frames)
            epoch_max_qs.extend(ep_max_qs)

        end_time = datetime.datetime.now()
        epoch_time = end_time - start_time

        epoch_stats = self.compute_validation_epoch_stats(
            epoch_episode_rewards,
            epoch_episode_nr_frames,
            epoch_max_qs,
            epoch_time,
        )
        return epoch_stats

    def compute_validation_epoch_stats(
        self,
        episode_rewards,
        episode_nr_frames,
        ep_max_qs,
        epoch_time,
    ) -> Dict:
        """Computes the statistics of the current validation epoch.

        Args:
            episode_rewards (List): list contraining the final reward of each episode in the current epoch.
            episode_nr_frames (List): list contraining the final number of frames of each episode in the current epoch.
            ep_max_qs (List): list contraining maximum Q values from the current epoch.
            epoch_time (float): How much time the epoch took to compute in seconds.

        Returns:
            Dict: Dictionary with the relevant statistics
        """
        stats = {}

        stats["frame_stamp"] = self.t

        stats["episode_rewards"] = self.get_vector_stats(episode_rewards)
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
        ep_frames = 0
        max_qs = []

        # Initialize the environment and start state
        s, info = self.validation_env.reset()
        s = torch.tensor(s, device=device).float()

        is_terminated = False
        while not is_terminated:
            action, max_q = self.select_action(
                s, self.t, self.num_actions, epsilon=self.validation_epsilon
            )
            s_prime, reward, is_terminated, truncated, info = self.validation_env.step(
                action
            )
            s_prime = torch.tensor(s_prime, device=device).float()

            max_qs.append(max_q)

            current_episode_reward += reward

            ep_frames += 1

            # Continue the process
            s = s_prime

        return (current_episode_reward, ep_frames, max_qs)

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

    def model_learn(self, sample, debug=True):
        """Compute the loss with TD learning."""
        states, actions, rewards, next_states, dones = sample

        # Convert inputs to tensors
        states = torch.stack(states, dim=0)
        actions = torch.FloatTensor(actions)  # Continuous actions
        rewards = torch.FloatTensor(rewards).unsqueeze(1)
        next_states = torch.stack(next_states, dim=0)
        dones = torch.FloatTensor(dones).unsqueeze(1)

        if self.reward_perception:
            rewards = rewards + self.reward_perception["shift"]

        self.policy_model.train()

        # Forward pass through the policy model
        q_vals, A_diag, b = self.policy_model(states)

        # Debug logs for tensor shapes
        if debug:
            print(f"q_vals shape: {q_vals.shape}")  # Expected: (batch_size, nr_betas)
            print(f"A_diag shape: {A_diag.shape}")  # Expected: (batch_size, nr_betas, nr_agents)
            print(f"b shape: {b.shape}")  # Expected: (batch_size, nr_betas, nr_agents)

        A_inv = 1.0 / A_diag  # Inverse of diagonal A
        if debug:
            print(f"A_inv shape: {A_inv.shape}")  # Expected: Same as A_diag

        w_star = -A_inv * b  # Compute \( w^* \)
        if debug:
            print(f"w_star shape: {w_star.shape}")  # Expected: (batch_size, nr_betas, nr_agents)

        # Normalize \( w^* \)
        w_star = torch.clamp(w_star, 0, 1)
        w_star = w_star / w_star.sum(dim=2, keepdim=True)

        # Compute \( b^T A^{-1} b \)
        b_T = b.transpose(1, 2)  # Shape: (batch_size, nr_agents, nr_betas)
        if debug:
            print(f"b_T shape: {b_T.shape}")  # Transposed b

        A_inv_b = A_inv * b  # Shape: (batch_size, nr_agents, nr_betas)
        if debug:
            print(f"A_inv_b shape: {A_inv_b.shape}")

        b_A_inv_b = torch.sum(b_T * A_inv_b, dim=1)  # Shape: (batch_size, nr_betas)
        if debug:
            print(f"b_A_inv_b shape: {b_A_inv_b.shape}")

        # Compute the Q-values for selected actions
        q_full = q_vals - 0.5 * b_A_inv_b  # Q-values for all \(\beta\)
        if debug:
            print(f"q_full shape: {q_full.shape}")  # Expected: (batch_size, nr_betas)
        selected_q_value = (q_full * actions).sum(dim=1, keepdim=True)

        # Forward pass through the target model for next states
        next_q_vals, next_A_diag, next_b = self.target_model(next_states)
        next_A_inv = 1.0 / next_A_diag

        # Compute \( b^T A^{-1} b \) for the target model
        next_b_T = next_b.transpose(1, 2)  # Shape: (batch_size, nr_agents, nr_betas)
        next_A_inv_b = next_A_inv * next_b  # Element-wise multiplication
        if debug:
            print(f"next_A_inv_b shape: {next_A_inv_b.shape}")  # Expected: (batch_size, nr_betas, nr_agents)

        # Collapse quadratic term along the agents dimension
        next_b_A_inv_b = torch.sum(next_b_T * next_A_inv_b, dim=1)  # Shape: (batch_size, nr_betas)
        if debug:
            print(f"next_b_A_inv_b shape: {next_b_A_inv_b.shape}")

        # Compute the Q-values for next states
        next_q_full = next_q_vals - 0.5 * next_b_A_inv_b  # Shape: (batch_size, nr_betas)
        if debug:
            print(f"next_q_full shape: {next_q_full.shape}")
        max_next_q_values = next_q_full.max(dim=1, keepdim=True)[0]

        # TD target
        expected_q_value = rewards + self.gamma * max_next_q_values * (1 - dones)

        # Compute loss
        if self.loss_function == "mse_loss":
            loss = F.mse_loss(selected_q_value, expected_q_value)
        elif self.loss_function == "smooth_l1_loss":
            loss = F.smooth_l1_loss(selected_q_value, expected_q_value)
        else:
            raise ValueError(f"Unsupported loss function: {self.loss_function}")

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        return loss.item()

def main():
    pass


if __name__ == "__main__":
    seed_everything(0)
    main()
    # play_game_visual("breakout")
