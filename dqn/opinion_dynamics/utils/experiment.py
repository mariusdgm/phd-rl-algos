import os

import random
import torch
import yaml
import datetime
import collections
import numpy as np
import pandas as pd

from pathlib import Path
from typing import Dict

from rl_envs_forge.envs.network_graph.network_graph import NetworkGraph


def seed_everything(seed):
    """
    Set the seed on everything I can think of.
    Hopefully this should ensure reproducibility.

    Credits: Florin
    """
    random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True


def create_path_to_experiment_folder(
    config: Dict,
    experiments_output_folder: str,
    timestamp_folder: str = None,
) -> str:
    """Build the path for the nested experiment structure:
    base_outputs / timestamp / experiment / environment / seed

    Args:
        config (Dict): Configuration of the experiment.
        experiments_output_folder (str): Root path for the folder where the outputs
                                        of paralelized experiments are stored.
        timestamp_folder (str, optional): Path to the previous top level output folder. If None, then a new top level folder
                                        is created with a string matching the current time. Defaults to None.

    Returns:
        str: The path to the folder that stores the output for this singular experiment
    """
    experiment = config["experiment_name"]
    env = config["environment"]
    seed = config["seed"]

    prev_output_not_expected = True

    if timestamp_folder is None:
        timestamp_folder = datetime.datetime.now().strftime(r"%Y_%m_%d-%H_%M_%S")
        prev_output_not_expected = False  # disable creation of prev folder

    exp_folder_path = os.path.join(
        experiments_output_folder,
        timestamp_folder,
        experiment,
        env,
        str(seed),
    )

    if prev_output_not_expected:
        Path(exp_folder_path).mkdir(parents=True, exist_ok=True)

    return exp_folder_path


def process_experiment(root_dir):
    rows = []

    for name in os.listdir(root_dir):
        experiment_path = os.path.join(root_dir, name)
        if os.path.isdir(experiment_path):
            for seed_name in os.listdir(experiment_path):
                seed_path = os.path.join(experiment_path, seed_name)
                if os.path.isdir(seed_path):
                    row_data = process_subexperiment(
                        seed_path, os.path.basename(root_dir)
                    )
                    for data in row_data:
                        data["seed"] = seed_name
                        data["experiment_name"] = name
                        data["sub_experiment_path"] = seed_path
                    rows.extend(row_data)

    # Create a DataFrame from the rows
    df = pd.DataFrame(rows)
    return df


def process_subexperiment(seed_folder_path, experiment_name):
    cfg_data = read_config(os.path.join(seed_folder_path, "cfg.yaml"), experiment_name)
    cfg_data["sub_experiment_path"] = seed_folder_path
    train_stats_file = find_train_stats_file(seed_folder_path)
    if train_stats_file:
        experiment_results = process_training_stats(train_stats_file, cfg_data)
        return experiment_results
    else:
        return []  # Return an empty list if no train stats file is found


def read_config(cfg_path, experiment_name):
    with open(cfg_path, "r") as file:
        config = yaml.safe_load(file)
        full_title = config.get("full_title", "")
        variable_part = remove_experiment_name(full_title, experiment_name)
        return parse_config_variables(variable_part)


def remove_experiment_name(full_title, experiment_name):
    to_remove = experiment_name + "_"
    return (
        full_title[len(to_remove) :].strip()
        if full_title.startswith(to_remove)
        else full_title
    )


def parse_config_variables(variable_str):
    variables = {}
    for part in variable_str.split(";"):
        if "=" in part:
            key, value = part.split("=", 1)
            key = f"sub_exp_cfg_{key.strip()}"  # Add prefix
            variables[key] = value.strip()
    return variables


def find_train_stats_file(folder_path):
    for file in os.listdir(folder_path):
        if file.endswith("_train_stats"):
            return os.path.join(folder_path, file)
    return None


def process_training_stats(train_stats_file, cfg_data):

    checkpoint = torch.load(train_stats_file, weights_only=False)
    training_stats = checkpoint.get("training_stats", [])
    validation_stats = checkpoint.get("validation_stats", [])
    redo_stats = checkpoint.get("redo_scores", [])

    stats_records = process_stats(training_stats, cfg_data, "training") + process_stats(
        validation_stats, cfg_data, "validation"
    )

    # Combine stats records with redo scores
    combined_records = []
    for record in stats_records:
        combined_record = record.copy()  # Copy the stats record
        combined_records.append(combined_record)

    return combined_records


def process_stats(stats, cfg_data, stats_type):
    records = []
    for epoch_stats in stats:
        record = {"epoch_type": stats_type}
        record.update(flatten(epoch_stats))  # Flatten the epoch_stats if it's nested
        record.update(cfg_data)  # Add configuration data
        records.append(record)
    return records


def flatten(d, parent_key="", sep="_"):
    items = []
    for k, v in d.items():
        new_key = parent_key + sep + k if parent_key else k
        if isinstance(v, collections.abc.MutableMapping):
            items.extend(flatten(v, new_key, sep=sep).items())
        else:
            items.append((new_key, v))
    return dict(items)


def create_adjacency_matrix_from_links(num_nodes, links):
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for link in links:
        from_node, to_node = link
        # adjacency_matrix[from_node, to_node] = 1
        adjacency_matrix[to_node, from_node] = 1

    return adjacency_matrix


### Env building code

# Small env, premade links
# def build_environment(random_initial_opinions=False):
#     links = [
#         (1, 3),
#         (3, 2),
#         (2, 3),
#         (2, 0),
#         (0, 2),
#         (1, 2),
#         (0, 1),
#         # (3, 4),
#         # (4, 3)
#     ]

#     num_agents = 4
#     connectivity_matrix = create_adjacency_matrix_from_links(num_nodes, links)
#     # connectivity_matrix = normalize_adjacency_matrix(connectivity_matrix)

#     if random_initial_opinions:
#         initial_opinions = np.random.uniform(low=0.0, high=1.0, size=num_nodes)

#     else:
#         initial_opinions = np.linspace(0.3, 0, num_agents)

#     # initial_opinions = np.linspace(0, 1, num_agents)

#     # initial_opinions = (np.mod(np.arange(0, 0.1 * num_agents, 0.1), 0.9)) + 0.1

#     env = NetworkGraph(
#         connectivity_matrix=connectivity_matrix,
#         initial_opinions=initial_opinions,
#         max_u=0.4,
#         budget=1000.0,
#         desired_opinion=1,
#         tau=0.1,
#         max_steps=50,
#         opinion_end_tolerance=0.05,
#         control_beta=0.4,
#         normalize_reward=True,
#         terminal_reward=0.5
#     )

#     env.reset()

#     return env


class EnvironmentFactory:
    def __init__(self):
        """
        Initializes the environment factory with a base configuration.
        """
        self.base_config = {
            "num_agents": 20,
            "max_u": 0.4,
            "budget": 1000.0,
            "desired_opinion": 1.0,
            "t_campaign": 1.0,
            "t_s": 0.1,
            "connection_prob_range": (0.05, 0.1),
            "bidirectional_prob": 0.1,
            "max_steps": 50,
            "opinion_end_tolerance": 0.05,
            "control_beta": 0.5,
            "normalize_reward": True,
            "terminal_reward": 0.5,
            "terminate_when_converged": False,
            "dynamics_model": "coca",  # or "laplacian"
            "seed": 42,
        }
        
        self.validation_versions = [0, 1, 2]

    def get_randomized_env(self, seed: int = None):
        """Returns a training environment with randomized opinions and optional seed."""
        config = self.base_config.copy()
        if seed is not None:
            config["seed"] = seed

        num_agents = config["num_agents"]
        initial_opinions = np.random.uniform(low=0.1, high=0.99, size=num_agents)
        config["initial_opinions"] = initial_opinions

        return NetworkGraph(**config)

    def get_validation_env(self, version: int = 0):
        """Returns a validation environment with controlled variation by version."""
        config = self.base_config.copy()
        num_agents = config["num_agents"]

        if version == 0:
            config["initial_opinions"] = np.linspace(0.3, 0.1, num_agents)
        elif version == 1:
            config["initial_opinions"] = np.linspace(0.4, 0.6, num_agents)
        elif version == 2:
            config["initial_opinions"] = np.linspace(0.1, 0.7, num_agents)
        else:
            raise ValueError(f"Unknown validation version: {version}")

        return NetworkGraph(**config)
    
def build_environment(random_initial_opinions=False):

    num_agents = 20

    if random_initial_opinions:
        initial_opinions = np.random.uniform(low=0.1, high=0.99, size=num_agents)

    else:
        initial_opinions = np.linspace(0.3, 0.1, num_agents)

    env = NetworkGraph(
        num_agents=num_agents,
        initial_opinions=initial_opinions,
        max_u=0.4,
        budget=1000.0,
        desired_opinion=1,
        t_campaign=1,
        t_s=0.1,
        connection_prob_range=(0.05, 0.1),
        bidirectional_prob=0.1,
        max_steps=50,
        opinion_end_tolerance=0.05,
        control_beta=0.4,
        normalize_reward=True,
        terminal_reward=0.5,
        seed=42,
        terminate_when_converged=False,
        dynamics_model="coca",
        # dynamics_model="laplacian",
    )

    env.reset()
    return env
