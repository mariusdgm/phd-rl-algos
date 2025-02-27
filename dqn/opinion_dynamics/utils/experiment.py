import os

import random
import torch
import numpy as np
import datetime

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


def create_adjacency_matrix_from_links(num_nodes, links):
    adjacency_matrix = np.zeros((num_nodes, num_nodes), dtype=int)
    for link in links:
        from_node, to_node = link
        # adjacency_matrix[from_node, to_node] = 1
        adjacency_matrix[to_node, from_node] = 1

    return adjacency_matrix


### Env building code
def build_environment():
    links = [
        (1, 3),
        (3, 2),
        (2, 3),
        (2, 0),
        (0, 2),
        (1, 2),
        (0, 1),
        # (3, 4),
        # (4, 3)
    ]

    num_nodes = 4
    connectivity_matrix = create_adjacency_matrix_from_links(num_nodes, links)
    # connectivity_matrix = normalize_adjacency_matrix(connectivity_matrix)

    initial_opinions = np.linspace(0.5, 0, num_nodes)
    # initial_opinions = np.linspace(0, 1, num_nodes)
    # initial_opinions = (np.mod(np.arange(0, 0.1 * num_nodes, 0.1), 0.9)) + 0.1

    env = NetworkGraph(
        connectivity_matrix=connectivity_matrix,
        initial_opinions=initial_opinions,
        max_u=0.5,
        budget=100.0,
        desired_opinion=0.9,
        tau=0.1,
        max_steps=100,
        opinion_end_tolerance=0.01,
        control_beta=0.2,
    )

    env.reset()

    return env
