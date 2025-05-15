import numpy as np

# import os, sys

# def get_dir_n_levels_up(path, n):
#     # Go up n levels from the given path
#     for _ in range(n):
#         path = os.path.dirname(path)
#     return path

# proj_root = get_dir_n_levels_up(os.path.abspath(__file__), 2)
# sys.path.append(proj_root)


def centrality_based_continuous_control(env, available_budget):
    """
    Compute a control action distributing a continuous budget based on centrality * deviation heuristic.

    Args:
        env: The environment instance with `opinions`, `desired_opinion`, `centralities`, and `max_u` attributes.
        available_budget (float): Total control budget to distribute.

    Returns:
        control_action (np.array): Control action array (N,) where 0 <= control_action[i] <= max_u
        controlled_nodes (list): List of indices of nodes that received some control
    """
    N = env.num_agents
    deviations = np.abs(env.opinions - env.desired_opinion)  # (N,)
    influence_powers = env.centralities * deviations        # (N,)
    agent_order = np.argsort(influence_powers)[::-1]         # Sort descending by power

    control_action = np.zeros(N)
    remaining_budget = available_budget

    controlled_nodes = []

    for agent_idx in agent_order:
        if remaining_budget <= 0:
            break

        assign_amount = min(float(env.max_u[agent_idx]), remaining_budget)
        control_action[agent_idx] = assign_amount
        controlled_nodes.append(agent_idx)
        remaining_budget -= assign_amount

    return control_action, controlled_nodes