import numpy as np
from itertools import product
from copy import deepcopy
from scipy.linalg import expm


def optimal_control_action(env, total_budget):
    """
    Implement the optimal control strategy using precomputed centralities and influence powers.

    Args:
        env (NetworkGraph): The NetworkGraph environment.
        total_budget (int): The total number of units of u_max to spend.

    Returns:
        tuple: (control input vector to be applied, remaining budget, indices of controlled agents)
    """
    centralities = env.centralities
    opinions = env.opinions
    desired_opinion = env.desired_opinion

    # Calculate influence power for each agent
    influence_powers = centralities * np.abs(desired_opinion - opinions)

    # Sort agents based on influence power in descending order
    sorted_indices = np.argsort(influence_powers)[::-1]

    # Initialize control inputs
    u = np.zeros(env.num_agents)
    remaining_budget = total_budget * env.max_u  # Convert the total budget into actual control units

    controlled_agents = []

    for idx in sorted_indices:
        if remaining_budget <= 0:
            break
        u[idx] = min(env.max_u, remaining_budget)
        remaining_budget -= u[idx]
        controlled_agents.append(idx)

    return u, remaining_budget, controlled_agents


def dynamic_programming_strategy(env, M, Q, step_duration):
    """
    Implement the dynamic programming algorithm to determine the optimal budget allocation across campaigns,
    using the actual dynamics of the NetworkGraph environment and the same control action selection method
    as used in the simulation (optimal_control_action).

    Args:
        env (NetworkGraph): The NetworkGraph environment.
        M (int): The number of campaigns.
        Q (int): The total budget (number of units of max_u).
        step_duration (float): Total duration of each campaign.

    Returns:
        list: The optimal budget allocation for each campaign.
        list: The indices of nodes being controlled in each campaign.
        list: The control inputs applied in each campaign.
        float: The final opinion error after applying the optimal policy.
    """
    # Main function body
    N = env.num_agents  # Number of agents
    ubar = env.max_u  # Maximum control input per agent
    d = env.desired_opinion  # Desired opinion

    # Discretize the opinion space
    nx = 10  # Number of discretization points
    opinion_grid = np.linspace(0, 1, nx)

    # Initialize the value function V and policy
    V = initialize_value_function(opinion_grid, M, Q, d)
    policy = [{} for _ in range(M)]  # Policy stores the best action at each state

    # Perform backward induction to fill V and policy
    perform_backward_induction(V, policy, env, opinion_grid, M, Q, step_duration)

    # Simulate the system using the optimal policy
    optimal_budget_allocation, nodes_controlled, control_inputs, final_opinion_error = simulate_optimal_policy(
        policy, env, opinion_grid, M, Q, step_duration
    )

    return (
        optimal_budget_allocation,
        nodes_controlled,
        control_inputs,
        final_opinion_error,
    )
    
def initialize_value_function(opinion_grid, M, Q, d):
    """
    Initialize the value function V with terminal costs.

    Args:
        nx (int): Number of discretization points.
        opinion_grid (np.ndarray): Discretized opinion space.
        M (int): Number of campaigns.
        Q (int): Total budget.
        d (float): Desired opinion.

    Returns:
        list: Initialized value function V.
    """
    V = [{} for _ in range(M + 1)]  # V[M] is the value function at the final stage

    # Terminal cost at stage M
    for ix, avg_opinion in enumerate(opinion_grid):
        for rem_budget in range(Q + 1):
            V[M][(ix, rem_budget)] = np.abs(avg_opinion - d)

    return V

def perform_backward_induction(V, policy, env, opinion_grid, M, Q, step_duration):
    """
    Perform the backward induction to fill the value function V and policy.

    Args:
        V (list): Value function initialized with terminal cost.
        policy (list): Policy to be filled during backward induction.
        env (NetworkGraph): The environment.
        opinion_grid (np.ndarray): Discretized opinion space.
        M (int): Number of campaigns.
        Q (int): Total budget.
        step_duration (float): Total duration of each campaign.
    """
    N = env.num_agents
    ubar = env.max_u
    d = env.desired_opinion

    for k in range(M - 1, -1, -1):
        print(f"Processing campaign {k + 1}/{M}")
        for ix, avg_opinion in enumerate(opinion_grid):
            for rem_budget in range(Q + 1):
                min_cost = float("inf")
                best_b = None
                best_u = None
                best_controlled_agents = None
                best_ix_next = None
                best_rem_budget_next = None

                for b in range(0, min(N, rem_budget) + 1):
                    # Create a temporary environment to simulate the step
                    temp_env = deepcopy(env)
                    temp_env.opinions = np.full(N, avg_opinion)

                    # Determine the control action using optimal_control_action
                    u, _, controlled_agents = optimal_control_action(temp_env, b)

                    # Simulate the step using the actual dynamics
                    opinions_after_step, _, _, _, _ = temp_env.step(
                        action=u,
                        step_duration=step_duration,
                    )

                    # Compute the new average opinion
                    avg_opinion_next = np.mean(opinions_after_step)
                    # Find the closest grid point
                    ix_next = np.argmin(np.abs(opinion_grid - avg_opinion_next))

                    # Remaining budget
                    rem_budget_next = rem_budget - b

                    # Immediate cost
                    cost = np.abs(avg_opinion_next - d)

                    # Total cost-to-go
                    future_cost = V[k + 1][(ix_next, rem_budget_next)]
                    total_cost = cost + future_cost

                    if total_cost < min_cost:
                        min_cost = total_cost
                        best_b = b
                        best_u = u.copy()
                        best_controlled_agents = controlled_agents.copy()
                        best_ix_next = ix_next
                        best_rem_budget_next = rem_budget_next

                # Store the minimum cost and best action
                V[k][(ix, rem_budget)] = min_cost
                policy[k][(ix, rem_budget)] = {
                    "b": best_b,
                    "u": best_u,
                    "controlled_agents": best_controlled_agents,
                    "ix_next": best_ix_next,
                    "rem_budget_next": best_rem_budget_next,
                }

def simulate_optimal_policy(policy, env, opinion_grid, M, Q, step_duration):
    """
    Simulate the system using the optimal policy derived from backward induction.

    Args:
        policy (list): Policy derived from backward induction.
        env (NetworkGraph): The environment.
        opinion_grid (np.ndarray): Discretized opinion space.
        M (int): Number of campaigns.
        Q (int): Total budget.
        step_duration (float): Total duration of each campaign.

    Returns:
        tuple: (optimal_budget_allocation, nodes_controlled, control_inputs, final_opinion_error)
    """
    N = env.num_agents
    ubar = env.max_u
    d = env.desired_opinion

    avg_opinion = np.mean(env.opinions)
    ix = np.argmin(np.abs(opinion_grid - avg_opinion))
    rem_budget = Q
    optimal_budget_allocation = []
    nodes_controlled = []
    control_inputs = []

    # Initialize the environment for simulation
    sim_env = deepcopy(env)
    sim_env.opinions = env.opinions.copy()

    for k in range(M):
        policy_entry = policy[k][(ix, rem_budget)]
        b = policy_entry["b"]
        u = policy_entry["u"]
        controlled_agents = policy_entry["controlled_agents"]
        ix_next = policy_entry["ix_next"]
        rem_budget_next = policy_entry["rem_budget_next"]

        # Record the best action
        optimal_budget_allocation.append(b)
        nodes_controlled.append(controlled_agents)
        control_inputs.append(u)

        # Apply the control action to the simulation environment
        opinions_after_step, _, _, _, _ = sim_env.step(
            action=u,
            step_duration=step_duration,
        )

        # Update state
        ix = ix_next
        rem_budget = rem_budget_next

    # Compute the final opinion error
    final_opinions = sim_env.opinions
    final_opinion_error = np.mean(np.abs(final_opinions - d))

    return optimal_budget_allocation, nodes_controlled, control_inputs, final_opinion_error

    
def run_dynamic_programming_campaigns(
    env,
    optimal_budget_allocation,
    step_duration,
    sampling_time,
    final_campaign_step_duration=None,
    final_campaign_sampling_time=None,
):
    """
    Run the campaigns using the optimal budget allocation from dynamic programming,
    applying the allocated budget in each campaign, and collecting opinions over time.
    Returns the affected nodes (controlled agents) in each campaign.

    Args:
        env (NetworkGraph): The NetworkGraph environment.
        optimal_budget_allocation (list): The budget allocation for each campaign.
        step_duration (float): Total duration of each campaign.
        sampling_time (float): Time interval at which to sample the opinions.
        final_campaign_step_duration (float, optional): Step duration for the final campaign.
        final_campaign_sampling_time (float, optional): Sampling time for the final campaign.

    Returns:
        np.ndarray: The opinions over time.
        np.ndarray: The time points corresponding to the opinions.
        list: List of arrays containing the indices of agents controlled in each campaign.
    """
    M = len(optimal_budget_allocation)  # Number of campaigns
    opinions_over_time = []
    time_points = []
    current_time = 0.0
    nodes_controlled_simulation = []

    # Compute initial influence powers and agent ordering
    deviations_initial = np.abs(env.opinions - env.desired_opinion)
    influence_powers_initial = env.centralities * deviations_initial
    agent_order = np.argsort(influence_powers_initial)[::-1]

    for k in range(M):
        print(f"Running campaign {k + 1}/{M}")
        # Determine the budget for this campaign
        budget_k = optimal_budget_allocation[k]

        # Determine the control action using the allocated budget and fixed agent order
        action = np.zeros(env.num_agents)
        if budget_k > 0:
            action[agent_order[:budget_k]] = env.max_u
        controlled_nodes = np.where(action == env.max_u)[0]
        nodes_controlled_simulation.append(controlled_nodes)

        # Determine the step_duration and sampling_time for this campaign
        if k == M - 1 and final_campaign_step_duration is not None:
            # Use the specified step_duration and sampling_time for the final campaign
            campaign_step_duration = final_campaign_step_duration
            campaign_sampling_time = final_campaign_sampling_time or sampling_time
        else:
            # Use the default step_duration and sampling_time
            campaign_step_duration = step_duration
            campaign_sampling_time = sampling_time

        # Run the campaign with sampling
        opinions_campaign, times_campaign = run_campaign_with_sampling(
            env,
            action=action,
            step_duration=campaign_step_duration,
            sampling_time=campaign_sampling_time,
        )

        # Append the opinions and time points
        if len(opinions_over_time) == 0:
            opinions_over_time = opinions_campaign
            time_points = times_campaign + current_time
        else:
            # Exclude the first time point to avoid duplicates
            opinions_over_time = np.vstack((opinions_over_time, opinions_campaign[1:]))
            time_points = np.concatenate(
                (time_points, times_campaign[1:] + current_time)
            )

        # Update current time
        current_time += campaign_step_duration

    return opinions_over_time, time_points, nodes_controlled_simulation


def compute_average_error(opinions_over_time, desired_opinion):
    """
    Compute the average error from the desired opinion over time.

    Args:
        opinions_over_time (np.ndarray): Array of opinions over time.
        desired_opinion (float): The desired opinion value.

    Returns:
        float: The average error.
    """
    # Compute the absolute error from the desired opinion at each time step
    errors = np.abs(opinions_over_time - desired_opinion)

    # Compute the mean error across all agents and time steps
    average_error = np.mean(errors)

    return average_error


def compute_final_average_error(opinions_over_time, desired_opinion):
    """
    Compute the average error from the desired opinion at the last time step.

    Args:
        opinions_over_time (np.ndarray): Array of opinions over time.
        desired_opinion (float): The desired opinion value.

    Returns:
        float: The average error at the final time step.
    """
    # Get the opinions at the final time step
    final_opinions = opinions_over_time[-1]

    # Compute the absolute error from the desired opinion at the final step
    final_error = np.mean(np.abs(final_opinions - desired_opinion))

    return final_error


def run_uncontrolled_experiment(env, num_steps):
    """
    Run the experiment with no control (uncontrolled case).

    Args:
        env (NetworkGraph): The NetworkGraph environment.
        num_steps (int): Total number of steps in the simulation.

    Returns:
        np.ndarray: The opinions over time.
    """
    # Initialize an array to store opinions over time
    opinions_over_time = np.zeros((num_steps, env.num_agents))

    # Run the simulation with no control input
    for i in range(num_steps):
        optimal_u = np.zeros(env.num_agents)
        opinions, reward, done, truncated, info = env.step(optimal_u)
        opinions_over_time[i] = opinions

    return opinions_over_time


def run_broadcast_strategy(env, total_budget, experiment_steps):
    """
    Run the experiment using the broadcast strategy (spending the entire budget at the initial time).

    Args:
        env (NetworkGraph): The NetworkGraph environment.
        total_budget (int): The total number of units of u_max to spend.
        experiment_steps (int): Total number of steps in the simulation.

    Returns:
        np.ndarray: The opinions over time.
        list: Budget distribution across campaigns (single campaign).
        list: Nodes affected in each campaign (single campaign).
    """
    # Initialize an array to store opinions over time
    opinions_over_time = np.zeros((experiment_steps, env.num_agents))

    # Lists to store budget distribution and affected nodes
    budget_distribution = []
    affected_nodes = []

    # Store opinions before applying the first control step
    opinions_over_time[0] = env.opinions.copy()

    # Calculate the control action using the total budget (units of u_max)
    optimal_action, remaining_budget, controlled_agents = optimal_control_action(env, total_budget)
    print(optimal_action)

    # Apply the control for the first step
    opinions, reward, done, truncated, info = env.step(optimal_action)
    opinions_over_time[1] = opinions

    # Track affected nodes and budget
    affected_nodes.append(list(np.where(optimal_action > 0)[0]))
    budget_distribution.append(np.sum(optimal_action))  # Budget used in the first step

    # No control applied for the remaining steps
    for i in range(2, experiment_steps):
        no_action_u = np.zeros(env.num_agents)  # No control input
        opinions, reward, done, truncated, info = env.step(no_action_u)
        opinions_over_time[i] = opinions

    return opinions_over_time, budget_distribution, affected_nodes


def run_campaign_with_sampling(
    env, action, step_duration, sampling_time
):
    """
    Run the experiment applying a given control action over a specified action duration,
    propagating the dynamics over a total step duration, and recording the opinions
    at specified sampling intervals, applying the action only once.

    Args:
        env (NetworkGraph): The NetworkGraph environment.
        action (np.ndarray): Control action to apply.
        step_duration (float): Total duration over which the opinions are propagated.
        sampling_time (float): Time interval at which to sample the opinions.

    Returns:
        np.ndarray: The opinions over time at the specified sampling intervals.
        np.ndarray: The corresponding time points.
    """

    # Initialize lists to store opinions over time
    num_samples = int(np.ceil(step_duration / sampling_time)) + 1  # +1 to include t=0
    opinions_over_time = np.zeros((num_samples, env.num_agents))
    time_points = np.zeros(num_samples)

    # Record initial opinions
    opinions_over_time[0] = env.opinions.copy()
    time_points[0] = 0.0

    
    # Simulate dynamics over the total step_duration, sampling at sampling_time intervals
    total_time = 0.0
    idx = 1  # Index for opinions_over_time

    while total_time < step_duration and idx < num_samples:
        # Determine the duration for this step
        dt = min(sampling_time, step_duration - total_time)

        propagation_duration = dt

        # Propagate opinions without applying control
        expL = expm(-env.L * propagation_duration)
        env.opinions = expL @ env.opinions
        env.opinions = np.clip(env.opinions, 0, 1)

        # Record the opinions
        total_time += dt
        opinions_over_time[idx] = env.opinions.copy()
        time_points[idx] = total_time
        idx += 1

    return opinions_over_time, time_points


def normalize_campaign_time(
    time_points, campaign_durations, step_duration, final_campaign_step_duration
):

    # Total number of campaigns
    M = len(campaign_durations)

    # Compute cumulative durations to find campaign boundaries
    cumulative_durations = np.cumsum([0] + campaign_durations)
    campaign_boundaries = cumulative_durations

    # Initialize the normalized time points array
    normalized_time_points = np.zeros_like(time_points)

    # Process each campaign
    for i in range(M):
        # Get start and end times for the campaign
        start_time = campaign_boundaries[i]
        end_time = campaign_boundaries[i + 1]

        # Find indices corresponding to this campaign
        start_idx = np.searchsorted(time_points, start_time, side="left")
        end_idx = np.searchsorted(time_points, end_time, side="right")

        # Extract campaign time points
        campaign_time = time_points[start_idx:end_idx]

        # Normalize time within the campaign to span from i to i+1
        if len(campaign_time) > 0:
            campaign_duration = campaign_time[-1] - campaign_time[0]
            if campaign_duration == 0:
                # If the campaign duration is zero, set all times to i
                normalized_campaign_time = np.full_like(campaign_time, i)
            else:
                # Normalize to [i, i+1]
                normalized_campaign_time = (
                    i + (campaign_time - campaign_time[0]) / campaign_duration
                )
            normalized_time_points[start_idx:end_idx] = normalized_campaign_time

    return normalized_time_points


# def dynamic_programming_strategy(env, M, Q, step_duration):
#     """
#     Implement the dynamic programming algorithm to determine the optimal budget allocation across campaigns,
#     using the actual dynamics of the NetworkGraph environment and the same control action selection method
#     as used in the simulation (optimal_control_action).

#     Args:
#         env (NetworkGraph): The NetworkGraph environment.
#         M (int): The number of campaigns.
#         Q (int): The total budget (number of units of max_u).
#         step_duration (float): Total duration of each campaign.

#     Returns:
#         list: The optimal budget allocation for each campaign.
#         list: The indices of nodes being controlled in each campaign.
#         list: The control inputs applied in each campaign.
#         float: The final opinion error after applying the optimal policy.
#     """
#     import numpy as np
#     from copy import deepcopy

#     def optimal_control_action(env, budget_k):
#         """
#         Determine the optimal control action given the current environment and budget.
#         Selects the top 'budget_k' agents based on their influence powers.

#         Args:
#             env (NetworkGraph): The environment.
#             budget_k (int): The budget allocated to control agents.

#         Returns:
#             np.ndarray: The control input vector.
#             np.ndarray: The indices of the agents being controlled.
#         """
#         N = env.num_agents
#         ubar = env.max_u
#         d = env.desired_opinion
#         # Compute influence powers
#         deviations = np.abs(env.opinions - d)
#         influence_powers = env.centralities * deviations
#         # Order agents based on influence powers
#         sorted_indices = np.argsort(influence_powers)[::-1]
#         # Initialize control input
#         u = np.zeros(N)
#         if budget_k > 0:
#             u[sorted_indices[:budget_k]] = ubar
#         return u, sorted_indices[:budget_k]

#     N = env.num_agents  # Number of agents
#     ubar = env.max_u  # Maximum control input per agent
#     d = env.desired_opinion  # Desired opinion

#     # Discretize the opinion space
#     nx = 10  # Number of discretization points
#     opinion_grid = np.linspace(0, 1, nx)

#     # Initialize the value function V[k][ix][rem_budget]
#     V = [{} for _ in range(M + 1)]  # V[M] is the value function at the final stage
#     policy = [{} for _ in range(M)]  # Policy stores the best action at each state

#     # Terminal cost at stage M
#     for ix, avg_opinion in enumerate(opinion_grid):
#         for rem_budget in range(Q + 1):
#             V[M][(ix, rem_budget)] = np.abs(avg_opinion - d)

#     # Backward induction
#     for k in range(M - 1, -1, -1):
#         print(f"Processing campaign {k + 1}/{M}")
#         for ix, avg_opinion in enumerate(opinion_grid):
#             for rem_budget in range(Q + 1):
#                 min_cost = float("inf")
#                 best_b = None
#                 best_u = None
#                 best_ix_next = None
#                 best_rem_budget_next = None

#                 for b in range(0, min(N, rem_budget) + 1):
#                     # Create a temporary environment to simulate the step
#                     temp_env = deepcopy(env)
#                     temp_env.opinions = np.full(N, avg_opinion)

#                     # Determine the control action using optimal_control_action
#                     u, _ = optimal_control_action(temp_env, b)

#                     # Simulate the step using the actual dynamics
#                     opinions_after_step, _, _, _, _ = temp_env.step(
#                         action=u,
#                         step_duration=step_duration,
#                     )

#                     # Compute the new average opinion
#                     avg_opinion_next = np.mean(opinions_after_step)
#                     # Find the closest grid point
#                     ix_next = np.argmin(np.abs(opinion_grid - avg_opinion_next))

#                     # Remaining budget
#                     rem_budget_next = rem_budget - b

#                     # Immediate cost
#                     cost = np.abs(avg_opinion_next - d)

#                     # Total cost-to-go
#                     future_cost = V[k + 1][(ix_next, rem_budget_next)]
#                     total_cost = cost + future_cost

#                     if total_cost < min_cost:
#                         min_cost = total_cost
#                         best_b = b
#                         best_u = u.copy()
#                         best_ix_next = ix_next
#                         best_rem_budget_next = rem_budget_next

#                 # Store the minimum cost and best action
#                 V[k][(ix, rem_budget)] = min_cost
#                 policy[k][(ix, rem_budget)] = {
#                     "b": best_b,
#                     "u": best_u,
#                     "ix_next": best_ix_next,
#                     "rem_budget_next": best_rem_budget_next,
#                 }

#     # Extract the optimal policy and simulate to compute final opinion error
#     avg_opinion = np.mean(env.opinions)
#     ix = np.argmin(np.abs(opinion_grid - avg_opinion))
#     rem_budget = Q
#     optimal_budget_allocation = []
#     nodes_controlled = []
#     control_inputs = []

#     # Initialize the environment for simulation
#     sim_env = deepcopy(env)
#     sim_env.opinions = env.opinions.copy()

#     for k in range(M):
#         policy_entry = policy[k][(ix, rem_budget)]
#         b = policy_entry["b"]
#         u = policy_entry["u"]
#         ix_next = policy_entry["ix_next"]
#         rem_budget_next = policy_entry["rem_budget_next"]

#         # Record the best action
#         optimal_budget_allocation.append(b)
#         controlled_nodes = np.where(u == ubar)[0]
#         nodes_controlled.append(controlled_nodes)
#         control_inputs.append(u)

#         # Apply the control action to the simulation environment
#         opinions_after_step, _, _, _, _ = sim_env.step(
#             action=u, step_duration=step_duration
#         )

#         # Update state
#         ix = ix_next
#         rem_budget = rem_budget_next

#     # Compute the final opinion error
#     final_opinions = sim_env.opinions
#     final_opinion_error = np.mean(np.abs(final_opinions - d))

#     return (
#         optimal_budget_allocation,
#         nodes_controlled,
#         control_inputs,
#         final_opinion_error,
#     )
