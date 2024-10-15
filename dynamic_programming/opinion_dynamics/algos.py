import numpy as np
from itertools import product
from copy import deepcopy
from scipy.linalg import expm
from scipy.interpolate import interp1d


def optimal_control_action(env, total_budget):
    """
    Implement the optimal control strategy using precomputed centralities and influence powers.

    Args:
        env (NetworkGraph): The NetworkGraph environment.
        total_budget (int): The total number of units of max_u to spend.

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
    remaining_budget = total_budget * env.max_u  # Total control units available

    controlled_agents = []

    epsilon = 1e-8  # Threshold for floating-point precision errors

    for idx in sorted_indices:
        if remaining_budget < epsilon:
            break  # Remaining budget is effectively zero
        control_input = min(env.max_u, remaining_budget)
        u[idx] = control_input
        remaining_budget -= control_input
        # Correct any negative remaining budget due to floating-point errors
        if remaining_budget < 0:
            remaining_budget = 0.0
        controlled_agents.append(idx)

    return u, remaining_budget, controlled_agents


def state_transition(env, x, u, step_duration):
    # Compute the controlled opinions
    opinions_controlled = u * env.desired_opinion + (1 - u) * x

    # Propagate opinions over the network
    expL = expm(-env.L * step_duration)
    opinions_next = expL @ opinions_controlled

    # Compute the next average opinion
    xplus = env.centralities @ opinions_next  # Assuming centralities sum to 1
    return xplus


def dynamic_programming_strategy(env, M, Q, step_duration):
    N = env.num_agents
    d = env.desired_opinion
    ubar = env.max_u
    x0 = np.mean(env.opinions)

    nx = 20  # Number of discretization points
    Xgrid = np.linspace(0, 1, nx)

    # Compute initial influence powers for the first iteration
    deviations = np.abs(env.opinions - d)
    influence_powers = env.centralities * deviations
    order0 = np.argsort(influence_powers)[::-1]

    # Agent order based on centrality for subsequent iterations
    order = np.argsort(env.centralities)[::-1]

    # Initialize value function V
    V = np.full((M + 1, nx, Q + 1), np.inf)
    # Final cost at stage M
    for ix, x in enumerate(Xgrid):
        V[M, ix, :] = 0  # No future cost

    # Initialize policy
    policy = [{} for _ in range(M)]

    # Backward induction
    for k in range(M - 1, -1, -1):
        print(f"DP step {k + 1}")
        for ix, x in enumerate(Xgrid):
            for rem in range(Q + 1):
                val = np.inf
                best_policy_entry = None

                for beta in range(0, min(rem, N) + 1):
                    u = np.zeros(N)
                    if beta > 0:
                        if k == M - 1:  # First iteration
                            u[order0[:beta]] = ubar
                        else:
                            u[order[:beta]] = ubar

                    xplus = state_transition(env, x, u, step_duration)
                    remplus = rem - beta

                    if remplus >= 0:
                        # Immediate cost at current stage
                        immediate_cost = abs(xplus - d)

                        # Future cost
                        if k < M - 1:
                            future_cost_function = interp1d(
                                Xgrid,
                                V[k + 1, :, remplus],
                                kind="linear",
                                fill_value="extrapolate",
                            )
                            vplus = future_cost_function(xplus)
                        else:
                            vplus = 0  # No future cost at the last stage

                        total_cost = immediate_cost + vplus

                        if total_cost < val:
                            val = total_cost
                            best_policy_entry = {
                                "beta": beta,
                                "u": u.copy(),
                                "xplus": xplus,
                                "remplus": remplus,
                                "controlled_agents": (
                                    order0[:beta] if k == M - 1 else order[:beta]
                                ),
                            }

                V[k, ix, rem] = val
                if best_policy_entry is not None:
                    policy_key = (ix, rem)
                    policy[k][policy_key] = best_policy_entry

    return policy, Xgrid, V


def extract_optimal_policy(policy, env, Xgrid, V, M, Q, step_duration):
    N = env.num_agents
    d = env.desired_opinion
    ubar = env.max_u
    rem = Q
    X = np.zeros(M + 1)
    X[0] = np.mean(env.opinions)
    optimal_budget_allocation = []
    nodes_controlled = []
    control_inputs = []

    # Compute initial influence powers for the first iteration
    deviations = np.abs(env.opinions - d)
    influence_powers = env.centralities * deviations
    order0 = np.argsort(influence_powers)[::-1]

    # Agent order based on centrality for subsequent iterations
    order = np.argsort(env.centralities)[::-1]

    total_cost = 0  # Initialize total cost

    for k in range(M):
        val = np.inf
        ustar = np.zeros(N)
        x = X[k]

        for beta in range(0, min(rem, N) + 1):
            u = np.zeros(N)
            if beta > 0:
                if k == 0:  # First iteration
                    u[order0[:beta]] = ubar
                else:
                    u[order[:beta]] = ubar

            xplus = state_transition(env, x, u, step_duration)
            remplus = rem - beta

            if remplus >= 0:
                # Immediate cost
                immediate_cost = abs(xplus - d)

                if k < M - 1:
                    future_cost_function = interp1d(
                        Xgrid,
                        V[k + 1, :, remplus],
                        kind="linear",
                        fill_value="extrapolate",
                    )
                    vplus = future_cost_function(xplus)
                else:
                    vplus = 0  # No future cost at the last stage

                total_cost_candidate = total_cost + immediate_cost + vplus

                if total_cost_candidate < val:
                    val = total_cost_candidate
                    xplusstar = xplus
                    remstar = remplus
                    ustar = u.copy()
                    betastar = beta
                    controlled_agents = np.where(u > 0)[0]
                    immediate_cost_star = immediate_cost

        # Update state and budget
        X[k + 1] = xplusstar
        rem = remstar
        total_cost += immediate_cost_star

        # Store the control action
        control_inputs.append(ustar)
        optimal_budget_allocation.append(betastar)
        nodes_controlled.append(controlled_agents)

    final_opinion_error = abs(X[-1] - d)
    return (
        optimal_budget_allocation,
        nodes_controlled,
        control_inputs,
        final_opinion_error,
        X,
    )


def compute_expected_value_for_budget_distribution(
    budget_allocation, env, M, step_duration
):
    """
    Simulate the opinion dynamics with a given budget distribution and compute the expected value.

    Args:
        budget_allocation (list): Budget allocated in each campaign.
        env (NetworkGraph): The environment instance.
        M (int): Number of campaigns.
        step_duration (float): Duration of each campaign.

    Returns:
        final_opinion_error (float): The absolute difference between the final average opinion and the desired opinion.
        total_cost (float): The sum of immediate costs across all campaigns.
        costs (list): Immediate cost at each campaign.
        X (list): Average opinion after each campaign.
    """
    N = env.num_agents
    d = env.desired_opinion
    ubar = env.max_u
    eigv = env.centralities
    eigv_normalized = eigv / np.sum(eigv)
    x0 = np.mean(env.opinions)
    uzero = np.zeros(N)
    X = np.zeros(M + 1)
    X[0] = x0
    total_cost = 0
    costs = []

    # Agent orders
    order = np.argsort(eigv)[::-1]
    order0 = np.argsort(eigv * np.abs(env.opinions - d))[::-1]
    rem = sum(budget_allocation)

    sim_env = deepcopy(env)
    sim_env.opinions = env.opinions.copy()

    for k in range(M):
        beta = budget_allocation[k]
        u = uzero.copy()
        beta_int = int(beta)  # Ensure beta is an integer number of agents
        if beta_int > 0:
            if k == 0:
                u[order0[:beta_int]] = ubar
            else:
                u[order[:beta_int]] = ubar
        # Apply control
        opinions_after_step, _, _, _, _ = sim_env.step(
            action=u,
            step_duration=step_duration,
        )
        sim_env.opinions = opinions_after_step.copy()
        xplus = np.mean(sim_env.opinions)
        # Compute immediate cost
        immediate_cost = abs(xplus - d)
        total_cost += immediate_cost
        costs.append(immediate_cost)
        X[k + 1] = xplus
    final_opinion_error = abs(X[-1] - d)
    return final_opinion_error, total_cost, costs, X


def run_dynamic_programming_campaigns(
    env,
    optimal_budget_allocation,
    step_duration,
    sampling_time,
    final_campaign_step_duration=None,
    final_campaign_sampling_time=None,
):
    M = len(optimal_budget_allocation)  # Number of campaigns
    opinions_over_time = []
    time_points = []
    current_time = 0.0
    nodes_controlled_simulation = []

    for k in range(M):
        # print(f"Running campaign {k + 1}/{M}")
        # Determine the budget for this campaign
        budget_k = optimal_budget_allocation[k]

        # Compute influence powers and agent ordering based on current opinions
        deviations = np.abs(env.opinions - env.desired_opinion)
        influence_powers = env.centralities * deviations
        agent_order = np.argsort(influence_powers)[::-1]

        # Determine the control action using the allocated budget and agent order
        action = np.zeros(env.num_agents)
        if budget_k > 0:
            action[agent_order[:budget_k]] = env.max_u
        controlled_nodes = np.where(action == env.max_u)[0]
        nodes_controlled_simulation.append(controlled_nodes)

        # Determine the step_duration and sampling_time for this campaign
        if k == M - 1 and final_campaign_step_duration is not None:
            campaign_step_duration = final_campaign_step_duration
            campaign_sampling_time = final_campaign_sampling_time or sampling_time
        else:
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
    optimal_action, remaining_budget, controlled_agents = optimal_control_action(
        env, total_budget
    )
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


def run_campaign_with_sampling(env, action, step_duration, sampling_time):
    # Initialize lists to store opinions over time
    num_samples = int(np.ceil(step_duration / sampling_time)) + 1  # +1 to include t=0
    opinions_over_time = np.zeros((num_samples, env.num_agents))
    time_points = np.zeros(num_samples)

    # Record initial opinions
    opinions_over_time[0] = env.opinions.copy()
    time_points[0] = 0.0

    # env.opinions = action * env.desired_opinion + (1 - action) * env.opinions
    # env.total_spent += np.sum(action)
    # env.current_step += 1

    env.step(action=action, step_duration=sampling_time)

    # Simulate dynamics over the total step_duration
    total_time = 0.0
    idx = 1  # Index for opinions_over_time

    while total_time < step_duration and idx < num_samples:
        # Determine the duration for this step
        dt = min(sampling_time, step_duration - total_time)

        # # Propagate opinions without applying control
        # expL = expm(-env.L * dt)
        # env.opinions = expL @ env.opinions
        # env.opinions = np.clip(env.opinions, 0, 1)

        env.step(action=np.zeros(env.num_agents), step_duration=dt)

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


# Closest result so far


# def dynamic_programming_strategy(env, M, Q, step_duration):
#     N = env.num_agents
#     d = env.desired_opinion
#     ubar = env.max_u
#     eigv = env.centralities / np.sum(env.centralities)
#     x0 = np.mean(env.opinions)

#     nx = 100  # Increase discretization points for better accuracy
#     Xgrid = np.linspace(0, 1, nx)

#     # Agents order based on eigenvector centrality
#     order = np.argsort(env.centralities)[::-1]
#     uzero = np.zeros(N)

#     # Initialize value function V
#     V = np.full((M + 1, nx, Q + 1), np.inf)
#     # Final cost at stage M
#     for ix, x in enumerate(Xgrid):
#         V[M, ix, :] = abs(x - d)

#     # Initialize policy
#     policy = [{} for _ in range(M)]

#     # Backward induction from stage M down to stage 2
#     for k in range(M - 1, 0, -1):
#         print(f"DP step {k + 1}")
#         for ix, x in enumerate(Xgrid):  # Current state
#             for rem in range(Q + 1):  # Remaining budget
#                 val = np.inf
#                 best_policy_entry = None
#                 for beta in range(0, min(rem, N) + 1):
#                     u = uzero.copy()
#                     if beta > 0:
#                         u[order[:beta]] = ubar
#                     # Next agent state
#                     xplus = np.dot(eigv, u * d + (1 - u) * x)
#                     # Remaining budget
#                     remplus = rem - beta
#                     # Interpolate the future cost
#                     future_cost_function = interp1d(
#                         Xgrid,
#                         V[k + 1, :, remplus],
#                         kind="linear",
#                         fill_value="extrapolate",
#                     )
#                     vplus = future_cost_function(xplus)
#                     total_cost = vplus
#                     if total_cost < val:
#                         val = total_cost
#                         V[k, ix, rem] = val
#                         best_policy_entry = {
#                             "beta": beta,
#                             "u": u.copy(),
#                             "xplus": xplus,
#                             "remplus": remplus,
#                             "controlled_agents": order[:beta],
#                         }
#                 if best_policy_entry is not None:
#                     policy_key = (ix, rem)
#                     policy[k][policy_key] = best_policy_entry
#     return policy, Xgrid, V


# def extract_optimal_policy(policy, env, Xgrid, V, M, Q, step_duration):
#     N = env.num_agents
#     d = env.desired_opinion
#     ubar = env.max_u
#     eigv = env.centralities / np.sum(env.centralities)
#     x0 = np.mean(env.opinions)
#     uzero = np.zeros(N)
#     # Agent orders
#     order = np.argsort(env.centralities)[::-1]
#     order0 = np.argsort(env.centralities * np.abs(env.opinions - d))[::-1]
#     rem = Q
#     X = np.zeros(M + 1)
#     X[0] = x0
#     optimal_budget_allocation = []
#     nodes_controlled = []
#     control_inputs = []

#     for k in range(M):
#         val = np.inf
#         ustar = uzero.copy()
#         best_policy_entry = None
#         for beta in range(0, min(rem, N) + 1):
#             u = uzero.copy()
#             if beta > 0:
#                 if k == 0:  # Stage 1
#                     u[order0[:beta]] = ubar
#                 else:
#                     u[order[:beta]] = ubar
#             xplus = np.dot(eigv, u * d + (1 - u) * X[k])
#             remplus = rem - beta
#             if remplus < 0 or remplus > Q:
#                 continue
#             if k < M - 1:
#                 future_cost_function = interp1d(
#                     Xgrid, V[k + 1, :, remplus], kind="linear", fill_value="extrapolate"
#                 )
#                 vplus = future_cost_function(xplus)
#             else:
#                 vplus = abs(xplus - d)
#             total_cost = vplus
#             if total_cost < val:
#                 val = total_cost
#                 xplusstar = xplus
#                 remstar = remplus
#                 ustar = u.copy()
#                 betastar = beta
#                 controlled_agents = np.where(u > 0)[0]
#         control_inputs.append(ustar)
#         X[k + 1] = xplusstar
#         optimal_budget_allocation.append(betastar)
#         nodes_controlled.append(controlled_agents)
#         rem = remstar
#     final_opinion_error = abs(X[-1] - d)
#     return (
#         optimal_budget_allocation,
#         nodes_controlled,
#         control_inputs,
#         final_opinion_error,
#         X,
#     )


# def compute_expected_value_for_budget_distribution(
#     budget_allocation, env, M, step_duration
# ):
#     """
#     Simulate the opinion dynamics with a given budget distribution and compute the expected value.

#     Args:
#         budget_allocation (list): Budget allocated in each campaign.
#         env (NetworkGraph): The environment instance.
#         M (int): Number of campaigns.
#         step_duration (float): Duration of each campaign.

#     Returns:
#         final_opinion_error (float): The absolute difference between the final average opinion and the desired opinion.
#         total_cost (float): The sum of immediate costs across all campaigns.
#         costs (list): Immediate cost at each campaign.
#         X (list): Average opinion after each campaign.
#     """
#     N = env.num_agents
#     d = env.desired_opinion
#     ubar = env.max_u
#     eigv = env.centralities
#     eigv_normalized = eigv / np.sum(eigv)
#     x0 = np.mean(env.opinions)
#     uzero = np.zeros(N)
#     X = np.zeros(M + 1)
#     X[0] = x0
#     total_cost = 0
#     costs = []

#     # Agent orders
#     order = np.argsort(eigv)[::-1]
#     order0 = np.argsort(eigv * np.abs(env.opinions - d))[::-1]
#     rem = sum(budget_allocation)

#     sim_env = deepcopy(env)
#     sim_env.opinions = env.opinions.copy()

#     for k in range(M):
#         beta = budget_allocation[k]
#         u = uzero.copy()
#         beta_int = int(beta)  # Ensure beta is an integer number of agents
#         if beta_int > 0:
#             if k == 0:
#                 u[order0[:beta_int]] = ubar
#             else:
#                 u[order[:beta_int]] = ubar
#         # Apply control
#         opinions_after_step, _, _, _, _ = sim_env.step(
#             action=u,
#             step_duration=step_duration,
#         )
#         sim_env.opinions = opinions_after_step.copy()
#         xplus = np.mean(sim_env.opinions)
#         # Compute immediate cost
#         immediate_cost = abs(xplus - d)
#         total_cost += immediate_cost
#         costs.append(immediate_cost)
#         X[k + 1] = xplus
#     final_opinion_error = abs(X[-1] - d)
#     return final_opinion_error, total_cost, costs, X
