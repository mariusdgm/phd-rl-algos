import numpy as np
from itertools import product
import matplotlib.pyplot as plt


def plot_opinions_over_time(opinions_over_time):
    """
    Plot the opinions of each agent over time.

    Args:
        opinions_over_time (np.ndarray): Array of shape (num_steps, num_agents) containing the opinions at each time step.
    """
    num_agents = opinions_over_time.shape[
        1
    ]  # Infer the number of agents from the array shape

    # Plot the opinions over time for each agent
    plt.figure(figsize=(10, 6))
    for agent_idx in range(num_agents):
        plt.plot(opinions_over_time[:, agent_idx], label=f"Agent {agent_idx}")

    plt.xlabel("Time Steps")
    plt.ylabel("Opinion")
    plt.title("Opinion Convergence Over Time")
    # plt.legend()
    plt.grid(True)
    plt.show()


def optimal_control_action(env, total_budget):
    """
    Implement the optimal control strategy using precomputed centralities and influence powers.

    Args:
        env (NetworkGraph): The NetworkGraph environment.
        total_budget (int): The total number of units of u_max to spend.

    Returns:
        np.ndarray: The control input vector to be applied.
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
    remaining_budget = (
        total_budget * env.max_u
    )  # Convert the total budget into actual control units

    for idx in sorted_indices:
        if remaining_budget <= 0:
            break
        u[idx] = min(env.max_u, remaining_budget)
        remaining_budget -= u[idx]

    return u, remaining_budget


def dynamic_programming_strategy(env, M, Q, action_duration, step_duration):
    """
    Implement the dynamic programming algorithm to determine the optimal budget allocation across campaigns.

    Args:
        env (NetworkGraph): The NetworkGraph environment.
        M (int): The number of campaigns.
        Q (int): The total budget (number of units of max_u).
        action_duration (float): Duration over which the action is applied in each campaign.
        step_duration (float): Total duration of each campaign.

    Returns:
        list: The optimal budget allocation for each campaign.
    """
    N = env.num_agents  # Number of agents
    ubar = env.max_u  # Maximum control input
    d = env.desired_opinion  # Desired opinion

    # Discretize the opinion space
    nx = 10  # Number of discretization points per agent
    opinion_grid = np.linspace(0, 1, nx)

    # Initialize the value function V[k][state][remaining_budget]
    V = [{} for _ in range(M + 1)]  # V[M] is the value function at the final stage

    # Compute the terminal cost at the final campaign (stage M)
    # For each possible state (opinions of agents) and remaining budget
    # Since the state space is large (opinions of all agents), we can consider the average opinion or some aggregated measure
    # For simplicity, we'll use the average opinion

    # Terminal cost
    for avg_opinion in opinion_grid:
        V[M][(avg_opinion, 0)] = np.abs(avg_opinion - d)

    # Backward induction
    for k in range(M - 1, -1, -1):
        print(f"Processing campaign {k}")
        V_k = V[k]
        V_k_plus_1 = V[k + 1]
        for avg_opinion in opinion_grid:
            for rem_budget in range(Q + 1):
                min_cost = float("inf")
                best_b = None
                best_next_state = None

                # Consider all possible budget allocations at this stage
                for b in range(
                    0, min(rem_budget, N) + 1
                ):  # b is the number of agents to control
                    # Simulate the effect of controlling 'b' agents
                    # Select 'b' most influential agents based on centralities and current opinions

                    # Create a temporary environment to simulate the step
                    temp_env = deepcopy(env)
                    temp_env.opinions = np.full(N, avg_opinion)

                    # Determine which agents to control
                    # Calculate influence power for each agent
                    influence_powers = temp_env.centralities * np.abs(
                        d - temp_env.opinions
                    )
                    sorted_indices = np.argsort(influence_powers)[::-1]

                    # Initialize control inputs
                    u = np.zeros(N)
                    indices_to_control = sorted_indices[:b]
                    u[indices_to_control] = ubar  # Apply maximum control input

                    # Simulate the step
                    opinions_after_step, _, _, _, _ = temp_env.step(
                        action=u,
                        action_duration=action_duration,
                        step_duration=step_duration,
                    )

                    # Compute the new average opinion
                    avg_opinion_next = np.mean(opinions_after_step)

                    # Round to nearest grid point
                    avg_opinion_next = min(
                        opinion_grid, key=lambda x: abs(x - avg_opinion_next)
                    )

                    # Remaining budget
                    rem_budget_next = rem_budget - b

                    # Immediate cost (control cost + deviation from desired opinion)
                    control_cost = 0.01 * np.sum(
                        u
                    )  # Assuming a small cost per unit of control
                    deviation_cost = np.abs(avg_opinion_next - d)
                    total_cost = control_cost + deviation_cost

                    # Total cost-to-go
                    future_cost = V_k_plus_1.get(
                        (avg_opinion_next, rem_budget_next), float("inf")
                    )
                    cost_to_go = total_cost + future_cost

                    if cost_to_go < min_cost:
                        min_cost = cost_to_go
                        best_b = b
                        best_next_state = (avg_opinion_next, rem_budget_next)

                # Store the minimum cost and best action
                V_k[(avg_opinion, rem_budget)] = min_cost

    # Extract the optimal policy
    # Start from the initial average opinion
    avg_opinion = np.mean(env.opinions)
    avg_opinion = min(opinion_grid, key=lambda x: abs(x - avg_opinion))
    rem_budget = Q
    optimal_budget_allocation = []

    for k in range(M):
        min_cost = float("inf")
        best_b = None
        best_next_state = None

        for b in range(0, min(rem_budget, N) + 1):
            # Simulate the effect of controlling 'b' agents
            temp_env = deepcopy(env)
            temp_env.opinions = np.full(N, avg_opinion)

            influence_powers = temp_env.centralities * np.abs(d - temp_env.opinions)
            sorted_indices = np.argsort(influence_powers)[::-1]

            u = np.zeros(N)
            indices_to_control = sorted_indices[:b]
            u[indices_to_control] = ubar

            opinions_after_step, _, _, _, _ = temp_env.step(
                action=u, action_duration=action_duration, step_duration=step_duration
            )

            avg_opinion_next = np.mean(opinions_after_step)
            avg_opinion_next = min(
                opinion_grid, key=lambda x: abs(x - avg_opinion_next)
            )
            rem_budget_next = rem_budget - b

            control_cost = 0.01 * np.sum(u)
            deviation_cost = np.abs(avg_opinion_next - d)
            total_cost = control_cost + deviation_cost

            future_cost = V[k + 1].get(
                (avg_opinion_next, rem_budget_next), float("inf")
            )
            cost_to_go = total_cost + future_cost

            if cost_to_go < min_cost:
                min_cost = cost_to_go
                best_b = b
                best_next_state = (avg_opinion_next, rem_budget_next)

        # Record the best action
        optimal_budget_allocation.append(best_b)
        avg_opinion, rem_budget = best_next_state

    return optimal_budget_allocation


def brute_force_strategy(env, M, Q):
    """
    Implement the brute force algorithm to determine the optimal budget allocation across campaigns.

    Args:
        env (NetworkGraph): The NetworkGraph environment.
        M (int): The number of campaigns.
        Q (int): The total budget.

    Returns:
        np.ndarray: The optimal budget allocation for each campaign.
    """
    N = env.num_agents  # Number of agents
    ubar = env.max_u  # Maximum control input
    v_rho = env.centralities  # Influence vector (precomputed centralities)
    d = env.desired_opinion  # Desired opinion
    initial_opinions = env.opinions.copy()  # Save the initial opinions

    def simulate_dynamics(b_allocation):
        """Simulate the dynamics for a given budget allocation."""
        opinions = initial_opinions.copy()

        for k, b_k in enumerate(b_allocation):
            # Allocate the budget using the optimal control strategy for each campaign
            u = optimal_control_action(env, budget=b_k)
            opinions = u * d + (1 - u) * opinions
            # Apply the network influence (evolution dynamics between campaigns)
            opinions += env.tau * (-env.L @ opinions)
            opinions = np.clip(opinions, 0, 1)  # Ensure opinions remain within [0, 1]

        return opinions

    def compute_cost(final_opinions):
        """Compute the cost as the mean deviation from the desired opinion."""
        return np.mean(np.abs(final_opinions - d))

    # Generate all possible budget allocations
    possible_budgets = list(range(min(N, Q) + 1))
    all_allocations = list(product(possible_budgets, repeat=M + 1))

    # Filter allocations to ensure the total budget does not exceed Q
    valid_allocations = [alloc for alloc in all_allocations if sum(alloc) <= Q]

    # Initialize best cost and allocation
    best_cost = float("inf")
    best_allocation = None

    # Brute force search
    for allocation in valid_allocations:
        final_opinions = simulate_dynamics(allocation)
        cost = compute_cost(final_opinions)

        if cost < best_cost:
            best_cost = cost
            best_allocation = allocation

    return np.array(best_allocation)


def run_experiment(
    env, num_steps, M, Q, sample_time, strategy="brute_force", campaign_length=0.5
):
    """
    Run the experiment to apply the computed campaign budgets and observe the opinions over time.

    Args:
        env (NetworkGraph): The NetworkGraph environment.
        num_steps (int): Total number of steps in the simulation.
        M (int): Number of campaigns.
        Q (int): Total budget.
        sample_time (float): Time step of the simulation.
        strategy (str): The strategy to use for budget allocation ('brute_force' or 'dp').
        campaign_length (float): The length of each campaign in continuous time.

    Returns:
        np.ndarray: The opinions over time.
        list: Budget distribution across campaigns.
        list: Nodes affected in each campaign.
    """
    # Compute the optimal budget allocation using the selected strategy
    if strategy == "brute_force":
        optimal_budget_allocation = brute_force_strategy(env, M, Q)
    elif strategy == "dp":
        optimal_budget_allocation = dynamic_programming_strategy(env, M, Q)
    else:
        raise ValueError("Invalid strategy selected. Choose 'brute_force' or 'dp'.")

    print(f"Optimal budget allocation ({strategy}):", optimal_budget_allocation)

    # Determine the number of steps for the given campaign length
    N = int(
        campaign_length / sample_time
    )  # Number of consecutive steps to apply optimal control

    # Compute the interval between campaigns (equidistantly spread)
    k = int((num_steps - M * N) / (M + 1))

    # Initialize an array to store opinions over time
    opinions_over_time = np.zeros((num_steps, env.num_agents))

    # Lists to store budget distribution and affected nodes
    budget_distribution = []
    affected_nodes = []

    # Run the simulation
    for i in range(num_steps):
        if i % k == 0 and len(optimal_budget_allocation) > 0:
            # Apply the optimal control for N consecutive steps
            current_budget = optimal_budget_allocation[0]
            optimal_budget_allocation = optimal_budget_allocation[1:]
            affected_nodes_in_campaign = []
            for j in range(N):
                if i + j < num_steps:
                    optimal_u = optimal_control_action(env, budget=current_budget)
                    opinions, reward, done, truncated, info = env.step(optimal_u)
                    opinions_over_time[i + j] = opinions
                    # Track affected nodes (non-zero actions)
                    affected_nodes_in_campaign = list(np.where(optimal_u > 0)[0])
            budget_distribution.append(current_budget)
            affected_nodes.append(affected_nodes_in_campaign)
            # Skip the next N-1 steps as they are already processed
            i += N - 1
        else:
            # Apply zero control input at other steps
            optimal_u = np.zeros(env.num_agents)
            opinions, reward, done, truncated, info = env.step(optimal_u)
            opinions_over_time[i] = opinions

    return opinions_over_time, budget_distribution, affected_nodes


def plot_budget_distribution(budget_distribution, affected_nodes):
    """
    Plot the budget distribution across campaigns with annotations for affected nodes.

    Args:
        budget_distribution (list): The budget allocated in each campaign.
        affected_nodes (list): The nodes affected in each campaign.
    """
    num_campaigns = len(budget_distribution)

    plt.figure(figsize=(12, 6))
    bars = plt.bar(range(num_campaigns), budget_distribution, color="skyblue")

    plt.xlabel("Campaign Number")
    plt.ylabel("Budget Allocated")
    plt.title("Budget Distribution Across Campaigns with Affected Nodes")

    for i, bar in enumerate(bars):
        height = bar.get_height()
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            height,
            ", ".join(map(str, affected_nodes[i])),
            ha="center",
            va="bottom",
            fontsize=10,
            rotation=90,
        )

    plt.show()


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
    optimal_action, remaining_budget = optimal_control_action(env, total_budget)
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
