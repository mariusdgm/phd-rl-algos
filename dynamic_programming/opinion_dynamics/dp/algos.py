import numpy as np
from itertools import product
from copy import deepcopy
from scipy.linalg import expm
from scipy.interpolate import interp1d
import numpy as np
from itertools import product, combinations
from copy import deepcopy


def run_policy_campaigns(env, policy, nx, step_duration, sampling_time, tolerance=0.01, max_campaigns=100):
    """
    Run the simulation using a given policy with a maximum number of campaigns.

    Args:
        env: The environment with network properties.
        policy: The control policy to be used.
        nx: Number of grid points per dimension (used for mapping states to grid indices).
        step_duration: Duration of each policy application.
        sampling_time: Time between samples during a campaign.
        tolerance: Tolerance for the mean opinion to stop the campaigns.
        max_campaigns: Maximum number of campaigns to run.

    Returns:
        opinions_over_time: Array of opinions over the entire simulation.
        time_points: Corresponding time points for each opinion snapshot.
        nodes_controlled_simulation: List of nodes controlled in each campaign.
    """
    N = env.num_agents
    opinions_over_time = []
    time_points = []
    current_time = 0.0
    nodes_controlled_simulation = []
    X = env.opinions.copy()
    grids = create_state_grid(N, nx)
    campaign_count = 0  # Initialize campaign count

    while True:
        # Increment campaign count
        campaign_count += 1

        # Map the current state to the closest grid point indices
        idx = tuple(np.abs(grids[i] - X[i]).argmin() for i in range(N))
        # Get the control action from the policy
        control_input = policy.get(idx, np.zeros(N))

        # Keep track of which nodes are controlled (non-zero control input)
        controlled_nodes = np.where(control_input > 0)[0]
        nodes_controlled_simulation.append(controlled_nodes)

        # Apply control and simulate over step_duration with sampling
        opinions_campaign, times_campaign = simulate_campaign_with_sampling(
            env, control_input, X, step_duration, sampling_time
        )

        # Adjust time points to include the cumulative time
        times_campaign += current_time

        # Append the opinions and time points from the current campaign
        if len(opinions_over_time) == 0:
            opinions_over_time = opinions_campaign
            time_points = times_campaign
        else:
            opinions_over_time = np.vstack((opinions_over_time, opinions_campaign[1:]))
            time_points = np.concatenate((time_points, times_campaign[1:]))

        # Update the current time and state
        current_time = time_points[-1]
        X = opinions_campaign[-1].copy()

        # Check if the mean opinion is within the tolerance
        mean_opinion_error = np.abs(np.mean(X) - env.desired_opinion)
        if mean_opinion_error <= tolerance:
            break

        # Check if maximum number of campaigns has been reached
        if campaign_count >= max_campaigns:
            print("Maximum number of campaigns reached.")
            break

    print(f"Simulation stopped after {campaign_count} campaigns.")
    return opinions_over_time, time_points, nodes_controlled_simulation

def simulate_campaign_with_sampling(env, control_input, X, step_duration, sampling_time):
    N = env.num_agents
    num_samples = int(np.ceil(step_duration / sampling_time)) + 1
    opinions_over_time = np.zeros((num_samples, N))
    time_points = np.zeros(num_samples)

    opinions_over_time[0] = X.copy()
    time_points[0] = 0.0
    current_time = 0.0

    # Simulate over the step_duration with sampling
    for idx in range(1, num_samples):
        dt = min(sampling_time, step_duration - current_time)
        current_time += dt

        # Apply control only at the first sample
        if idx == 1:
            control = control_input
        else:
            control = np.zeros(N)

        # Update the state using the dynamics
        X = env.compute_dynamics(X, control, dt)
        X = np.clip(X, 0, 1)

        # Record the opinions and time
        opinions_over_time[idx] = X.copy()
        time_points[idx] = current_time

    return opinions_over_time, time_points

def generate_full_control_policy(env, nx):
    """
    Generate a control policy that applies maximum control to all agents in all states.

    Args:
        env: The environment with network properties.
        nx: Number of grid points per dimension (used for state grid creation).

    Returns:
        policy: A dictionary mapping state indices to control actions.
    """
    N = env.num_agents
    ubar = env.max_u
    grids = create_state_grid(N, nx)
    grid_shape = tuple(len(grid) for grid in grids)

    # Define the full control action (maximum control for all agents)
    full_control_action = np.full(N, ubar)
    policy = {}

    # Map every state in the grid to the full control action
    for idx in np.ndindex(grid_shape):
        policy[idx] = full_control_action.copy()

    return policy
    
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

        controlled_agents.append(idx)

    return u, remaining_budget, controlled_agents


def dynamic_programming_multiplicative(env, M, TB):
    """
    Generalized multiplicative dynamic programming for long-stage control.

    Args:
        env (NetworkGraph): The environment instance.
        M (int): Number of campaigns.
        TB (int): Total budget.

    Returns:
        V (ndarray): Value function of shape (M+1, TB+1).
        B (ndarray): Decision table for optimal budget allocation.
        order, order0 (list): Sorted agent orders based on centrality and initial deviation.
    """
    N = env.num_agents
    ubar = env.max_u
    xd = env.desired_opinion
    x0 = np.copy(env.opinions)
    eigv = env.centralities

    # Agent orderings based on centralities and initial influence powers
    order = np.argsort(eigv)[::-1]  # Descending order of centralities
    score0 = eigv * np.abs(x0 - xd)
    order0 = np.argsort(score0)[::-1]  # Descending order of score0

    # Function f(beta) and f0(beta) for dynamic programming
    def f(beta):
        beta = min(beta, N)
        return 1 - ubar * np.sum(eigv[order[:beta]])

    def f0(beta):
        beta = min(beta, N)
        return np.sum(score0) - ubar * np.sum(score0[order0[:beta]])

    # Initialize value function V and decision table B
    V = np.full((M + 1, TB + 1), np.nan)
    B = np.full((M + 1, TB + 1), np.nan)

    # At the final stage, spend the whole remaining budget
    for rem in range(TB + 1):
        V[M, rem] = np.log(f(rem))
        B[M, rem] = rem

    # Backward induction from stage M-1 down to 1
    for k in range(M - 1, 0, -1):
        print(f"DPMULT step {k}")
        for rem in range(TB + 1):  # Loop over remaining budget
            val = np.inf
            betastar = 0
            for beta in range(0, min(rem, N) + 1):
                # Value of investing beta when remaining budget is rem
                if k > 1:
                    q = np.log(f(beta)) + V[k + 1, rem - beta]
                else:
                    q = np.log(f0(beta)) + V[k + 1, rem - beta]
                if q < val:
                    val = q
                    betastar = beta
            V[k, rem] = val
            B[k, rem] = betastar

    return V, B, order, order0


def forward_propagation_multiplicative(env, V, B, order, order0, M, TB):
    """
    Forward propagation from given initial state using dynamic programming results.

    Args:
        env (NetworkGraph): The environment instance.
        V (ndarray): Value function from dynamic programming.
        B (ndarray): Decision table for optimal budget allocation.
        order (list): Agent order based on centrality.
        order0 (list): Agent order based on initial influence.
        M (int): Number of campaigns.
        TB (int): Total budget.

    Returns:
        BETA (list): Optimal budget allocations.
        X (ndarray): State of the network after each campaign.
        U (ndarray): Control inputs applied to the agents.
        final_cost (float): Final cost (discrepancy from target opinion).
    """
    N = env.num_agents
    ubar = env.max_u
    xd = env.desired_opinion
    eigv = env.centralities

    X = np.zeros((N, M + 1))
    BETA = np.zeros(M, dtype=int)
    U = np.zeros((N, M))
    X[:, 0] = env.opinions.copy()  # Initial opinions
    rem = TB

    for k in range(M):
        BETA[k] = int(B[k + 1, rem])  # Budget allocation for stage k
        if k == 0:
            U[order0[: BETA[k]], k] = ubar
        else:
            U[order[: BETA[k]], k] = ubar

        # Update the opinions with influence and network dynamics
        X[:, k + 1] = np.dot(eigv, (U[:, k] * xd + (1 - U[:, k]) * X[:, k]))
        rem -= BETA[k]

    # Calculate the final cost based on the discrepancy from the desired opinion
    final_cost = np.abs(np.mean(X[:, -1]) - xd)
    return BETA, U, X, final_cost

def create_state_grid(N, nx):
    grid_range = np.linspace(0, 1, nx)
    grids = [grid_range for _ in range(N)]
    return grids

def interpolate_to_grid(value, grid):
    """Interpolate a value onto a grid to find fractional index."""
    if value <= grid[0]:
        return 0
    elif value >= grid[-1]:
        return len(grid) - 1
    else:
        lower_idx = np.searchsorted(grid, value) - 1
        lower_val, upper_val = grid[lower_idx], grid[lower_idx + 1]
        return lower_idx + (value - lower_val) / (upper_val - lower_val)

def dynamic_programming_high_dim(env, M, TB, nx=10, gamma=1):
    env = deepcopy(env)
    N = env.num_agents
    ubar = env.max_u
    xd = env.desired_opinion
    grids = create_state_grid(N, nx)
    grid_shape = tuple(len(grid) for grid in grids)
    
    V = np.full((M + 1,) + grid_shape + (TB + 1,), np.inf)
    B_dict = {}

    grid_points = np.array(np.meshgrid(*grids, indexing='ij')).T.reshape(-1, N)
    terminal_cost = np.abs(grid_points - xd).mean(axis=1).reshape(grid_shape)
    
    V[M, ..., :] = np.repeat(terminal_cost[..., np.newaxis], TB + 1, axis=-1)

    for k in range(M - 1, -1, -1):
        print(f"\nDP step {k}")
        for rem in range(TB + 1):
            for idx in np.ndindex(grid_shape):
                current_states = np.array([grids[i][idx[i]] for i in range(N)])
                val = np.inf
                best_control, best_nodes = None, None

                for beta_used in range(0, min(rem, N) + 1):
                    if rem - beta_used < 0:
                        continue

                    for target_nodes in combinations(range(N), beta_used):
                        control_input = np.zeros(N)
                        for node in target_nodes:
                            control_input[node] = ubar
                        
                        # Directly compute new states and map to nearest grid indices
                        new_states = env.compute_dynamics(current_states, control_input, step_duration=30)
                        new_states_clipped = np.clip(new_states, 0, 1)

                        # Get nearest grid indices rather than interpolating
                        next_idx = tuple(np.abs(grids[i] - new_states_clipped[i]).argmin() for i in range(N))
                        future_cost = V[k + 1][next_idx + (rem - beta_used,)]

                        immediate_cost = np.abs(new_states_clipped - xd).mean()
                        total_cost = immediate_cost + gamma * future_cost

                        if total_cost < val:
                            val = total_cost
                            best_control = control_input
                            best_nodes = target_nodes
                            
                if best_control is not None:
                    V[(k,) + idx + (rem,)] = val
                    B_dict[(k,) + idx + (rem,)] = {'control': best_control, 'nodes': best_nodes}

    return V, B_dict


def forward_propagation_high_dim(env, V, B_dict, M, TB, nx=10):
    env = deepcopy(env)
    N = env.num_agents
    grids = create_state_grid(N, nx)
    X = np.zeros((N, M + 1))
    U = np.zeros((N, M))
    BETA_hd = np.zeros(M, dtype=int)
    X[:, 0] = env.opinions.copy()
    rem = TB
    xd = env.desired_opinion
    ubar = env.max_u

    total_cost = 0

    for k in range(M):
        current_states = X[:, k]
        idx = tuple(np.abs(grids[i] - current_states[i]).argmin() for i in range(N))

        key = (k,) + idx + (rem,)
        control_data = B_dict.get(key, {'control': np.zeros(N), 'nodes': []})
        control_input, target_nodes = control_data['control'], control_data['nodes']
        U[:, k] = control_input

        new_states = env.compute_dynamics(current_states, control_input, step_duration=100)
        X[:, k + 1] = np.clip(new_states, 0, 1)

        beta_k = len(target_nodes)
        BETA_hd[k] = beta_k
        rem -= beta_k

        immediate_cost = np.mean(np.abs(new_states - xd))
        next_idx = tuple(np.abs(grids[i] - new_states[i]).argmin() for i in range(N))

        try:
            future_cost = V[k + 1][next_idx + (rem,)]
        except IndexError:
            future_cost = np.inf
            print(f"Warning: Index error at stage {k}, next_idx={next_idx}, rem={rem}")

        total_cost += immediate_cost + future_cost

        print(f"Stage {k} -> rem={rem}, beta_k={beta_k}, control_input={control_input}")
        print(f"  current_states={current_states}, new_states={new_states}")
        print(f"  immediate_cost={immediate_cost}, future_cost={future_cost}")
        print(f"  Total cost so far: {total_cost}")

        if rem < 0 or beta_k > TB:
            print(f"Warning: Budget inconsistency detected at stage {k}. rem={rem}, beta_k={beta_k}")
            break

    final_cost = np.mean(np.abs((X[:, -1] - xd)))
    print(f"Final X: {X[:, -1]}")
    print(f"Final Cost from forward propagation: {final_cost}")
    print(f"Total Expected Cost (from V and propagation): {total_cost}")
    
    return X, U, BETA_hd, final_cost

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
    x0 = np.copy(env.opinions)
    uzero = np.zeros(N)
    X = np.zeros(M + 1)
    X[0] = np.mean(x0)
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

def dynamic_programming_with_grid(env, M, TB, nx=10, epsilon=1e-8):
    """
    Implement dynamic programming with state grid and budget.

    Args:
        env (NetworkGraph): The environment instance.
        M (int): Number of campaigns.
        K (int): Total budget.
        nx (int): Number of discretization points on the grid for x.

    Returns:
        V (ndarray): Value function of shape (M+1, nx, K+1).
        order, order0 (list): Sorted agent orders based on centrality and initial deviation.
    """
    N = env.num_agents
    ubar = env.max_u
    x0 = np.mean(env.opinions)
    xd = env.desired_opinion
    eigv = env.centralities

    # Grid for the state
    Xgrid = np.linspace(0, 1, nx)

    # Agent orderings based on centralities and initial influence powers
    order = np.argsort(eigv)[::-1]  # Descending order of centralities
    score0 = eigv * np.abs(x0 - xd)
    order0 = np.argsort(score0)[::-1]  # Descending order of score0

    uzero = np.zeros(N, dtype=np.float64)

    # Initialize value function V
    V = np.full((M + 1, nx, TB + 1), np.inf)

    # Final cost for the last stage (adjust to Python indexing)
    for ix in range(nx):
        V[M, ix, :] = np.abs(Xgrid[ix] - xd)

    # Backward induction from stage M-1 down to stage 2
    for k in range(M - 1, 0, -1):
        print(f"DP step {k}")
        for ix in range(nx):  # Loop over grid points for the current state
            for rem in range(TB + 1):  # Loop over remaining budget
                x = Xgrid[ix]
                val = np.inf
                u = uzero.copy()
                for beta in range(0, min(rem, N) + 1):
                    if beta > 0:
                        u[order[beta - 1]] = ubar
                    xplus = np.dot(eigv, u * xd + (1 - u) * x)
                    # xplus = np.clip(
                    #     xplus, Xgrid[0], Xgrid[-1]
                    # )
                    remplus = rem - beta

                    # Interpolate future cost
                    vplus = interp1d(
                        Xgrid,
                        V[k + 1, :, remplus],
                        kind="linear",
                        # fill_value="extrapolate",
                    )(xplus)
                    # vplus = np.interp(xplus, Xgrid, V[k + 1, :, remplus])
                    val = min(val, vplus)

                V[k, ix, rem] = val

    return V, order, order0


def forward_propagation_with_grid(env, V, order, order0, M, TB, nx=10, epsilon=1e-8):
    """
    Forward propagation from given initial state using dynamic programming results.

    Args:
        env (NetworkGraph): The environment instance.
        V (ndarray): Value function from dynamic programming.
        order (list): Agent order based on centrality.
        order0 (list): Agent order based on initial influence.
        M (int): Number of campaigns.
        TB (int): Total budget.
        nx (int): Number of discretization points on the grid for x.

    Returns:
        BETA (list): Optimal budget allocations.
        X (ndarray): State of the network after each campaign.
        U (ndarray): Control inputs applied to the agents.
        final_cost (float): Final cost (discrepancy from target opinion).
    """
    N = env.num_agents
    ubar = env.max_u
    x0 = np.mean(env.opinions)
    xd = env.desired_opinion
    eigv = env.centralities
    Xgrid = np.linspace(0, 1, nx)  # State grid

    X = np.zeros((N, M + 1))
    BETA = np.zeros(M, dtype=int)
    U = np.zeros((N, M))
    X[:, 0] = env.opinions.copy()  # Initial opinions
    rem = TB
    uzero = np.zeros(N, dtype=np.float64)
    # xplusstar = np.inf
    # betastar = 1
    # remstar = 0

    for k in range(M):
        val = np.inf
        ustar = uzero.copy()
        u = uzero.copy()
        for beta in range(0, min(rem, N - 1) + 1):
            if beta > 0:
                if k == 0:
                    u[order0[beta - 1]] = ubar
                else:
                    u[order[beta - 1]] = ubar
            xplus = np.dot(eigv, u * xd + (1 - u) * X[:, k])
            # xplus = np.clip(
            #     xplus, Xgrid[0], Xgrid[-1]
            # )
            remplus = rem - beta

            vplus = interp1d(
                Xgrid,
                V[k + 1, :, remplus],
                kind="linear",
                # fill_value="extrapolate"
            )(xplus)
            # vplus = np.interp(xplus, Xgrid, V[k + 1, :, remplus])
            if vplus < val:  # save best solution
                val = vplus
                xplusstar = xplus
                remstar = remplus
                ustar = u.copy()
                betastar = beta

        U[:, k] = ustar
        X[:, k + 1] = xplusstar
        BETA[k] = betastar
        rem = remstar

    final_cost = np.abs(np.mean(X[:, -1]) - xd)
    return BETA, U, X, final_cost