import matplotlib.pyplot as plt
import numpy as np
import math


def interpolate_opinion_trajectory(env, opinions_over_time, actions, n_substeps=10):
    """
    Interpolate intermediate opinions between control steps using env.compute_dynamics.

    Args:
        env: The environment (must implement compute_dynamics).
        opinions_over_time (np.ndarray): Shape (T+1, N), opinions at each main step.
        actions (List[np.ndarray]): List of control actions (length T).
        n_substeps (int): Number of substeps per control step.

    Returns:
        interpolated_opinions (np.ndarray): Shape ((T * n_substeps + 1), N).
        interpolated_times (np.ndarray): Time points per substep.
    """
    dt = env.tau / n_substeps
    interpolated = []
    times = []

    # --- Append initial unmodified state at time 0 ---
    interpolated.append(opinions_over_time[0].copy())
    times.append(0.0)

    for t in range(len(actions)):
        x_start = opinions_over_time[t]
        u = actions[t]

        # Apply impulse control (treated as instantaneous at t * tau)
        x = env.compute_dynamics(
            current_state=x_start, control_action=u, step_duration=0.0
        )
        interpolated.append(x.copy())
        times.append(t * env.tau)

        # return None, None
        # Propagate over substeps
        for k in range(1, n_substeps + 1):
            x = env.compute_dynamics(
                current_state=x, control_action=np.zeros_like(u), step_duration=dt
            )
            interpolated.append(x.copy())
            times.append(t * env.tau + k * dt)

    return np.array(interpolated), np.array(times)


def plot_action_heatmap(actions, step_labels=None, target_xticks=12):
    """
    Plot a heatmap of actions over time, with agents on the Y-axis.
    X-axis shows standardized intervals (…, 0–5–10–… or 0–10–20–…).

    Args:
        actions (array-like): shape (T, N)
        step_labels (list[str] or None): optional custom labels per step (length T)
        target_xticks (int): target maximum number of visible x ticks
    """
    actions = np.asarray(actions)
    T, N = actions.shape

    # Heatmap data: agents on Y
    data = actions.T  # (N, T)
    x_edges = np.arange(T + 1)
    y_edges = np.arange(N + 1)

    fig, ax = plt.subplots(figsize=(12, 6))
    mesh = ax.pcolormesh(x_edges, y_edges, data, cmap="viridis", shading="auto")

    cbar = plt.colorbar(mesh, ax=ax)
    cbar.set_label("Control Magnitude")

    # ---------- standardized x ticks (nice steps: 1, 2, 5 × 10^k) ----------
    if T > 1:
        # desired raw step in indices
        raw = max(1, int(math.ceil(T / max(3, target_xticks))))
        k = int(math.floor(math.log10(raw)))
        base = 10**k
        for m in (1, 2, 5, 10):
            step = m * base
            if step >= raw:
                break
        tick_idx = np.arange(0, T, step, dtype=int)
        # ensure we include the last index if far from the last tick
        if (
            len(tick_idx) == 0
            or tick_idx[-1] < T - 1
            and (T - 1) - tick_idx[-1] >= step // 2
        ):
            tick_idx = np.append(tick_idx, T - 1)
    else:
        tick_idx = np.array([0], dtype=int)

    # Build labels: numeric indices or provided step_labels
    if step_labels is None:
        xtick_labels = [str(i) for i in tick_idx]
    else:
        # clamp in case user passed fewer/more labels
        xtick_labels = [str(step_labels[i]) for i in tick_idx]

    # ticks at cell centers -> +0.5
    ax.set_xticks(tick_idx + 0.5)
    ax.set_xticklabels(xtick_labels)
    ax.set_xlabel("Time Step")
    ax.set_xlim(0, T)

    # Y-axis (agents) at cell centers
    agent_labels = [f"N{i}" for i in range(N)]
    ax.set_yticks(np.arange(N) + 0.5)
    ax.set_yticklabels(agent_labels)
    ax.set_ylabel("Nodes")

    ax.set_title("Control Actions Heatmap (Agents on Y-axis)")
    ax.grid(visible=True, axis="y", color="white", linestyle="--", linewidth=0.5)

    # If labels still feel tight, uncomment:
    # plt.setp(ax.get_xticklabels(), fontsize=9)

    plt.tight_layout()
    plt.show()


def plot_campaign_budgets(optimal_budget_allocation, nodes_controlled, control_inputs):
    """
    Plot the budget allocation across campaigns and display the nodes controlled in each campaign.

    Args:
        optimal_budget_allocation (list): The optimal budget allocation for each campaign.
        nodes_controlled (list): List of arrays containing indices of nodes controlled in each campaign.
        control_inputs (list): List of arrays containing control inputs applied in each campaign.
    """
    control_inputs = control_inputs.copy()
    M = len(optimal_budget_allocation)  # Number of campaigns
    campaigns = np.arange(1, M + 1)  # Campaign numbers for the x-axis

    # Calculate the total budget used in each campaign from control_inputs
    total_budgets = [np.sum(u) for u in control_inputs]

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(campaigns, total_budgets, color="skyblue", edgecolor="black")

    # Set labels and title
    plt.xlabel("Campaign")
    plt.ylabel("Total Budget Used")
    plt.title("Budget Allocation Across Campaigns")

    # Display the indices of the nodes controlled above each bar
    for idx, bar in enumerate(bars):
        # Get the height of the bar (total budget)
        height = bar.get_height()

        # Get the indices of nodes controlled in this campaign
        nodes = nodes_controlled[idx]

        if len(nodes) > 0:
            # Sort the nodes
            nodes = sorted(nodes)
            # Create a string of node indices separated by commas
            nodes_str = ", ".join(map(str, nodes))
        else:
            nodes_str = "No nodes controlled"

        # Display the node indices above the bar
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(total_budgets) * 0.02,  # Slightly above the bar
            nodes_str,
            ha="center",
            va="bottom",
            fontsize=10,
            rotation=0,
        )

    plt.xticks(campaigns)  # Set x-axis ticks to campaign numbers
    plt.ylim(0, max(total_budgets) * 1.1)  # Set y-axis limit for better spacing

    plt.tight_layout()
    plt.show()


def plot_campaign_budgets_with_order(optimal_budget_allocation, order, order0, umax):
    """
    Plot the budget allocation across campaigns and display the nodes controlled in each campaign.

    Args:
        optimal_budget_allocation (list): The optimal budget allocation for each campaign.
        order (list): Sorted indices of nodes by centrality for campaigns after the first.
        order0 (list): Sorted indices of nodes by initial influence powers for the first campaign.
        umax (float): The maximum control input that can be applied to a node.
    """
    M = len(optimal_budget_allocation)  # Number of campaigns
    campaigns = np.arange(1, M + 1)  # Campaign numbers for the x-axis

    control_inputs = []
    nodes_controlled = []

    # Iterate through each campaign and build the control inputs and nodes controlled
    for k in range(M):
        # Select the appropriate order (order0 for the first campaign, order for others)
        if k == 0:
            current_order = order0
        else:
            current_order = order

        # Determine the number of nodes to control based on the budget allocation
        budget_k = optimal_budget_allocation[k]

        # Apply maximum control input to the top nodes according to the budget
        u = np.zeros(len(current_order))
        u[current_order[:budget_k]] = umax

        # Store control inputs and nodes controlled for this campaign
        control_inputs.append(u)
        nodes_controlled.append(current_order[:budget_k])

    # Calculate the total budget used in each campaign from control_inputs
    total_budgets = [np.sum(u) for u in control_inputs]

    # Create the bar plot
    plt.figure(figsize=(10, 6))
    bars = plt.bar(campaigns, total_budgets, color="skyblue", edgecolor="black")

    # Set labels and title
    plt.xlabel("Campaign")
    plt.ylabel("Total Budget Used")
    plt.title("Budget Allocation Across Campaigns")

    # Display the indices of the nodes controlled above each bar
    for idx, bar in enumerate(bars):
        # Get the height of the bar (total budget)
        height = bar.get_height()

        # Get the indices of nodes controlled in this campaign
        nodes = nodes_controlled[idx]

        if len(nodes) > 0:
            # Sort the nodes
            nodes = sorted(nodes)
            # Create a string of node indices separated by commas
            nodes_str = ", ".join(map(str, nodes))
        else:
            nodes_str = "No nodes controlled"

        # Display the node indices above the bar
        plt.text(
            bar.get_x() + bar.get_width() / 2,
            height + max(total_budgets) * 0.02,  # Slightly above the bar
            nodes_str,
            ha="center",
            va="bottom",
            fontsize=10,
            rotation=0,
        )

    plt.xticks(campaigns)  # Set x-axis ticks to campaign numbers
    plt.ylim(0, max(total_budgets) * 1.1)  # Set y-axis limit for better spacing

    plt.tight_layout()
    plt.show()


def plot_opinions_over_time(opinions_over_time, time_points=None, title=None):
    """
    Plot the opinions of each agent over time.

    Args:
        opinions_over_time (np.ndarray): Array of shape (num_steps, num_agents) containing the opinions at each time step.
        time_points (np.ndarray, optional): Array of time points corresponding to each sample.
                                            If None, time steps are used as the x-axis.
        title (str, optional): Custom title for the plot. Defaults to "Opinion Convergence Over Time".
    """
    num_agents = opinions_over_time.shape[1]

    if time_points is None:
        time_points = np.arange(opinions_over_time.shape[0])

    plt.figure(figsize=(12, 6))
    for agent_idx in range(num_agents):
        plt.plot(
            time_points, opinions_over_time[:, agent_idx], label=f"Agent {agent_idx}"
        )

    plt.xlabel("Time" if time_points is not None else "Time Steps")
    plt.ylabel("Opinion")
    plt.title(title if title is not None else "Opinion Convergence Over Time")
    plt.grid(True)
    plt.tight_layout()
    plt.show()


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


def visualize_policy_from_env(policy, env, nx, node_index):
    """
    Visualize the policy with respect to the value of a specific node and
    the mean opinion of the network. Colors now represent the action value
    continuously via a colorbar.
    """
    N = env.num_agents  # Number of agents
    grids = [np.linspace(0, 1, nx) for _ in range(N)]  # Uniform grid in [0, 1]

    node_opinions = []  # X-axis: Opinion of the specific node
    mean_opinions = []  # Y-axis: Mean opinion of the network
    action_values = []  # Store the continuous action for this node

    # Convert policy dict states -> scatter plot data
    for idx, action in policy.items():
        # Map state indices to actual grid values
        state = [grids[i][idx[i]] for i in range(N)]
        node_opinion = state[node_index]
        mean_opinion = np.mean(state)

        node_opinions.append(node_opinion)
        mean_opinions.append(mean_opinion)
        action_values.append(action[node_index])  # Actual action (0..max_u)

    plt.figure(figsize=(10, 6))
    # Use a scatter plot colored by action value with a colormap
    sc = plt.scatter(
        node_opinions,
        mean_opinions,
        c=action_values,  # color by the node's action
        cmap="viridis",  # choose any colormap you like
        s=50,
        alpha=0.8,
    )

    # Add a colorbar showing the range of action values
    cbar = plt.colorbar(sc)
    cbar.set_label("Action Value")

    plt.xlabel(f"Opinion of Node {node_index}")
    plt.ylabel("Mean Opinion of Network")
    plt.title(f"Policy Visualization for Node {node_index}")
    plt.grid(True)
    plt.show()
