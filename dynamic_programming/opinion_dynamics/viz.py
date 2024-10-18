import matplotlib.pyplot as plt
import numpy as np


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

def plot_opinions_over_time(opinions_over_time, time_points=None):
    """
    Plot the opinions of each agent over time.

    Args:
        opinions_over_time (np.ndarray): Array of shape (num_steps, num_agents) containing the opinions at each time step.
        time_points (np.ndarray, optional): Array of time points corresponding to each sample.
                                            If None, time steps are used as the x-axis.
    """
    num_agents = opinions_over_time.shape[
        1
    ]  # Infer the number of agents from the array shape

    if time_points is None:
        # Use time steps as the x-axis
        time_points = np.arange(opinions_over_time.shape[0])

    # Plot the opinions over time for each agent
    plt.figure(figsize=(12, 6))
    for agent_idx in range(num_agents):
        plt.plot(
            time_points, opinions_over_time[:, agent_idx], label=f"Agent {agent_idx}"
        )

    plt.xlabel("Time" if time_points is not None else "Time Steps")
    plt.ylabel("Opinion")
    plt.title("Opinion Convergence Over Time")
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
