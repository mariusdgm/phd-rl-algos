import numpy as np
from itertools import product
import matplotlib.pyplot as plt

def plot_opinions_over_time(opinions_over_time):
    """
    Plot the opinions of each agent over time.

    Args:
        opinions_over_time (np.ndarray): Array of shape (num_steps, num_agents) containing the opinions at each time step.
    """
    num_agents = opinions_over_time.shape[1]  # Infer the number of agents from the array shape

    # Plot the opinions over time for each agent
    plt.figure(figsize=(10, 6))
    for agent_idx in range(num_agents):
        plt.plot(opinions_over_time[:, agent_idx], label=f'Agent {agent_idx + 1}')

    plt.xlabel('Time Steps')
    plt.ylabel('Opinion')
    plt.title('Opinion Convergence Over Time with Zero Control Input')
    # plt.legend()
    plt.grid(True)
    plt.show()

def optimal_control_action(env, budget):
    """
    Implement the optimal control strategy using precomputed centralities and influence powers.

    Args:
        env (NetworkGraph): The NetworkGraph environment.
        
    Returns:
        np.ndarray: The control input vector to be applied.
    """
    
    centralities = env.centralities
    opinions = env.opinions
    desired_opinion = env.desired_opinion

    # Calculate influence power for each agent
    influence_powers = centralities * np.abs(desired_opinion - opinions)

    # # Sort agents based on influence power in descending order
    sorted_indices = np.argsort(influence_powers)[::-1]

    # Initialize control inputs
    u = np.zeros(env.num_agents)
    remaining_budget = budget

    for idx in sorted_indices:
        if remaining_budget <= 0:
            break
        if u[idx] < env.max_u:
            u[idx] = min(env.max_u, remaining_budget)
            remaining_budget -= u[idx]

    return u


def dynamic_programming_strategy(env, M, Q):
    """
    Implement the dynamic programming algorithm to determine the optimal budget allocation across campaigns.

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
    x_t0 = env.opinions  # Initial opinions
    d = env.desired_opinion  # Desired opinion
    
    # Function f0(b0) for the first campaign
    def f0(b0, v_rho, x_t0, d, ubar):
        term1 = np.sum(v_rho[:b0] * (1 - ubar) * np.abs(x_t0[:b0] - d))
        term2 = np.sum(v_rho[b0:] * np.abs(x_t0[b0:] - d))
        return term1 + term2

    # Function f(b) for subsequent campaigns
    def f(b, v_rho, ubar):
        return 1 - ubar * np.sum(v_rho[:b])

    # Initialize the value function Vk
    V = np.zeros((M + 1, Q + 1))  # V[k, r] represents Vk(rk)

    # Base case: the last campaign
    for r in range(Q + 1):
        V[M, r] = np.log(f(r, v_rho, ubar))

    # Backward pass for intermediate campaigns
    for k in range(M - 1, -1, -1):
        for r in range(Q + 1):
            V[k, r] = np.min([np.log(f(b, v_rho, ubar)) + V[k + 1, r - b] for b in range(min(N, r) + 1)])

    # Initial campaign calculation
    V0 = np.min([np.log(f0(b0, v_rho, x_t0, d, ubar)) + V[1, Q - b0] for b0 in range(min(N, Q) + 1)])

    # Forward pass to find the optimal budget allocations
    b_star = np.zeros(M + 1, dtype=int)
    b_star[0] = np.argmin([np.log(f0(b0, v_rho, x_t0, d, ubar)) + V[1, Q - b0] for b0 in range(min(N, Q) + 1)])

    for k in range(1, M):
        b_star[k] = np.argmin([np.log(f(b, v_rho, ubar)) + V[k + 1, Q - np.sum(b_star[:k]) - b] for b in range(min(N, Q - np.sum(b_star[:k])) + 1)])

    b_star[M] = Q - np.sum(b_star[:M])

    return b_star

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
    best_cost = float('inf')
    best_allocation = None
    
    # Brute force search
    for allocation in valid_allocations:
        final_opinions = simulate_dynamics(allocation)
        cost = compute_cost(final_opinions)
        
        if cost < best_cost:
            best_cost = cost
            best_allocation = allocation
    
    return np.array(best_allocation)