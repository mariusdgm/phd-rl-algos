import matplotlib.pyplot as plt
import numpy as np

def plot_gamblers_policy(policy, max_capital):
    """
    Plots the policy for the Gambler's Problem.

    Args:
        policy (dict): A dictionary mapping states (capital amounts) to actions (stakes).
        max_capital (int): The maximum amount of capital.
    """
    policy_array = np.zeros(max_capital + 1, dtype=int)
    for capital in range(max_capital + 1):
        policy_array[capital] = policy.get(capital, 0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(max_capital + 1), policy_array, marker='o')
    plt.xlabel('Capital')
    plt.ylabel('Stake')
    plt.title('Optimal Policy (Stake at Each Capital Level)')
    plt.grid(True)
    plt.show()

def plot_gamblers_value_function(V, max_capital):
    """
    Plots the value function for the Gambler's Problem.

    Args:
        V (dict): A dictionary mapping states (capital amounts) to values (floats).
        max_capital (int): The maximum amount of capital.
    """
    value_array = np.zeros(max_capital + 1)
    for capital in range(max_capital + 1):
        value_array[capital] = V.get(capital, 0)

    plt.figure(figsize=(10, 6))
    plt.plot(range(max_capital + 1), value_array, marker='o')
    plt.xlabel('Capital')
    plt.ylabel('Value')
    plt.title('Value Function for Each Capital Level')
    plt.grid(True)
    plt.show()