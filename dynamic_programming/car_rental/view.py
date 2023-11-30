import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
import numpy as np

def get_text_color(background_color):
    """Determine text color (black or white) based on background color intensity."""
    return "white" if sum(background_color[:3]) < 1.5 else "black"


def plot_car_rental_policy(policy, max_cars):
    """
    Plots the policy grid for Jack's Car Rental problem.

    Args:
        policy (dict): A dictionary mapping states (tuples) to actions (ints).
        max_cars (int): The maximum number of cars at each location.
    """
    policy_grid = np.zeros((max_cars + 1, max_cars + 1), dtype=int)

    for cars_at_A in range(max_cars + 1):
        for cars_at_B in range(max_cars + 1):
            state = (cars_at_A, cars_at_B)
            policy_grid[max_cars - cars_at_B, cars_at_A] = policy.get(
                state, 0
            )  # Flipping rows for correct orientation

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("viridis")
    im = ax.imshow(policy_grid, cmap=cmap, interpolation="nearest")

    ax.set_xlabel("Number of Cars at Location A")
    ax.set_ylabel("Number of Cars at Location B")
    ax.set_xticks(range(max_cars + 1))
    ax.set_yticks(range(max_cars + 1))
    ax.set_xticklabels(range(max_cars + 1))
    ax.set_yticklabels(
        reversed(range(max_cars + 1))
    )  # Reversed for correct orientation

    # Loop over data dimensions and create text annotations.
    for i in range(max_cars + 1):
        for j in range(max_cars + 1):
            color = cmap(policy_grid[i, j] / policy_grid.max())
            text_color = get_text_color(color)
            text = ax.text(
                j, i, policy_grid[i, j], ha="center", va="center", color=text_color
            )

    ax.set_title("Optimal Policy (Action to Take at Each State)")
    fig.colorbar(im, ax=ax)
    plt.show()

def plot_car_rental_value_function(V, max_cars):
    """
    Plots the value function grid for Jack's Car Rental problem.

    Args:
        V (dict): A dictionary mapping states (tuples) to values (floats).
        max_cars (int): The maximum number of cars at each location.
    """
    value_grid = np.zeros((max_cars + 1, max_cars + 1))

    for cars_at_A in range(max_cars + 1):
        for cars_at_B in range(max_cars + 1):
            state = (cars_at_A, cars_at_B)
            value_grid[max_cars - cars_at_B, cars_at_A] = V.get(state, 0)  # Flipping rows for correct orientation

    fig, ax = plt.subplots(figsize=(8, 8))
    cmap = plt.get_cmap("viridis")
    norm = mcolors.Normalize(vmin=value_grid.min(), vmax=value_grid.max(), clip=True)
    im = ax.imshow(value_grid, cmap=cmap, norm=norm, interpolation="nearest")

    ax.set_xlabel("Number of Cars at Location A")
    ax.set_ylabel("Number of Cars at Location B")
    ax.set_xticks(range(max_cars + 1))
    ax.set_yticks(range(max_cars + 1))
    ax.set_xticklabels(range(max_cars + 1))
    ax.set_yticklabels(reversed(range(max_cars + 1)))  # Reversed for correct orientation

    # Adjust font size based on the number of cells
    font_size = max(4, 30 / np.sqrt(max_cars))  # Decrease font size for larger grids

    # Loop over data dimensions and create text annotations.
    for i in range(max_cars + 1):
        for j in range(max_cars + 1):
            color = cmap(norm(value_grid[i, j]))
            text_color = get_text_color(color)
            formatted_value = f"{value_grid[i, j]:.1f}"  # One decimal place
            text = ax.text(j, i, formatted_value, ha="center", va="center", color=text_color, fontsize=font_size)

    ax.set_title("State-Value Function (V) for Each State")
    fig.colorbar(im, ax=ax)
    plt.show()