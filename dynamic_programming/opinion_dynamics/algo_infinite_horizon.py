import numpy as np
from itertools import product
from scipy.interpolate import RegularGridInterpolator


def create_state_grid(N, nx):
    grid_range = np.linspace(0, 1, nx)
    grids = [grid_range.copy() for _ in range(N)]
    return grids


def initialize_value_function(N, nx):
    grid_shape = tuple([nx] * N)
    return np.zeros(grid_shape)

def is_terminal(next_state, env):
    return np.abs(np.mean(next_state) - env.desired_opinion) <= env.opinion_end_tolerance

def reward_function(x, u, d, beta):
    return -np.abs(d - x).sum() - beta * np.sum(u)


def build_interpolator(V, grids):
    """
    Build a global interpolator for the current value function V.
    """
    return RegularGridInterpolator(grids, V, bounds_error=False, fill_value=None)


def value_iteration(
    env, nx=10, gamma=1.0, beta=0.0, step_duration=3.0, max_iterations=1000, tol=1e-6
):
    N = env.num_agents
    d = env.desired_opinion

    grids = create_state_grid(N, nx)
    grid_shape = tuple(len(grid) for grid in grids)

    V = initialize_value_function(N, nx)

    action_levels = [0, env.max_u/2, env.max_u]  # 3 levels
    control_actions = list(product(action_levels, repeat=N))

    for iteration in range(max_iterations):
        V_new = np.zeros_like(V)
        max_diff = 0

        interpolator = build_interpolator(V, grids)

        for idx in np.ndindex(grid_shape):
            current_state = np.array([grids[i][idx[i]] for i in range(N)])
            best_value = -np.inf

            for control in control_actions:
                control_input = np.array(control)
                next_state = env.compute_dynamics(
                    current_state, control_input, step_duration
                )
                next_state = np.clip(next_state, 0, 1)

                # next_idx = tuple(np.abs(grids[i] - next_state[i]).argmin() for i in range(N))
                # future_value = V[next_idx]
                if is_terminal(next_state, env):
                    future_value = 0
                else:
                    future_value = interpolator(next_state.reshape(1, -1))[0]

                immediate_reward = reward_function(
                    current_state, control_input, d, beta
                )

                total_value = immediate_reward + gamma * future_value
                if total_value > best_value:
                    best_value = total_value

            V_new[idx] = best_value
            max_diff = max(max_diff, abs(V_new[idx] - V[idx]))

        if max_diff < tol:
            print(f"Converged in {iteration + 1} iterations.")
            break

        V = V_new

    return V


def extract_policy(env, V, nx=10, gamma=1.0, beta=0.0, step_duration=3.0):
    N = env.num_agents
    d = env.desired_opinion

    grids = create_state_grid(N, nx)
    grid_shape = tuple(len(grid) for grid in grids)

    action_levels = [0, env.max_u/2, env.max_u]  # 3 levels
    control_actions = list(product(action_levels, repeat=N))
    policy = {}

    interpolator = build_interpolator(V, grids)

    for idx in np.ndindex(grid_shape):
        current_state = np.array([grids[i][idx[i]] for i in range(N)])
        best_value = -np.inf
        best_action = None

        for control in control_actions:
            control_input = np.array(control)
            next_state = env.compute_dynamics(
                current_state, control_input, step_duration
            )
            next_state = np.clip(next_state, 0, 1)

            # next_idx = tuple(np.abs(grids[i] - next_state[i]).argmin() for i in range(N))
            # future_value = V[next_idx]
            if is_terminal(next_state, env):
                future_value = 0
            else: 
                future_value = interpolator(next_state.reshape(1, -1))[0]

            immediate_reward = reward_function(current_state, control_input, d, beta)
            total_value = immediate_reward + gamma * future_value

            if total_value > best_value:
                best_value = total_value
                best_action = control_input

        policy[idx] = best_action if best_action is not None else np.zeros(N)

    return policy