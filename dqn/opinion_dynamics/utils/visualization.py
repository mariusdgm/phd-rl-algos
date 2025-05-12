import numpy as np

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

    for t in range(len(actions)):
        x_start = opinions_over_time[t]
        u = actions[t]

        # Apply impulse control: this is effectively part of the dynamics in compute_dynamics
        x = env.compute_dynamics(current_state=x_start, control_action=u, step_duration=0.0)

        for k in range(n_substeps):
            time = t * env.tau + k * dt
            interpolated.append(x.copy())
            times.append(time)

            # Apply small dynamics update via env
            x = env.compute_dynamics(current_state=x, control_action=np.zeros_like(u), step_duration=dt)

    # Append final state
    interpolated.append(opinions_over_time[-1])
    times.append((len(actions)) * env.tau)

    return np.array(interpolated), np.array(times)