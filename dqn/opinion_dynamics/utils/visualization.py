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

    # --- Append initial unmodified state at time 0 ---
    interpolated.append(opinions_over_time[0].copy())
    times.append(0.0)

    for t in range(len(actions)):
        x_start = opinions_over_time[t]
        u = actions[t]

        # Apply impulse control (treated as instantaneous at t * tau)
        x = env.compute_dynamics(current_state=x_start, control_action=u, step_duration=0.0)
        interpolated.append(x.copy())
        times.append(t * env.tau)

        # Propagate over substeps
        for k in range(1, n_substeps + 1):
            x = env.compute_dynamics(current_state=x, control_action=np.zeros_like(u), step_duration=dt)
            interpolated.append(x.copy())
            times.append(t * env.tau + k * dt)

    return np.array(interpolated), np.array(times)