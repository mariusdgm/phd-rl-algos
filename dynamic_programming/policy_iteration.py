import numpy as np


def policy_evaluation(states, policy, V, t_r_dict, gamma, theta):
    while True:
        delta = 0
        for state in states:
            state_v = V[state]
            action = policy[state]
            next_state, reward, done = t_r_dict.get((state, action), (None, 0, True))

            if next_state is None or done:  # Handling terminal states
                V[state] = reward
            else:
                V[state] = reward + gamma * V[next_state]

            delta = max(delta, abs(state_v - V[state]))

        if delta < theta:
            break
    return V


def policy_improvement(states, actions, policy, V, t_r_dict, gamma):
    policy_stable = True
    for state in states:
        old_action = policy[state]

        q_values = []
        for action in actions:
            next_state, reward, done = t_r_dict.get((state, action), (None, 0, True))

            if next_state is None or done:  # Handling terminal states
                q_value = reward
            else:
                q_value = reward + gamma * V[next_state]
            q_values.append(q_value)

        policy[state] = actions[np.argmax(q_values)]

        if old_action != policy[state]:
            policy_stable = False

    return policy, policy_stable


def policy_iteration(t_r_dict, policy, V, states, actions, gamma=0.9, theta=1e-6):

    V = policy_evaluation(states, policy, V, t_r_dict, gamma, theta)
    policy, policy_stable = policy_improvement(
        states, actions, policy, V, t_r_dict, gamma
    )

    return policy, V, policy_stable


def find_optimal_policy(t_r_dict, gamma=0.9, theta=1e-6):
    states = list(set([s for s, _ in t_r_dict.keys()]))
    actions = list(set([a for _, a in t_r_dict.keys()]))

    # Initialize value function and policy arbitrarily
    V = {state: 0 for state in states}
    policy = {state: np.random.choice(actions) for state in states}

    policy_stable = False
    while not policy_stable:
        policy, V, policy_stable = policy_iteration(
            t_r_dict, policy, V, states, actions, gamma, theta
        )

    return policy, V
