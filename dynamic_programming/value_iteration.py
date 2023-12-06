import numpy as np


def value_iteration_v(t_r_dict, states, actions, gamma=0.9, theta=1e-6):
    V = {state: 0 for state in states}

    while True:
        delta = 0
        for state in states:
            v = V[state]

            # One step lookahead to find the best action value
            values = []
            for action in actions:
                next_state, reward, done = t_r_dict.get(
                    (state, action), (None, 0, True)
                )
                if next_state is None or done:
                    values.append(reward)
                else:
                    values.append(reward + gamma * V[next_state])

            V[state] = max(values)
            delta = max(delta, abs(v - V[state]))

        if delta < theta:
            break

    policy = {
        state: max(
            actions,
            key=lambda a: t_r_dict.get((state, a), (None, 0, True))[1]
            + gamma * V.get(t_r_dict.get((state, a), (None, 0, True))[0], 0),
        )
        for state in states
    }

    return policy, V


def value_iteration_q(t_r_dict, states, actions, gamma=0.9, theta=1e-6):
    Q = {state: {action: 0 for action in actions} for state in states}

    while True:
        delta = 0
        for state in states:
            for action in actions:
                q = Q[state][action]

                # One step lookahead to find the new Q value for the current state-action pair
                next_state, reward, done = t_r_dict.get(
                    (state, action), (None, 0, True)
                )
                if next_state is None or done:
                    Q[state][action] = reward
                else:
                    Q[state][action] = reward + gamma * max(
                        [Q[next_state][a] for a in actions]
                    )

                delta = max(delta, abs(q - Q[state][action]))

        if delta < theta:
            break

    policy = {state: max(actions, key=lambda a: Q[state][a]) for state in states}

    return policy, Q


### Stochastic case


def value_iteration_v_stochastic(t_r_dict, states, actions, gamma=0.9, theta=1e-6):
    V = {state: 0 for state in states}

    while True:
        delta = 0
        for state in states:
            v = V[state]
            action_values = []

            for action in actions:
                if (state, action) in t_r_dict:
                    win_state, lose_state, reward = t_r_dict[(state, action)]

                    # Assuming uniform probability for win and lose
                    win_prob = 0.5
                    lose_prob = 0.5

                    # Calculate expected value for the action
                    expected_value = win_prob * (reward + gamma * V.get(win_state, 0)) + \
                                     lose_prob * (gamma * V.get(lose_state, 0))
                    action_values.append(expected_value)

            V[state] = max(action_values, default=0)
            delta = max(delta, abs(v - V[state]))

        if delta < theta:
            break

    # Derive policy from the value function
    policy = {state: actions[np.argmax(
                [0.5 * (t_r_dict.get((state, a), (0, 0, 0))[2] + gamma * V.get(t_r_dict.get((state, a), (0, 0, 0))[0], 0)) +
                 0.5 * (gamma * V.get(t_r_dict.get((state, a), (0, 0, 0))[1], 0)) 
                 for a in actions], default=0)] 
              for state in states}

    return policy, V
