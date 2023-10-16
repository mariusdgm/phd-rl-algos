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
            + gamma * V.get(t_r_dict.get((state, a), (None, 0, True))[0], 0)
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
