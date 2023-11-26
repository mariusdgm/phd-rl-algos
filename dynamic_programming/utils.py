import numpy as np


def create_random_policy(states, actions):
    num_actions = len(actions)
    action_prob = 1.0 / num_actions  # Uniform probability for each action

    policy = {}
    for state in states:
        policy[state] = {action: action_prob for action in actions}

    return policy


def extract_V_from_Q(Q, states):
    return {state: max(Q[state].values()) for state in states}
