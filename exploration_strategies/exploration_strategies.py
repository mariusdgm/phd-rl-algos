import numpy as np


# 1. Epsilon-Greedy Strategy
def epsilon_greedy(q_values, epsilon):
    if np.random.rand() < epsilon:
        return np.random.choice(len(q_values))  # Explore: random action
    else:
        return np.argmax(q_values)  # Exploit: best known action


# 2. Optimistic Initial Values
def initialize_q_values(num_arms, initial_value):
    return np.full(num_arms, initial_value, dtype=float)


# 3. Upper Confidence Bound (UCB)
def ucb_selection(q_values, counts, timestep, c=2):
    ucb_values = np.zeros_like(q_values, dtype=float)
    for i in range(len(q_values)):
        if counts[i] == 0:
            ucb_values[i] = float("inf")
        else:
            adjusted_timestep = timestep + 1e-5  # Adjust timestep to avoid log(0)
            ucb_values[i] = q_values[i] + c * np.sqrt(
                np.log(adjusted_timestep) / counts[i]
            )
    return np.argmax(ucb_values)


# 4. Gradient Bandit Algorithm
def softmax_preferences(preferences):
    exp_prefs = np.exp(preferences - np.max(preferences))  # Numerical stability
    return exp_prefs / np.sum(exp_prefs)


def select_action_gradient_bandit(preferences):
    action_probabilities = softmax_preferences(preferences)
    return np.random.choice(len(preferences), p=action_probabilities)


### Integrating strategies
class StrategyHandler:
    def __init__(self, strategy, num_arms, **strategy_params):
        self.strategy = strategy
        self.num_arms = num_arms
        initial_value = (
            0 if strategy != "optimistic" else strategy_params.get("initial_value", 0)
        )
        self.q_values = initialize_q_values(num_arms, initial_value)
        self.counts = np.zeros(num_arms)  # Count of selections for each arm
        self.preferences = np.zeros(num_arms)  # For gradient bandit
        self.total_reward = 0  # Total reward, for average reward in gradient bandit
        self.strategy_params = strategy_params

    def select_action(self, timestep):
        if self.strategy in ["epsilon_greedy", "optimistic"]:
            return epsilon_greedy(
                self.q_values, self.strategy_params.get("epsilon", 0.1)
            )
        elif self.strategy == "ucb":
            return ucb_selection(
                self.q_values, self.counts, timestep, self.strategy_params.get("c", 2)
            )
        elif self.strategy == "gradient_bandit":
            return select_action_gradient_bandit(self.preferences)
        else:
            raise ValueError(f"Unknown strategy: {self.strategy}")

    def update(self, chosen_arm, reward):
        self.counts[chosen_arm] += 1
        self.total_reward += reward

        alpha = self.strategy_params.get("alpha", 0.1)  # Fixed learning rate

        if self.strategy in ["epsilon_greedy", "optimistic", "ucb"]:
            # Update Q-values using fixed learning rate
            self.q_values[chosen_arm] += alpha * (reward - self.q_values[chosen_arm])
        elif self.strategy == "gradient_bandit":
            # Update preferences
            baseline = self.total_reward / sum(self.counts)
            prob = softmax_preferences(self.preferences)
            self.preferences -= (
                self.strategy_params.get("alpha", 0.1) * (reward - baseline) * prob
            )
            self.preferences[chosen_arm] += self.strategy_params.get("alpha", 0.1) * (
                reward - baseline
            )
