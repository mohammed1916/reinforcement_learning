import numpy as np
import gym
from gym import spaces


# Define a custom environment for the multi-armed bandit (same as previous)
class MultiArmedBanditEnv(gym.Env):
    def _init_(self, n_arms=10):
        super(MultiArmedBanditEnv, self)._init_()
        self.n_arms = n_arms
        self.action_space = spaces.Discrete(n_arms)  # Each action corresponds to an arm
        self.observation_space = spaces.Discrete(1)  # No meaningful observations
        self.reward_probs = np.random.rand(n_arms)  # Random probabilities of reward for each arm

    def reset(self):
        return 0  # Return a dummy observation

    def step(self, action):
        reward = 1 if np.random.rand() < self.reward_probs[action] else 0  # Reward based on arm's probability
        return 0, reward, False, {}  # No terminal state or additional info


# Softmax algorithm
def softmax(x):
    e_x = np.exp(x - np.max(x))  # Subtract max to prevent overflow
    return e_x / e_x.sum(axis=0)


def softmax_bandit(env, n_steps=1000, tau=1.0):
    n_arms = env.action_space.n
    action_values = np.zeros(n_arms)  # Estimated value of each action
    action_counts = np.zeros(n_arms)  # Count of each action taken

    total_reward = 0

    for step in range(n_steps):
        # Calculate probabilities using softmax
        probabilities = softmax(action_values / tau)
        action = np.random.choice(n_arms, p=probabilities)  # Choose action based on probabilities

        _, reward, _, _ = env.step(action)
        total_reward += reward

        # Update action value estimates
        action_counts[action] += 1
        action_values[action] += (reward - action_values[action]) / action_counts[action]

    return action_values, total_reward


# Create the environment and run the algorithm
env = MultiArmedBanditEnv(n_arms=10)
action_values, total_reward = softmax_bandit(env, n_steps=1000, tau=1.0)

print("Estimated action values:", action_values)
print("Total reward collected:", total_reward)