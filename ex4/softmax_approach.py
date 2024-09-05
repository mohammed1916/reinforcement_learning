import numpy as np

class SoftmaxBandit:
    def _init_(self, n_arms, temperature, true_rewards):
        self.n_arms = n_arms
        self.temperature = temperature  # Temperature parameter for softmax
        self.true_rewards = true_rewards  # True mean rewards for each arm
        self.arm_counts = np.zeros(n_arms)  # Count of times each arm was pulled
        self.arm_values = np.zeros(n_arms)  # Estimated value of each arm

    def select_arm(self):
        # Calculate the probabilities using the softmax formula
        exp_values = np.exp(self.arm_values / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        
        # Choose an arm based on the computed probabilities
        return np.random.choice(self.n_arms, p=probabilities)

    def update_estimates(self, chosen_arm, reward):
        # Increment the count of the chosen arm
        self.arm_counts[chosen_arm] += 1
        
        # Update the estimated value of the chosen arm using incremental formula
        n = self.arm_counts[chosen_arm]
        value = self.arm_values[chosen_arm]
        # Update rule: Q_new = Q_old + (reward - Q_old) / n
        self.arm_values[chosen_arm] = value + (reward - value) / n

    def run(self, n_steps):
        total_reward = 0
        rewards = np.zeros(n_steps)
        
        for step in range(n_steps):
            # Select an arm using softmax strategy
            chosen_arm = self.select_arm()
            
            # Simulate the reward for the chosen arm based on the true reward distribution
            reward = np.random.binomial(1, self.true_rewards[chosen_arm])
            
            # Update the estimates for the chosen arm
            self.update_estimates(chosen_arm, reward)
            
            # Update total reward and store reward of this step
            total_reward += reward
            rewards[step] = reward
        
        return total_reward, rewards

# Parameters
n_arms = 5                     # Number of arms
temperature = 0.1              # Temperature parameter for softmax
true_rewards = [0.1, 0.5, 0.7, 0.3, 0.9]  # True probabilities of reward for each arm
n_steps = 1000                 # Number of steps

# Create an instance of the bandit with softmax strategy
bandit = SoftmaxBandit(n_arms, temperature, true_rewards)

# Run the bandit algorithm for the specified number of steps
total_reward, rewards = bandit.run(n_steps)

# Output the results
print(f"Total Reward: {total_reward}")
print(f"Estimated Values of Arms: {bandit.arm_values}")
print(f"Counts of Arms Pulled: {bandit.arm_counts}")