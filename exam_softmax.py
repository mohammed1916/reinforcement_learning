import numpy as np
import matplotlib.pyplot as plt

class EpsilonGreedyBandit:
    def __init__(self, n_arms, epsilon):
        self.n_arms = n_arms
        self.epsilon = epsilon
        self.arm_counts = np.zeros(n_arms)  
        self.arm_values = np.zeros(n_arms)  
    
    def select_arm(self):
        if np.random.rand() < self.epsilon:
            return np.random.randint(self.n_arms)
        else:
            return np.argmax(self.arm_values)
    
    def update_estimates(self, chosen_arm, reward):
        self.arm_counts[chosen_arm] += 1
        
        n = self.arm_counts[chosen_arm]
        value = self.arm_values[chosen_arm]
        self.arm_values[chosen_arm] = value + (reward - value) / n
    
    def run(self, n_steps):
        total_reward = 0
        rewards = np.zeros(n_steps)
        for step in range(n_steps):
            chosen_arm = self.select_arm()
            reward = np.random.randint(1,6)
            self.update_estimates(chosen_arm, reward)
            total_reward += reward
            rewards[step] = reward
        return total_reward, rewards

n_arms = 3                    
epsilon = 0.1                 
n_steps = 7                
bandit = EpsilonGreedyBandit(n_arms, epsilon)


total_reward, rewards = bandit.run(n_steps)

print(f"Total Reward: {total_reward}")
print(f"Estimated Values of Arms: {bandit.arm_values}")
print(f"Counts of Arms Pulled: {bandit.arm_counts}")
print(f"Rewards: {len(rewards)}")
print(f"Rewards: {rewards}")

x = np.arange(1, n_steps+1)
plt.plot(x, rewards)
plt.xlabel('Day')
plt.ylabel('Reward')
plt.title('Rewards over Days using Softmax')
plt.yticks(np.arange(1, 6))
plt.show()