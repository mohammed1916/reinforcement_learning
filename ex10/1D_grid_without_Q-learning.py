import gym
from gym import spaces
import numpy as np
import random

class OneDGridEnv(gym.Env):
    def __init__(self, grid_size=11, goal_state=10):
        super(OneDGridEnv, self).__init__()
        self.grid_size = grid_size
        self.goal_state = goal_state
        self.action_space = spaces.Discrete(2)  # Two actions: 0 = "left", 1 = "right"
        self.observation_space = spaces.Discrete(self.grid_size)  # 1D grid of grid_size cells
        self.state = 0  # Start at position 0

    def reset(self):
        self.state = 0  # Reset to start
        return self.state

    def step(self, action):
        if action == 0:  # Move left
            self.state = max(0, self.state - 1)
        elif action == 1:  # Move right
            self.state = min(self.grid_size - 1, self.state + 1)

        # Calculate reward
        reward = 10 if self.state == self.goal_state else -1

        # Check if the episode is done
        done = self.state == self.goal_state

        return self.state, reward, done, {}

    def render(self):
        grid = ["_"] * self.grid_size
        grid[self.state] = "A"  # Mark the agent's current position
        grid[self.goal_state] = "G"  # Mark the goal position
        print(" ".join(grid))

# Instantiate the environment
env = OneDGridEnv()
# Run a random agent in the environment for 10 episodes
for episode in range(10):
    state = env.reset()
    done = False

    while not done:
        action = env.action_space.sample()  # Choose a random action
        next_state, reward, done, _ = env.step(action)
        env.render()
        print("Reward: ", reward)

        state = next_state

# # Create a numpy array to store the episode trajectories
# trajectories = np.zeros((10, env.grid_size))

# for episode in range(10):
#     state = env.reset()
#     done = False

#     while not done:
#         action = env.action_space.sample()  # Choose a random action
#         next_state, reward, done, _ = env.step(action)
#         env.render()
#         print("Reward: ", reward)

#         state = next_state

#         # Update the trajectory array
#         trajectories[episode, state] = 1

# # Visualize the episode trajectories
# print("Episode Trajectories:")
# print(trajectories)