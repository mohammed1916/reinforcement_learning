import gym
from gym import spaces
import numpy as np
import random
import matplotlib.pyplot as plt

# Define the custom GridWorld environment
class GridWorldEnv(gym.Env):
    """
    Custom GridWorld Environment following OpenAI Gym interface.
    The agent starts at (0,0) and aims to reach (4,4). There are hole states that terminate the episode with negative rewards.
    """
    metadata = {'render.modes': ['human']}

    def __init__(self):
        super(GridWorldEnv, self).__init__()
        
        # Define action and observation space
        # Actions: 0=Up, 1=Down, 2=Left, 3=Right
        self.action_space = spaces.Discrete(4)
        
        # Observation space: Tuple of (row, col)
        self.observation_space = spaces.Tuple((
            spaces.Discrete(5),  # Rows: 0 to 4
            spaces.Discrete(5)   # Columns: 0 to 4
        ))
        
        # Grid dimensions
        self.BOARD_ROWS = 5
        self.BOARD_COLS = 5
        
        # Define start, win, and hole states
        self.START = (0, 0)
        self.WIN_STATE = (4, 4)
        self.HOLE_STATES = [(1, 0), (3, 1), (4, 2), (1, 3)]
        
        self.reset()
    
    def reset(self, seed=None, options=None):
        """Reset the environment to the initial state."""
        super().reset(seed=seed)
        self.state = self.START
        self.done = False
        self.total_reward = 0
        return self.state, {}
    
    def step(self, action):
        """Execute one time step within the environment."""
        if self.done:
            raise RuntimeError("Episode has ended. Please reset the environment.")
        
        row, col = self.state
        
        # Define movement based on action
        if action == 0:   # Up
            next_row, next_col = row - 1, col
        elif action == 1: # Down
            next_row, next_col = row + 1, col
        elif action == 2: # Left
            next_row, next_col = row, col - 1
        elif action == 3: # Right
            next_row, next_col = row, col + 1
        else:
            raise ValueError("Invalid Action")
        
        # Check boundaries
        if 0 <= next_row < self.BOARD_ROWS and 0 <= next_col < self.BOARD_COLS:
            self.state = (next_row, next_col)
        # Else, stay in the same state
        
        reward = self.get_reward(self.state)
        self.total_reward += reward
        
        # Check if done
        if self.state == self.WIN_STATE or self.state in self.HOLE_STATES:
            self.done = True
        
        info = {}
        return self.state, reward, self.done, False, info
    
    def get_reward(self, state):
        """Get reward based on the current state."""
        if state in self.HOLE_STATES:
            return -5
        elif state == self.WIN_STATE:
            return 1
        else:
            return -1
    
    def render(self, mode='human'):
        """Render the current state of the environment."""
        grid = [['O' for _ in range(self.BOARD_COLS)] for _ in range(self.BOARD_ROWS)]
        for hole in self.HOLE_STATES:
            grid[hole[0]][hole[1]] = 'H'
        grid[self.WIN_STATE[0]][self.WIN_STATE[1]] = 'W'
        grid[self.state[0]][self.state[1]] = 'A'  # Agent's position
        
        print("\n".join([' '.join(row) for row in grid]))
        print()
    
    def close(self):
        pass

# Define the Agent class
class Agent:
    def __init__(self, env):
        self.env = env
        self.actions = list(range(env.action_space.n))  # [0,1,2,3]
        
        # Q-Learning parameters
        self.alpha = 0.5       # Learning rate
        self.gamma = 0.9       # Discount factor
        self.epsilon = 0.1     # Exploration rate
        
        # Initialize Q-table as a dictionary
        self.Q = {}
        for row in range(env.BOARD_ROWS):
            for col in range(env.BOARD_COLS):
                for action in self.actions:
                    self.Q[(row, col, action)] = 0.0
        
        # To store rewards per episode for plotting
        self.plot_rewards = []
    
    def choose_action(self, state):
        """Choose an action based on epsilon-greedy policy."""
        if random.random() < self.epsilon:
            action = random.choice(self.actions)
        else:
            q_values = [self.Q[(state[0], state[1], a)] for a in self.actions]
            max_q = max(q_values)
            # In case multiple actions have the same max Q-value, choose randomly among them
            actions_with_max_q = [a for a, q in zip(self.actions, q_values) if q == max_q]
            action = random.choice(actions_with_max_q)
        return action
    
    def learn(self, episodes):
        """Run Q-Learning for a specified number of episodes."""
        for episode in range(1, episodes + 1):
            state, _ = self.env.reset()
            done = False
            total_reward = 0
            
            while not done:
                action = self.choose_action(state)
                next_state, reward, done, _, _ = self.env.step(action)
                total_reward += reward
                
                # Update Q-value
                old_q = self.Q[(state[0], state[1], action)]
                next_q_values = [self.Q[(next_state[0], next_state[1], a)] for a in self.actions]
                max_next_q = max(next_q_values) if not done else 0
                new_q = (1 - self.alpha) * old_q + self.alpha * (reward + self.gamma * max_next_q)
                self.Q[(state[0], state[1], action)] = new_q
                
                state = next_state
            
            self.plot_rewards.append(total_reward)
            
            # Optionally print progress
            if episode % 100 == 0:
                print(f"Episode {episode}/{episodes} - Total Reward: {total_reward}")
        
        print("Training completed.")
    
    def plot_rewards_graph(self):
        """Plot the rewards over episodes."""
        plt.figure(figsize=(12, 6))
        plt.plot(self.plot_rewards)
        plt.xlabel('Episodes')
        plt.ylabel('Total Reward')
        plt.title('Total Reward per Episode')
        plt.grid(True)
        plt.show()
    
    def show_values(self):
        """Display the maximum Q-value for each state."""
        print("Maximum Q-values for each state:")
        for row in range(self.env.BOARD_ROWS):
            print('-----------------------------------------------')
            row_values = '| '
            for col in range(self.env.BOARD_COLS):
                q_vals = [self.Q[(row, col, a)] for a in self.actions]
                max_q = max(q_vals)
                row_values += f"{max_q:.2f}".ljust(8) + ' | '
            print(row_values)
        print('-----------------------------------------------')
    
    def show_policy(self):
        """Display the policy derived from the Q-table."""
        action_mapping = {
            0: '↑',  # Up
            1: '↓',  # Down
            2: '←',  # Left
            3: '→'   # Right
        }
        print("Policy derived from Q-values:")
        for row in range(self.env.BOARD_ROWS):
            print('---------------------------------')
            row_policy = '| '
            for col in range(self.env.BOARD_COLS):
                if (row, col) == self.env.WIN_STATE:
                    row_policy += ' W '.ljust(5) + ' | '
                elif (row, col) in self.env.HOLE_STATES:
                    row_policy += ' H '.ljust(5) + ' | '
                else:
                    q_vals = [self.Q[(row, col, a)] for a in self.actions]
                    max_q = max(q_vals)
                    best_actions = [a for a, q in zip(self.actions, q_vals) if q == max_q]
                    action = random.choice(best_actions)
                    row_policy += f" {action_mapping[action]} ".ljust(5) + ' | '
            print(row_policy)
        print('---------------------------------')

# Main function to run the training and display results
def main():
    # Initialize the custom GridWorld environment
    env = GridWorldEnv()
    
    # Initialize the agent with the environment
    agent = Agent(env)
    
    # Define the number of training episodes
    episodes = 3000  # Increased to ensure better convergence
    
    # Train the agent using Q-Learning
    agent.learn(episodes)
    
    # Plot the rewards over episodes
    agent.plot_rewards_graph()
    
    # Show the learned Q-values
    agent.show_values()
    
    # Optionally, show the derived policy
    agent.show_policy()
    
    # Demonstrate the learned policy by running one episode
    state, _ = env.reset()
    env.render()
    done = False
    total_reward = 0
    while not done:
        # Choose the best action based on Q-values
        q_vals = [agent.Q[(state[0], state[1], a)] for a in agent.actions]
        max_q = max(q_vals)
        best_actions = [a for a, q in zip(agent.actions, q_vals) if q == max_q]
        action = random.choice(best_actions)
        
        next_state, reward, done, _, _ = env.step(action)
        total_reward += reward
        env.render()
        state = next_state
    
    print(f"Demonstration Episode - Total Reward: {total_reward}")

if __name__ == "__main__":
    main()
