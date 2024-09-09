# import gym
# env = gym.make("CartPole-v1")
# observation, info = env.reset(seed=42)

# for _ in range(1000):
#     action = env.action_space.sample()
#     observation, reward, terminated, truncated, info = env.step(action)

#     if terminated or truncated:
#         observation, info = env.reset()
# env.close()
import gym

# Create the environment
env = gym.make('CartPole-v1', render_mode="rgb_array")

# Reset the environment to the initial state
observation = env.reset()

for t in range(1000):
    # Render the environment (this will show a pop-up window)
    env.render()
    
    # Take a random action
    action = env.action_space.sample()
    
    # Apply the action to the environment
    observation, reward, done, truncated, info = env.step(action)
    
    # Print the step information
    print(f"Step {t}: Action {action}, Observation {observation}, Reward {reward}, Done {done}, Truncated {truncated}")
    
    # Check if the episode is finished
    if done or truncated:
        print("Episode finished after {} timesteps".format(t+1))
        break

# Close the environment
env.close()
