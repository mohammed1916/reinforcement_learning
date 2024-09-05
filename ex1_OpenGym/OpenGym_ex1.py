import gym
env = gym.make("CartPole-v1")
observation, info = env.reset(seed=42)

for _ in range(1000):
    action = env.action_space.sample()
    observation, reward, terminated, truncated, info = env.step(action)

    if terminated or truncated:
        observation, info = env.reset()
env.close()

# import gym

# # Create the environment
# env = gym.make('CartPole-v1',render_mode="rgb_array")

# # Reset the environment
# state = env.reset()

# # Run the simulation
# done = False
# while not done:
#     # Choose an action
#     action = env.action_space.sample()

#     # Take the action in the environment
#     next_state, reward, terminated, truncated, info  = env.step(action)
#     # print("S:   ",s)
#     # print("len(S)",len(s))
#     # Update the state
#     state = next_state

#     # Render the environment (optional)
#     env.render()

# # Close the environment
# env.close()

