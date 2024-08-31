# import numpy as np

# P = np.array([(.9,0),(.1,0),(0,0),(0,0)],
#              [(.5,0),(.3,-3),(0.2,-2),(0,0)],
#              [(0,0),(0.4,-1),(0,0),(0.6,10)],
#              [(0,0),(0,0),(0,0),(0,0)],
#              )

import numpy as np

def compute_state_values(transition_reward_matrix, gamma=0.9, tolerance=1e-6, final_state=3):
    """
    Computes the state values for a Markov Reward Process using the value iteration method.
    Stops if the final state is reached.

    :param transition_reward_matrix: A 2D numpy array where each element is a tuple (probability, reward).
    :param gamma: The discount factor, typically between 0 and 1.
    :param tolerance: The convergence tolerance for the value iteration.
    :param final_state: The index of the final state where we stop the iteration.
    :return: A numpy array of state values.
    """
    num_states = len(transition_reward_matrix)
    state_values = np.zeros(num_states)
    iteration = 0

    while True:
        iteration += 1
        new_state_values = np.zeros(num_states)
        print(f"Iteration {iteration}:")

        for s in range(num_states):
            expected_reward = 0
            expected_value = 0
            
            for s_prime in range(num_states):
                probability, reward = transition_reward_matrix[s, s_prime]
                expected_reward += probability * reward
                expected_value += probability * state_values[s_prime]

            new_state_values[s] = expected_reward + gamma * expected_value
            print(f"  State {s}: Reward = {expected_reward}, Value = {new_state_values[s]}")
        
        # Print the state values for this iteration
        print("  New State Values:", new_state_values)
        
        # Check for convergence
        if np.max(np.abs(new_state_values - state_values)) < tolerance:
            print("Convergence reached.")
            break

        # Stop if the final state has been reached
        if state_values[final_state] != 0 and new_state_values[final_state] == 0:
            print(f"Stopping as the final state {final_state} has been reached.")
            break
        
        state_values = new_state_values

    return state_values

def main():
    # Define the transition probability and reward matrix
    transition_reward_matrix = np.array([
        [(0.9, 0), (0.1, 0), (0, 0), (0, 0)],
        [(0.5, 0), (0.3, -3), (0.2, -2), (0, 0)],
        [(0, 0), (0.4, -1), (0, 0), (0.6, 10)],
        [(0, 0), (0, 0), (0, 0), (0, 0)]
    ], dtype=object)

    # Discount factor
    gamma = 0.9
    final_state = 3  # Index of the final state (4th state)

    print("Starting Value Iteration...")
    state_values = compute_state_values(transition_reward_matrix, gamma, final_state=final_state)
    print("Final State Values:", state_values)

if __name__ == "__main__":
    main()
