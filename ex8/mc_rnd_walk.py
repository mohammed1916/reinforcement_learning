import random
import matplotlib.pyplot as plt

def monte_carlo_random_walk(num_steps, num_simulations):
    final_positions = []

    for _ in range(num_simulations):
        position = 0  # Start at the origin
        for _ in range(num_steps):
            # Randomly move left (-1) or right (+1)
            step = random.choice([-1, 1])
            position += step
        final_positions.append(position)

    return final_positions

# Parameters
num_steps = 1000         # Number of steps in each walk
num_simulations = 10000  # Number of simulations

# Run the Monte Carlo random walk
final_positions = monte_carlo_random_walk(num_steps, num_simulations)

# Plotting the results
plt.hist(final_positions, bins=50, edgecolor='black', density=True)
plt.title('1D Random Walk: Distribution of Final Positions')
plt.xlabel('Final Position')
plt.ylabel('Probability Density')
plt.grid(True)
plt.show()