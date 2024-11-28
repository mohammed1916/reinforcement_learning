import random

def monte_carlo_pi(num_samples):
    inside_circle = 0

    for _ in range(num_samples):
        # Generate random x and y points between -1 and 1
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        
        # Check if the point is inside the unit circle
        if x**2 + y**2 <= 1:
            inside_circle += 1

    # Calculate the approximation of pi
    pi_approx = 4 * inside_circle / num_samples
    return pi_approx

# Set the number of samples for the simulation
num_samples = 1000000
pi_value = monte_carlo_pi(num_samples)
print(f"Approximated value of pi: {pi_value}")