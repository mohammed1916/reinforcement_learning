import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Parameters
alpha = 0.1  # Learning rate
gamma = 0.9  # Discount factor

# Sample state features (ϕ(s)) and reward
state_dim = 4  # Dimension of state feature space
state = np.random.rand(state_dim)  # Example state
print(state)
reward = 1.0  # Example reward

# Define a simple linear model in Keras
model = Sequential([
    Dense(1, input_dim=state_dim, use_bias=False, activation='linear')
])

# Use SGD with a small learning rate
optimizer = SGD(learning_rate=alpha)
model.compile(optimizer=optimizer, loss='mean_squared_error')

# Compute the value V(s) using the current model
V_s = model.predict(state.reshape(1, -1))[0][0]
print("V(s):", V_s)

# Calculate TD error δ = r - V(s)
delta = reward - V_s
print("TD error (δ):", delta)

# Manual update for weights: θ = θ + α ⋅ δ ⋅ ϕ(s)
prev_weights = model.get_weights()[0]
print("Previous weights (θ): \n", prev_weights)
weights = prev_weights +  alpha * delta * state.reshape(-1, 1)
print("Updated weights (θ): \n", weights)
model.set_weights([weights])

# Print updated weights
print("Updated weights (θ): \n", model.get_weights()[0])
