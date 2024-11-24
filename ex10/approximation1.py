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

# Calculate TD error δ = r - V(s)
delta = reward - V_s

# Manual update for weights: θ = θ + α ⋅ δ ⋅ ϕ(s)
weights = model.get_weights()[0]
weights += alpha * delta * state.reshape(-1, 1)
model.set_weights([weights])

# Print updated weights
print("Updated weights (θ): \n", model.get_weights()[0])
