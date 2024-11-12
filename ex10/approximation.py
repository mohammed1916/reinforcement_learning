import numpy as np
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense
from tensorflow.keras.optimizers import SGD

# Set the initial parameters
alpha = 0.1    # Learning rate
reward = 10    # Constant reward received in each interaction
phi_s = np.array([1])  # Feature vector for State A (assumed to be 1 for simplicity)

# Initializing the model
model = Sequential([
    Dense(1, input_dim=1, activation='linear', use_bias=False)  # Linear model with weight theta
])
model.compile(optimizer=SGD(learning_rate=alpha), loss='mean_squared_error')

# Initial weight (theta) will be set to 0 by default
print("Initial theta:", model.get_weights()[0][0][0])

# Interaction 1
# Forward pass to get V(s) (initial prediction)
V_s = model.predict(phi_s)
delta = reward - V_s  # TD error

# Update the model using delta * phi(s) as the target
model.fit(phi_s, V_s + delta, verbose=0)

# Print updated theta after first interaction
print("Updated theta after Interaction 1:", model.get_weights()[0][0][0])

# Interaction 2
# Forward pass again with updated theta
V_s = model.predict(phi_s)
delta = reward - V_s  # TD error

# Update the model using delta * phi(s) as the target
model.fit(phi_s, V_s + delta, verbose=0)

# Print updated theta after second interaction
print("Updated theta after Interaction 2:", model.get_weights()[0][0][0])

# Final estimated value function V(s) for State A
V_s_final = model.predict(phi_s)
print("Estimated Value Function for State A:", V_s_final[0][0])
