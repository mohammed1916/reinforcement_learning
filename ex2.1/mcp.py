import numpy as np

P = np.array([[.3, .7],[.2,.8]])

print("\nTransition matrix = ",P)

S = np.array([0.5, 0.5])

print("Steady state probabilities: ")
for i in range(10):
    S = np.dot(S,P)
    print(f'Iter {i}: Probability vector S = {S}')
print(f'Final vector S = {S}')