"""RL_lab_3
Reference: https://colab.research.google.com/drive/1VSvZZK1eINmpjNioAvYlzjsH_QzUWbEK
"""

import numpy as np

P = np.array([[0.2,0.8],[0.6,0.4]])
print("Transition matrix:\n",P)

S = np.array([0.6,0.4])

for i in range(10):
    S = np.dot(S,P)
    print("\nIteration {0}. Probability vector S = {1}". format(i,S))

print("\nFinal probability vector = ",S)

"""Problem 2

"""

from sympy import symbols, Eq, solve

T = np.array([[0.3,0.2,0.5],[0.4,0,0],[0.1,0,0.9]])
print("Transition matrix:\n",T)

R = np.array([[-3,-2,0,0],[-2,0,0,10],[-3,0,0,0],[0,0,0,0]])
print("Reward matrix:\n",R)

x,y,z = symbols("x,y,z")

initial_state = np.array([x,y,z])

initial_equation = np.dot(initial_state,T)
print(initial_equation)

initial_equation[1]

eq1 = Eq(initial_equation[0], x)
# eq1

eq2 = Eq(initial_equation[1], y)
# eq2

eq3 = Eq(initial_equation[2], z)
# eq3

eq4 = Eq((x+y+z), 1)
# eq4

solutions = solve([eq1, eq2, eq3, eq4], (x, y, z, 1))
solutions

print("solutions", [solutions[i][:3] for i in range(len(solutions))])

print("solutions", [solutions[0][:3]])

print("solutions", [solutions[1][:3]])

print("solutions", [solutions[2][:3]])

