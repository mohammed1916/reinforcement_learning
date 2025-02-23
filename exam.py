from sympy import symbols, Eq, solve
import numpy as np

T = np.array([
    [0.7,0.3],
    [0.6,0.4],
])

# Create equation
x, y = symbols('x y')
initial_state = np.array([x,y]) 
solutions = np.dot(initial_state,T)
print("Initial Equation 1: ",solutions[0])
print("Initial Equation 2: ",solutions[1])
eq1 = Eq(solutions[0], x)
eq2 = Eq(solutions[1], y)
eq3 = Eq((x+y), 1)
solutions = solve([eq1, eq2, eq3], (x, y, 1))   
# for i in range(len(solutions)):
#     print("Initial probability vector S = ", solutions[i][:2])
# solutions = [np.array(s) for s in solutions]
pi = solutions[1][:2] # Remove 1 from (0.666666666666667, 0.333333333333333, 1.00000000000000)
print("Initial probability vector S = ", pi)

# res1 =[]
for i in range(3):
    # T = np.dot(pi, T)
    pi = np.dot(pi, T)
    # print(pi)
    print("After year: {0}. Probability vector S = {1}". format(i,pi))
    # T = res1

print("After 3rd year: ",pi)
