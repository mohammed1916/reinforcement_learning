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
print("Initial Equation: ",solutions)
eq1 = Eq(solutions[0], x)
eq2 = Eq(solutions[1], y)
eq3 = Eq((x+y), 1)
solutions = solve([eq1, eq2, eq3], (x, y, 1))   
# for i in range(len(solutions)):
#     print("Initial probability vector S = ", solutions[i][:2])
# solutions = [np.array(s) for s in solutions]
pi = solutions[1][:2]
print(pi)

# res1 =[]
for i in range(3):
    pi = np.dot(pi, T)
    print(pi)
    # print("After year: {0}. Probability vector S = {1}". format(i,res1))
    # T = res1

print("After 3",pi)
