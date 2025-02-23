import numpy as np
p=np.array([[0.3,0.7],[0.2,0.8]])
print("Transition matrix:\n",p)
initial_state=([0.5,0.5])
print(initial_state)
for i in range(10):
    S=np.dot(initial_state,p)
    print("\nIter{0}.Probability vector S={1}".format(i,S))
print("\nFinal Vector S={0}".format(S))
import numpy as np
p=np.array([[1/2,1/2],[1/3,2/3]])
print("Transition matrix:\n",p)
initial_state=([1,0])
print("\nInitial_state:\n",initial_state)
S=initial_state
for i in range(5):
    S=np.dot(S,p)
    print("\nIterations:",i)
    print(S)
print("\nFinal value:\n",S)