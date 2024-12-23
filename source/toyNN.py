from microGrad import Value
from random import random, randint
# let's test the datastructure!
"""    
What is it capable of? 
1. It can do a step to update it's weights through the network automatically to minimize the Loss...
2. Let's define the loss as: L = (y - (mx + c))**2
3. Let us initialise m and c randomly, and then do a step to update the weights to minimize the loss...
4. Let us fix the target m and c as 3 and 4. we are trying to get the line y = 3x+4
5. two sample points at x = 1, and at x = 2, y = 7, and y = 10, or let's say x is generated randomly at runtime :)
"""
def f(x): return 3*x**2 + 12*x + 5 # 3 points are needed to uniquely identify a quadratic equation
from tqdm import tqdm
epochs = 10000
a = Value(random())
b = Value(random())
c = Value(random())
lr = 0.01
X = [Value(x, no_grad=True) for x in [-1,0,1]]
Y = [Value(f(x.val), no_grad=True) for x in X]
print(f"X: {[x.val for x in X]}, Y: {[y.val for y in Y]}")
L = sum([(y - (a*x**2 + b*x + c))**2 for x,y in zip(X,Y)])
for i in tqdm(range(epochs)):
    L.step(lr)
    if i%100==0: print(f"Epoch: {i}, Loss: {L.val:.2f}, equation: {a.val:.2f}x^2 + {b.val:.2f}x + {c.val:.2f}")
print(f"Final equation: {a.val:.2f}x^2 + {b.val:.2f}x + {c.val:.2f}")