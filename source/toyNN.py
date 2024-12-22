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
def f(x): return 8*x**2 + 12*x + 3 # 3 points are needed to uniquely identify a quadratic equation
from tqdm import tqdm
epochs = 10000
a = Value(random())
b = Value(random())
c = Value(random())
lr = 0.01
x1 = Value(-1, no_grad=True)
x2 = Value(0, no_grad=True)
x3 = Value(1, no_grad=True)
y1 = Value(f(x1.val), no_grad=True)
y2 = Value(f(x2.val), no_grad=True)
y3 = Value(f(x3.val), no_grad=True)
print(f"x1: {x1.val}, y1: {y1.val}, x2: {x2.val}, y2: {y2.val}, x3: {x3.val}, y3: {y3.val}")
for i in tqdm(range(epochs)):
    L = (y1 - (a*x1**2 + b*x1+c))**2 + (y2 - (a*x2**2 + b*x2 + c))**2 + (y3 - (a*x3**2 + b*x3 + c))**2 # TODO: this can  be moved to a forward pass...
    L.step(lr)
    if i%100==0: print(f"Epoch: {i}, Loss: {L.val:.2f}, equation: {a.val:.2f}x^2 + {b.val:.2f}x + {c.val:.2f}")

print(f"y1: {y1.val}, predicted: {a*x1**2 + b*x1 + c}")
print(f"y2: {y2.val}, predicted: {a*x2**2 + b*x2 + c}")
print(f"y3: {y3.val}, predicted: {a*x3**2 + b*x3 + c}")
