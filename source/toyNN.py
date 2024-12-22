from microGrad import Value
from random import random
# let's test the datastructure!
"""    
What is it capable of? 
1. It can do a step to update it's weights through the network automatically to minimize the Loss...
2. Let's define the loss as: L = (y - (mx + c))**2
3. Let us initialise m and c randomly, and then do a step to update the weights to minimize the loss...
4. Let us fix the target m and c as 3 and 4. we are trying to get the line y = 3x+4
5. two sample points at x = 1, and at x = 2, y = 7, and y = 10, or let's say x is generated randomly at runtime :)
"""
def f(x): return 7*x + 12
from tqdm import tqdm
epochs = 1000
m = Value(random())
c = Value(random())
lr = 0.01
for i in tqdm(range(epochs)):
    x = random()
    y = f(x)
    L = (y - (m*x + c))**2
    L.step(lr*10)
    if i%100==0: print(f"Epoch: {i}, Loss: {L.val:.2f}, m: {m.val:.2f}, c: {c.val:.2f}")
