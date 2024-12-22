"""
#Goal: build a tool to create a connected structure of the model of nodes with a inbuilt gradient feature that tells each node
 how to update itself(take it's next step) to better fit the data.

Thoughts:
1. L = w*x + b; That's how loss functions are usually in some sense. We need to compute the gradient of L wrt w and b. x is the input... So, we are optimising the weights so that we have the least loss for the inputs of this form...
2. in the general case, we have change-needed: Delta = f(x, w1, w2, ... wn) where x is the input to that level/layer and w1, w2, ... wn are the tunable parameters in that level.
3. Our goal is to have a tool that can compute the gradient of Delta wrt w1, w2, ... wn will have terms from (x and wi's)
So, in this context, we need to know what L is, and then gradient of each level wrt to the next one(in the general case).
3. The idea is to reduce the LOSS, which means the layer just before needs to know how to change!
"""
import torch
import matplotlib.pyplot as plt
import numpy as np
import random

# Toy task: generate samples of y from x, where y = 3x + 4, and build a model to predict y from x.
# We will use a multi layer perceptron but without any activation function(we will add it later), and that model should generalize Universally!
def generate_linear_data(samples = 1, m = 3, c = 4, sd = 0.05):
    x = np.random.rand(samples)
    y = m*x + c + np.random.rand(samples)*sd
    return x, y

# We don't even many samples, let's build gradient descent with just 2 samples to uniquely identify m and c
x1 = 2; x2 = 5; m = 5; c = 7
# goal to gradient descent our way to find m and c, given x and y
random.seed(5)
mp = random.random(); cp = random.random()
lr = 0.02
x = x1 # let's say
y = m * x + c
yp = mp * x + cp
dyp_dm = x
dyp_dc = 1
L = (y - yp)**2
dL_dm = 2 * (y - yp) * (-dyp_dm)
dL_dc = 2 * (y - yp) * (-dyp_dc)
mp = mp - lr * dL_dm
cp = cp - lr * dL_dc

# Let's do this for 500 iterations
for epoch in range(501):
    dmp = 0; dcp = 0
    L = 0
    batch_update = False
    for x in [x1, x2]:
        y = m * x + c
        yp = mp * x + cp
        dyp_dm = x
        dyp_dc = 1
        dL_dm = 2 * (y - yp) * (-dyp_dm)
        dL_dc = 2 * (y - yp) * (-dyp_dc)
        if batch_update:
            dmp += - lr * dL_dm
            dcp += - lr * dL_dc
        else:
            mp -= lr * dL_dm
            cp -= lr * dL_dc
        L += (y - yp)**2
    if batch_update:
        mp += dmp
        cp += dcp
    L /= 2
    if epoch%100==0: print(f"Epoch: {epoch}, Loss: {L:.2f}, mLearnt: {mp:.2f}, cLearnt: {cp:.2f}")
print(f"Actual m: {m}, Actual c: {c}")
# We have proved that 2 examples are sufficient to uniquely identify m and c.
# Start with random m and c, and then gradient descent your way to find the actual m and c.

# Let's build useful tools for gradient descent, with pre-defined functions and operations...
# Take this step for example: dL_dc = 2 * (y - yp) * (-dyp_dc), let's give a structure to this!
# let's break down what is happening:
    # 1. yp = mp * x + cp
    # 2. L = (y - yp)**2
    # 3. Let l = y - yp
    # 4. L = l**2
    # 5. dL_dl = 2 * l; dl_dyp = -1; dyp_dmp = x; dyp_dc = 1
    # 6. dL_dmp = dL_dl * dl_dyp * dyp_dmp
    # 7. dL_dc = dL_dl * dl_dyp * dyp_dc
# what we have learnt is that to compute gradient at a level, information needs to flow from the L all the way to the level!
# Let's define dL_dx as grad_x
# L = l**2, grad_l = 2*l
# l = y - yp, grad_yp = grad_l*-1
# yp = mp*x + cp, grad_mp = grad_yp*x, grad_cp = grad_yp*1
# You see, that each level, the definition of the forward pass also gives the definition of the backward pass! Let's build a data structure to capture this!


"""
Thinking about the data structure :)
x -> forward pass -> y
define early as towards the x, and late as towards the y
1. Each intermediate node is formed out of an operation of atmost two(WLOG) earlier nodes say e1 and e2
2. We can now define the grad_e1 = grad_new(all later levels) * dnew_de1(this level); similarly for e2
3. We have to distinguish the X node to be non-tunable, similarly the structure should clearly know what the L node is...
4. We will call the backward() pass on the L node, and it should update all the nodes in the structure.

A point that I later realized is that a node can have multiple children... And each of them will have gradients...
So, the grad of the node will come from the sum of gradients through each of the children...
"""