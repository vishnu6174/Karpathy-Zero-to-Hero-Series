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


class Value:
    def __init__(self, val, op=None):
        self.val = val
        self.grad = 0 # grad definition: dL/dx

# Toy task: generate samples of y from x, where y = 3x + 4, and build a model to predict y from x.
# We will use a multi layer perceptron but without any activation function(we will add it later), and that model should generalize Universally!
def generate_linear_data(samples = 1000, m = 3, c = 4, sd = 0.05):
    x = torch.randn(samples)
    y = m*x + c + torch.randn(samples)*sd
    return x, y

x_linear, y_linear = generate_linear_data(1000, 3, 4, 0.05)
print(x_linear, y_linear)
plt.scatter(x_linear, y_linear)
plt.show()