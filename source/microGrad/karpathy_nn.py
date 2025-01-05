import random
from karpathy_engine import Value
import matplotlib.pyplot as plt

class Module:

    def zero_grad(self):
        for p in self.parameters():
            p.grad = 0

    def parameters(self):
        return []

class Neuron(Module):

    def __init__(self, nin, nonlin=True):
        self.w = [Value(random.uniform(-1,1)) for _ in range(nin)]
        self.b = Value(0)
        self.nonlin = nonlin

    def __call__(self, x):
        act = sum((wi*xi for wi,xi in zip(self.w, x)), self.b)
        #80% Relu, 10% square, 10% cube
        if self.nonlin:
            # weighted sampling
            # if random.random() < 0.8:
            #     return act.square()
            return act.relu()
        else:
            return act

    def parameters(self):
        return self.w + [self.b]

    def __repr__(self):
        return f"{'ReLU' if self.nonlin else 'Linear'}Neuron({len(self.w)})"

class Layer(Module):

    def __init__(self, nin, nout, **kwargs):
        self.neurons = [Neuron(nin, **kwargs) for _ in range(nout)]

    def __call__(self, x):
        out = [n(x) for n in self.neurons]
        return out[0] if len(out) == 1 else out

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

    def __repr__(self):
        return f"Layer of [{', '.join(str(n) for n in self.neurons)}]"

class MLP(Module):

    def __init__(self, nin, nouts):
        sz = [nin] + nouts
        self.layers = [Layer(sz[i], sz[i+1], nonlin=i!=len(nouts)-1) for i in range(len(nouts))]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]

    def __repr__(self):
        return f"MLP of [{', '.join(str(layer) for layer in self.layers)}]"

# Trying on an example:
def f(x, sd=0):
    return 3*x**2 + 2*x + 1 + random.gauss(0, sd)
xrange = (-10, 10)
samples = 300 # get 100 samples by splitting the range into 100 parts
d = (xrange[1] - xrange[0]) / samples
X = [xrange[0] + i*d for i in range(samples)]
Y = [f(x) for x in X]

# Let's train a simple MLP to predict this function:
# Initialize the MLP
mlp = MLP(2, [2,2,1])
# Training loop
learning_rate = 0.005
for epoch in range(101):
    # Forward pass
    Ypred = [mlp([Value(x), Value(x**2)]) for x in X]

    # Compute loss (Mean Squared Error)
    loss = sum((y_pred - y)**2 for y_pred, y in zip(Ypred, Y)) / len(Y)

    # Backward pass
    mlp.zero_grad()
    loss.backward()

    # Update parameters
    for p in mlp.parameters():
        p.data -= learning_rate * p.grad

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss.data}")
# Plotting the results
plt.scatter(X, Y, label='True Function')
plt.scatter(X, [y_pred.data for y_pred in Ypred], label='Predicted Function')
plt.legend()
plt.show()

