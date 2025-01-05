from microGrad import Value
from random import random, randint
import matplotlib.pyplot as plt
from tqdm import tqdm

def generate_data(f, num_samples=1000, x_range=(-100, 100)):
    X = [randint(*x_range) for _ in range(num_samples)]
    Y = [f(x) for x in X]
    return X, Y

def initialize_layer(input_size, output_size):
    W = [[Value(random()) for _ in range(input_size)] for _ in range(output_size)]
    B = [Value(random()) for _ in range(output_size)]
    return W, B

def relu(inputs):
    return [node.relu() for node in inputs]

def tanh(inputs):
    return [node.tanh_activation() for node in inputs]

def forward_layer(inputs, weights, biases, activation=None):
    if activation is None: activation = lambda x: x
    layer = [sum([w * inp for w, inp in zip(weights[i], inputs)]) + biases[i] for i in range(len(biases))]
    return activation(layer)

def build_and_train_nn(X, Y, epochs=800, lr=0.00001, neurons=4):
    Xtrue = Value(random(), no_grad=True)
    Ytrue = Value(random(), no_grad=True)

    W1, B1 = initialize_layer(1, neurons)
    W2, B2 = initialize_layer(neurons, neurons)
    W4, B4 = initialize_layer(neurons, 1)

    for i in tqdm(range(epochs)):
        ind = randint(0, len(X) - 1)
        Xtrue.val = X[ind]
        Ytrue.val = Y[ind]

        H1 = forward_layer([Xtrue], W1, B1, activation=tanh)

        H2 = forward_layer(H1, W2, B2, activation=relu)

        Ypred = forward_layer(H2, W4, B4, activation=None)[0]

        L = (Ytrue - Ypred) ** 2
        L.step(lr)

        if i % 100 == 0:
            print(f"Epoch: {i}, Loss: {L.val:.2f}, Ypred: {Ypred.val:.2f}, Ytrue: {Ytrue.val:.2f}")

    return W1, B1, W2, B2, W4, B4

def plot_predictions(X, W1, B1, W2, B2, W4, B4):
    preds = []
    Xtrue = Value(random(), no_grad=True)
    for x in X:
        Xtrue.val = x
        H1 = forward_layer([Xtrue], W1, B1)

        H2 = forward_layer(H1, W2, B2)

        Ypred = forward_layer(H2, W4, B4)[0]
        Ypred.forward_prop()
        preds.append(Ypred.val)

    plt.scatter(X, preds)
    plt.show()

def toyNN():
    def f(x): return x*3+7
    X, Y = generate_data(f)
    W1, B1, W2, B2, W4, B4 = build_and_train_nn(X, Y)
    plot_predictions(X, W1, B1, W2, B2, W4, B4)

toyNN()