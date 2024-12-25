from microGrad import Value
from random import random, randint, sample
# let's test the datastructure!
"""    
What is it capable of? 
1. It can do a step to update it's weights through the network automatically to minimize the Loss...
2. Let's define the loss as: L = (y - (mx + c))**2
3. Let us initialise m and c randomly, and then do a step to update the weights to minimize the loss...
4. Let us fix the target m and c as 3 and 4. we are trying to get the line y = 3x+4
5. two sample points at x = 1, and at x = 2, y = 7, and y = 10, or let's say x is generated randomly at runtime :)
"""
from tqdm import tqdm
def estimate_coefficients():
    def f(x): return 3 * x ** 2 + 12 * x + 5  # 3 points are needed to uniquely identify a quadratic equation
    epochs = 500
    a = Value(random())
    b = Value(random())
    c = Value(random())
    lr = 0.01
    X = [Value(x, no_grad=True) for x in [-1, 0, 1]]
    Y = [Value(f(x.val), no_grad=True) for x in X]
    print(f"X: {[x.val for x in X]}, Y: {[y.val for y in Y]}")
    L = sum([(y - (a * x ** 2 + b * x + c)) ** 2 for x, y in zip(X, Y)])
    for i in tqdm(range(epochs)):
        L.step(lr)
        if i % 100 == 0: print(f"Epoch: {i}, Loss: {L.val:.2f}, equation: {a.val:.2f}x^2 + {b.val:.2f}x + {c.val:.2f}")
    print(f"Final equation: {a.val:.2f}x^2 + {b.val:.2f}x + {c.val:.2f} vs actual: 3x^2 + 12x + 5")

"""
Now, let us build an actual toy neural network for f(x) = 8*x**3 + 4*x**2 + 3*x + 2
not by estimating the coefficients, but using a neural network and training it to learn the function
"""
def toyNN():
    def f(x, sd): return x*3
    X = [randint(-100, 100) for _ in range(1000)]
    Y = [f(x, 0) for x in X]
    # plotting the data
    import matplotlib.pyplot as plt
    # plt.scatter(X, Y)
    # plt.show()
    epochs = 800
    lr = 0.00001
    # let's build a neural network with 3 hidden layers of 10 neurons each
    Xtrue = Value(random(), no_grad=True) # just a place holder value, it will keep getting updated based on the input...
    Ytrue = Value(random(), no_grad=True)
    ind = randint(0, len(X))
    Xtrue.val = X[ind]
    Ytrue.val = Y[ind]
    neurons = 4
    # we need to define the weights and biases that lead us to the hidden layers and eventually to the output layer
    W1 = [Value(random()) for _ in range(neurons)]
    B1 = [Value(random()) for _ in range(neurons)]
    # now the hidden layer 1 is just a linear combination of the input layer and the weights and biases
    x = Xtrue
    H1 = [w*x + b for w, b in zip(W1, B1)]
    # normalize each hidden layer to 0 mean and 1 variance
    # H1 = [(h - sum(H1)/len(H1))/sum([(h - sum(H1)/len(H1))**2 for h in H1])**0.5 for h in H1]
    # pass it through a ReLU activation function
    H1 = [node.relu() for node in H1]
    print(f"Hidden layer 1: {[h.val for h in H1]}")
    # now let's define the weights and biases for the next hidden layer, it's shape is 10x10
    W2 = [[Value(random()) for _ in range(neurons)] for _ in range(neurons)]
    B2 = [Value(random()) for _ in range(neurons)]
    # now the hidden layer 2 is just a linear combination of the hidden layer 1 and the weights and biases
    H2 = [sum([w*h for w, h in zip(W2[i], H1)]) + B2[i] for i in range(neurons)]
    # H2 = [(h - sum(H2)/len(H2))/sum([(h - sum(H2)/len(H2))**2 for h in H2])**0.5 for h in H2]
    H2 = [node.relu() for node in H2]
    print(f"Shape of W2: {len(W2)}x{len(W2[0])}, Shape of H2: {len(H2)}")
    # # now let's define the weights and biases for the next hidden layer, it's shape is 10x10
    # W3 = [[Value(random()) for _ in range(neurons)] for _ in range(neurons)]
    # B3 = [Value(random()) for _ in range(neurons)]
    # # now the hidden layer 3 is just a linear combination of the hidden layer 2 and the weights and biases
    # H3 = [sum([w*h for w, h in zip(W3[i], H2)]) + B3[i] for i in range(neurons)]
    # # H3 = [(h - sum(H3)/len(H3))/sum([(h - sum(H3)/len(H3))**2 for h in H3])**0.5 for h in H3]
    # # H3 = [node.relu() for node in H3]
    #
    # print(f"Shape of W3: {len(W3)}x{len(W3[0])}, Shape of H3: {len(H3)}")
    # Finally, the output layer is just a linear combination of the hidden layer 3 and the weights and biases
    W4 = [Value(random()) for _ in range(neurons)]
    B4 = Value(random())
    Ypred = sum([w*h for w, h in zip(W4, H2)]) + B4
    # now let's define the loss function
    L = (Ytrue - Ypred) ** 2
    for i in tqdm(range(epochs)):
        ind = randint(0, len(X)-1)
        Xtrue.val = X[ind]
        Ytrue.val = Y[ind]
        L.step(lr)
        if i % 1 == 0: print(f"Epoch: {i}, Loss: {L.val:.2f}, Ypred: {Ypred.val:.2f}, Ytrue: {Ytrue.val:.2f}")

    # let's plot the data
    # get predictions for all the data points from the model...
    preds = []
    for x in X:
        Xtrue.val = x
        Ypred.forward_prop()
        preds.append(Ypred.val)
    plt.scatter(X, preds)
    plt.show()

toyNN()