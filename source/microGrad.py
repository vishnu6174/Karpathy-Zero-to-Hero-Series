from random import random
class Value:
    def __init__(self, val, children = None, op = None, grad = 0):
        self.val = val
        self.children = children
        self.op = op
        self.grad = grad
        self.backward = lambda: None

    def __repr__(self):
        return f"Value(data={self.val}, grad={self.grad})"
    
    # Construction of the nodes from elementary operations
    def __add__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        res = Value(self.val + other.val, children = [self, other], op = "+")
        def backward():
            # res = self + other                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                                            9
            # dL/dself = dL/dres * dres/dself 
            self.grad += res.grad
            other.grad += res.grad
            #print(f"self: {self}, other: {other}")
        res.backward = backward
        # every node should know the backward pass that needs to be invoked, when necessary!
        return res
    
    def __mul__(self, other):
        other = Value(other) if not isinstance(other, Value) else other
        res = Value(self.val * other.val, children = [self, other], op = "*")
        def backward():
            # in this step:
            # res = self * other
            # dL/dself = dL/dres * dres/dself ## dL/dres is easier to calculate as it is closer to L
            self.grad += res.grad * other.val # += because self can have other children too
            other.grad += res.grad * self.val
        res.backward = backward
        return res
    
    def __pow__(self, power):
        assert isinstance(power,(int, float)), "Only integer/float powers are supported"
        res = Value(self.val**power, children = [self], op = f"**{power}")
        def backward():
            # in this step: res = self**power
            # dL/dself = dL/dres * dres/dself
            self.grad += res.grad * power * self.val**(power-1)
        res.backward = backward
        return res
    
    # this backward pass when called will do the entire backpropagation starting from this node! dL/dnode calculation
    def back_prop(self):
        self.grad = 1 # dL/dL = 1
        # let us do a dfs topo sort to do the backpropagation
        topo = self.topo_sort()
        for node in topo:
            node.backward()
        
    def topo_sort(self):
        topo = []
        visited = set()
        def dfs(node):
            if node in visited: return
            if node.children:
                for child in node.children:
                    dfs(child)
            topo.append(node)
            visited.add(node)
        dfs(self)
        topo.reverse()
        return topo

    def step(self, lr):
        # set up the gradients and then update the values
        topo = self.topo_sort()
        self.back_prop()
        for node in topo:
            node.val -= lr * node.grad
            node.grad = 0 # reset the gradients for the next iteration

    def __rmul__(self, other):
        # cases like 2*Value(3)
        return self.__mul__(other)

    def __sub__(self, other):
        return self + (-1)*other

    def __neg__(self):
        # Eg: -Value(3)
        return -1*self # will work because __rmul__ is defined
    
    def __truediv__(self, other):
        return self * other**-1 # will work because __pow__ is defined
    
    def __rtruediv__(self, other):
        # other/self
        return other * self**-1
    
    def __radd__(self, other):
        # other + self
        return self + other
    
    def __rsub__(self, other):
        # other - self
        return -1*self + other
    # let us assume that cases like Val**Val are 2**Val are not supported for now...