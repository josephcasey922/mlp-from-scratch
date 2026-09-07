import numpy as np

class Sigmoid:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = 1 / (1 + np.exp(-x))
        return self.output

    def backward(self,grad_output):
        grad = self.output * (1 - self.output)
        return grad * grad_output


class ReLU:
    def __init__(self):
        self.mask = None

    def forward(self, x):
        self.mask = x > 0
        return np.maximum(0, x)

    def backward(self, grad_output):
        return grad_output * self.mask

class Tanh:
    def __init__(self):
        self.output = None

    def forward(self, x):
        self.output = np.tanh(x)
        return self.output

    def backward(self, grad_output):
        return grad_output * (1 - self.output ** 2)

class LeakyReLU:
    def __init__(self, alpha=0.01):
        self.alpha = alpha
        self.mask = None

    def forward(self,x):
        self.mask = np.where(x >= 0, 1, self.alpha)
        return self.mask * x

    def backward(self, grad_output):
        return grad_output * self.mask

class Softmax:
    def __init__(self):
        self.output = None

    def forward(self, x):
        shifted_x = x - np.max(x, axis=1, keepdims=True)
        self.output = np.exp(shifted_x) / np.sum(np.exp(shifted_x), axis=1, keepdims=True)
        return self.output

    def backward(self, grad_output):
        grad_input = np.zeros_like(self.output)
        for i in range(self.output.shape[0]):
            j = np.diag(self.output[i]) - np.outer(self.output[i], self.output[i])
            grad_input[i] = j.T @ grad_output[i]
        return grad_input



