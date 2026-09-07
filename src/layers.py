import numpy as np

class Linear:
    def __init__(self, input_dim, output_dim):
        rng = np.random.default_rng()
        self.input_dim = input_dim
        self.output_dim = output_dim

        # He initialization with ReLU
        self.W = rng.normal(loc = 0.0, scale = np.sqrt(2.0 / self.input_dim),
                            size = (self.input_dim, self.output_dim))
        #add Xavier initialization
        # self.W = . . .

        self.b = np.zeros(self.output_dim)
        self.dw = None
        self.db = None

    def forward(self, x):
        #self.batch_size = x.shape[0]
        self.x = x
        z = self.x @ self.W + self.b
        return z

    def backward(self, grad_output):
        self.dw = self.x.T @ grad_output
        self.db = np.sum(grad_output, axis=0)
        dx = grad_output @ self.W.T
        return dx