class Network:
    def __init__(self, layers):
        self.layers = layers

    def forward(self, x):
        output = x
        for layer in self.layers:
            output = layer.forward(output)

        return output

    def backward(self, grad_output):
        grad = grad_output
        for layer in reversed(self.layers):
            grad = layer.backward(grad)
        return grad
