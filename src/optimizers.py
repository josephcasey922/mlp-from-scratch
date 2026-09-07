import numpy as np

class SGD:
    def __init__(self, learning_rate):
        self.learning_rate = learning_rate

    def step(self, layers):
        for layer in layers:
            if hasattr(layer, 'W') and hasattr(layer, 'b'):
                layer.W -= self.learning_rate * layer.dw
                layer.b -= self.learning_rate * layer.db