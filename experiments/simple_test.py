import numpy as np

from src.layers import Linear
from src.activations import ReLU, Softmax
from src.losses import CrossEntropyLoss
from src.optimizers import SGD
from src.network import Network

X = np.array([
    [-2.0, -1.0],
    [-1.5, -2.0],
    [-1.0, -1.5],
    [ 1.0,  1.5],
    [ 1.5,  2.0],
    [ 2.0,  1.0]
])

y = np.array([0, 0, 0, 1, 1, 1])


# one hidden layer model
network = Network([
    Linear(2, 4),
    ReLU(),
    Linear(4, 2),
    Softmax()
])

# loss
loss_fn = CrossEntropyLoss()

# optimizer
optimizer = SGD(learning_rate=0.01)

#training
for epoch in range(100):
    #forward
    pred = network.forward(X)

    # compute loss
    loss = loss_fn.forward(pred, y)

    #backprop
    grad = loss_fn.backward()
    network.backward(grad)

    # optimize
    optimizer.step(network.layers)

    if epoch % 10 == 0:
        print(epoch, loss)