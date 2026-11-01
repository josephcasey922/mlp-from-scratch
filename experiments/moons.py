from sklearn.datasets import make_moons
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

import numpy as np

from src.layers import Linear
from src.activations import ReLU, Softmax
from src.losses import CrossEntropyLoss
from src.optimizers import SGD
from src.network import Network

X, y = make_moons(
    n_samples=500,
    noise=0.15,
    random_state=1
)

X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.2,
    random_state=1,
    stratify=y
)

scaler = StandardScaler()

X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

network = Network([
    Linear(2, 8),
    ReLU(),
    Linear(8, 8),
    ReLU(),
    Linear(8, 2),
    Softmax()
])

loss_fn = CrossEntropyLoss()
optimizer = SGD(learning_rate=0.01)

#training
for epoch in range(1000):
    #forward
    pred = network.forward(X_train)

    # compute loss
    loss = loss_fn.forward(pred, y_train)

    #backprop
    grad = loss_fn.backward()
    network.backward(grad)

    # optimize
    optimizer.step(network.layers)

    if epoch % 10 == 0:
        print(epoch, loss)

pred = network.forward(X_test)
predicted_classes = np.argmax(pred, axis=1)

accuracy = np.mean(predicted_classes == y_test)

print("Test accuracy:", accuracy)