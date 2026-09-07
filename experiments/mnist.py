import numpy as np

from sklearn.datasets import fetch_openml
from sklearn.model_selection import train_test_split

from src.layers import Linear
from src.activations import ReLU, Softmax
from src.losses import CrossEntropyLoss
from src.optimizers import SGD
from src.network import Network

# settings
RANDOM_SEED = 1

TRAIN_SIZE = 10000
TEST_SIZE = 2000

BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 0.01


# load mnist and normalize data

mnist = fetch_openml(
    "mnist_784",
    version=1,
    as_frame=False,
    parser="auto"
)

X = mnist.data.astype(np.float64)
y = mnist.target.astype(np.int64)

X /= 255.0



# split train/test
X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    train_size=TRAIN_SIZE,
    test_size=TEST_SIZE,
    random_state=RANDOM_SEED,
    stratify=y
)

print("Training shape:", X_train.shape)
print("Test shape:", X_test.shape)



network = Network([
    Linear(784, 128),
    ReLU(),

    Linear(128, 64),
    ReLU(),

    Linear(64, 10),
    Softmax()
])


loss_fn = CrossEntropyLoss()

optimizer = SGD( learning_rate=LEARNING_RATE)


# for randomizing batch selections
rng = np.random.default_rng(RANDOM_SEED)


# train
for epoch in range(EPOCHS):

    # Shuffle training examples
    indices = rng.permutation(len(X_train))

    X_train_shuffled = X_train[indices]
    y_train_shuffled = y_train[indices]

    epoch_loss = 0.0
    num_batches = 0


    for start in range(0, len(X_train), BATCH_SIZE):

        end = start + BATCH_SIZE

        X_batch = X_train_shuffled[start:end]
        y_batch = y_train_shuffled[start:end]


        # Forward

        pred = network.forward(X_batch)

        loss = loss_fn.forward(pred, y_batch)

        # Backward

        grad = loss_fn.backward()

        network.backward(grad)


        #optimize

        optimizer.step(network.layers)

        # start adding loss after first optimization step
        epoch_loss += loss
        num_batches += 1



    # average loss
    average_loss = epoch_loss / num_batches


    #train accuracy
    train_probabilities = network.forward(X_train)
    train_predictions = np.argmax(train_probabilities, axis=1)
    train_accuracy = np.mean(train_predictions == y_train)

    # test accuracy
    test_probabilities = network.forward(X_test)
    test_predictions = np.argmax(test_probabilities, axis=1)
    test_accuracy = np.mean( test_predictions == y_test)


    print(
        f"Epoch {epoch + 1:2d}/{EPOCHS} "
        f"| Loss: {average_loss:.4f} "
        f"| Train Acc: {train_accuracy:.4f} "
        f"| Test Acc: {test_accuracy:.4f}"
    )



print("\nTraining complete.")

# final results
test_probabilities = network.forward(X_test)
test_predictions = np.argmax(test_probabilities,axis=1)
final_accuracy = np.mean(test_predictions == y_test)

print(
    f"Final test accuracy: {final_accuracy * 100:.2f}%"
)