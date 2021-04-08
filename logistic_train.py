import numpy as np
from tqdm import tqdm


def sigmoid(x):
    return 1 / (1 + np.exp(-x))


def xent(predictions, labels):
    return -(
        labels * np.log(predictions) + (1 - labels) * np.log(1 - predictions)
    ).mean()


def grad(predictions, labels, data):
    g = ((predictions - labels) * data).sum(axis=0)
    return np.expand_dims(g, 1)


def logistic_train(data, labels, epsilon, maxiter=1000, lr=0.001):
    N, d = data.shape

    weights = np.random.rand(d, 1)
    prev_prediction = np.random.rand(N, 1)
    pbar = tqdm(range(maxiter))
    for iteration in pbar:

        # Prediction
        prediction = sigmoid(data @ weights)
        error = xent(prediction, labels)

        # Gradient step
        g = grad(prediction, labels, data)
        weights = weights - lr * g

        # Stop condition
        prediction_diff = (np.abs(prediction - prev_prediction)).mean()
        if epsilon is not None and prediction_diff < epsilon:
            print(f"Stopping after {iteration} iterations.")
            break
        else:
            prev_prediction = prediction

        predictions = predict(data, weights)
        correct = (predictions == labels).sum()

        # Update progress
        pbar.set_description(
            f"Error: {error:.3f}, Change in pred: {prediction_diff:.5f}, Weight Norm: {np.linalg.norm(weights):.2f}, Accuracy: {correct / labels.shape[0]}"
        )

    return weights


def predict(data, weights):
    return np.where(sigmoid(data @ weights) > 0.5, 1, 0)


if __name__ == "__main__":
    TRAIN_SIZE = 2000

    # Load data
    with open("spam_data/data.txt") as f:
        data = np.loadtxt(f)

    with open("spam_data/labels.txt") as f:
        labels = np.loadtxt(f)
    labels = np.expand_dims(labels, axis=-1)

    # Add feature for bias weights
    data = np.concatenate([data, np.ones((data.shape[0], 1))], axis=1)

    # Split data
    train_x = data[:TRAIN_SIZE, :]
    train_y = labels[:TRAIN_SIZE, :]

    test_x = data[TRAIN_SIZE:, :]
    test_y = labels[TRAIN_SIZE:, :]

    for n in [200, 500, 800, 1000, 1500, 2000]:
        weights = logistic_train(train_x[:n, :], train_y[:n, :], 1e-6, maxiter=1000)

        predictions = predict(test_x, weights)
        correct = (predictions == test_y).sum()
        print(f"n={n}, accuracy : {correct / test_y.shape[0]}")
