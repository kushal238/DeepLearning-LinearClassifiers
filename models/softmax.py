"""Softmax model."""

import numpy as np


class Softmax:
    def __init__(self, n_class: int, lr: float, epochs: int, reg_const: float):
        """Initializing a new classifier.

        Parameters:
            n_class: the number of classes
            lr: the learning rate
            epochs: the number of epochs to train for
            reg_const: the regularization constant
        """
        self.w = None  # TODO: change this
        self.lr = lr
        self.epochs = epochs
        self.reg_const = reg_const
        self.n_class = n_class
        self.mean = None 
        self.std = None 
        self.decay_rate = 0.1

    def standardize(self, X, training=False):
        if training: 
            self.mean = np.mean(X, axis=0)
            self.std = np.std(X, axis=0)
        return (X - self.mean) / self.std

    def calc_gradient(self, X_train: np.ndarray, y_train: np.ndarray) -> np.ndarray:
        """Calculating gradient of the softmax loss.

        Inputs have dimension D, there are C classes, and we operate on
        mini-batches of N examples.

        Parameters:
            X_train: a numpy array of shape (N, D) containing a mini-batch
                of data
            y_train: a numpy array of shape (N,) containing training labels;
                y[i] = c means that X[i] has label c, where 0 <= c < C

        Returns:
            gradient with respect to weights w; an array of same shape as w
        """
        n_samples, n_features = X_train.shape
        gradient = np.zeros_like(self.w)
        T = 0.9  # Temperature

        for i in range(n_samples):
            scores = np.dot(X_train[i], self.w.T) # xi * w
            scores_max = np.max(scores) # max(xi * w)
            exp_scores = np.exp((scores - scores_max) / T) # exp(xi * w - max(xi * w)
            sum_exp_scores = np.sum(exp_scores) # sum(exp(xi * w - max(xi * w))
            probs = exp_scores / sum_exp_scores # softmax

            for c in range(self.n_class):
                if c == y_train[i]:
                    gradient[c] += (probs[c] - 1) * X_train[i] # p * xi - xi
                else:
                    gradient[c] += probs[c] * X_train[i] # p * xi

        gradient /= n_samples
        return gradient

    def get_acc(self, pred, y_test):
        return np.sum(y_test == pred) / len(y_test) * 100
    
    def get_test_acc(self, X_val, y_val):
        # get the predictions
        pred_test = self.predict(X_val)
        acc_test = self.get_acc(pred_test, y_val)

        return np.array([acc_test])
    
    def train(self, X_train: np.ndarray, y_train: np.ndarray, X_val, y_val):
        """Training the classifier.

        Hint: operate on mini-batches of data for SGD.

        Parameters:
            X_train: a numpy array of shape (N, D) containing training data;
                N examples with D dimensions
            y_train: a numpy array of shape (N,) containing training labels
        """
        np.random.seed(0)
        initial_lr = self.lr
        X_train = self.standardize(X_train, training=True)
        n_samples, n_features = X_train.shape
        self.w = np.random.rand(self.n_class, n_features)
        batch_size = 5000 # maybe 5000
        max_acc = 0
        for epoch in range(self.epochs):
            for batch in range(0, n_samples, batch_size):
                end = min(batch + batch_size, n_samples)
                X_batch = X_train[batch:end]
                y_batch = y_train[batch:end]
                gradient = self.calc_gradient(X_batch, y_batch)
                self.w -= self.lr * gradient
            # self.lr = (1/(1+30*epoch)) * self.lr
            self.lr = np.exp(-self.decay_rate * epoch) * initial_lr
            accuracy = self.get_acc(self.predict(X_val), y_val)
            if accuracy > max_acc:
                # store self.w to a file called svm_weights.npy
                max_acc = accuracy
                np.save('svm_weights.npy', self.w)

            print(f"Epoch {epoch + 1}/{self.epochs}, Accuracy: {accuracy:.2f}")


    def predict(self, X_test: np.ndarray) -> np.ndarray:
        """using the trained weights to predict labels for test data points.

        Parameters:
            X_test: a numpy array of shape (N, D) containing testing data;
                N examples with D dimensions

        Returns:
            predicted labels for the data in X_test; a 1-dimensional array of
                length N, where each element is an integer giving the predicted
                class.
        """
        X_test = self.standardize(X_test)
        scores = np.dot(X_test, self.w.T)
        y_pred = np.argmax(scores, axis=1)
        return y_pred
