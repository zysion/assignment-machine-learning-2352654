"""W03. Rosenblatt perceptron.

Interface (Appendix A):
    perceptron_update(w, b, x, y, lr) -> (w, b)
        One update on one example; y is -1 or +1. If y * (w @ x + b) <= 0 then
        w <- w + lr * y * x and b <- b + lr * y, otherwise w, b are returned unchanged.
    class Perceptron(lr=1.0, max_epochs=100, seed=0)
        fit(X, y) with y in {-1, +1}; returns self. Attributes w_, b_.
        predict(X) -> array of -1 / +1.
"""
import numpy as np


def perceptron_update(w, b, x, y, lr):
    if y * (w @ x + b) <= 0:
        w = w + lr * y * x
        b = b + lr * y
    return w, b
    raise NotImplementedError


class Perceptron:
    def __init__(self, lr=1.0, max_epochs=100, seed=0):
        self.lr, self.max_epochs, self.seed = lr, max_epochs, seed

    def fit(self, X, y):
        w = np.zeros(X.shape[1])
        b = 0.0
        for i in range(self.max_epochs):
            for xi, yi in zip(X, y):
                w, b = perceptron_update(w, b, xi, yi, self.lr)   
        self.w_, self.b_ = w, b
        return self
        raise NotImplementedError

    def predict(self, X):
        z = X @ self.w_ + self.b_
        return np.where(z > 0, 1, -1)
        raise NotImplementedError
