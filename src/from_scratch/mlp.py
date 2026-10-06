"""W04. Multilayer perceptron with an explicit forward and backward pass.

Interface (Appendix A). params is a dict {"W1", "b1", ..., "WL", "bL"}; W_l has shape
(n_in, n_out). The output layer is softmax; the loss is mean cross-entropy. The hidden
activation is your choice (state it in MODEL_LOG.md).
    init_params(sizes, seed) -> params       sizes = [d, h1, ..., K]
    forward(params, X) -> (P, cache)          P has shape (n, K), rows sum to 1
    loss(params, X, Y) -> float               Y is one-hot (n, K); mean cross-entropy
    backward(params, cache, Y) -> grads       same keys and shapes as params, gradient of loss
"""
import numpy as np

def activation(z):
    return np.where(z > 0, z, 0)  # ReLU

def output_activation(z):
    exp_z = np.exp(z)
    return exp_z / np.sum(exp_z, axis=1, keepdims=True)  # softmax

def init_params(sizes, seed=0):
    rng = np.random.default_rng(seed)
    params = {}
    for i in range(1, len(sizes)):
        params[f'W{i}'] = rng.normal(0, 0.1, (sizes[i-1], sizes[i]))
        params[f'b{i}'] = np.zeros(sizes[i])
    return params
    raise NotImplementedError


def forward(params, X):
    A = X
    cache = {'A0': A}
    n_layers = len(params) // 2  # have w and b
    for i in range(1, n_layers):
        A = activation(A @ params[f'W{i}'] + params[f'b{i}'])
        cache[f'A{i}'] = A
    A = output_activation(A @ params[f'W{n_layers}'] + params[f'b{n_layers}'])
    cache[f'A{n_layers}'] = A
    return A, cache
    raise NotImplementedError


def loss(params, X, Y):
    #mean cross-entropy loss
    A, _ = forward(params, X)
    return -np.mean(np.sum(Y * np.log(np.clip(A, 1e-12, 1.0)), axis=1))
    raise NotImplementedError

def backward(params, cache, Y):
    P = cache[f'A{len(params) // 2}']
    grads = {}
    dZ = P - Y
    n_layers = len(params) // 2
    for i in range(n_layers, 0, -1):
        A_prev = cache[f'A{i-1}']
        grads[f'W{i}'] = A_prev.T @ dZ / Y.shape[0]
        grads[f'b{i}'] = np.mean(dZ, axis=0)
        if i > 1:
            dA_prev = dZ @ params[f'W{i}'].T
            dZ = dA_prev * (A_prev > 0)  # derivative of ReLU
    return grads
    raise NotImplementedError
