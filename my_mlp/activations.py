import numpy as np

def sigmoid(x):
    """Sigmoid activation function."""
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))  # Clip to prevent overflow

def sigmoid_derivative(x):
    """Derivative of the sigmoid function."""
    s = sigmoid(x)
    return s * (1 - s)

def tanh(x):
    """Hyperbolic tangent activation function."""
    return np.tanh(x)

def tanh_derivative(x):
    """Derivative of the tanh function."""
    return 1 - np.tanh(x) ** 2 