import numpy as np

# --- Activation Functions Module ---
def sigmoid(x):
    """Compute the sigmoid activation function."""
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """Compute the derivative of the sigmoid function."""
    return x * (1 - x)

def relu(x):
    """Compute the ReLU activation function."""
    return np.maximum(0, x)

def relu_derivative(x):
    """Compute the derivative of the ReLU function."""
    return np.where(x > 0, 1, 0)

def linear(x):
    """Compute the linear activation function."""
    return x

def linear_derivative(x):
    """Compute the derivative of the linear function."""
    return np.ones_like(x)