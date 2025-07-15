# --- Class for a layer in a neural network ---
import numpy as np
from activation_functions import sigmoid, relu, linear, sigmoid_derivative, relu_derivative, linear_derivative

class DenseLayer():
    def __init__(self, input_size, output_size, activation_function=sigmoid):
        """
        Initialize a layer with given input size, output size, and activation function.
        :param input_size: Number of inputs to the layer
        :param output_size: Number of outputs from the layer
        :param activation_function: Activation function to use (default is sigmoid)
        """
        # --- Validate input parameters ---
        if input_size <= 0 or output_size <= 0:
            raise ValueError("Input size and output size must be positive integers.")
        if activation_function not in [sigmoid, relu, linear]:
            raise ValueError("Invalid activation function. Choose from sigmoid, relu, or linear.")
        
        # --- Initialize layer properties ---
        self.input_size = input_size
        self.output_size = output_size
        self.activation_function = activation_function
        self.activation_derivative = {
            sigmoid: sigmoid_derivative,
            relu: relu_derivative,
            linear: linear_derivative
        }[activation_function]
        self.weights = self.initialize_weights()
        self.biases = self.initialize_biases()

    def initialize_weights(self):
        # Initialize weights with small random values
        return np.random.randn(self.input_size, self.output_size) * 0.01

    def initialize_biases(self):
        # Initialize biases to zero
        return np.zeros((1, self.output_size))

    def forward(self, inputs):
        self.inputs = inputs
        self.z = np.dot(inputs, self.weights) + self.biases
        self.a = self.activation_function(self.z)
        return self.a

    def backward(self, grad, learning_rate):
        # Compute the gradient of the activation function
        activation_grad = grad * self.activation_derivative(self.a)

        # Compute gradients for weights and biases
        self.weights_gradient = np.dot(self.inputs.T, activation_grad)  # Shape: (input_size, output_size)
        self.biases_gradient = np.sum(activation_grad, axis=0, keepdims=True)  # Shape: (1, output_size)

        # Compute the gradient to propagate to the previous layer
        grad_input = np.dot(activation_grad, self.weights.T)  # Shape: (batch_size, input_size)

        return grad_input