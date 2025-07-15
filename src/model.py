# --- Class for Neural Network Model ---
import numpy as np

class NeuralNetwork():
    def __init__(self, layers=[]):
        """
        Initialize the neural network with a list of layers.
        :param layers: List of Layer objects
        """
        self.layers = layers

    def predict(self, inputs):
        """
        Perform a forward pass through the network.
        :param inputs: Input data for the network
        :return: Output of the network
        """
        for layer in self.layers:
            inputs = layer.forward(inputs)
        return inputs[0][0]

    def train(self, training_data, epochs, learning_rate):
        """
        Train the neural network using backpropagation.
        :param training_data: Tuple of (inputs, targets) or a single array to split
        :param epochs: Number of training iterations
        :param learning_rate: Learning rate for weight updates
        """

        # --- Validate and preprocess training data ---
        if isinstance(training_data, list):
            # Convert list of tuples to a NumPy array
            training_data = np.array(training_data)

        print("Training data shape:", training_data.shape)

        # --- Split training data ---
        if isinstance(training_data, np.ndarray) and training_data.shape[1] == 3:
            inputs = training_data[:, :2]  # First two columns as inputs
            targets = training_data[:, 2:]  # Last column as targets
        elif isinstance(training_data, tuple) and len(training_data) == 2:
            inputs, targets = training_data
        else:
            raise ValueError("Invalid training_data format. Expected a tuple (inputs, targets), "
                             "a NumPy array with shape (n_samples, 3), or a list of tuples.")

        # --- Start training loop ---
        for epoch in range(epochs):
            # --- Forward propagation ---
            outputs = self.predict(inputs)

            # --- Compute loss (mean squared error) ---
            loss = np.mean((outputs - targets) ** 2)
            print(f"Epoch {epoch + 1}/{epochs}, Loss: {loss:.4f}")

            # --- Backward propagation ---
            grad = (2.0 / targets.shape[0]) * (outputs - targets)

            for layer in reversed(self.layers):
                grad = layer.backward(grad, learning_rate)

            for layer in self.layers:
                layer.weights -= learning_rate * layer.weights_gradient
                layer.biases -= learning_rate * layer.biases_gradient

    def evaluate(self, test_data):
        """
        Evaluiert das neuronale Netz und gibt den mittleren quadratischen Fehler (MSE) zurück.
        :param test_data: Tupel aus (inputs, targets)
        :return: Mittlerer quadratischer Fehler (MSE) des Modells
        """
        inputs, targets = test_data
        predictions = self.predict(inputs)
        # Berechne den MSE, genau wie in der Trainingsschleife
        loss = np.mean((predictions - targets) ** 2)
        return loss
    
    def save(self, filename, path='models/'):
        """
        Save the model to a file.
        :param filename: Name of the file to save the model
        """
        import pickle
        with open(path + filename, 'wb') as f:
            pickle.dump(self, f)

    def load(self, filename, path='models/'):
        """
        Load the model from a file.
        :param filename: Name of the file to load the model from
        """
        import pickle
        with open(path + filename, 'rb') as f:
            model = pickle.load(f)
        return model