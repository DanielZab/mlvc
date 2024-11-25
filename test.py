import numpy as np

class MLP:
    def __init__(self, input_size, hidden_size, output_size, learning_rate=0.01):
        # Initialize weights and biases
        self.learning_rate = learning_rate
        self.weights_input_hidden = np.random.randn(input_size, hidden_size) * 0.1
        self.bias_hidden = np.zeros((1, hidden_size))
        self.weights_hidden_output = np.random.randn(hidden_size, output_size) * 0.1
        self.bias_output = np.zeros((1, output_size))

    @staticmethod
    def sigmoid(x):
        """Sigmoid activation function."""
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def sigmoid_derivative(x):
        """Derivative of the sigmoid activation function."""
        return MLP.sigmoid(x) * (1 - MLP.sigmoid(x))

    @staticmethod
    def mean_squared_error(y_true, y_pred):
        """Mean squared error loss."""
        return np.mean((y_true - y_pred) ** 2)

    @staticmethod
    def mean_squared_error_derivative(y_true, y_pred):
        """Derivative of the MSE loss."""
        return y_pred - y_true

    def forward(self, X):
        """Forward pass through the network."""
        # Hidden layer
        self.hidden_input = np.dot(X, self.weights_input_hidden) + self.bias_hidden
        self.hidden_output = self.sigmoid(self.hidden_input)

        # Output layer
        self.output_input = np.dot(self.hidden_output, self.weights_hidden_output) + self.bias_output
        self.output = self.sigmoid(self.output_input)

        return self.output

    def backpropagation(self, X, y_true):
        """Backward pass through the network to update weights."""
        # Calculate the loss derivative
        loss_derivative = self.mean_squared_error_derivative(y_true, self.output) # output (4,1), loss derivative (4,1)

        # Output layer gradients
        output_error = loss_derivative * self.sigmoid_derivative(self.output) # (4,1)
        d_weights_hidden_output = np.dot(self.hidden_output.T, output_error) # (4,1), (4,4), (4,1)
        d_bias_output = np.sum(output_error, axis=0, keepdims=True)

        # Hidden layer gradients
        hidden_error = np.dot(output_error, self.weights_hidden_output.T) * self.sigmoid_derivative(self.hidden_output)
        d_weights_input_hidden = np.dot(X.T, hidden_error)
        d_bias_hidden = np.sum(hidden_error, axis=0, keepdims=True)

        # Update weights and biases
        self.weights_hidden_output -= self.learning_rate * d_weights_hidden_output
        self.bias_output -= self.learning_rate * d_bias_output
        self.weights_input_hidden -= self.learning_rate * d_weights_input_hidden
        self.bias_hidden -= self.learning_rate * d_bias_hidden

    def train(self, X, y_true, epochs=1000):
        """Train the MLP using the forward and backpropagation functions."""
        for epoch in range(epochs):
            # Forward pass
            y_pred = self.forward(X)

            # Backward pass
            self.backpropagation(X, y_true)

            # Calculate and print loss every 100 epochs
            if epoch % 100 == 0:
                loss = self.mean_squared_error(y_true, y_pred)
                print(f"Epoch {epoch}, Loss: {loss:.4f}")

    def predict(self, X):
        """Make predictions using the trained model."""
        return self.forward(X)

# Example usage
if __name__ == "__main__":
    # Define dataset (XOR problem)
    X = np.array([[0, 0], [0, 1], [1, 0], [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Initialize and train the MLP
    mlp = MLP(input_size=2, hidden_size=4, output_size=1, learning_rate=0.1)
    mlp.train(X, y, epochs=1000)

    # Make predictions
    predictions = mlp.predict(X)
    print("Predictions:")
    print(predictions)
