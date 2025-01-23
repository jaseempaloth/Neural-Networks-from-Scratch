import numpy as np


def sigmoid(x):
    # Sigmoid function, 1 / (1 + exp(-x)), values are squashed between 0 and 1
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    return x * (1 - x)

def relu(x):
    # ReLU function, max(0, x), values are squashed between 0 and infinity
    return np.maximum(0, x)

def relu_derivative(x):
    return np.where(x > 0, 1, 0)

def softmax(x):
    # Softmax function, exp(x) / sum(exp(x)), values are squashed between 0 and 1
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # Subtract max for numerical stability
    return exp_x / exp_x.sum(axis=0, keepdims=True)

class NeuralNetwork:
    def __init__(self, input_size, hidden_size, output_size, activation='sigmoid'):
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.output_size = output_size
        self.activation = activation

        # He initialization for ReLU
        self.hidden_weights = np.random.randn(hidden_size, input_size) * np.sqrt(2.0/input_size)
        self.hidden_bias = np.zeros(hidden_size)
        
        # Xavier initialization for sigmoid/softmax
        self.output_weights = np.random.randn(output_size, hidden_size) * np.sqrt(1.0/hidden_size)
        self.output_bias = np.zeros(output_size)
        
        self.best_loss = float('inf')
        self.patience_counter = 0
    
    def forward(self, X):
        # Hidden layer computation
        self.hidden = np.dot(self.hidden_weights, X.T) + self.hidden_bias[:, np.newaxis]
        self.hidden_output = relu(self.hidden)
        
        # Output layer computation
        self.output = np.dot(self.output_weights, self.hidden_output) + self.output_bias[:, np.newaxis]
        
        if self.activation == 'sigmoid':
            self.activated_output = sigmoid(self.output)
            return self.activated_output
        else:
            self.activated_output = softmax(self.output)
            return self.activated_output
    
    def backward(self, X, y, learning_rate):
        batch_size = X.shape[0]
        
        # Calculate output layer error
        if self.activation == 'sigmoid':
            output_error = y.T - self.activated_output
            output_delta = output_error * sigmoid_derivative(self.activated_output)
        else:  # softmax
            output_delta = y.T - self.activated_output

        # Calculate hidden layer error
        hidden_error = np.dot(self.output_weights.T, output_delta)
        hidden_delta = hidden_error * relu_derivative(self.hidden_output)

        # Update weights and biases with batch normalization
        self.output_weights += learning_rate * np.dot(output_delta, self.hidden_output.T) / batch_size
        self.output_bias += learning_rate * np.sum(output_delta, axis=1) / batch_size
        self.hidden_weights += learning_rate * np.dot(hidden_delta, X) / batch_size
        self.hidden_bias += learning_rate * np.sum(hidden_delta, axis=1) / batch_size

    def calculate_loss(self, y_true, y_pred):
        if self.activation == 'sigmoid':
            return np.mean(np.square(y_true.T - y_pred))
        else:  # Cross-entropy loss for softmax
            return -np.mean(y_true.T * np.log(y_pred + 1e-15))

    def train(self, X, y, epochs, learning_rate, batch_size=32, patience=5):
        n_samples = X.shape[0]
        
        for i in range(epochs):
            # Mini-batch processing
            indices = np.random.permutation(n_samples)
            for start_idx in range(0, n_samples, batch_size):
                batch_indices = indices[start_idx:start_idx + batch_size]
                X_batch = X[batch_indices]
                y_batch = y[batch_indices]
                
                predictions = self.forward(X_batch)
                self.backward(X_batch, y_batch, learning_rate)
            
            # Calculate error on full dataset
            if i % 10 == 0:
                predictions = self.forward(X)
                current_loss = self.calculate_loss(y, predictions)
                print(f"Epoch {i}, Loss: {current_loss:.6f}")
                
                # Early stopping
                if current_loss < self.best_loss:
                    self.best_loss = current_loss
                    self.patience_counter = 0
                else:
                    self.patience_counter += 1
                    if self.patience_counter >= patience:
                        print(f"Early stopping at epoch {i}")
                        break

    def predict(self, X):
        predictions = self.forward(X)
        if self.activation == 'sigmoid':
            return (predictions > 0.5).T.astype(int)
        return np.argmax(predictions, axis=0)