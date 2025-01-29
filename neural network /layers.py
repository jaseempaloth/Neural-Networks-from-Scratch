import numpy as np
from .activations import relu, sigmoid, tanh, softmax

class DenseLayer:
    def __init__(self, input_size, output_size, activation=None):
        self.weights = np.random.randn(output_size, input_size) * np.sqrt(2.0/input_size)
        self.biases = np.zeros(output_size)
        self.activation = activation
    
    def forward(self, X):
        self.z = np.dot(self.weights, X.T) + self.biases[:, np.newaxis]

        if self.activation == 'relu':
            self.a = relu(self.z)
        elif self.activation == 'sigmoid':
            self.a = sigmoid(self.z)
        elif self.activation == 'tanh':
            self.a = tanh(self.z)
        elif self.activation == 'softmax':
            self.a = softmax(self.z)
        else:
            self.a = self.z # Linear activation
        return self.a.T
