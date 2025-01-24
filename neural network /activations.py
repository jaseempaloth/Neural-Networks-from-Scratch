import numpy as np

# Activation functions and their derivatives

def relu(x):
    """
    ReLU function, max(0, x), values are squashed between 0 and infinity
    ReLU is Used primarily in hidden layers
    """
    return np.maximum(0, x)

def relu_derivative(x):
    """
    Derivative of the ReLU function
    """
    return np.where(x > 0, 1, 0)

def leaky_relu(x):
    """
    Leaky ReLU function, max(0.1 * x, x), values are squashed between 0 and infinity
    Leaky ReLU is used for hidden layers
    Popular variant of ReLU that addresses the "dying ReLU" problem
    Instead of zero, it has a small slope for negative values (typically 0.01)
    """
    return np.maximum(0.1 * x, x)

def leaky_relu_derivative(x):
    """
    Derivative of the Leaky ReLU function
    """
    return np.where(x > 0, 1, 0.1)

def tanh(x):
    """
    Hyperbolic tangent function, (exp(x) - exp(-x)) / (exp(x) + exp(-x)), values are squashed between -1 and 1
    Tanh is used for hidden layers, Often performs better than sigmoid in hidden layers
    """
    return (np.exp(x) - np.exp(-x)) / (np.exp(x) + np.exp(-x))

def tanh_derivative(x):
    """
    Derivative of the tanh function
    """
    return 1 - np.tanh(x)**2

def sigmoid(x):
    """
    Sigmoid function, 1 / (1 + exp(-x)), values are squashed between 0 and 1
    Sigmoid is mainly used for output layer for binary classification
    """
    return 1 / (1 + np.exp(-x))

def sigmoid_derivative(x):
    """
    Derivative of the sigmoid function
    """
    return x * (1 - x)

def softmax(x):
    """
    Softmax function, exp(x) / sum(exp(x)), values are squashed between 0 and 1
    Softmax standard choice for output layer in multi-class classification
    """
    exp_x = np.exp(x - np.max(x, axis=0, keepdims=True))  # Subtract max for numerical stability
    return exp_x / exp_x.sum(axis=0, keepdims=True)

def softmax_derivative(x):
    """
    Derivative of the softmax function
    """
    return softmax(x) * (1 - softmax(x))

def log_softmax(x):
    """
    Natural logarithm of the softmax function
    """
    return np.log(softmax(x))

def log_softmax_derivative(x):
    """
    Derivative of the log softmax function
    """
    return softmax(x)

def elu(x):
    """
    Exponential linear unit function, exp(x) - 1 if x < 0, x otherwise
    ELU is used for hidden layers
    """
    return np.where(x > 0, x, np.exp(x) - 1)

def elu_derivative(x):
    """
    Derivative of the ELU function
    """
    return np.where(x > 0, 1, np.exp(x))

def selu(x):
    """
    Scaled Exponential Linear Unit function, alpha * (exp(x) - 1) if x < 0, alpha * x otherwise
    SELU is used for hidden layers
    """
    return np.where(x > 0, x, alpha * (np.exp(x) - 1))

def selu_derivative(x):
    """
    Derivative of the SELU function
    """
    alpha = 1.6732632423543772848170429916717
    return np.where(x > 0, 1, alpha * np.exp(x))

def swish(x):
    """
    Swish function, x / (1 + exp(-x)), values are squashed between 0 and 1
    Swish is used for hidden layers
    """
    return x / (1 + np.exp(-x))

def swish_derivative(x):
    """
    Derivative of the Swish function
    """
    return swish(x) * (1 - swish(x))

def gelu(x):
    """
    Gaussian Error Linear Unit function, 0.5 * x * (1 + tanh(sqrt(2 / pi) * (x + 0.044715 * x**3)))
    GELU is used for hidden layers
    """
    return 0.5 * x * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))

def gelu_derivative(x):
    """
    Derivative of the GELU function
    """
    return 0.5 * (1 + np.tanh(np.sqrt(2 / np.pi) * (x + 0.044715 * x**3)))
