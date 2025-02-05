import numpy as np

class Optimizer:
    def update(self, layer):
        raise NotImplementedError

class SGD(Optimizer):
    def __init__(self, learning_rate=0.01):
        self.learning_rate = learning_rate

    def update(self, layer):
        layer.weights += self.learning_rate * layer.weights_grad
        layer.biases += self.learning_rate * layer.biases_grad
    
   
class Adam(Optimizer):
    def __init__(self, learning_rate=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.learning_rate = learning_rate
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.t = 0
        self.m = 0
        self.v = 0

    def update(self, layer):
        self.t += 1
        self.m = self.beta1 * self.m + (1 - self.beta1) * layer.weights_grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * layer.weights_grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        layer.weights += self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)
        
        self.m = self.beta1 * self.m + (1 - self.beta1) * layer.biases_grad
        self.v = self.beta2 * self.v + (1 - self.beta2) * layer.biases_grad**2
        m_hat = self.m / (1 - self.beta1**self.t)
        v_hat = self.v / (1 - self.beta2**self.t)
        layer.biases += self.learning_rate * m_hat / (np.sqrt(v_hat) + self.epsilon)