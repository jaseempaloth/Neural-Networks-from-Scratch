import numpy as np

class Activation:
    """Base class for all activation functions."""

    def __call__(self, x: Tensor) -> Tensor:
        """Apply the activation function."""
        return self.forward(x)
    
    def parameters(self):
        """Return an empty list since activations don't have parameters."""
        return []
    
    def forward(self, x: Tensor) -> Tensor:
        """
        Forward pass of the activation function.
        
        Args:
            x: Input tensor.
            
        Returns:
            Output tensor after applying the activation function.
        """
        raise NotImplementedError("Subclasses must implement forward method.")
    

class ReLU(Activation):
    """
    Rectified Linear Unit activation function.
    
    f(x) = max(0, x)
    """
    
