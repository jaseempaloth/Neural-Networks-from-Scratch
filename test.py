import numpy as np
from neural_network import NeuralNetwork
from sklearn.datasets import make_classification, make_moons
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def test_binary_classification():
    # Generate a binary classification dataset
    X, y = make_moons(n_samples=1000, noise=0.1)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert labels to one-hot encoded format
    y_train_one_hot = np.eye(2)[y_train]
    y_test_one_hot = np.eye(2)[y_test]
    
    # Create and train the neural network
    nn = NeuralNetwork(input_size=2, hidden_size=10, output_size=2, activation='softmax')
    nn.train(X_train, y_train_one_hot, epochs=100, learning_rate=0.01)
    
    # Make predictions
    predictions = nn.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Binary Classification Accuracy: {accuracy:.4f}")

def test_multiclass_classification():
    # Generate a multi-class classification dataset
    X, y = make_classification(n_samples=1000, n_features=20, n_classes=3, n_clusters_per_class=1)
    X = StandardScaler().fit_transform(X)
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    
    # Convert labels to one-hot encoded format
    y_train_one_hot = np.eye(3)[y_train]
    y_test_one_hot = np.eye(3)[y_test]
    
    # Create and train the neural network
    nn = NeuralNetwork(input_size=20, hidden_size=15, output_size=3, activation='softmax')
    nn.train(X_train, y_train_one_hot, epochs=100, learning_rate=0.01)
    
    # Make predictions
    predictions = nn.predict(X_test)
    accuracy = np.mean(predictions == y_test)
    print(f"Multi-class Classification Accuracy: {accuracy:.4f}")

if __name__ == "__main__":
    print("Testing Binary Classification...")
    test_binary_classification()
    
    print("\nTesting Multi-class Classification...")
    test_multiclass_classification()
