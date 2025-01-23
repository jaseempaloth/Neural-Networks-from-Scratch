import numpy as np
from neural_network import NeuralNetwork

def main():
    network = NeuralNetwork(input_size=2, hidden_size=3, output_size=1)

    # XOR input and target
    X = np.array([[0, 0],
                  [0, 1],
                  [1, 0],
                  [1, 1]])
    y = np.array([[0], [1], [1], [0]])

    # Train the network
    network.train(X, y, epochs=1000, learning_rate=0.1, batch_size=2, patience=5)

    # Make predictions
    predictions = network.predict(X)
    print("Predictions:", predictions)

    # Varify predictions
    print("Expected:", y)
    accuracy = np.mean(predictions == y)
    print(f"Accuracy: {accuracy * 100:.2f}%")

if __name__ == "__main__":
    main()


