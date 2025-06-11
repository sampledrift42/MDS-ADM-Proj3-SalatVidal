import numpy as np
from .activations import sigmoid, sigmoid_derivative
from tqdm import tqdm

class MLP:
    def __init__(self, layer_sizes, learning_rate=0.1):
        """
        Initialize a Multi-Layer Perceptron.
        
        Args:
            layer_sizes (list): List of integers representing the number of neurons in each layer
            learning_rate (float): Learning rate for gradient descent
        """
        self.layer_sizes = layer_sizes
        self.learning_rate = learning_rate
        self.weights = []
        self.biases = []
        
        # Initialize weights and biases
        for i in range(len(layer_sizes) - 1):
            # He initialization
            self.weights.append(
                np.random.randn(layer_sizes[i], layer_sizes[i + 1]) * np.sqrt(2.0 / layer_sizes[i])
            )
            self.biases.append(np.zeros((1, layer_sizes[i + 1])))
    
    def forward(self, X):
        """
        Perform forward propagation.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            tuple: (activations, pre_activations) for each layer
        """
        activations = [X]
        pre_activations = []
        
        for w, b in zip(self.weights, self.biases):
            z = np.dot(activations[-1], w) + b
            pre_activations.append(z)
            activations.append(sigmoid(z))
            
        return activations, pre_activations
    
    def predict(self, X):
        """
        Make predictions for input data.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
            
        Returns:
            numpy.ndarray: Predicted class labels
        """
        activations, _ = self.forward(X)
        return activations[-1]

    def binary_cross_entropy_loss(self, y_true, y_pred):
        """
        Compute binary cross entropy loss.
        
        Args:
            y_true (numpy.ndarray): True labels
            y_pred (numpy.ndarray): Predicted probabilities
            
        Returns:
            float: Binary cross entropy loss
        """
        epsilon = 1e-15  # Small constant to prevent log(0)
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))

    def binary_cross_entropy_gradient(self, y_true, y_pred):
        """
        Compute gradient of binary cross entropy loss.
        
        Args:
            y_true (numpy.ndarray): True labels
            y_pred (numpy.ndarray): Predicted probabilities
            
        Returns:
            numpy.ndarray: Gradient of loss with respect to predictions
        """
        epsilon = 1e-15
        y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
        return -y_true / y_pred + (1 - y_true) / (1 - y_pred)

    def backpropagate(self, X, y_true):
        """
        Perform backpropagation to compute gradients and update weights.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
            y_true (numpy.ndarray): True labels of shape (n_samples, n_outputs)
            
        Returns:
            float: Current loss value
        """
        # Forward pass
        activations, pre_activations = self.forward(X)
        
        # Initialize gradients
        weight_gradients = [np.zeros_like(w) for w in self.weights]
        bias_gradients = [np.zeros_like(b) for b in self.biases]
        
        # Compute output layer error
        output_error = self.binary_cross_entropy_gradient(y_true, activations[-1])
        delta = output_error * sigmoid_derivative(pre_activations[-1])
        
        # Backpropagate error
        for l in range(len(self.weights) - 1, -1, -1):
            # Compute gradients
            weight_gradients[l] = np.dot(activations[l].T, delta) / X.shape[0]
            bias_gradients[l] = np.mean(delta, axis=0, keepdims=True)
            
            if l > 0:  # Don't compute delta for input layer
                # Compute error for previous layer
                delta = np.dot(delta, self.weights[l].T) * sigmoid_derivative(pre_activations[l-1])
        
        # Update weights and biases
        for l in range(len(self.weights)):
            self.weights[l] -= self.learning_rate * weight_gradients[l]
            self.biases[l] -= self.learning_rate * bias_gradients[l]
        
        # Compute and return current loss
        return self.binary_cross_entropy_loss(y_true, activations[-1])

    def train(self, X, y, epochs, batch_size=32, verbose=True):
        """
        Train the neural network using mini-batch gradient descent.
        
        Args:
            X (numpy.ndarray): Input data of shape (n_samples, n_features)
            y (numpy.ndarray): True labels of shape (n_samples, n_outputs)
            epochs (int): Number of training epochs
            batch_size (int): Size of mini-batches
            verbose (bool): Whether to print training progress
            
        Returns:
            list: Training loss history
        """
        n_samples = X.shape[0]
        loss_history = []
        
        # Create progress bar
        pbar = tqdm(range(epochs), desc="Training", disable=not verbose)
        
        for epoch in pbar:
            # Shuffle data
            indices = np.random.permutation(n_samples)
            X_shuffled = X[indices]
            y_shuffled = y[indices]
            
            epoch_loss = 0
            
            # Mini-batch training
            for i in range(0, n_samples, batch_size):
                X_batch = X_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                
                # Perform backpropagation
                batch_loss = self.backpropagate(X_batch, y_batch)
                epoch_loss += batch_loss * len(X_batch)
            
            # Compute average loss for the epoch
            epoch_loss /= n_samples
            loss_history.append(epoch_loss)
            
            # Update progress bar with current loss
            if verbose:
                pbar.set_postfix({'loss': f'{epoch_loss:.4f}'})
        
        return loss_history 