import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import os
from .neural_network import MLP

def create_logical_data(function_name):
    """
    Create training data for logical functions.
    
    Args:
        function_name (str): Name of the logical function ('AND', 'OR', or 'XOR')
        
    Returns:
        tuple: (X, y) where X is input data and y is target values
    """
    # Input combinations for 2-input logical functions
    X = np.array([
        [0, 0],
        [0, 1],
        [1, 0],
        [1, 1]
    ])
    
    # Target values based on the logical function
    if function_name == 'AND':
        y = np.array([[0], [0], [0], [1]])
    elif function_name == 'OR':
        y = np.array([[0], [1], [1], [1]])
    elif function_name == 'XOR':
        y = np.array([[0], [1], [1], [0]])
    else:
        raise ValueError("function_name must be 'AND', 'OR', or 'XOR'")
    
    return X, y

def plot_decision_boundary(model, X, y, title, save_path):
    """
    Plot the decision boundary of the neural network.
    
    Args:
        model: Trained MLP model
        X (numpy.ndarray): Input data
        y (numpy.ndarray): Target values
        title (str): Plot title (not used, kept for compatibility)
        save_path (str): Path to save the figure
    """
    # Create a mesh grid
    x_min, x_max = X[:, 0].min() - 0.5, X[:, 0].max() + 0.5
    y_min, y_max = X[:, 1].min() - 0.5, X[:, 1].max() + 0.5
    xx, yy = np.meshgrid(np.arange(x_min, x_max, 0.01),
                        np.arange(y_min, y_max, 0.01))
    
    # Predict for each point in the mesh grid
    Z = model.predict(np.c_[xx.ravel(), yy.ravel()])
    Z = (Z > 0.5).reshape(xx.shape)
    
    # Set font to match LaTeX
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    
    # Plot the decision boundary
    plt.figure(figsize=(8, 6))
    plt.contourf(xx, yy, Z, alpha=0.3, cmap='RdYlBu')
    scatter = plt.scatter(X[:, 0], X[:, 1], c=y.ravel(), 
                         cmap='RdYlBu', s=200, edgecolor='black', linewidth=2)
    plt.xlabel('Input 1', fontsize=11)
    plt.ylabel('Input 2', fontsize=11)
    
    # Add grid
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # Set axis limits with some padding
    plt.xlim(x_min, x_max)
    plt.ylim(y_min, y_max)
    
    # Save the figure
    plt.savefig(save_path, format='pdf', bbox_inches='tight', dpi=300)
    plt.close()

def save_predictions_table(X, y, predictions, save_path):
    """
    Save predictions to a CSV file.
    
    Args:
        X (numpy.ndarray): Input data
        y (numpy.ndarray): True labels
        predictions (numpy.ndarray): Model predictions
        save_path (str): Path to save the CSV file
    """
    # Create DataFrame
    df = pd.DataFrame({
        'Input 1': X[:, 0],
        'Input 2': X[:, 1],
        'Target': y.ravel(),
        'Predicted': predictions.ravel()
    })
    
    # Save to CSV
    df.to_csv(save_path, index=False)

def train_and_evaluate(function_name, hidden_units=4, epochs=1000):
    """
    Train and evaluate the neural network on a logical function.
    
    Args:
        function_name (str): Name of the logical function
        hidden_units (int): Number of hidden units
        epochs (int): Number of training epochs
    """
    # Create training data
    X, y = create_logical_data(function_name)
    
    # Create and train the model
    model = MLP(layer_sizes=[2, hidden_units, 1], learning_rate=0.1)
    loss_history = model.train(X, y, epochs=epochs, batch_size=4, verbose=True)
    
    # Set font to match LaTeX
    plt.rcParams['font.family'] = 'serif'
    plt.rcParams['font.size'] = 11
    
    # Plot and save training loss
    plt.figure(figsize=(8, 4))
    plt.plot(loss_history)
    plt.xlabel('Epoch', fontsize=11)
    plt.ylabel('Loss', fontsize=11)
    plt.savefig(f'doc/img/{function_name.lower()}_loss.pdf', format='pdf', bbox_inches='tight')
    plt.close()
    
    # Plot and save decision boundary
    plot_decision_boundary(
        model, X, y,
        f'{function_name} Decision Boundary',
        f'doc/img/{function_name.lower()}_boundary.pdf'
    )
    
    # Save predictions table
    predictions = model.predict(X)
    save_predictions_table(
        X, y, predictions,
        f'doc/tab/{function_name.lower()}_predictions.csv'
    )
    
    # Print predictions
    print(f"\n{function_name} Predictions:")
    print("Input | Target | Predicted")
    print("-" * 25)
    for i in range(len(X)):
        print(f"{X[i]} | {y[i][0]:.0f} | {predictions[i][0]:.4f}")

def main():
    """Run tests for all logical functions."""
    # Create output directories if they don't exist
    os.makedirs('doc/img', exist_ok=True)
    os.makedirs('doc/tab', exist_ok=True)
    
    # Test AND function
    print("\nTraining AND function...")
    train_and_evaluate('AND', hidden_units=4, epochs=1000)
    
    # Test OR function
    print("\nTraining OR function...")
    train_and_evaluate('OR', hidden_units=4, epochs=1000)
    
    # Test XOR function (needs more hidden units)
    print("\nTraining XOR function...")
    train_and_evaluate('XOR', hidden_units=8, epochs=2000)

if __name__ == "__main__":
    main() 