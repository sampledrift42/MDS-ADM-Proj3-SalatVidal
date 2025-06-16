# My MLP - Neural Network Implementation

A Python package implementing a Multi-Layer Perceptron (MLP) from scratch, including backpropagation and gradient checking. The implementation is demonstrated through learning logical functions (AND, OR, XOR) with comprehensive analysis of the learning process and decision boundaries.

## Features

- Implementation of backpropagation from scratch
- Support for multiple layers and neurons
- Numerical gradient checking for verification
- Built-in support for learning logical functions (AND, OR, XOR)
- Comprehensive visualization of training progress and decision boundaries
- Detailed analysis of network performance and predictions

## Installation

```bash
pip install -e .
```

## Dependencies

- numpy>=1.21.0
- matplotlib>=3.4.0
- tqdm>=4.65.0
- PyQt6>=6.4.0
- pandas>=1.3.0

## Usage

### Basic Usage

```python
from my_mlp import MLP

# Create a neural network with 2 input neurons, 4 hidden neurons, and 1 output neuron
model = MLP(layer_sizes=[2, 4, 1], learning_rate=0.1)

# Train the model
loss_history = model.train(X, y, epochs=1000, batch_size=4, verbose=True)

# Make predictions
predictions = model.predict(X_test)
```

### Learning Logical Functions

The package includes built-in support for learning logical functions:

```python
from my_mlp.test_logical_functions import train_and_evaluate

# Train and evaluate AND function
train_and_evaluate('AND', hidden_units=4, epochs=1000)

# Train and evaluate OR function
train_and_evaluate('OR', hidden_units=4, epochs=1000)

# Train and evaluate XOR function (requires more hidden units)
train_and_evaluate('XOR', hidden_units=8, epochs=2000)
```

## Project Structure

- `my_mlp/`: Main package directory
  - `__init__.py`: Package initialization
  - `neural_network.py`: MLP implementation with backpropagation
  - `activations.py`: Activation functions (sigmoid, tanh)
  - `test_logical_functions.py`: Implementation of logical function tests
- `doc/`: Documentation and analysis
  - `img/`: Generated plots and visualizations
  - `tab/`: Generated prediction tables
  - `main.tex`: LaTeX document with analysis
- `setup.py`: Package configuration
- `requirements.txt`: Project dependencies

## Analysis

The project includes a comprehensive analysis of the network's performance on logical functions, including:
- Training loss curves
- Decision boundary visualizations
- Detailed prediction tables
- Performance analysis and interpretation

The analysis is documented in the LaTeX document (`doc/main.tex`) and includes generated figures and tables in the `doc/img/` and `doc/tab/` directories. Use `pdflatex` to compile it to .pdf, or use the included Makefile (simply type `make`). 
