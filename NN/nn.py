"""
Pure Python Fully Connected Neural Network

A parametrized implementation of a feedforward neural network with:
- Configurable architecture (layers, neurons per layer)
- Multiple activation functions
- Gradient descent training with backpropagation
- Manual weight inspection and updates
"""

import numpy as np
from typing import List, Tuple, Callable, Optional


# =============================================================================
# Activation Functions and Their Derivatives
# =============================================================================

def sigmoid(x: np.ndarray) -> np.ndarray:
    return 1 / (1 + np.exp(-np.clip(x, -500, 500)))


def sigmoid_derivative(x: np.ndarray) -> np.ndarray:
    s = sigmoid(x)
    return s * (1 - s)


def relu(x: np.ndarray) -> np.ndarray:
    return np.maximum(0, x)


def relu_derivative(x: np.ndarray) -> np.ndarray:
    return (x > 0).astype(float)


def tanh(x: np.ndarray) -> np.ndarray:
    return np.tanh(x)


def tanh_derivative(x: np.ndarray) -> np.ndarray:
    return 1 - np.tanh(x) ** 2


def linear(x: np.ndarray) -> np.ndarray:
    return x


def linear_derivative(x: np.ndarray) -> np.ndarray:
    return np.ones_like(x)


def softmax(x: np.ndarray) -> np.ndarray:
    exp_x = np.exp(x - np.max(x, axis=1, keepdims=True))
    return exp_x / np.sum(exp_x, axis=1, keepdims=True)


ACTIVATIONS = {
    'sigmoid': (sigmoid, sigmoid_derivative),
    'relu': (relu, relu_derivative),
    'tanh': (tanh, tanh_derivative),
    'linear': (linear, linear_derivative),
}


# =============================================================================
# Loss Functions
# =============================================================================

def mse_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    return np.mean((y_pred - y_true) ** 2)


def mse_loss_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    return 2 * (y_pred - y_true) / y_true.shape[0]


def cross_entropy_loss(y_pred: np.ndarray, y_true: np.ndarray) -> float:
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return -np.mean(y_true * np.log(y_pred) + (1 - y_true) * np.log(1 - y_pred))


def cross_entropy_derivative(y_pred: np.ndarray, y_true: np.ndarray) -> np.ndarray:
    epsilon = 1e-15
    y_pred = np.clip(y_pred, epsilon, 1 - epsilon)
    return (y_pred - y_true) / (y_pred * (1 - y_pred) * y_true.shape[0])


LOSSES = {
    'mse': (mse_loss, mse_loss_derivative),
    'cross_entropy': (cross_entropy_loss, cross_entropy_derivative),
}


# =============================================================================
# Layer Class
# =============================================================================

class Layer:
    """A single fully connected layer."""

    def __init__(
            self,
            input_size: int,
            output_size: int,
            activation: str = 'relu',
            weight_init: str = 'xavier'
    ):
        self.input_size = input_size
        self.output_size = output_size
        self.activation_name = activation

        if activation not in ACTIVATIONS:
            raise ValueError(f"Unknown activation: {activation}. Choose from {list(ACTIVATIONS.keys())}")

        self.activation, self.activation_derivative = ACTIVATIONS[activation]

        # Weight initialization
        if weight_init == 'xavier':
            scale = np.sqrt(2.0 / (input_size + output_size))
        elif weight_init == 'he':
            scale = np.sqrt(2.0 / input_size)
        else:
            scale = 0.01

        self.weights = np.random.randn(input_size, output_size) * scale
        self.biases = np.zeros((1, output_size))

        # Cache for backpropagation
        self.input_cache = None
        self.z_cache = None  # pre-activation values

        # Gradients
        self.weight_gradients = None
        self.bias_gradients = None

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the layer."""
        self.input_cache = x
        self.z_cache = x @ self.weights + self.biases
        return self.activation(self.z_cache)

    def backward(self, grad_output: np.ndarray) -> np.ndarray:
        """
        Backward pass through the layer.

        Args:
            grad_output: Gradient from the next layer (dL/da)

        Returns:
            Gradient to pass to the previous layer
        """
        # Gradient through activation: dL/dz = dL/da * da/dz
        grad_z = grad_output * self.activation_derivative(self.z_cache)

        # Compute gradients for weights and biases
        self.weight_gradients = self.input_cache.T @ grad_z
        self.bias_gradients = np.sum(grad_z, axis=0, keepdims=True)

        # Gradient to pass to previous layer: dL/dx = dL/dz * dz/dx
        grad_input = grad_z @ self.weights.T

        return grad_input

    def update_weights(self, learning_rate: float):
        """Update weights using computed gradients."""
        self.weights -= learning_rate * self.weight_gradients
        self.biases -= learning_rate * self.bias_gradients

    def set_weights(self, weights: np.ndarray, biases: np.ndarray):
        """Manually set weights and biases."""
        if weights.shape != self.weights.shape:
            raise ValueError(f"Weight shape mismatch: expected {self.weights.shape}, got {weights.shape}")
        if biases.shape != self.biases.shape:
            raise ValueError(f"Bias shape mismatch: expected {self.biases.shape}, got {biases.shape}")
        self.weights = weights.copy()
        self.biases = biases.copy()

    def get_weights(self) -> Tuple[np.ndarray, np.ndarray]:
        """Get current weights and biases."""
        return self.weights.copy(), self.biases.copy()

    def __repr__(self):
        return f"Layer({self.input_size} -> {self.output_size}, activation={self.activation_name})"


# =============================================================================
# Neural Network Class
# =============================================================================

class NeuralNetwork:
    """
    Fully connected feedforward neural network.

    Example:
        # Create a network with 2 inputs, two hidden layers (8 and 4 neurons), and 1 output
        nn = NeuralNetwork(
            layer_sizes=[2, 8, 4, 1],
            activations=['relu', 'relu', 'sigmoid']
        )
    """

    def __init__(
            self,
            layer_sizes: List[int],
            activations: Optional[List[str]] = None,
            loss: str = 'mse',
            weight_init: str = 'xavier'
    ):
        """
        Initialize the neural network.

        Args:
            layer_sizes: List of layer sizes [input_size, hidden1, hidden2, ..., output_size]
            activations: List of activation functions for each layer (excluding input)
                        If None, uses ReLU for hidden layers and sigmoid for output
            loss: Loss function ('mse' or 'cross_entropy')
            weight_init: Weight initialization method ('xavier', 'he', or 'small')
        """
        if len(layer_sizes) < 2:
            raise ValueError("Need at least input and output layer sizes")

        self.layer_sizes = layer_sizes
        self.num_layers = len(layer_sizes) - 1

        # Default activations
        if activations is None:
            activations = ['relu'] * (self.num_layers - 1) + ['sigmoid']

        if len(activations) != self.num_layers:
            raise ValueError(f"Need {self.num_layers} activations, got {len(activations)}")

        # Loss function
        if loss not in LOSSES:
            raise ValueError(f"Unknown loss: {loss}. Choose from {list(LOSSES.keys())}")
        self.loss_fn, self.loss_derivative = LOSSES[loss]
        self.loss_name = loss

        # Create layers
        self.layers: List[Layer] = []
        for i in range(self.num_layers):
            layer = Layer(
                input_size=layer_sizes[i],
                output_size=layer_sizes[i + 1],
                activation=activations[i],
                weight_init=weight_init
            )
            self.layers.append(layer)

        # Training history
        self.history = {'loss': [], 'val_loss': []}

    def forward(self, x: np.ndarray) -> np.ndarray:
        """Forward pass through the entire network."""
        for layer in self.layers:
            x = layer.forward(x)
        return x

    def predict(self, x: np.ndarray) -> np.ndarray:
        """Alias for forward pass (inference mode)."""
        return self.forward(x)

    def backward(self, y_pred: np.ndarray, y_true: np.ndarray):
        """Backward pass to compute all gradients."""
        # Start with loss gradient
        grad = self.loss_derivative(y_pred, y_true)

        # Backpropagate through layers in reverse
        for layer in reversed(self.layers):
            grad = layer.backward(grad)

    def update_weights(self, learning_rate: float):
        """Update all weights using computed gradients."""
        for layer in self.layers:
            layer.update_weights(learning_rate)

    def train_step(self, x: np.ndarray, y: np.ndarray, learning_rate: float) -> float:
        """
        Perform a single training step.

        Args:
            x: Input data (batch_size, input_size)
            y: Target data (batch_size, output_size)
            learning_rate: Learning rate for gradient descent

        Returns:
            Loss value for this batch
        """
        # Forward pass
        y_pred = self.forward(x)

        # Compute loss
        loss = self.loss_fn(y_pred, y)

        # Backward pass
        self.backward(y_pred, y)

        # Update weights
        self.update_weights(learning_rate)

        return loss

    def fit(
            self,
            x_train: np.ndarray,
            y_train: np.ndarray,
            epochs: int = 100,
            learning_rate: float = 0.01,
            batch_size: Optional[int] = None,
            validation_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
            verbose: bool = True,
            print_every: int = 10
    ) -> dict:
        """
        Train the network on data.

        Args:
            x_train: Training inputs
            y_train: Training targets
            epochs: Number of training epochs
            learning_rate: Learning rate for gradient descent
            batch_size: Mini-batch size (None for full batch)
            validation_data: Optional (x_val, y_val) tuple
            verbose: Print progress
            print_every: Print every N epochs

        Returns:
            Training history dictionary
        """
        n_samples = x_train.shape[0]

        if batch_size is None:
            batch_size = n_samples

        self.history = {'loss': [], 'val_loss': []}

        for epoch in range(epochs):
            # Shuffle data
            indices = np.random.permutation(n_samples)
            x_shuffled = x_train[indices]
            y_shuffled = y_train[indices]

            # Mini-batch training
            epoch_losses = []
            for i in range(0, n_samples, batch_size):
                x_batch = x_shuffled[i:i + batch_size]
                y_batch = y_shuffled[i:i + batch_size]
                loss = self.train_step(x_batch, y_batch, learning_rate)
                epoch_losses.append(loss)

            avg_loss = np.mean(epoch_losses)
            self.history['loss'].append(avg_loss)

            # Validation
            if validation_data is not None:
                x_val, y_val = validation_data
                val_pred = self.predict(x_val)
                val_loss = self.loss_fn(val_pred, y_val)
                self.history['val_loss'].append(val_loss)

            # Progress
            if verbose and (epoch + 1) % print_every == 0:
                msg = f"Epoch {epoch + 1}/{epochs} - Loss: {avg_loss:.6f}"
                if validation_data is not None:
                    msg += f" - Val Loss: {val_loss:.6f}"
                print(msg)

        return self.history

    def compute_loss(self, x: np.ndarray, y: np.ndarray) -> float:
        """Compute loss without updating weights."""
        y_pred = self.forward(x)
        return self.loss_fn(y_pred, y)

    # =========================================================================
    # Manual Weight Access Methods
    # =========================================================================

    def get_layer_weights(self, layer_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get weights and biases for a specific layer."""
        if layer_index < 0 or layer_index >= len(self.layers):
            raise IndexError(f"Layer index {layer_index} out of range [0, {len(self.layers) - 1}]")
        return self.layers[layer_index].get_weights()

    def set_layer_weights(self, layer_index: int, weights: np.ndarray, biases: np.ndarray):
        """Manually set weights and biases for a specific layer."""
        if layer_index < 0 or layer_index >= len(self.layers):
            raise IndexError(f"Layer index {layer_index} out of range [0, {len(self.layers) - 1}]")
        self.layers[layer_index].set_weights(weights, biases)

    def get_all_weights(self) -> List[Tuple[np.ndarray, np.ndarray]]:
        """Get all weights and biases as a list of (weights, biases) tuples."""
        return [layer.get_weights() for layer in self.layers]

    def set_all_weights(self, all_weights: List[Tuple[np.ndarray, np.ndarray]]):
        """Set all weights from a list of (weights, biases) tuples."""
        if len(all_weights) != len(self.layers):
            raise ValueError(f"Expected {len(self.layers)} weight tuples, got {len(all_weights)}")
        for i, (weights, biases) in enumerate(all_weights):
            self.layers[i].set_weights(weights, biases)

    def get_gradients(self, layer_index: int) -> Tuple[np.ndarray, np.ndarray]:
        """Get computed gradients for a specific layer (after backward pass)."""
        layer = self.layers[layer_index]
        return layer.weight_gradients.copy(), layer.bias_gradients.copy()

    def manual_weight_update(
            self,
            layer_index: int,
            weight_delta: np.ndarray,
            bias_delta: np.ndarray
    ):
        """Manually update weights by adding deltas."""
        layer = self.layers[layer_index]
        layer.weights += weight_delta
        layer.biases += bias_delta

    def summary(self):
        """Print network architecture summary."""
        print("=" * 60)
        print("Neural Network Summary")
        print("=" * 60)
        print(f"Loss function: {self.loss_name}")
        print("-" * 60)

        total_params = 0
        for i, layer in enumerate(self.layers):
            n_params = layer.weights.size + layer.biases.size
            total_params += n_params
            print(f"Layer {i}: {layer} - {n_params:,} parameters")

        print("-" * 60)
        print(f"Total parameters: {total_params:,}")
        print("=" * 60)

    def __repr__(self):
        return f"NeuralNetwork(layers={self.layer_sizes}, loss={self.loss_name})"


# =============================================================================
# Demo / Example Usage
# =============================================================================

if __name__ == "__main__":
    np.random.seed(42)
    test_net = NeuralNetwork(
        layer_sizes=[9, 5, 2],
        activations=['relu', 'sigmoid'],
        loss='mse'
    )

    X = np.random.uniform(0, 1, 900).reshape(100, 9)

    predictions = test_net.predict(X)

    print(predictions.shape)
    print(np.min(predictions[:, 0]), np.mean(predictions[:, 0]), np.max(predictions[:, 0]))
    print(np.min(predictions[:, 0]), np.mean(predictions[:, 0]), np.max(predictions[:, 0]))

    print([(l[0].shape, l[1].shape) for l in test_net.get_all_weights()])