"""
RNN implementation for stock price prediction.

This module contains a custom implementation of a vanilla RNN (Recurrent Neural Network)
built from scratch using PyTorch. The implementation includes both single RNN layers
and a multi-layer RNN model suitable for time series prediction tasks.

Classes:
    RNNLayer: A single RNN layer implementing the basic recurrent computation
    RNN: A multi-layer RNN model with training, evaluation, and prediction methods
"""

import torch
import torch.nn as nn
import numpy as np
from torch.utils.data import DataLoader
import config


class RNNLayer(nn.Module):
    """
    A single RNN layer implementing the basic recurrent computation.

    This layer applies the standard RNN formula:
    h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b_ih)

    Args:
        input_size (int): The number of expected features in the input x
        hidden_size (int): The number of features in the hidden state h
    """

    def __init__(self, input_size, hidden_size):
        super().__init__()
        self.input_size = input_size
        self.hidden_size = hidden_size

        # Linear transformations for RNN computation
        # W_ih: input-to-hidden weight matrix with bias
        self.input_transform = nn.Linear(input_size, hidden_size, bias=True)
        # W_hh: hidden-to-hidden weight matrix without bias (bias already in input_transform)
        self.hidden_transform = nn.Linear(hidden_size, hidden_size, bias=False)

        # Initialize weights using appropriate initialization schemes
        self.reset_parameters()

    def reset_parameters(self):
        """
        Initialize the layer parameters using appropriate initialization schemes.

        - Xavier uniform for input transformation weights (good for tanh activation)
        - Zero initialization for bias terms
        - Orthogonal initialization for hidden weights (helps with gradient flow)
        """
        nn.init.xavier_uniform_(self.input_transform.weight)
        nn.init.zeros_(self.input_transform.bias)
        nn.init.orthogonal_(self.hidden_transform.weight)

    def forward(self, x, h_prev):
        """
        Forward pass through the RNN layer.

        Computes: h_t = tanh(W_ih * x_t + W_hh * h_{t-1} + b_ih)

        Args:
            x (torch.Tensor): Input tensor of shape (batch_size, input_size)
            h_prev (torch.Tensor): Previous hidden state of shape (batch_size, hidden_size)

        Returns:
            torch.Tensor: New hidden state of shape (batch_size, hidden_size)
        """
        # Apply linear transformations and combine with tanh activation
        h_t = torch.tanh(self.input_transform(x) + self.hidden_transform(h_prev))
        return h_t


class RNN(nn.Module):
    """
    Multi-layer RNN for time series prediction.

    This model stacks multiple RNN layers and includes dropout for regularization.
    It's designed specifically for stock price prediction tasks where we want to
    predict future values based on historical sequences.

    Args:
        input_size (int): Number of input features (e.g., OHLCV dimensions)
        hidden_size (int): Size of the hidden state in each RNN layer
        num_layers (int): Number of stacked RNN layers
        output_size (int): Number of output predictions (default: 1 for single stock price)
        dropout_rate (float): Dropout probability for regularization (default: 0.2)
    """

    def __init__(
        self,
        input_size: int,
        hidden_size: int,
        num_layers: int,
        output_size: int = 1,
        dropout_rate: float = 0.2,
    ):
        super().__init__()
        # Store model hyperparameters
        self.input_size = input_size
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.output_size = output_size
        self.dropout_rate = dropout_rate

        # Set device from config
        self.device = torch.device(config.device)

        # Create stack of RNN layers
        # First layer takes input_size, subsequent layers take hidden_size
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            layer_input_size = input_size if i == 0 else hidden_size
            self.layers.append(RNNLayer(layer_input_size, hidden_size))

        # Regularization and output layers
        self.dropout = nn.Dropout(dropout_rate)
        self.fc = nn.Linear(hidden_size, output_size)  # Final prediction layer

        # Initialize final layer weights
        nn.init.xavier_uniform_(self.fc.weight)
        nn.init.zeros_(self.fc.bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Forward pass through the multi-layer RNN.

        Processes the entire sequence and returns prediction based on final hidden state.

        Args:
            x (torch.Tensor): Input sequences of shape (batch_size, seq_len, input_size)

        Returns:
            torch.Tensor: Predictions of shape (batch_size, output_size)
        """
        batch_size, seq_len, _ = x.size()

        # Initialize hidden states for all layers to zeros
        h = [
            torch.zeros(batch_size, self.hidden_size, device=self.device)
            for _ in range(self.num_layers)
        ]

        # Process sequence step by step
        for t in range(seq_len):
            # Get input at current time step
            inp = x[:, t, :]

            # Pass through each RNN layer
            for layer_idx, rnn_layer in enumerate(self.layers):
                # Update hidden state for current layer
                h[layer_idx] = rnn_layer(inp, h[layer_idx])
                # Output of current layer becomes input to next layer
                inp = h[layer_idx]
                # Apply dropout between layers (not after final layer)
                if self.dropout_rate > 0 and layer_idx < self.num_layers - 1:
                    inp = self.dropout(inp)

        # Use final hidden state from last layer for prediction
        last_output = h[-1]

        # Apply final dropout and linear transformation
        out = self.dropout(last_output)
        out = self.fc(out)
        return out

    def predict(self, data_loader: DataLoader):
        """
        Generate predictions for a dataset.

        Sets model to evaluation mode and processes all batches in the data loader
        to generate predictions along with ground truth values.

        Args:
            data_loader (DataLoader): DataLoader containing test/validation data

        Returns:
            tuple: (predictions, actuals) as numpy arrays
                - predictions: Model predictions of shape (total_samples, output_size)
                - actuals: Ground truth values of shape (total_samples, output_size)
        """
        self.eval()  # Set model to evaluation mode
        predictions = []
        actuals = []

        with torch.no_grad():  # Disable gradient computation for efficiency
            for batch_x, batch_y in data_loader:
                # Move data to device
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Generate predictions
                outputs = self(batch_x)

                # Store results (move back to CPU for numpy conversion)
                predictions.append(outputs.cpu().numpy())
                actuals.append(batch_y.cpu().numpy())

        # Concatenate all batches into single arrays
        return np.concatenate(predictions), np.concatenate(actuals)

    def evaluate(self, data_loader: DataLoader, loss_func):
        """
        Evaluate model performance on a dataset.

        Computes average loss across all batches in the data loader.

        Args:
            data_loader (DataLoader): DataLoader containing evaluation data
            loss_func: Loss function to use for evaluation (e.g., MSELoss)

        Returns:
            dict: Dictionary containing the average loss
        """
        self.eval()  # Set model to evaluation mode
        total_loss = 0
        num_batches = 0

        with torch.no_grad():  # Disable gradient computation
            for batch_x, batch_y in data_loader:
                # Move data to device
                batch_x = batch_x.to(self.device)
                batch_y = batch_y.to(self.device)

                # Forward pass and loss computation
                outputs = self(batch_x)
                loss = loss_func(outputs, batch_y)

                # Accumulate loss
                total_loss += loss.item()
                num_batches += 1

        # Calculate average loss
        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        return {"loss": avg_loss}

    def train_epoch(self, train_loader: DataLoader, optimizer, loss_func):
        """
        Train the model for one epoch.

        Performs forward pass, backward pass, and parameter updates for all batches.
        Includes gradient clipping to prevent exploding gradients.

        Args:
            train_loader (DataLoader): DataLoader containing training data
            optimizer: Optimizer for parameter updates (e.g., Adam, SGD)
            loss_func: Loss function for training (e.g., MSELoss)

        Returns:
            dict: Dictionary containing the average training loss for the epoch
        """
        self.train()  # Set model to training mode
        total_loss = 0
        num_batches = 0

        for batch_x, batch_y in train_loader:
            # Move data to device
            batch_x = batch_x.to(self.device)
            batch_y = batch_y.to(self.device)

            # Reset gradients
            optimizer.zero_grad()

            # Forward pass
            outputs = self(batch_x)
            loss = loss_func(outputs, batch_y)

            # Backward pass
            loss.backward()

            # Clip gradients to prevent exploding gradient problem
            torch.nn.utils.clip_grad_norm_(self.parameters(), max_norm=1.0)

            # Update parameters
            optimizer.step()

            # Accumulate loss for averaging
            total_loss += loss.item()
            num_batches += 1

        # Calculate average loss for the epoch
        avg_loss = total_loss / num_batches if num_batches > 0 else float("inf")
        return {"train_loss": avg_loss}

    def save_model(self, filepath: str):
        """
        Save the model state and hyperparameters to a file.

        Saves both the trained weights and the model configuration needed
        to reconstruct the model architecture.

        Args:
            filepath (str): Path where to save the model file
        """
        model_state = {
            "model_state_dict": self.state_dict(),  # Trained weights and biases
            "input_size": self.input_size,  # Model architecture parameters
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout_rate": self.dropout_rate,
            "model_type": self.__class__.__name__,  # For verification during loading
        }
        torch.save(model_state, filepath)
        print(f"Model saved to {filepath}")

    @classmethod
    def load_model(cls, filepath: str, model_class):
        """
        Load a saved model from a file.

        Reconstructs the model architecture using saved hyperparameters and
        loads the trained weights.

        Args:
            filepath (str): Path to the saved model file
            model_class: The model class to instantiate (should be RNN)

        Returns:
            RNN: Loaded model with trained weights
        """
        # Load model state dictionary
        model_state = torch.load(filepath, map_location="cpu")

        # Reconstruct model with saved hyperparameters
        model = model_class(
            input_size=model_state["input_size"],
            hidden_size=model_state["hidden_size"],
            num_layers=model_state["num_layers"],
            output_size=model_state["output_size"],
            dropout_rate=model_state["dropout_rate"],
        )

        # Load trained weights
        model.load_state_dict(model_state["model_state_dict"])
        print(f"Model loaded from {filepath}")
        return model

    def get_model_info(self):
        """
        Get comprehensive information about the model.

        Returns a dictionary containing model architecture details and parameter counts.
        Useful for model comparison and debugging.

        Returns:
            dict: Dictionary containing model information including:
                - Architecture parameters (sizes, layers, etc.)
                - Parameter counts (total and trainable)
                - Device information
        """
        # Count total and trainable parameters
        total_params = sum(p.numel() for p in self.parameters())
        trainable_params = sum(p.numel() for p in self.parameters() if p.requires_grad)

        return {
            "model_type": self.__class__.__name__,
            "input_size": self.input_size,
            "hidden_size": self.hidden_size,
            "num_layers": self.num_layers,
            "output_size": self.output_size,
            "dropout_rate": self.dropout_rate,
            "total_parameters": total_params,
            "trainable_parameters": trainable_params,
            "device": str(self.device),
        }
