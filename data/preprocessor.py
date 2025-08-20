import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset, DataLoader
import config
from data.data_loader import StockAwareDataLoader


class StockDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray):
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]


class StockAwareDataset(StockDataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, stock_labels: list):
        super().__init__(sequences, targets)
        self.stock_labels = stock_labels

        # Group indices by stock
        self.stock_groups = {}
        for idx, stock in enumerate(stock_labels):
            if stock not in self.stock_groups:
                self.stock_groups[stock] = []
            self.stock_groups[stock].append(idx)

    def get_stock_batch(self, stock: str, batch_size: int):
        """Get a batch of sequences from a specific stock"""
        if stock not in self.stock_groups or len(self.stock_groups[stock]) == 0:
            raise ValueError(f"No sequences available for stock {stock}")

        indices = np.random.choice(
            self.stock_groups[stock],
            min(batch_size, len(self.stock_groups[stock])),
            replace=False,
        )
        return self.sequences[indices], self.targets[indices]


class StockPreprocessor:
    """
    Stock data preprocessing with support for both legacy and stock-aware processing.

    This class provides two processing approaches that can be controlled via config.py:

    1. LEGACY PROCESSING (use_stock_aware_processing = False):
       - Uses global StandardScaler across all stocks
       - Can cause artificial price/volume jumps between stocks in training batches
       - Uses standard PyTorch DataLoader with random batching

    2. STOCK-AWARE PROCESSING (use_stock_aware_processing = True) - RECOMMENDED:
       - Uses individual StandardScaler for each stock
       - Eliminates artificial scaling jumps between different stocks
       - Ensures batches contain sequences from only one stock (prevents temporal discontinuity)
       - Better training stability and performance

    Configuration:
        Set config.use_stock_aware_processing = True/False to control processing method
    """

    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_columns = []

    def create_sequences(self, data: pd.DataFrame, target_col: str = "close"):
        """
        LEGACY METHOD: Creates sequences with global scaling across all stocks.

        This method applies StandardScaler globally across all stocks, which can cause
        artificial price/volume jumps between different stocks in the training data.

        For better results, consider using create_sequences_by_symbol() instead,
        which applies stock-specific scaling.

        Args:
            data: Combined DataFrame containing all stock data
            target_col: Column name for target variable (default: "close")

        Returns:
            tuple: (X_sequences, y_sequences) as numpy arrays
        """
        # Drop unwanted columns if they exist
        if "capital gains" in data.columns:
            data = data.drop(columns=["capital gains"])

        data_clean = data.dropna().copy()

        feature_cols = [
            col
            for col in data_clean.columns
            if col not in ["symbol", target_col]
            and data_clean[col].dtype in ["float64", "int64"]
        ]

        self.feature_columns = feature_cols

        # Global scaling (legacy approach)
        X_scaled = self.scaler.fit_transform(data_clean[feature_cols])
        y_scaled = self.target_scaler.fit_transform(data_clean[[target_col]])

        X_sequences = []
        y_sequences = []

        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i - self.sequence_length : i])
            y_sequences.append(y_scaled[i])

        print(
            f"Created {len(X_sequences)} sequences with {len(feature_cols)} features using global scaling"
        )
        return np.array(X_sequences), np.array(y_sequences)

    def _create_sequences_single_stock(
        self, symbol_data: pd.DataFrame, target_col: str, symbol: str
    ):
        """Create sequences for a single stock with stock-specific scaling"""
        # Drop unwanted columns if they exist
        if "capital gains" in symbol_data.columns:
            symbol_data = symbol_data.drop(columns=["capital gains"])

        data_clean = symbol_data.dropna().copy()

        feature_cols = [
            col
            for col in data_clean.columns
            if col not in ["symbol", target_col]
            and data_clean[col].dtype in ["float64", "int64"]
        ]

        self.feature_columns = feature_cols

        # Create individual scalers for this stock
        stock_scaler = StandardScaler()
        stock_target_scaler = StandardScaler()

        # Scale this stock's data independently
        X_scaled = stock_scaler.fit_transform(data_clean[feature_cols])
        y_scaled = stock_target_scaler.fit_transform(data_clean[[target_col]])

        # Store scalers for later use
        self.stock_scalers[symbol] = stock_scaler
        self.stock_target_scalers[symbol] = stock_target_scaler

        X_sequences = []
        y_sequences = []

        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i - self.sequence_length : i])
            y_sequences.append(y_scaled[i])

        print(f"Created {len(X_sequences)} sequences with {len(feature_cols)} features")
        return np.array(X_sequences), np.array(y_sequences)

    def create_sequences_by_symbol(self, data: pd.DataFrame, target_col: str = "close"):
        """
        RECOMMENDED METHOD: Creates sequences with stock-specific scaling.

        This method applies StandardScaler individually to each stock, eliminating
        artificial price/volume jumps between different stocks. It also tracks which
        sequences belong to which stocks for stock-aware batching.

        Benefits over create_sequences():
        - No artificial scaling jumps between stocks (SPY $400 vs AAPL $150)
        - Enables stock-aware batching (prevents temporal discontinuity)
        - Better training stability and performance

        Args:
            data: Combined DataFrame containing all stock data
            target_col: Column name for target variable (default: "close")

        Returns:
            tuple: (X_sequences, y_sequences, stock_indices) where stock_indices
                   tracks which stock each sequence belongs to
        """
        all_sequences = []
        all_targets = []
        stock_indices = []  # Track which sequences belong to which stock
        self.stock_scalers = {}  # Store scalers for each stock
        self.stock_target_scalers = {}

        for symbol in data["symbol"].unique():
            symbol_data = data[data["symbol"] == symbol].copy()
            symbol_data = symbol_data.sort_index()

            print(f"Processing {symbol}: {len(symbol_data)} data points")

            if len(symbol_data) < self.sequence_length + 1:
                print(
                    f"Warning: Not enough data for {symbol} ({len(symbol_data)} points), skipping"
                )
                continue

            sequences, targets = self._create_sequences_single_stock(
                symbol_data, target_col, symbol
            )

            if len(sequences) > 0:
                all_sequences.append(sequences)
                all_targets.append(targets)
                # Track which sequences belong to this stock
                stock_indices.extend([symbol] * len(sequences))
                print(f"Created {len(sequences)} sequences for {symbol}\n")

        if all_sequences:
            return (
                np.concatenate(all_sequences),
                np.concatenate(all_targets),
                stock_indices,
            )
        else:
            print("Error: No sequences created from any symbol")
            return np.array([]), np.array([]), []

    def prepare_data(
        self,
        data: pd.DataFrame,
        test_size: float = 0.15,
        validation_size: float = 0.15,
        target_col: str = "close",
    ):
        """
        Prepare data for training using either stock-aware or legacy processing.

        The processing method is controlled by config.use_stock_aware_processing:
        - True: Uses stock-specific scaling and enables stock-aware batching
        - False: Uses global scaling with standard batching (legacy behavior)

        Args:
            data: Combined DataFrame containing all stock data
            test_size: Fraction of data for testing
            validation_size: Fraction of data for validation
            target_col: Column name for target variable

        Returns:
            dict: Data dictionary with train/val/test splits and metadata
        """
        if config.use_stock_aware_processing:
            print(
                "Using stock-aware processing (stock-specific scaling + stock-aware batching)"
            )
            X, y, stock_indices = self.create_sequences_by_symbol(data, target_col)

            if len(X) == 0:
                raise ValueError("No sequences created. Check your data.")

            train_size = 1 - test_size - validation_size

            X_temp, X_test, y_temp, y_test, stock_temp, stock_test = train_test_split(
                X,
                y,
                stock_indices,
                test_size=test_size,
                shuffle=False,
                random_state=config.random_seed,
            )

            val_size_adjusted = validation_size / (train_size + validation_size)
            X_train, X_val, y_train, y_val, stock_train, stock_val = train_test_split(
                X_temp,
                y_temp,
                stock_temp,
                test_size=val_size_adjusted,
                shuffle=False,
                random_state=config.random_seed,
            )

            print(
                f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
            )

            return {
                "X_train": X_train,
                "y_train": y_train,
                "stock_train": stock_train,
                "X_val": X_val,
                "y_val": y_val,
                "stock_val": stock_val,
                "X_test": X_test,
                "y_test": y_test,
                "stock_test": stock_test,
                "input_size": X_train.shape[2],
                "scaler": self.scaler,
                "target_scaler": self.target_scaler,
                "stock_scalers": self.stock_scalers,
                "stock_target_scalers": self.stock_target_scalers,
                "feature_columns": self.feature_columns,
            }
        else:
            print("Using legacy processing (global scaling + standard batching)")
            X, y = self.create_sequences(data, target_col)

            if len(X) == 0:
                raise ValueError("No sequences created. Check your data.")

            train_size = 1 - test_size - validation_size

            X_temp, X_test, y_temp, y_test = train_test_split(
                X,
                y,
                test_size=test_size,
                shuffle=False,
                random_state=config.random_seed,
            )

            val_size_adjusted = validation_size / (train_size + validation_size)
            X_train, X_val, y_train, y_val = train_test_split(
                X_temp,
                y_temp,
                test_size=val_size_adjusted,
                shuffle=False,
                random_state=config.random_seed,
            )

            print(
                f"Data split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
            )

            return {
                "X_train": X_train,
                "y_train": y_train,
                "X_val": X_val,
                "y_val": y_val,
                "X_test": X_test,
                "y_test": y_test,
                "input_size": X_train.shape[2],
                "scaler": self.scaler,
                "target_scaler": self.target_scaler,
                "feature_columns": self.feature_columns,
            }

    # Utils

    def create_dataloaders(self, data_dict, batch_size: int = 32):
        """
        Create appropriate data loaders based on the data processing method used.

        Automatically detects whether stock-aware or legacy processing was used
        by checking for stock metadata in the data dictionary.

        Args:
            data_dict: Data dictionary returned by prepare_data()
            batch_size: Batch size for training

        Returns:
            dict: DataLoaders for train/val/test splits
        """
        if "stock_train" in data_dict:
            # Stock-aware processing was used - create stock-aware datasets and loaders
            print(
                "Creating stock-aware dataloaders (batches contain sequences from same stock)"
            )

            train_dataset = StockAwareDataset(
                data_dict["X_train"], data_dict["y_train"], data_dict["stock_train"]
            )
            val_dataset = StockAwareDataset(
                data_dict["X_val"], data_dict["y_val"], data_dict["stock_val"]
            )
            test_dataset = StockAwareDataset(
                data_dict["X_test"], data_dict["y_test"], data_dict["stock_test"]
            )

            return {
                "train": StockAwareDataLoader(train_dataset, batch_size, shuffle=True),
                "val": StockAwareDataLoader(val_dataset, batch_size, shuffle=False),
                "test": StockAwareDataLoader(test_dataset, batch_size, shuffle=False),
            }
        else:
            # Legacy processing was used - create standard datasets and loaders
            print("Creating standard dataloaders (legacy batching)")

            train_dataset = StockDataset(data_dict["X_train"], data_dict["y_train"])
            val_dataset = StockDataset(data_dict["X_val"], data_dict["y_val"])
            test_dataset = StockDataset(data_dict["X_test"], data_dict["y_test"])

            return {
                "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
                "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
            }

    def inverse_transform_target(self, scaled_target: np.ndarray, stock_indices: np.ndarray = None) -> np.ndarray:
        """
        Inverse transform target values. Handles both legacy and stock-aware processing.
        
        Args:
            scaled_target: Scaled target values to inverse transform
            stock_indices: Stock indices for each prediction (for stock-aware processing)
        
        Returns:
            Original scale target values
        """
        if hasattr(self, 'stock_target_scalers') and stock_indices is not None:
            # Stock-aware processing: use stock-specific scalers
            result = np.zeros_like(scaled_target)
            
            # Convert to numpy array if it's a list
            if isinstance(stock_indices, list):
                stock_indices = np.array(stock_indices)
            
            for stock_symbol in np.unique(stock_indices):
                mask = stock_indices == stock_symbol
                
                if np.sum(mask) == 0:
                    continue
                    
                scaler = self.stock_target_scalers[stock_symbol]
                
                result[mask] = scaler.inverse_transform(
                    scaled_target[mask].reshape(-1, 1)
                )
            
            return result
        else:
            # Legacy processing: use global scaler
            if hasattr(self, 'target_scaler'):
                return self.target_scaler.inverse_transform(
                    scaled_target.reshape(-1, 1)
                ).flatten()
            else:
                # Fallback: try to use the first stock's scaler if available
                if hasattr(self, 'stock_target_scalers') and self.stock_target_scalers:
                    first_scaler = list(self.stock_target_scalers.values())[0]
                    return first_scaler.inverse_transform(
                        scaled_target.reshape(-1, 1)
                    ).flatten()
                else:
                    raise ValueError("No target scaler available for inverse transformation")
