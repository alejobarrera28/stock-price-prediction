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
    Preprocesses stock data into sequences for training.

    Supports two processing modes:
    - Global scaling: One scaler for all stocks
    - Stock-aware: Individual scalers per stock and single stock batching (recommended)

    Mode controlled by config.use_stock_aware_processing
    """

    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.scaler = StandardScaler()
        self.target_scaler = StandardScaler()
        self.feature_columns = []

    def _create_sequences_global(self, data: pd.DataFrame, target_col: str = "close"):
        """
        Creates sequences using global scaling across all stocks.

        Args:
            data: Combined DataFrame containing all stock data
            target_col: Target column name

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
        """Processes a single stock's data with individual scaling."""
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

    def _create_sequences_stock_aware(
        self, data: pd.DataFrame, target_col: str = "close"
    ):
        """
        Creates sequences with individual scaling per stock.

        Args:
            data: Combined DataFrame containing all stock data
            target_col: Target column name

        Returns:
            tuple: (X_sequences, y_sequences, stock_indices)
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
        Prepares data for training with train/val/test splits.

        Processing method (stock-aware scaling and batching) controlled by config.use_stock_aware_processing.

        Args:
            data: Combined DataFrame containing all stock data
            test_size: Fraction for testing
            validation_size: Fraction for validation
            target_col: Target column name

        Returns:
            dict: Data dictionary with splits and metadata
        """
        if config.use_stock_aware_processing:
            print(
                "\nUsing stock-aware processing (stock-specific scaling + stock-aware batching)\n"
            )
            X, y, stock_indices = self._create_sequences_stock_aware(data, target_col)

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
            print("Using global scaling processing")
            X, y = self._create_sequences_global(data, target_col)

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

    def create_dataloaders(self, data_dict, batch_size: int = 32):
        """
        Creates DataLoaders based on processing method used.

        Args:
            data_dict: Data dictionary from prepare_data()
            batch_size: Batch size for training

        Returns:
            dict: DataLoaders for train/val/test splits
        """
        if "stock_train" in data_dict:
            print("Creating stock-aware dataloaders")

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
            print("Creating standard dataloaders")

            train_dataset = StockDataset(data_dict["X_train"], data_dict["y_train"])
            val_dataset = StockDataset(data_dict["X_val"], data_dict["y_val"])
            test_dataset = StockDataset(data_dict["X_test"], data_dict["y_test"])

            return {
                "train": DataLoader(train_dataset, batch_size=batch_size, shuffle=True),
                "val": DataLoader(val_dataset, batch_size=batch_size, shuffle=False),
                "test": DataLoader(test_dataset, batch_size=batch_size, shuffle=False),
            }

    # Utils

    def inverse_transform_target(
        self, scaled_target: np.ndarray, stock_indices: np.ndarray = None
    ) -> np.ndarray:
        """
        Inverse transforms scaled target values back to original scale.

        Args:
            scaled_target: Scaled target values
            stock_indices: Stock indices for stock-aware processing

        Returns:
            Original scale target values
        """
        if hasattr(self, "stock_target_scalers") and stock_indices is not None:
            # Use stock-specific scalers
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
            # Use global scaler
            if hasattr(self, "target_scaler"):
                return self.target_scaler.inverse_transform(
                    scaled_target.reshape(-1, 1)
                ).flatten()
            else:
                # Fallback: try to use the first stock's scaler if available
                if hasattr(self, "stock_target_scalers") and self.stock_target_scalers:
                    first_scaler = list(self.stock_target_scalers.values())[0]
                    return first_scaler.inverse_transform(
                        scaled_target.reshape(-1, 1)
                    ).flatten()
                else:
                    raise ValueError(
                        "No target scaler available for inverse transformation"
                    )
