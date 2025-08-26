import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import torch
from torch.utils.data import Dataset
import config
from data.data_loader import StockAwareDataLoader


class StockAwareDataset(Dataset):
    def __init__(self, sequences: np.ndarray, targets: np.ndarray, stock_labels: list):
        super().__init__()
        self.sequences = torch.FloatTensor(sequences)
        self.targets = torch.FloatTensor(targets)
        self.stock_labels = stock_labels

        # Group indices by stock
        self.stock_groups = {}
        for idx, stock in enumerate(stock_labels):
            if stock not in self.stock_groups:
                self.stock_groups[stock] = []
            self.stock_groups[stock].append(idx)

    def __len__(self):
        return len(self.sequences)

    def __getitem__(self, idx):
        return self.sequences[idx], self.targets[idx]

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
    """Preprocesses stock data into sequences for training."""

    def __init__(self, sequence_length: int = 60):
        self.sequence_length = sequence_length
        self.feature_columns = ["close"]


    def _create_sequences_single_stock(
        self, symbol_data: pd.DataFrame, target_col: str, symbol: str
    ):
        """Processes a single stock's data with individual scaling."""
        data_clean = symbol_data.dropna().copy()
        close_data = data_clean[[target_col]].values
        
        stock_scaler = StandardScaler()
        stock_target_scaler = StandardScaler()
        
        X_scaled = stock_scaler.fit_transform(close_data)
        y_scaled = stock_target_scaler.fit_transform(close_data)
        
        self.stock_scalers[symbol] = stock_scaler
        self.stock_target_scalers[symbol] = stock_target_scaler
        
        X_sequences = []
        y_sequences = []
        
        for i in range(self.sequence_length, len(X_scaled)):
            X_sequences.append(X_scaled[i - self.sequence_length : i])
            y_sequences.append(y_scaled[i])
        
        print(f"Created {len(X_sequences)} sequences for {symbol}")
        return np.array(X_sequences), np.array(y_sequences)

    def _create_sequences_stock_aware(
        self, data: pd.DataFrame, target_col: str = "close"
    ):
        """Creates sequences with individual scaling per stock."""
        all_sequences = []
        all_targets = []
        stock_indices = []
        self.stock_scalers = {}
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
                stock_indices.extend([symbol] * len(sequences))

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
        """Prepares data for training with train/val/test splits."""
        print("\nUsing stock-aware processing\n")
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

        # Validate that we're using only close price
        assert X_train.shape[2] == 1, f"Expected input_size=1 (close price only), got {X_train.shape[2]}"
        assert len(self.feature_columns) == 1, f"Expected 1 feature column, got {len(self.feature_columns)}"
        assert self.feature_columns[0] == "close", f"Expected 'close' feature, got {self.feature_columns[0]}"

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
            "stock_scalers": self.stock_scalers,
            "stock_target_scalers": self.stock_target_scalers,
            "feature_columns": self.feature_columns,
        }

    def create_dataloaders(self, data_dict, batch_size: int = 32):
        """Creates stock-aware DataLoaders for train/val/test splits."""
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

    # Utils

    def inverse_transform_target(
        self, scaled_target: np.ndarray, stock_indices: np.ndarray = None
    ) -> np.ndarray:
        """Inverse transforms scaled target values back to original scale."""
        result = np.zeros_like(scaled_target)

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
