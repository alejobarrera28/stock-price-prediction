import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
import torch
from torch.utils.data import Dataset
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
        self, symbol_data: pd.DataFrame, target_col: str, symbol: str, 
        train_scaler: StandardScaler = None, train_target_scaler: StandardScaler = None
    ):
        """Processes a single stock's data with individual scaling."""
        data_clean = symbol_data.dropna().copy()
        close_data = data_clean[[target_col]].values
        
        # Use provided scalers (fitted on training data) or create new ones
        if train_scaler is not None:
            stock_scaler = train_scaler
            stock_target_scaler = train_target_scaler
            X_scaled = stock_scaler.transform(close_data)
            y_scaled = stock_target_scaler.transform(close_data)
        else:
            # This is training data - fit new scalers
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
        self, data: pd.DataFrame, target_col: str = "close", is_training: bool = True
    ):
        """Creates sequences with individual scaling per stock."""
        all_sequences = []
        all_targets = []
        all_dates = []
        stock_indices = []
        
        if is_training:
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

            # Use existing scalers for val/test data
            train_scaler = None if is_training else self.stock_scalers.get(symbol)
            train_target_scaler = None if is_training else self.stock_target_scalers.get(symbol)
            
            if not is_training and train_scaler is None:
                print(f"Warning: No training scaler found for {symbol} (likely not in training period), skipping")
                continue

            sequences, targets = self._create_sequences_single_stock(
                symbol_data, target_col, symbol, train_scaler, train_target_scaler
            )

            if len(sequences) > 0:
                # Extract dates from index for each sequence (date of the target point)
                dates = symbol_data.index[self.sequence_length:].values
                
                all_sequences.append(sequences)
                all_targets.append(targets)
                all_dates.extend(dates)
                stock_indices.extend([symbol] * len(sequences))

        if all_sequences:
            return (
                np.concatenate(all_sequences),
                np.concatenate(all_targets),
                all_dates,
                stock_indices,
            )
        else:
            print("Error: No sequences created from any symbol")
            return np.array([]), np.array([]), [], []

    def prepare_data(
        self,
        data: pd.DataFrame,
        test_size: float = 0.15,
        validation_size: float = 0.15,
        target_col: str = "close",
    ):
        """Prepares data for training with proper temporal splits by date and no data leakage."""
        print("\nUsing stock-aware processing with date-based splitting (no data leakage)\n")
        
        # First, split data by dates to prevent leakage
        dates = pd.to_datetime(data.index, utc=True).tz_convert(None)
        min_date = dates.min()
        max_date = dates.max()
        date_range = max_date - min_date
        
        train_cutoff = min_date + date_range * (1 - test_size - validation_size)
        val_cutoff = min_date + date_range * (1 - test_size)
        
        print(f"Date range: {min_date.date()} to {max_date.date()}")
        print(f"Train period: {min_date.date()} to {train_cutoff.date()}")
        print(f"Validation period: {train_cutoff.date()} to {val_cutoff.date()}")
        print(f"Test period: {val_cutoff.date()} to {max_date.date()}")
        
        # Split data by dates
        train_mask = dates < train_cutoff
        val_mask = (dates >= train_cutoff) & (dates < val_cutoff)
        test_mask = dates >= val_cutoff
        
        train_data = data[train_mask].copy()
        val_data = data[val_mask].copy()
        test_data = data[test_mask].copy()
        
        print(f"Raw data split - Train: {len(train_data)}, Val: {len(val_data)}, Test: {len(test_data)}")
        
        # Process training data and fit scalers
        print("\n=== Processing Training Data (fitting scalers) ===")
        X_train, y_train, train_dates, stock_train = self._create_sequences_stock_aware(
            train_data, target_col, is_training=True
        )
        
        # Process validation data using training scalers
        print("\n=== Processing Validation Data (using training scalers) ===")
        X_val, y_val, val_dates, stock_val = self._create_sequences_stock_aware(
            val_data, target_col, is_training=False
        )
        
        # Process test data using training scalers
        print("\n=== Processing Test Data (using training scalers) ===")
        X_test, y_test, test_dates, stock_test = self._create_sequences_stock_aware(
            test_data, target_col, is_training=False
        )
        
        if len(X_train) == 0:
            raise ValueError("No training sequences created. Check your data.")

        print(
            f"\nSequence split - Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}"
        )
        
        # Verify each split contains data from available stocks
        all_stocks = set(train_data['symbol'].unique()) | set(val_data['symbol'].unique()) | set(test_data['symbol'].unique())
        train_stocks = set(stock_train) if stock_train else set()
        val_stocks = set(stock_val) if stock_val else set() 
        test_stocks = set(stock_test) if stock_test else set()
        
        print(f"Stocks in training: {len(train_stocks)}/{len(all_stocks)} - {sorted(train_stocks)}")
        print(f"Stocks in validation: {len(val_stocks)}/{len(all_stocks)} - {sorted(val_stocks)}")
        print(f"Stocks in test: {len(test_stocks)}/{len(all_stocks)} - {sorted(test_stocks)}")
        
        # Validate that we're using only close price
        if len(X_train) > 0:
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
            "input_size": X_train.shape[2] if len(X_train) > 0 else 1,
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

            if stock_symbol not in self.stock_target_scalers:
                print(f"Warning: No scaler found for {stock_symbol} during inverse transform")
                result[mask] = scaled_target[mask]  # Return as-is if no scaler
            else:
                scaler = self.stock_target_scalers[stock_symbol]
                result[mask] = scaler.inverse_transform(
                    scaled_target[mask].reshape(-1, 1)
                )

        return result
