import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score


class StockPredictionMetrics:
    def __init__(self):
        self.metrics_history = []

    @staticmethod
    def _calculate_mse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_squared_error(y_true, y_pred)

    @staticmethod
    def _calculate_mae(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return mean_absolute_error(y_true, y_pred)

    @staticmethod
    def _calculate_rmse(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return np.sqrt(mean_squared_error(y_true, y_pred))

    @staticmethod
    def _calculate_mape(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        y_true, y_pred = np.array(y_true), np.array(y_pred)
        mask = y_true != 0
        return np.mean(np.abs((y_true[mask] - y_pred[mask]) / y_true[mask])) * 100

    @staticmethod
    def _calculate_r2(y_true: np.ndarray, y_pred: np.ndarray) -> float:
        return r2_score(y_true, y_pred)

    @staticmethod
    def _calculate_directional_accuracy(
        y_true: np.ndarray, y_pred: np.ndarray
    ) -> float:
        if len(y_true) < 2:
            return 0.0

        true_direction = np.diff(y_true) > 0
        pred_direction = np.diff(y_pred) > 0

        return np.mean(true_direction == pred_direction) * 100

    def calculate_all_metrics(self, y_true: np.ndarray, y_pred: np.ndarray):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        metrics = {
            "mae": self._calculate_mae(y_true, y_pred),
            "rmse": self._calculate_rmse(y_true, y_pred),
            "mape": self._calculate_mape(y_true, y_pred),
            "r2": self._calculate_r2(y_true, y_pred),
            "directional_accuracy": self._calculate_directional_accuracy(
                y_true, y_pred
            ),
        }

        return metrics

    def log_metrics(self, model_name: str, epoch: int, metrics):
        log_entry = {
            "model": model_name,
            "epoch": epoch,
            "timestamp": pd.Timestamp.now(),
            **metrics,
        }
        self.metrics_history.append(log_entry)
