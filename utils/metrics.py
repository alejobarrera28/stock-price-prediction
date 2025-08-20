import numpy as np
import pandas as pd
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import torch


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

    @staticmethod
    def _calculate_sharpe_ratio(
        returns: np.ndarray, risk_free_rate: float = 0.02
    ) -> float:
        if np.std(returns) == 0:
            return 0.0
        return (
            (np.mean(returns) - risk_free_rate / 252) / np.std(returns) * np.sqrt(252)
        )

    @staticmethod
    def _calculate_max_drawdown(prices: np.ndarray) -> float:
        peak = np.maximum.accumulate(prices)
        drawdown = (prices - peak) / peak
        return np.min(drawdown) * 100

    def calculate_all_metrics(
        self, y_true: np.ndarray, y_pred: np.ndarray, include_financial: bool = True
    ):
        y_true = np.array(y_true).flatten()
        y_pred = np.array(y_pred).flatten()

        metrics = {
            "mse": self._calculate_mse(y_true, y_pred),
            "mae": self._calculate_mae(y_true, y_pred),
            "rmse": self._calculate_rmse(y_true, y_pred),
            "mape": self._calculate_mape(y_true, y_pred),
            "r2": self._calculate_r2(y_true, y_pred),
            "directional_accuracy": self._calculate_directional_accuracy(
                y_true, y_pred
            ),
        }

        if include_financial and len(y_true) > 1:
            returns_true = np.diff(y_true) / y_true[:-1]
            returns_pred = np.diff(y_pred) / y_pred[:-1]

            metrics.update(
                {
                    "sharpe_ratio_true": self._calculate_sharpe_ratio(returns_true),
                    "sharpe_ratio_pred": self._calculate_sharpe_ratio(returns_pred),
                    "max_drawdown_true": self._calculate_max_drawdown(y_true),
                    "max_drawdown_pred": self._calculate_max_drawdown(y_pred),
                }
            )

        return metrics

    def compare_models(self, results, metric: str = "rmse") -> pd.DataFrame:
        comparison_data = []

        for model_name, model_results in results.items():
            y_true = model_results["y_true"]
            y_pred = model_results["y_pred"]

            metrics = self.calculate_all_metrics(y_true, y_pred)
            metrics["model"] = model_name
            comparison_data.append(metrics)

        df = pd.DataFrame(comparison_data)
        return df.sort_values(
            by=metric,
            ascending=True if metric in ["mse", "mae", "rmse", "mape"] else False,
        )

    def log_metrics(self, model_name: str, epoch: int, metrics):
        log_entry = {
            "model": model_name,
            "epoch": epoch,
            "timestamp": pd.Timestamp.now(),
            **metrics,
        }
        self.metrics_history.append(log_entry)
