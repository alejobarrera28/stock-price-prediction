import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import pandas as pd
import os
from tqdm import tqdm
import argparse
import matplotlib.pyplot as plt

import config
from data.data_loader import StockDataLoader
from data.preprocessor import StockPreprocessor
from models.rnn import RNN
from utils.metrics import StockPredictionMetrics
from utils.visualization import StockVisualization


class StockTrainer:
    def __init__(self, config=None):
        self.config = config
        self.device = torch.device(config.device)

        self.metrics = StockPredictionMetrics()
        self.visualizer = StockVisualization()

        torch.manual_seed(config.random_seed)
        np.random.seed(config.random_seed)

    def load_and_preprocess_data(self, force_reload: bool = False):
        print("\nLoading and preprocessing data...\n")

        data_loader = StockDataLoader(symbols=config.stocks, period=config.data_period)

        if force_reload or not os.path.exists(
            f"{config.data_save_path}/raw_stock_data.csv"
        ):
            data_loader.fetch_all_data()
            data_loader.save_data()

        raw_data = data_loader.load_data()
        print(f"Loaded data shape: {raw_data.shape}")

        preprocessor = StockPreprocessor(sequence_length=config.sequence_length)

        data_dict = preprocessor.prepare_data(
            raw_data, test_size=config.test_size, validation_size=config.validation_size
        )

        dataloaders = preprocessor.create_dataloaders(
            data_dict, batch_size=config.batch_size
        )

        return data_dict, dataloaders, preprocessor

    def create_model(self, input_size: int):
        model = RNN(
            input_size=input_size,
            hidden_size=config.hidden_size,
            num_layers=config.num_layers,
            dropout_rate=config.dropout_rate,
        )

        return model.to(self.device)

    def train_model(self, model, dataloaders, model_name: str):
        loss_func = nn.MSELoss()
        optimizer = optim.Adam(model.parameters(), lr=config.learning_rate)
        scheduler = optim.lr_scheduler.ReduceLROnPlateau(
            optimizer, patience=5, factor=0.5
        )

        train_history = {"train_loss": [], "val_loss": []}

        for epoch in tqdm(range(config.num_epochs), desc=f"Training {model_name}"):
            train_metrics = model.train_epoch(
                dataloaders["train"], optimizer, loss_func
            )
            val_metrics = model.evaluate(dataloaders["val"], loss_func)

            train_loss = train_metrics["train_loss"]
            val_loss = val_metrics["loss"]

            train_history["train_loss"].append(train_loss)
            train_history["val_loss"].append(val_loss)

            scheduler.step(val_loss)

            self.metrics.log_metrics(
                model_name,
                epoch,
                {
                    "train_loss": train_loss,
                    "val_loss": val_loss,
                    "learning_rate": optimizer.param_groups[0]["lr"],
                },
            )

            if (epoch + 1) % int(config.num_epochs/10) == 0:
                print(
                    f"{model_name} - Epoch {epoch+1}: "
                    f"Train Loss: {train_loss:.6f}, Val Loss: {val_loss:.6f}"
                )

        model.save_model(
            f"{config.model_save_path}/{model_name.lower().replace(' ', '_')}.pth"
        )

        return model, train_history

    def evaluate_model(
        self, model, dataloaders, model_name: str, preprocessor, data_dict
    ):
        # For evaluation, use the original test data directly to preserve order
        X_test = torch.FloatTensor(data_dict["X_test"]).to(self.device)
        y_test = data_dict["y_test"]
        stock_test = data_dict.get("stock_test", None)

        # Get predictions directly without DataLoader to preserve order
        model.eval()
        with torch.no_grad():
            test_predictions = model(X_test).cpu().numpy()

        test_actuals = y_test

        test_predictions_orig = preprocessor.inverse_transform_target(
            test_predictions, stock_test
        )
        test_actuals_orig = preprocessor.inverse_transform_target(
            test_actuals, stock_test
        )

        metrics = self.metrics.calculate_all_metrics(
            test_actuals_orig, test_predictions_orig
        )

        print(f"\n{model_name} Test Metrics:")
        for metric, value in metrics.items():
            print(f"  {metric}: {value:.4f}")

        return {
            "y_true": test_actuals_orig,
            "y_pred": test_predictions_orig,
            "metrics": metrics,
        }


def main():
    parser = argparse.ArgumentParser(description="Train stock prediction models")
    parser.add_argument(
        "--epochs", type=int, default=None, help="Number of training epochs"
    )
    parser.add_argument(
        "--force-reload", action="store_true", help="Force reload data from yfinance"
    )

    args = parser.parse_args()

    if args.epochs:
        config.num_epochs = args.epochs

    trainer = StockTrainer(config)

    data_dict, dataloaders, preprocessor = trainer.load_and_preprocess_data(
        args.force_reload
    )
    input_size = data_dict["input_size"]
    print(f"Input size: {input_size}")

    print(f"\n{'='*50}")
    print(f"Training Model")
    print(f"{'='*50}")

    model = trainer.create_model(input_size)
    model_name = "RNN"
    print(f"Model info: {model.get_model_info()}\n")

    trained_model, history = trainer.train_model(model, dataloaders, model_name)

    results = trainer.evaluate_model(
        trained_model, dataloaders, model_name, preprocessor, data_dict
    )

    trainer.visualizer.plot_training_history(
        history,
        model_name,
        save_path=f"plots/{model_name.replace(' ', '_')}_training_history.png",
    )

    trainer.visualizer.plot_predictions_vs_actual(
        results["y_true"][:200],
        results["y_pred"][:200],
        model_name,
        save_path=f"plots/{model_name.replace(' ', '_')}_predictions.png",
    )

    print(f"\nPlots saved to plots/ directory")


if __name__ == "__main__":
    main()
