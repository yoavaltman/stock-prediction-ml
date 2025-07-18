import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import joblib
import os

from StockDataAnalyzer import StockAnalyzer
from prepare_data import clean_data
from split_data import split_dataframe
from model_train import (
    train_lstm_model, train_linear_model,
    evaluate_model
)

from sklearn.preprocessing import StandardScaler


def save_metrics_row(model_name, dataset, mse, r2, filename="results/metrics.csv"):
    row = {
        "model": model_name,
        "dataset": dataset,
        "mse": mse,
        "r2_score": r2,
    
    }
    df = pd.DataFrame([row])
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    if os.path.exists(filename):
        df.to_csv(filename, mode='a', header=False, index=False)
    else:
        df.to_csv(filename, index=False)



def load_and_prepare_data():
    analyzer = StockAnalyzer('SPY', '1998-01-01', '2025-07-09', '1d')
    analyzer.analyze_stock()
    df = analyzer.get_stock_data()
    df_clean = clean_data(df)

    actual_date = df_clean['date']
    actual_close = df_clean['Close']
    base_close = df_clean['Close'].copy()
   

    df_train, df_val, df_test = split_dataframe(df_clean)
    feature_cols = df_clean.columns.difference(['date', 'target_price', 'Close'])

    scaler_X = StandardScaler()
    df_train[feature_cols] = scaler_X.fit_transform(df_train[feature_cols])
    df_val[feature_cols] = scaler_X.transform(df_val[feature_cols])
    df_test[feature_cols] = scaler_X.transform(df_test[feature_cols])

    X_train = df_train.drop(columns=['date', 'target_price', 'Close'])
    X_val = df_val.drop(columns=['date', 'target_price', 'Close'])
    y_train = df_train['target_price']
    y_val = df_val['target_price']

    base_train = base_close.iloc[:len(df_train)]
    base_val = base_close.iloc[len(df_train):len(df_train) + len(df_val)]

    return (X_train, y_train, base_train), (X_val, y_val, base_val), actual_date, actual_close, feature_cols, df_train, df_val, df_test

def train_and_evaluate_models(X_train, y_train, base_train, X_val, y_val, base_val, actual_date, actual_close):
    models_to_run = []

    print("\n Training LSTM...")
    lstm_result = train_lstm_model(X_train, y_train, X_val, y_val, base_train=base_train, base_val=base_val)
    models_to_run.append(("LSTM", lstm_result))
    lstm_result[0].save("models/lstm_model.h5")

    print("\n Training Linear Regression...")
    linear_result = train_linear_model(X_train, y_train, X_val, y_val, base_val=base_val)
    models_to_run.append(("Linear Regression", linear_result))
    joblib.dump(linear_result[0], "models/linear_model.pkl")

    for model_name, results in models_to_run:
        print(f"\n Evaluating {model_name}...")

        if model_name == "LSTM":
            model, _, window_size, val_preds, y_val_true = results
        else:
            model, val_preds, y_val_true = results
            window_size = 90  # manually assign for alignment

        mse, r2 = evaluate_model(val_preds, y_val_true, label=model_name)

        plot_model_results(actual_date, y_val_true, val_preds, model_name, len(X_train) + window_size)

        save_metrics_row(model_name, "validation", mse, r2)



def plot_model_results(actual_date, actual_prices, predicted_prices, label, start_idx):
    """
    Plots predicted vs actual prices on scatter and time series plots,
    assuming actual_prices and predicted_prices are both aligned future prices.
    """
    plot_predicted_vs_actual_prices(actual_prices, predicted_prices, label=label)
    plot_price_vs_predicted(actual_date, actual_prices, predicted_prices, start_idx, label=label)
    plot_residuals(actual_prices, predicted_prices, label=label)


def plot_price_vs_predicted(dates, actual_prices, predicted_prices, start_idx, label=""):
    aligned_dates = dates.iloc[start_idx: start_idx + len(predicted_prices)]
    aligned_actual = actual_prices[:len(predicted_prices)]  

    plt.figure(figsize=(12, 5))
    plt.plot(aligned_dates, aligned_actual, label="Actual Price")
    plt.plot(aligned_dates, predicted_prices, label="Predicted Price")
    plt.xlabel("Date")
    plt.ylabel("Price")
    plt.title(f"Actual vs Predicted Price ({label})")
    plt.legend()
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.show()


def plot_predicted_vs_actual_prices(actual_prices, predicted_prices, label=""):
    plt.figure(figsize=(8, 5))
    plt.scatter(actual_prices, predicted_prices, alpha=0.5)
    plt.plot([actual_prices.min(), actual_prices.max()],
             [actual_prices.min(), actual_prices.max()], 'r--', label='Ideal (y = x)')
    plt.xlabel("Actual Price")
    plt.ylabel("Predicted Price")
    plt.title(f"Predicted vs Actual Prices ({label})")
    plt.legend()
    plt.grid(True)
    plt.tight_layout()
    plt.show()

def plot_residuals(y_true, y_pred, label=""):
    residuals = y_pred - y_true
    plt.figure(figsize=(10, 4))
    plt.scatter(y_pred, residuals, alpha=0.5)
    plt.axhline(0, color='red', linestyle='--')
    plt.xlabel("Predicted Price")
    plt.ylabel("Residual")
    plt.title(f"Residuals vs Predicted Prices ({label})")
    plt.tight_layout()
    plt.show()


if __name__ == "__main__":
    (X_train, y_train, base_train), (X_val, y_val, base_val), actual_date, actual_close, _, _, _, _= load_and_prepare_data()

    train_and_evaluate_models(X_train, y_train, base_train, X_val, y_val, base_val, actual_date, actual_close)
