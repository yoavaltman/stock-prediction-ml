import numpy as np
import pandas as pd
import joblib
import os
from tensorflow.keras.models import load_model
from tensorflow.keras.metrics import MeanSquaredError

from run_pipeline import (
    load_and_prepare_data,
    plot_model_results, save_metrics_row
)

from model_train import (
    make_sequences,
    evaluate_model
)


def save_predictions_csv(dates, actual, predicted, model_label, filename):
    df = pd.DataFrame({
        'date': dates.values,
        'actual_price': actual,
        'predicted_price': predicted
    })
    os.makedirs(os.path.dirname(filename), exist_ok=True)
    df.to_csv(filename, index=False)
    print(f"✅ Saved {model_label} predictions to {filename}")


if __name__ == "__main__":
    np.random.seed(42)
    window_size = 90

    os.makedirs("results/plots", exist_ok=True)

    # Load data and scalers
    (X_train, y_train, base_train), (X_val, y_val, base_val), actual_date, _, _, df_train, df_val, df_test = load_and_prepare_data()

    # Prepare test set
    X_test = df_test.drop(columns=['date', 'target_price', 'Close'])
    y_test = df_test['target_price']
    base_test = df_test['Close']  # Used to reconstruct actual prices

    # === LSTM Evaluation ===
    print("\nEvaluating LSTM on Test Set...")
    lstm_model = load_model("models/lstm_model.h5", custom_objects={"mse": MeanSquaredError()})

    X_seq_test, y_seq_test, _, base_seq_test = make_sequences(X_test, y_test, window_size, base_prices=base_test)
    lstm_preds_log = lstm_model.predict(X_seq_test, verbose=0).flatten()
    lstm_preds = base_seq_test * np.exp(lstm_preds_log)
    lstm_actual = base_seq_test * np.exp(y_seq_test)

    mse, r2 = evaluate_model(lstm_preds, lstm_actual, label="LSTM (Test)")
    save_metrics_row("LSTM", "test", mse, r2)

    start_idx = len(X_train) + len(X_val) + window_size
    plot_model_results(actual_date, lstm_actual, lstm_preds, label="LSTM (Test)", start_idx=start_idx)

    lstm_dates = actual_date[start_idx: start_idx + len(lstm_preds)]
    save_predictions_csv(lstm_dates, lstm_actual, lstm_preds, "LSTM", "results/lstm_predictions.csv")


    # === Linear Regression Evaluation ===
    print("\nEvaluating Linear Regression on Test Set...")
    linear_model = joblib.load("models/linear_model.pkl")

    linear_preds_log = linear_model.predict(X_test)
    linear_preds = base_test.values * np.exp(linear_preds_log)
    linear_actual = base_test.values * np.exp(y_test.values)

    mse, r2 = evaluate_model(linear_preds, linear_actual, label="Linear (Test)")
    save_metrics_row("Linear Regression", "test", mse, r2)

    start_idx = len(X_train) + len(X_val)
    plot_model_results(actual_date, linear_actual, linear_preds, label="Linear (Test)", start_idx=start_idx)

    linear_dates = actual_date[start_idx: start_idx + len(linear_preds)]
    save_predictions_csv(linear_dates, linear_actual, linear_preds, "Linear Regression", "results/linear_predictions.csv")
