#TODO
#Analyze directional accuracy -> what does it mean on 30 day log predictions
#create a metrics csv
#find a better graph of test lstm using same model



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



if __name__ == "__main__":
    np.random.seed(42)
    window_size = 90

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
    plot_model_results(actual_date, lstm_actual, lstm_preds, "LSTM (Test)", start_idx)

    # Save LSTM predictions to CSV
    lstm_dates = actual_date[start_idx: start_idx + len(lstm_preds)]
    lstm_df = pd.DataFrame({
        'date': lstm_dates.values,
        'actual_price': lstm_actual,
        'predicted_price': lstm_preds
    })
    lstm_df.to_csv("lstm_predictions.csv", index=False)

    print("✅ Saved LSTM predictions to results/lstm_predictions.csv")


    # === Linear Regression Evaluation ===
    print("\nEvaluating Linear Regression on Test Set...")
    linear_model = joblib.load("models/linear_model.pkl")

    linear_preds_log = linear_model.predict(X_test)
    linear_preds = base_test.values * np.exp(linear_preds_log)
    linear_actual = base_test.values * np.exp(y_test.values)

    mse, r2 = evaluate_model(linear_preds, linear_actual, label="Linear (Test)")
    save_metrics_row("Linear Regression", "test", mse, r2)


    start_idx = len(X_train) + len(X_val)
    plot_model_results(actual_date, linear_actual, linear_preds, "Linear (Test)", start_idx)

    # Save Linear predictions to CSV
    linear_dates = actual_date[start_idx: start_idx + len(linear_preds)]
    linear_df = pd.DataFrame({
        'date': linear_dates.values,
        'actual_price': linear_actual,
        'predicted_price': linear_preds
    })
    linear_df.to_csv("linear_predictions.csv", index=False)
    print("✅ Saved Linear predictions to results/linear_predictions.csv")


