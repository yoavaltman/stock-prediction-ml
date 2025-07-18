import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping
import tensorflow.keras.backend as K


from sklearn.metrics import mean_squared_error, r2_score
from sklearn.linear_model import LinearRegression

# Set random seeds
np.random.seed(42)
tf.random.set_seed(42)

def make_sequences(X, y, window_size, base_prices=None):
    Xs, ys, indices, bases = [], [], [], []
    for i in range(len(X) - window_size - 1):
        Xs.append(X.iloc[i:(i + window_size)].values)
        ys.append(y.iloc[i + window_size])
        indices.append(X.index[i + window_size])
        if base_prices is not None:
            bases.append(base_prices.iloc[i + window_size])
    bases_out = np.array(bases) if base_prices is not None else None
    return np.array(Xs), np.array(ys), np.array(indices), bases_out



def train_lstm_model(X_train, y_train, X_val, y_val, base_train, base_val):
    window_size = 90
    units1 = 128
    units2 = 128
    learning_rate = 0.0001
    batch_size = 64
    epochs = 200

    # Create sequences
    X_seq_train, y_seq_train, _, base_seq_train = make_sequences(X_train, y_train, window_size, base_prices=base_train)
    X_seq_val, y_seq_val, _, base_seq_val = make_sequences(X_val, y_val, window_size, base_prices=base_val)

    if len(X_seq_train) == 0 or len(X_seq_val) == 0:
        raise ValueError("Not enough data to form sequences with current window size.")

    input_shape = X_seq_train.shape[1:]

    # Model
    model = Sequential([
        Input(shape=input_shape),
        LSTM(units1, return_sequences=True),
        Dropout(0.2),
        LSTM(units2),
        Dropout(0.2),
        Dense(1, activation='linear', kernel_initializer='he_uniform')
    ])

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss='mse',
        metrics=["mae"]
    )

    early_stop = EarlyStopping(monitor="val_loss", patience=50, restore_best_weights=True)

    history = model.fit(
        X_seq_train, y_seq_train,
        validation_data=(X_seq_val, y_seq_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=[early_stop],
        verbose=0
    )

    # Predict log returns
    pred_log_returns = model.predict(X_seq_val, verbose=0).flatten()
    actual_log_returns = y_seq_val

    # Reconstruct future prices from current price
    predicted_prices = base_seq_val * np.exp(pred_log_returns)
    actual_prices = base_seq_val * np.exp(actual_log_returns)

    return model, history, window_size, predicted_prices, actual_prices

def train_linear_model(X_train, y_train, X_val, y_val, base_val):
    model = LinearRegression()
    model.fit(X_train, y_train)

    pred_log_returns = model.predict(X_val)
    actual_log_returns = y_val.values

    predicted_prices = base_val.values * np.exp(pred_log_returns)
    actual_prices = base_val.values * np.exp(actual_log_returns)

    return model, predicted_prices, actual_prices

def evaluate_model(price_preds, price_actual, label=""):
    mse = mean_squared_error(price_actual, price_preds)
    r2 = r2_score(price_actual, price_preds)
    print(f"{label} Price MSE: {mse:.6f}")
    print(f"{label} Price R²: {r2:.4f}")
    return mse, r2
