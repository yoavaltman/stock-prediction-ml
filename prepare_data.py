import pandas as pd
import numpy as np

def clean_data(df):
    """
    Prepares StockAnalyzer output for regression:
    - Drops early rows with rolling artifacts
    - Adds 'date' column from index
    - Fills missing values
    - Adds log price as regression target
    - Keeps only selected features and target
    """
    selected_features = ['Close',
        'trend5', 'trend10', 'trend50',
        'vol_10d', 'vol_50d',
        'rsi', 'macd_histogram',
        'atr',
        'sma_ratio_5_50', 'sma_ratio_5_20',
        'breadth_20',
        'drawdown_from_peak', 'vwap_dev',
        'trend_slope_50', 'ema_trend_slope',
        'is_high_vol', 'ema_50', 'obv', 'vol_skew', 'daily_return',
        'zscore_30', 'percentile_30', 'norm_price_30', 'cum_return'
    ]

    df = df.copy()

    # Drop first rows with NaNs from rolling windows
    df = df.iloc[65:].reset_index()
    df['date'] = df['Date'].dt.date
    df = df.drop(columns=['Date'])

    # Fill NA with forward/back fill then 0
    df = df.ffill().bfill().fillna(0)

    #target
    df['target_price'] = np.log(df['Close'].shift(-30)) - np.log(df['Close'])
    df = df.dropna(subset=['target_price'])

    # Keep final columns
    keep_cols = ['date', 'target_price'] + selected_features

   
    return df[[col for col in keep_cols if col in df.columns]]

