# Stock Price Prediction Using Machine Learning

This project compares two approaches for forecasting stock prices: a recurrent neural network (LSTM) and a traditional linear regression model. The models are trained to predict **30-day log returns** for the SPY ETF using historical closing prices. The goal is to evaluate how each model performs on unseen data using backtest-style evaluation.

---

## Overview

- **Target**: 30-day log returns of SPY (S&P 500 ETF)
- **Models**:
  - LSTM using TensorFlow
  - Linear Regression using scikit-learn
- **Evaluation**: R² score and Mean Squared Error (MSE)
- **Output**: Predicted log returns, reconstructed price predictions, and visualizations

---

## Project Structure

```
stock-prediction-ml/
│
├── model_train.py         # Model training logic (LSTM + Linear)
├── run_pipeline.py        # Trains models and evaluates on validation set
├── run_test.py            # Evaluates trained models on test set
├── prepare_data.py        # Cleans and formats raw historical SPY data
├── split_data.py          # Train/val/test splitting using rolling window
├── StockDataAnalyzer.py   # Data scraping and analysis helper
│
├── models/                # Saved trained models (LSTM .h5, linear .pkl)
├── results/
│   ├── plots/             # All saved evaluation plots
│   ├── metrics.csv        # R² and MSE scores for each model and dataset
│   ├── lstm_predictions.csv     # LSTM test predictions
│   └── linear_predictions.csv   # Linear test predictions
│
├── requirements.txt       # Core dependencies
├── .gitignore             # Prevents committing models/results
└── README.md              # Project overview
```

---

## How to Use

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Run training & validation pipeline:
   ```bash
   python run_pipeline.py
   ```

3. Run on test set:
   ```bash
   python run_test.py
   ```

The `results/` folder will contain evaluation metrics (CSV) and plots for both models on both validation and test data.

---

## Sample Output Plots

**LSTM: Validation Price vs Prediction**

![LSTM Val](results/plots/lstm_validation_price_vs_predicted.png)

**Linear Regression: Validation Price vs Prediction**

![LR Val](results/plots/linear_validation_price_vs_predicted.png)

**LSTM: Residuals**

![LSTM Residuals](results/plots/lstm_validation_residuals.png)

---

## Key Observations

- **LSTM** captures non-linear price trends and performs better on volatile segments, but can underpredict large jumps.
- **Linear Regression** is simpler and more stable but fails to adapt to changing patterns in the data.
- Both models are evaluated using R² and MSE to measure prediction accuracy on log returns.

This project highlights the limitations of basic ML models in financial forecasting and serves as a starting point for more advanced time series work.

---

## License

This project is open-source under the MIT License.
