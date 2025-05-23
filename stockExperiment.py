# optimized_stock_model.py
import yfinance as yf
import pandas as pd
import numpy as np
import joblib
import os
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import TimeSeriesSplit, GridSearchCV
from sklearn.metrics import precision_score

# Enhanced Data Downloading
def download_data(ticker):
    filename = f"{ticker}.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        df = yf.Ticker(ticker).history(period="max")
        df.to_csv(filename)
    df.index = pd.to_datetime(df.index, utc=True)
    df.drop(["Dividends", "Stock Splits"], axis=1, inplace=True, errors='ignore')
    return df

# Advanced Feature Engineering
def add_features(df):
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)

    df["RSI"] = compute_RSI(df["Close"], 14)
    df["MACD"], df["MACD_signal"] = compute_MACD(df["Close"])

    horizons = [5, 20, 50, 200]
    for h in horizons:
        df[f"Close_Ratio_{h}"] = df["Close"] / df["Close"].rolling(h).mean()
        df[f"Trend_{h}"] = df["Close"].diff(h)

    df["News_Sentiment"] = get_sentiment_data(df.index)
    return df.dropna()

# Helper functions for advanced indicators
def compute_RSI(series, period):
    delta = series.diff()
    gain = np.where(delta > 0, delta, 0)
    loss = np.where(delta < 0, -delta, 0)
    avg_gain = pd.Series(gain).rolling(window=period).mean()
    avg_loss = pd.Series(loss).rolling(window=period).mean()
    rs = avg_gain / avg_loss
    return 100 - (100 / (1 + rs))

def compute_MACD(series, slow=26, fast=12, signal=9):
    exp1 = series.ewm(span=fast, adjust=False).mean()
    exp2 = series.ewm(span=slow, adjust=False).mean()
    macd = exp1 - exp2
    signal_line = macd.ewm(span=signal, adjust=False).mean()
    return macd, signal_line

# Placeholder for sentiment fetching
def get_sentiment_data(dates):
    # Integrate sentiment analysis from news API or sentiment CSV
    # Placeholder returning neutral sentiment
    return pd.Series(0, index=dates)

# Optimized Model Training with Hyperparameter Tuning
def optimize_model(X, y):
    tscv = TimeSeriesSplit(n_splits=5)
    params = {
        'n_estimators': [100, 200],
        'max_depth': [10, 20],
        'min_samples_split': [10, 20]
    }
    grid_search = GridSearchCV(RandomForestClassifier(random_state=1), params, cv=tscv, scoring='precision')
    grid_search.fit(X, y)
    return grid_search.best_estimator_

# Predict and backtest
def backtest(df, predictors, start=1000, step=250):
    predictions = []
    for i in range(start, len(df), step):
        train, test = df.iloc[:i], df.iloc[i:i+step]
        model = optimize_model(train[predictors], train["Target"])
        preds = model.predict(test[predictors])
        predictions.append(pd.DataFrame({"Target": test["Target"], "Predictions": preds}, index=test.index))
    return pd.concat(predictions)

# Main function
def run_stock_prediction(ticker):
    print(f"Processing {ticker}")
    df = download_data(ticker)
    df = add_features(df)

    predictors = ["RSI", "MACD", "MACD_signal", "News_Sentiment"] + \
                 [f"Close_Ratio_{h}" for h in [5, 20, 50, 200]] + \
                 [f"Trend_{h}" for h in [5, 20, 50, 200]]

    if df.empty:
        print(f"No data after feature engineering for {ticker}. Skipping...")
        return None

    start_idx = max(int(len(df) * 0.4), 200)  # Ensures sufficient initial training data
    predictions = backtest(df, predictors, start=start_idx)
    
    if predictions.empty:
        print(f"Insufficient predictions generated for {ticker}.")
        return None

    precision = precision_score(predictions["Target"], predictions["Predictions"])
    print(f"{ticker} Precision: {precision:.4f}")
    return predictions


# Run for specified tickers
tickers = ["^GSPC", "AAPL", "MSFT", "IBM"]
results = {}
for ticker in tickers:
    results[ticker] = run_stock_prediction(ticker)