import yfinance as yf
import pandas as pd
import os
import joblib
from sklearn.ensemble import RandomForestClassifier

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

def add_pattern_features(df):
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Close"].rolling(20).std()
    df["Momentum"] = df["Close"] - df["Close"].shift(10)
    return df.dropna()

def label_patterns(df):
    df["Future_Close"] = df["Close"].shift(-5)
    df["Pattern_Label"] = (df["Future_Close"] > df["Close"]).astype(int)
    return df.dropna()

def train_pattern_model(ticker):
    print(f"Training pattern model for {ticker}....")
    df = download_data(ticker)
    df = add_pattern_features(df)
    df = label_patterns(df)

    features = ["SMA_20", "SMA_50", "Volatility", "Momentum"]
    X = df[features]
    y = df["Pattern_Label"]

    model = RandomForestClassifier(n_estimators=100, random_state=42)
    model.fit(X, y)
    joblib.dump(model, f"{ticker}_pattern_model.pkl")
    print(f"Pattern model saved as {ticker}_pattern_model.pkl")

if __name__ == "__main__":
    tickers = ["^GSPC", "AAPL", "MSFT", "IBM"]
    for ticker in tickers:
        train_pattern_model(ticker)