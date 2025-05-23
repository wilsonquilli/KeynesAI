import pandas as pd
import joblib
import os
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import precision_score
from static.boomCrash import BoomCrashModel


def download_data(ticker):
    filename = f"{ticker}.csv"
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        import yfinance as yf
        df = yf.Ticker(ticker).history(period="max")
        df.to_csv(filename)
    df.index = pd.to_datetime(df.index, utc=True)
    df.drop(["Dividends", "Stock Splits"], axis=1, inplace=True, errors='ignore')
    return df


def add_features(df):
    df["Tomorrow"] = df["Close"].shift(-1)
    df["Target"] = (df["Tomorrow"] > df["Close"]).astype(int)
    df = df.loc["1990-01-01":].copy()
    return df

def label_patterns(df):
    df["Target"] = (df["Close"].shift(-5) > df["Close"]).astype(int)
    return df

def add_horizon_features(df, horizons):
    for horizon in horizons:
        rolling_averages = df.rolling(horizon).mean()
        df[f"Close_Ratio_{horizon}"] = df["Close"] / rolling_averages["Close"]
        df[f"Trend_{horizon}"] = df.shift(1).rolling(horizon).sum()["Target"]
    df = df.dropna(subset=df.columns[df.columns != "Tomorrow"])
    return df


def add_pattern_features(df):
    df["SMA_20"] = df["Close"].rolling(20).mean()
    df["SMA_50"] = df["Close"].rolling(50).mean()
    df["Volatility"] = df["Close"].rolling(20).std()
    df["Momentum"] = df["Close"] - df["Close"].shift(10)
    return df.dropna()


def predict(train, test, predictors, model, thresholds):
    model.fit(train[predictors], train["Target"])
    probs = model.predict_proba(test[predictors])[:, 1]
    preds = (probs >= thresholds).astype(int)
    preds = pd.Series(preds, index=test.index, name="Predictions")
    combined = pd.concat([test["Target"], preds], axis=1)
    return combined


def backtest(df, model, predictors, ticker, start=2500, step=250):
    all_predictions = []

    try:
        pattern_model = joblib.load(f"{ticker}_pattern_model.pkl")
        boom_crash_model = joblib.load("boom_crash_model.pkl")
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        return None

    pattern_features = ["SMA_20", "SMA_50", "Volatility", "Momentum"]

    for i in range(start, df.shape[0], step):
        train = df.iloc[0:i].copy()
        test = df.iloc[i:(i + step)].copy()

        pattern_probs = pattern_model.predict_proba(test[pattern_features])[:, 1]
        thresholds = []

        for date, p in zip(test.index, pattern_probs):
            phase = boom_crash_model.get_market_phase(date)
            threshold = 0.6

            if p < 0.4:
                threshold += 0.1
            elif p > 0.6:
                threshold -= 0.1

            if phase == "boom":
                threshold -= 0.05
            elif phase == "crash":
                threshold += 0.05

            thresholds.append(max(0.35, min(0.65, threshold)))

        predictions = predict(train, test, predictors, model, thresholds)
        all_predictions.append(predictions)
    return pd.concat(all_predictions)


def generate_future_dates(last_date, periods=252):
    """Generate future trading dates"""
    from pandas.tseries.offsets import BDay
    return pd.date_range(start=last_date + BDay(1), periods=periods, freq='B')


def predict_future(df, model, predictors, ticker, periods=252):
    """Predict the next 252 trading days"""
    try:
        pattern_model = joblib.load(f"{ticker}_pattern_model.pkl")
        boom_crash_model = joblib.load("boom_crash_model.pkl")
    except FileNotFoundError as e:
        print(f"Model file not found: {e}")
        return None

    # Generate future dates
    last_date = df.index[-1]
    future_dates = generate_future_dates(last_date, periods)

    # Create future dataframe
    future_df = pd.DataFrame(index=future_dates)

    # Initialize with last known values
    last_row = df.iloc[-1].copy()

    predictions = []
    price_path = [last_row['Close']]

    for i, date in enumerate(future_dates):
        # Create a new row based on previous values
        new_row = last_row.copy()
        new_row.name = date

        # Update features that depend on previous prices
        new_row['Close'] = price_path[-1]  # Start with last price

        # Calculate pattern features
        if i >= 50:  # Need enough history for SMA_50
            window = price_path[-50:]
            new_row['SMA_20'] = np.mean(price_path[-20:])
            new_row['SMA_50'] = np.mean(window)
            new_row['Volatility'] = np.std(price_path[-20:])
            new_row['Momentum'] = price_path[-1] - price_path[-11]
        else:
            # For early days without enough history, use approximations
            new_row['SMA_20'] = np.mean(price_path[-min(20, len(price_path)):])
            new_row['SMA_50'] = np.mean(price_path)
            new_row['Volatility'] = np.std(price_path)
            new_row['Momentum'] = price_path[-1] - price_path[0] if len(price_path) > 10 else 0

        # Calculate horizon features
        for horizon in [2, 5, 60, 250, 1000]:
            if i >= horizon:
                rolling_avg = np.mean(price_path[-horizon:])
                new_row[f"Close_Ratio_{horizon}"] = price_path[-1] / rolling_avg
                # For trend, we'd need the actual target history which we don't have
                # So we'll use a simplified version
                new_row[f"Trend_{horizon}"] = np.sum(predictions[-horizon:]) if len(predictions) >= horizon else 0.5
            else:
                new_row[f"Close_Ratio_{horizon}"] = 1.0
                new_row[f"Trend_{horizon}"] = 0.5

        # Get pattern probability
        pattern_features = ["SMA_20", "SMA_50", "Volatility", "Momentum"]
        if i >= 50:  # Need enough history for pattern recognition
            pattern_probs = pattern_model.predict_proba(pd.DataFrame([new_row[pattern_features]]))[:, 1][0]
        else:
            pattern_probs = 0.5

        # Determine threshold
        phase = boom_crash_model.get_market_phase(date)
        threshold = 0.6

        if pattern_probs < 0.4:
            threshold += 0.1
        elif pattern_probs > 0.6:
            threshold -= 0.1

        if phase == "boom":
            threshold -= 0.05
        elif phase == "crash":
            threshold += 0.05

        threshold = max(0.35, min(0.65, threshold))

        # Make prediction
        prob = model.predict_proba(pd.DataFrame([new_row[predictors]]))[:, 1][0]
        prediction = 1 if prob >= threshold else 0
        predictions.append(prediction)

        # Update price based on prediction (simplified model)
        change = 0.005 if prediction == 1 else -0.003  # 0.5% up or 0.3% down
        new_price = price_path[-1] * (1 + change)
        price_path.append(new_price)

        # Update last_row for next iteration
        last_row = new_row.copy()
        last_row['Close'] = new_price

    # Create result dataframe
    result = pd.DataFrame({
        'Date': future_dates,
        'Predicted_Close': price_path[1:],  # Skip the initial price
        'Predicted_Direction': predictions
    }).set_index('Date')

    return result


def run_for_stock(ticker):
    print(f"\nProcessing {ticker}...")
    df = download_data(ticker)
    df = add_features(df)
    df = add_pattern_features(df)

    horizons = [2, 5, 60, 250, 1000]
    df = add_horizon_features(df, horizons)

    predictors = [f"Close_Ratio_{h}" for h in horizons] + [f"Trend_{h}" for h in horizons]

    model = RandomForestClassifier(n_estimators=150, min_samples_split=50, random_state=1)
    df = df.dropna()

    # Backtest for validation
    predictions = backtest(df, model, predictors, ticker)
    if predictions is not None:
        precision = precision_score(predictions["Target"], predictions["Predictions"])
        print(f"{ticker} Backtest Precision: {precision:.4f}")

        # Generate future predictions
        future_predictions = predict_future(df, model, predictors, ticker)
        if future_predictions is not None:
            # Save predictions to CSV
            future_predictions.to_csv(f"{ticker}_future_predictions.csv")
            print(f"Saved 252-day predictions for {ticker} to {ticker}_future_predictions.csv")

            # Print summary
            print(f"{ticker} 252-Day Prediction Summary:")
            print(f"Starting Price: {df['Close'].iloc[-1]:.2f}")
            print(f"Predicted Final Price: {future_predictions['Predicted_Close'].iloc[-1]:.2f}")
            change_pct = (future_predictions['Predicted_Close'].iloc[-1] / df['Close'].iloc[-1] - 1) * 100
            print(f"Predicted Change: {change_pct:.2f}%")
            up_days = future_predictions['Predicted_Direction'].sum()
            print(
                f"Predicted Up Days: {up_days}/{len(future_predictions)} ({up_days / len(future_predictions) * 100:.1f}%)")

            return future_predictions
    return None


if __name__ == "__main__":
    required_files = ["boom_crash_model.pkl"]
    tickers = ["^GSPC", "AAPL", "MSFT", "IBM"]
    required_files.extend([f"{ticker}_pattern_model.pkl" for ticker in tickers])

    missing_files = [f for f in required_files if not os.path.exists(f)]

    if missing_files:
        print("Missing model files. Please run:")
        if "boom_crash_model.pkl" in missing_files:
            print("- boomCrash.py to create the boom/crash model")
        if any("_pattern_model.pkl" in f for f in missing_files):
            print("- chart.py to create the pattern models")
    else:
        results = {}
        for ticker in tickers:
            results[ticker] = run_for_stock(ticker)
