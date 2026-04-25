import os
import numpy as np
import pandas as pd
import requests
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score

try:
    from backend.boomCrash import BoomCrashModel
except ModuleNotFoundError:
    from boomCrash import BoomCrashModel


DEFAULT_HORIZONS = [2, 5, 10, 20, 60, 120, 250]
DATA_CACHE_DIR = os.path.dirname(__file__)
NASDAQ_BASE_URL = "https://api.nasdaq.com/api"
HISTORY_REQUEST_TIMEOUT = 8
HISTORY_HEADERS = {
    "User-Agent": (
        "Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) "
        "AppleWebKit/537.36 (KHTML, like Gecko) Chrome/123.0 Safari/537.36"
    ),
    "Accept": "application/json, text/plain, */*",
    "Origin": "https://www.nasdaq.com",
    "Referer": "https://www.nasdaq.com/",
}


def _is_plain_stock_ticker(ticker):
    return ticker.isalnum() and 1 <= len(ticker) <= 10


def _nasdaq_get(path, params=None):
    response = requests.get(
        f"{NASDAQ_BASE_URL}/{path}",
        params=params or {},
        headers=HISTORY_HEADERS,
        timeout=HISTORY_REQUEST_TIMEOUT,
    )
    response.raise_for_status()
    payload = response.json()
    status = payload.get("status") or {}
    if status.get("rCode") not in (None, 200):
        errors = status.get("bCodeMessage") or []
        error_text = "; ".join(item.get("errorMessage", "") for item in errors if item.get("errorMessage"))
        raise RuntimeError(error_text or f"Nasdaq API error for {path}.")
    return payload


def _parse_numeric_text(value):
    cleaned = str(value or "").replace("$", "").replace(",", "").strip()
    return float(cleaned) if cleaned else np.nan


def _download_nasdaq_history(ticker):
    start_date = (pd.Timestamp.utcnow() - pd.Timedelta(days=3650)).date().isoformat()
    payload = _nasdaq_get(
        f"quote/{ticker}/historical",
        {
            "assetclass": "stocks",
            "fromdate": start_date,
            "limit": "5000",
        },
    )
    rows = (payload.get("data") or {}).get("tradesTable", {}).get("rows") or []
    if len(rows) < 60:
        raise RuntimeError(f"Not enough Nasdaq history returned for {ticker}.")

    records = []
    for row in reversed(rows):
        records.append(
            {
                "Date": pd.to_datetime(row.get("date"), format="%m/%d/%Y", utc=True),
                "Open": _parse_numeric_text(row.get("open")),
                "High": _parse_numeric_text(row.get("high")),
                "Low": _parse_numeric_text(row.get("low")),
                "Close": _parse_numeric_text(row.get("close")),
                "Volume": _parse_numeric_text(row.get("volume")),
            }
        )

    df = pd.DataFrame.from_records(records).set_index("Date").sort_index()
    df["Volume"] = df["Volume"].fillna(0)
    return df.dropna(subset=["Open", "High", "Low", "Close"])


def download_data(ticker):
    filename = os.path.join(DATA_CACHE_DIR, f"{ticker}.csv")
    if os.path.exists(filename):
        df = pd.read_csv(filename, index_col=0, parse_dates=True)
    else:
        try:
            if _is_plain_stock_ticker(ticker):
                df = _download_nasdaq_history(ticker)
            else:
                raise RuntimeError("Ticker requires non-Nasdaq fallback.")
        except Exception as nasdaq_err:
            import yfinance as yf

            try:
                df = yf.Ticker(ticker).history(period="10y", auto_adjust=False)
            except Exception as err:
                raise RuntimeError(
                    f"Unable to download market history for {ticker}: Nasdaq error: {nasdaq_err}; Yahoo error: {err}"
                ) from err

        if df.empty:
            raise RuntimeError(f"Unable to download market history for {ticker}. No data returned.")
        df.to_csv(filename)

    df.index = pd.to_datetime(df.index, utc=True)
    df.drop(["Dividends", "Stock Splits"], axis=1, inplace=True, errors="ignore")
    return df


def _safe_series(series):
    return series.replace([np.inf, -np.inf], np.nan)


def _compute_rsi(series, period):
    delta = series.diff()
    gain = delta.clip(lower=0)
    loss = -delta.clip(upper=0)
    avg_gain = gain.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, np.nan)
    return 100 - (100 / (1 + rs))


def _compute_macd(series):
    ema_fast = series.ewm(span=12, adjust=False).mean()
    ema_slow = series.ewm(span=26, adjust=False).mean()
    macd = ema_fast - ema_slow
    signal = macd.ewm(span=9, adjust=False).mean()
    hist = macd - signal
    return macd, signal, hist


def _compute_atr(df, period=14):
    high_low = df["High"] - df["Low"]
    high_close = (df["High"] - df["Close"].shift(1)).abs()
    low_close = (df["Low"] - df["Close"].shift(1)).abs()
    true_range = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
    return true_range.ewm(alpha=1 / period, min_periods=period, adjust=False).mean()


def add_features(df):
    df = df.sort_index().copy()
    close = df["Close"]
    previous_close = close.shift(1)
    boom_crash_model = BoomCrashModel()
    phase_map = {"crash": -1.0, "neutral": 0.0, "boom": 1.0}

    df["Tomorrow"] = close.shift(-1)
    df["Target"] = (df["Tomorrow"] > close).astype(int)
    df["Pattern_Target"] = (close.shift(-5) > close).astype(int)
    df["Return_1"] = close.pct_change()
    df["Return_5"] = close.pct_change(5)
    df["Return_20"] = close.pct_change(20)
    df["Log_Return_1"] = np.log(close / previous_close)
    df["Gap"] = (df["Open"] - previous_close) / previous_close
    df["Intraday_Range"] = (df["High"] - df["Low"]) / close.replace(0, np.nan)
    df["Body_Size"] = (df["Close"] - df["Open"]) / df["Open"].replace(0, np.nan)
    df["Volume_Change"] = df["Volume"].pct_change()
    df["Cycle_Score"] = [
        phase_map[boom_crash_model.get_market_phase(date)] for date in df.index
    ]

    return df.loc["1990-01-01":].copy()


def add_pattern_features(df):
    df = df.copy()
    close = df["Close"]
    volume = df["Volume"].replace(0, np.nan)
    returns = close.pct_change()

    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    rolling_std_20 = returns.rolling(20).std()
    rolling_std_60 = returns.rolling(60).std()
    atr_14 = _compute_atr(df, 14)
    macd, macd_signal, macd_hist = _compute_macd(close)

    df["SMA_20"] = sma_20
    df["SMA_50"] = sma_50
    df["Volatility"] = rolling_std_20
    df["Momentum"] = close.pct_change(10)

    df["RSI_14"] = _compute_rsi(close, 14)
    df["RSI_28"] = _compute_rsi(close, 28)
    df["MACD"] = macd
    df["MACD_signal"] = macd_signal
    df["MACD_hist"] = macd_hist
    df["ATR_14"] = atr_14
    df["ATR_Ratio"] = atr_14 / close.replace(0, np.nan)
    df["SMA_20_Ratio"] = close / sma_20
    df["SMA_50_Ratio"] = close / sma_50
    df["SMA_200_Ratio"] = close / sma_200
    df["EMA_12_Ratio"] = close / ema_12
    df["EMA_26_Ratio"] = close / ema_26
    df["EMA_50_Ratio"] = close / ema_50
    df["Volatility_20"] = rolling_std_20
    df["Volatility_60"] = rolling_std_60
    df["Momentum_10"] = close.pct_change(10)
    df["Momentum_20"] = close.pct_change(20)
    df["Volume_Ratio_20"] = volume / volume.rolling(20).mean()
    df["Volume_Ratio_60"] = volume / volume.rolling(60).mean()
    df["Drawdown_20"] = close / close.rolling(20).max() - 1
    df["Trend_Strength_20"] = sma_20 / sma_50 - 1

    bollinger_std = close.rolling(20).std()
    lower_band = sma_20 - (2 * bollinger_std)
    upper_band = sma_20 + (2 * bollinger_std)
    band_width = (upper_band - lower_band).replace(0, np.nan)
    df["Bollinger_Position"] = (close - lower_band) / band_width
    df["Bollinger_Width"] = band_width / sma_20.replace(0, np.nan)

    return df


def label_patterns(df):
    labeled = df.copy()
    labeled["Pattern_Target"] = (labeled["Close"].shift(-5) > labeled["Close"]).astype(int)
    return labeled


def add_horizon_features(df, horizons):
    df = df.copy()
    returns = df["Close"].pct_change()

    for horizon in horizons:
        rolling_close = df["Close"].rolling(horizon)
        df[f"Close_Ratio_{horizon}"] = df["Close"] / rolling_close.mean()
        df[f"Trend_{horizon}"] = df["Target"].shift(1).rolling(horizon).mean()
        df[f"Pct_Return_{horizon}"] = df["Close"].pct_change(horizon)
        df[f"Rolling_Volatility_{horizon}"] = returns.rolling(horizon).std()

    df = df.replace([np.inf, -np.inf], np.nan)
    keep_columns = [column for column in df.columns if column != "Tomorrow"]
    return df.dropna(subset=keep_columns)


def prepare_model_frame(df, horizons=None):
    horizons = horizons or DEFAULT_HORIZONS
    prepared = add_features(df)
    prepared = add_pattern_features(prepared)
    prepared = add_horizon_features(prepared, horizons)
    return prepared


def get_pattern_predictors():
    return [
        "RSI_14",
        "RSI_28",
        "MACD",
        "MACD_signal",
        "MACD_hist",
        "ATR_Ratio",
        "SMA_20_Ratio",
        "SMA_50_Ratio",
        "SMA_200_Ratio",
        "EMA_12_Ratio",
        "EMA_26_Ratio",
        "EMA_50_Ratio",
        "Volatility_20",
        "Volatility_60",
        "Momentum_10",
        "Momentum_20",
        "Volume_Ratio_20",
        "Volume_Ratio_60",
        "Gap",
        "Intraday_Range",
        "Body_Size",
        "Drawdown_20",
        "Trend_Strength_20",
        "Bollinger_Position",
        "Bollinger_Width",
    ]


def get_main_predictors(horizons=None):
    horizons = horizons or DEFAULT_HORIZONS
    predictors = get_pattern_predictors() + [
        "Return_1",
        "Return_5",
        "Return_20",
        "Log_Return_1",
        "Volume_Change",
        "Cycle_Score",
    ]

    for horizon in horizons:
        predictors.extend(
            [
                f"Close_Ratio_{horizon}",
                f"Trend_{horizon}",
                f"Pct_Return_{horizon}",
                f"Rolling_Volatility_{horizon}",
            ]
        )

    return list(dict.fromkeys(predictors))


def build_pattern_model():
    return RandomForestClassifier(
        n_estimators=250,
        max_depth=8,
        min_samples_split=20,
        min_samples_leaf=5,
        class_weight="balanced_subsample",
        random_state=42,
        n_jobs=-1,
    )


def build_main_model():
    return RandomForestClassifier(
        n_estimators=400,
        max_depth=10,
        min_samples_split=25,
        min_samples_leaf=5,
        max_features="sqrt",
        class_weight="balanced_subsample",
        random_state=1,
        n_jobs=-1,
    )


def _fit_pattern_model(train):
    pattern_predictors = get_pattern_predictors()
    pattern_train = train.dropna(subset=pattern_predictors + ["Pattern_Target"]).copy()
    if pattern_train.empty or pattern_train["Pattern_Target"].nunique() < 2:
        return None

    pattern_model = build_pattern_model()
    pattern_model.fit(pattern_train[pattern_predictors], pattern_train["Pattern_Target"])
    return pattern_model


def _build_thresholds(test, pattern_probs, boom_crash_model):
    thresholds = []

    for date, pattern_probability, volatility in zip(
        test.index,
        pattern_probs,
        test["Volatility_20"].fillna(test["Volatility_20"].median()),
    ):
        threshold = 0.54

        if pattern_probability >= 0.65:
            threshold -= 0.06
        elif pattern_probability <= 0.35:
            threshold += 0.06

        phase = boom_crash_model.get_market_phase(date)
        if phase == "boom":
            threshold -= 0.03
        elif phase == "crash":
            threshold += 0.03

        if volatility > test["Volatility_20"].median():
            threshold += 0.02

        thresholds.append(max(0.40, min(0.65, threshold)))

    return np.array(thresholds)


def predict(train, test, predictors, model, thresholds):
    model.fit(train[predictors], train["Target"])
    probabilities = model.predict_proba(test[predictors])[:, 1]
    predictions = (probabilities >= np.array(thresholds)).astype(int)
    predictions = pd.Series(predictions, index=test.index, name="Predictions")
    probabilities = pd.Series(probabilities, index=test.index, name="Probability")
    return pd.concat([test["Target"], predictions, probabilities], axis=1)


def backtest(df, model, predictors, ticker=None, start=None, step=125):
    all_predictions = []
    boom_crash_model = BoomCrashModel()
    start = start or max(1000, int(df.shape[0] * 0.55))

    for i in range(start, df.shape[0], step):
        train = df.iloc[:i].copy()
        test = df.iloc[i : (i + step)].copy()
        if test.empty:
            continue

        pattern_model = _fit_pattern_model(train)
        if pattern_model is None:
            pattern_probs = np.full(test.shape[0], 0.5)
        else:
            pattern_probs = pattern_model.predict_proba(test[get_pattern_predictors()])[:, 1]

        thresholds = _build_thresholds(test, pattern_probs, boom_crash_model)
        predictions = predict(train, test, predictors, build_main_model(), thresholds)
        all_predictions.append(predictions)

    if not all_predictions:
        return None

    return pd.concat(all_predictions)


def generate_future_dates(last_date, periods=252):
    from pandas.tseries.offsets import BDay

    return pd.date_range(start=last_date + BDay(1), periods=periods, freq="B")


def _latest_indicator_values(simulation_df, signal_history, date, horizons):
    close = simulation_df["Close"]
    volume = simulation_df["Volume"].replace(0, np.nan)
    open_series = simulation_df["Open"]
    high = simulation_df["High"]
    low = simulation_df["Low"]

    latest = {
        "Return_1": close.pct_change().iloc[-1],
        "Return_5": close.pct_change(5).iloc[-1],
        "Return_20": close.pct_change(20).iloc[-1],
        "Log_Return_1": np.log(close.iloc[-1] / close.iloc[-2]) if len(close) > 1 else 0,
        "Gap": (open_series.iloc[-1] - close.iloc[-2]) / close.iloc[-2] if len(close) > 1 else 0,
        "Intraday_Range": (high.iloc[-1] - low.iloc[-1]) / close.iloc[-1] if close.iloc[-1] else 0,
        "Body_Size": (close.iloc[-1] - open_series.iloc[-1]) / open_series.iloc[-1] if open_series.iloc[-1] else 0,
        "Volume_Change": volume.pct_change().iloc[-1] if len(volume) > 1 else 0,
    }

    rsi_14 = _compute_rsi(close, 14)
    rsi_28 = _compute_rsi(close, 28)
    macd, macd_signal, macd_hist = _compute_macd(close)
    atr_14 = _compute_atr(simulation_df, 14)
    sma_20 = close.rolling(20).mean()
    sma_50 = close.rolling(50).mean()
    sma_200 = close.rolling(200).mean()
    ema_12 = close.ewm(span=12, adjust=False).mean()
    ema_26 = close.ewm(span=26, adjust=False).mean()
    ema_50 = close.ewm(span=50, adjust=False).mean()
    returns = close.pct_change()
    vol_20 = returns.rolling(20).std()
    vol_60 = returns.rolling(60).std()
    boll_std = close.rolling(20).std()
    boll_lower = sma_20 - (2 * boll_std)
    boll_upper = sma_20 + (2 * boll_std)

    latest.update(
        {
            "RSI_14": rsi_14.iloc[-1],
            "RSI_28": rsi_28.iloc[-1],
            "MACD": macd.iloc[-1],
            "MACD_signal": macd_signal.iloc[-1],
            "MACD_hist": macd_hist.iloc[-1],
            "ATR_Ratio": atr_14.iloc[-1] / close.iloc[-1] if close.iloc[-1] else 0,
            "SMA_20_Ratio": close.iloc[-1] / sma_20.iloc[-1],
            "SMA_50_Ratio": close.iloc[-1] / sma_50.iloc[-1],
            "SMA_200_Ratio": close.iloc[-1] / sma_200.iloc[-1],
            "EMA_12_Ratio": close.iloc[-1] / ema_12.iloc[-1],
            "EMA_26_Ratio": close.iloc[-1] / ema_26.iloc[-1],
            "EMA_50_Ratio": close.iloc[-1] / ema_50.iloc[-1],
            "Volatility_20": vol_20.iloc[-1],
            "Volatility_60": vol_60.iloc[-1],
            "Momentum_10": close.pct_change(10).iloc[-1],
            "Momentum_20": close.pct_change(20).iloc[-1],
            "Volume_Ratio_20": volume.iloc[-1] / volume.rolling(20).mean().iloc[-1],
            "Volume_Ratio_60": volume.iloc[-1] / volume.rolling(60).mean().iloc[-1],
            "Drawdown_20": close.iloc[-1] / close.rolling(20).max().iloc[-1] - 1,
            "Trend_Strength_20": sma_20.iloc[-1] / sma_50.iloc[-1] - 1,
            "Bollinger_Position": (close.iloc[-1] - boll_lower.iloc[-1]) / (boll_upper.iloc[-1] - boll_lower.iloc[-1]),
            "Bollinger_Width": (boll_upper.iloc[-1] - boll_lower.iloc[-1]) / sma_20.iloc[-1],
            "Cycle_Score": {"crash": -1.0, "neutral": 0.0, "boom": 1.0}[BoomCrashModel().get_market_phase(date)],
        }
    )

    for horizon in horizons:
        latest[f"Close_Ratio_{horizon}"] = close.iloc[-1] / close.rolling(horizon).mean().iloc[-1]
        latest[f"Trend_{horizon}"] = np.mean(signal_history[-horizon:])
        latest[f"Pct_Return_{horizon}"] = close.pct_change(horizon).iloc[-1]
        latest[f"Rolling_Volatility_{horizon}"] = returns.rolling(horizon).std().iloc[-1]

    latest_series = pd.Series(latest, dtype="float64")
    return _safe_series(latest_series).fillna(0.0)


def predict_future(df, model, predictors, ticker, periods=252, horizons=None):
    horizons = horizons or DEFAULT_HORIZONS
    boom_crash_model = BoomCrashModel()
    future_dates = generate_future_dates(df.index[-1], periods)
    pattern_model = _fit_pattern_model(df)

    simulation_df = df[["Open", "High", "Low", "Close", "Volume"]].copy()
    signal_history = df["Target"].fillna(0).astype(int).tolist()
    predicted_prices = []
    predicted_directions = []

    for date in future_dates:
        last_close = simulation_df["Close"].iloc[-1]
        recent_volume = simulation_df["Volume"].tail(20).mean()
        simulation_df.loc[date, "Open"] = last_close
        simulation_df.loc[date, "High"] = last_close * 1.004
        simulation_df.loc[date, "Low"] = last_close * 0.996
        simulation_df.loc[date, "Close"] = last_close
        simulation_df.loc[date, "Volume"] = recent_volume

        feature_row = _latest_indicator_values(simulation_df, signal_history, date, horizons)
        pattern_features = feature_row[get_pattern_predictors()].to_frame().T
        pattern_probability = (
            pattern_model.predict_proba(pattern_features)[:, 1][0]
            if pattern_model is not None
            else 0.5
        )
        threshold = _build_thresholds(
            pd.DataFrame([feature_row], index=[date]),
            np.array([pattern_probability]),
            boom_crash_model,
        )[0]

        main_probability = model.predict_proba(feature_row[predictors].to_frame().T)[:, 1][0]
        predicted_direction = int(main_probability >= threshold)
        predicted_directions.append(predicted_direction)
        signal_history.append(predicted_direction)

        price_change = 0.008 if predicted_direction == 1 else -0.005
        new_close = last_close * (1 + price_change)
        simulation_df.loc[date, "Close"] = new_close
        simulation_df.loc[date, "High"] = max(simulation_df.loc[date, "Open"], new_close) * 1.002
        simulation_df.loc[date, "Low"] = min(simulation_df.loc[date, "Open"], new_close) * 0.998
        predicted_prices.append(new_close)

    result = pd.DataFrame(
        {
            "Date": future_dates,
            "Predicted_Close": predicted_prices,
            "Predicted_Direction": predicted_directions,
        }
    ).set_index("Date")
    return result


def run_for_stock(ticker):
    print(f"\nProcessing {ticker}...")
    horizons = DEFAULT_HORIZONS
    df = prepare_model_frame(download_data(ticker), horizons)
    predictors = get_main_predictors(horizons)
    model = build_main_model()

    predictions = backtest(df, model, predictors, ticker)
    if predictions is None:
        print(f"Unable to backtest {ticker}.")
        return None

    accuracy = accuracy_score(predictions["Target"], predictions["Predictions"])
    precision = precision_score(predictions["Target"], predictions["Predictions"], zero_division=0)
    print(f"{ticker} Backtest Accuracy: {accuracy:.4f}")
    print(f"{ticker} Backtest Precision: {precision:.4f}")

    model.fit(df[predictors], df["Target"])
    future_predictions = predict_future(df, model, predictors, ticker, horizons=horizons)
    future_predictions.to_csv(f"{ticker}_future_predictions.csv")
    print(f"Saved 252-day predictions for {ticker} to {ticker}_future_predictions.csv")
    print(f"{ticker} 252-Day Prediction Summary:")
    print(f"Starting Price: {df['Close'].iloc[-1]:.2f}")
    print(f"Predicted Final Price: {future_predictions['Predicted_Close'].iloc[-1]:.2f}")
    change_pct = (future_predictions["Predicted_Close"].iloc[-1] / df["Close"].iloc[-1] - 1) * 100
    print(f"Predicted Change: {change_pct:.2f}%")
    up_days = future_predictions["Predicted_Direction"].sum()
    print(f"Predicted Up Days: {up_days}/{len(future_predictions)} ({up_days / len(future_predictions) * 100:.1f}%)")
    return future_predictions


if __name__ == "__main__":
    for ticker in ["^GSPC", "AAPL", "MSFT", "IBM"]:
        run_for_stock(ticker)
