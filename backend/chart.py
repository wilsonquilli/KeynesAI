import joblib

try:
    from backend.stock import (
        add_features,
        add_pattern_features,
        build_pattern_model,
        download_data,
        get_pattern_predictors,
    )
except ModuleNotFoundError:
    from stock import (
        add_features,
        add_pattern_features,
        build_pattern_model,
        download_data,
        get_pattern_predictors,
    )


def label_patterns(df):
    labeled = df.copy()
    labeled["Pattern_Target"] = (labeled["Close"].shift(-5) > labeled["Close"]).astype(int)
    return labeled


def train_pattern_model(ticker):
    print(f"Training pattern model for {ticker}...")
    df = download_data(ticker)
    df = add_features(df)
    df = add_pattern_features(df)
    df = label_patterns(df)

    predictors = get_pattern_predictors()
    training_df = df.dropna(subset=predictors + ["Pattern_Target"]).copy()
    if training_df.empty or training_df["Pattern_Target"].nunique() < 2:
        raise RuntimeError(f"Not enough training data to build a pattern model for {ticker}.")

    model = build_pattern_model()
    model.fit(training_df[predictors], training_df["Pattern_Target"])
    joblib.dump(model, f"{ticker}_pattern_model.pkl")
    print(f"Pattern model saved as {ticker}_pattern_model.pkl")


if __name__ == "__main__":
    for ticker in ["^GSPC", "AAPL", "MSFT", "IBM"]:
        train_pattern_model(ticker)
