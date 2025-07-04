import yfinance as yf
import pandas as pd
import joblib
import ta
import os
import matplotlib.pyplot as plt


def predict_btc_price(model_folder="models", feature_file="feature_columns.txt", trend_model_file="models/btc_trend_classifier.pkl"):
    # Step 1: Download BTC data
    btc_raw = yf.download('BTC-USD', period='60d', interval='1d', auto_adjust=False)

    # Step 2: Flatten and clean
    btc = pd.DataFrame(index=btc_raw.index)
    for col in ['Open', 'High', 'Low', 'Close', 'Adj Close', 'Volume']:
        if col in btc_raw.columns:
            btc[col.lower()] = btc_raw[col].squeeze()
        else:
            raise ValueError(f"Missing column: {col}")
    btc.dropna(inplace=True)

    # Step 3: Add technical indicators
    btc = ta.add_all_ta_features(btc, open='open', high='high', low='low', close='close', volume='volume', fillna=True)
    latest = btc.tail(1).copy()

    # Step 4: Load feature columns
    if not os.path.exists(feature_file):
        raise FileNotFoundError("feature_columns.txt not found")

    with open(feature_file, "r") as f:
        expected_columns = [line.strip() for line in f.readlines()]

    # Step 5: Predict trend
    if not os.path.exists(trend_model_file):
        raise FileNotFoundError("Trend classifier model not found")

    trend_model = joblib.load(trend_model_file)
    X_trend = latest[[col for col in expected_columns if col != 'predicted_trend']]
    predicted_trend = trend_model.predict(X_trend)[0]
    latest['predicted_trend'] = predicted_trend

    X_live = latest[expected_columns]
    current_price = latest['close'].values[0]
    forecasts = []

    for i in range(1, 8):
        model_path = os.path.join(model_folder, f"btc_stacked_regressor_t+{i}.pkl")
        if not os.path.exists(model_path):
            forecasts.append({"day": f"t+{i}", "error": "Model not found"})
            continue

        model = joblib.load(model_path)
        predicted_price = model.predict(X_live)[0]
        percent_change = ((predicted_price - current_price) / current_price) * 100
        direction = "📈 UP" if predicted_price > current_price else "📉 DOWN or FLAT"

        forecasts.append({
            "day": f"t+{i}",
            "predicted_price": round(predicted_price, 2),
            "percent_change": round(percent_change, 2),
            "direction": direction
        })

    return {
        "current_price": round(current_price, 2),
        "forecast": forecasts
    }


# Run the forecast and print
if __name__ == '__main__':
    result = predict_btc_price()
    print("\n📊 7-Day Bitcoin Forecast:")
    for day in result["forecast"]:
        if "error" in day:
            print(f" {day['day']}: ❌ {day['error']}")
        else:
            print(f" {day['day']}: ${day['predicted_price']} ({day['percent_change']}%) → {day['direction']}")
