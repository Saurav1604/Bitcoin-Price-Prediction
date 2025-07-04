import pandas as pd
import joblib
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor, StackingRegressor, VotingClassifier, RandomForestClassifier
from sklearn.linear_model import Ridge
from xgboost import XGBRegressor, XGBClassifier

# Load dataset
df = pd.read_csv("data/bitcoin_with_indicators.csv")

# Drop non-numeric columns if any
if 'date' in df.columns:
    df.drop(columns=['date'], inplace=True)

# Create targets t+1 to t+7 and trend
for i in range(1, 8):
    df[f'target_t+{i}'] = df['close'].shift(-i)
df['trend'] = (df['close'].shift(-1) > df['close']).astype(int)
df.dropna(inplace=True)

# Define features for trend classification
all_targets = [f'target_t+{i}' for i in range(1, 8)]
trend_features = df.select_dtypes(include='number').drop(columns=all_targets + ['trend']).columns.tolist()

X_trend = df[trend_features]
y_trend = df['trend']

# Train trend classifier
rf_cls = RandomForestClassifier(n_estimators=100, random_state=42)
xgb_cls = XGBClassifier(use_label_encoder=False, eval_metric='logloss')
voting_cls = VotingClassifier(estimators=[('rf', rf_cls), ('xgb', xgb_cls)], voting='soft')
voting_cls.fit(X_trend, y_trend)
joblib.dump(voting_cls, "models/btc_trend_classifier.pkl")

# Add predicted trend
df['predicted_trend'] = voting_cls.predict(X_trend)
final_features = trend_features + ['predicted_trend']
X = df[final_features]

# Save features used
with open("feature_columns.txt", "w") as f:
    for col in final_features:
        f.write(col + "\n")

# Train and save 7 stacked regressors
for i in range(1, 8):
    y = df[f'target_t+{i}']
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

    rf = RandomForestRegressor(n_estimators=100, random_state=42)
    xgb = XGBRegressor(n_estimators=100, random_state=42)
    stacked = StackingRegressor(estimators=[('rf', rf), ('xgb', xgb)], final_estimator=Ridge())

    print(f"\n🔁 Training model for t+{i}...")
    stacked.fit(X_train, y_train)
    joblib.dump(stacked, f"models/btc_stacked_regressor_t+{i}.pkl")
    print(f"✅ Saved: models/btc_stacked_regressor_t+{i}.pkl")