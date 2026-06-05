import os
import pandas as pd
import matplotlib.pyplot as plt

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


os.makedirs("figures", exist_ok=True)

df = pd.read_csv("data/PJME_prepared.csv")
df["Datetime"] = pd.to_datetime(df["Datetime"])
df = df.sort_values("Datetime")

df["hour"] = df["Datetime"].dt.hour
df["dayofweek"] = df["Datetime"].dt.dayofweek
df["month"] = df["Datetime"].dt.month
df["lag_24"] = df["PJME_MW"].shift(24)
df["lag_168"] = df["PJME_MW"].shift(168)

df = df.dropna()

features = ["hour", "dayofweek", "month", "lag_24", "lag_168"]

X = df[features]
y = df["PJME_MW"]

split_idx = int(len(df) * 0.8)

X_train = X.iloc[:split_idx]
X_test = X.iloc[split_idx:]

y_train = y.iloc[:split_idx]
y_test = y.iloc[split_idx:]

model = XGBRegressor(
    n_estimators=100,
    max_depth=6,
    learning_rate=0.1,
    random_state=42
)

model.fit(X_train, y_train)

pred = model.predict(X_test)

mae = mean_absolute_error(y_test, pred)

print(f"MAE XGBoost = {mae:.2f}")

plt.figure(figsize=(15, 5))
plt.plot(y_test.values[:300], label="Actual")
plt.plot(pred[:300], label="XGBoost Forecast")
plt.title("Actual vs XGBoost Forecast")
plt.xlabel("Time")
plt.ylabel("PJME MW")
plt.legend()
plt.savefig("figures/xgboost_forecast_vs_actual.png", dpi=300, bbox_inches="tight")
plt.close()

errors = y_test - pred

plt.figure(figsize=(12, 5))
plt.hist(errors, bins=50)
plt.title("Distribution of XGBoost Forecast Errors")
plt.xlabel("Error")
plt.ylabel("Frequency")
plt.savefig("figures/xgboost_error_distribution.png", dpi=300, bbox_inches="tight")
plt.close()

print("Figures saved successfully")