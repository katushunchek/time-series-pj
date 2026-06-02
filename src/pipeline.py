import pandas as pd

from xgboost import XGBRegressor
from sklearn.metrics import mean_absolute_error


def load_data(path: str) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["Datetime"] = pd.to_datetime(df["Datetime"])
    df = df.sort_values("Datetime")
    return df


def create_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()

    df["hour"] = df["Datetime"].dt.hour
    df["dayofweek"] = df["Datetime"].dt.dayofweek
    df["month"] = df["Datetime"].dt.month

    df["lag_24"] = df["PJME_MW"].shift(24)
    df["lag_168"] = df["PJME_MW"].shift(168)

    df = df.dropna()

    return df


def train_test_split_time(df: pd.DataFrame, test_size: float = 0.2):
    split_idx = int(len(df) * (1 - test_size))

    features = ["hour", "dayofweek", "month", "lag_24", "lag_168"]

    X = df[features]
    y = df["PJME_MW"]

    X_train = X.iloc[:split_idx]
    X_test = X.iloc[split_idx:]

    y_train = y.iloc[:split_idx]
    y_test = y.iloc[split_idx:]

    return X_train, X_test, y_train, y_test


def train_model(X_train, y_train):
    model = XGBRegressor(
        n_estimators=100,
        max_depth=6,
        learning_rate=0.1,
        random_state=42
    )

    model.fit(X_train, y_train)

    return model


def evaluate_model(model, X_test, y_test) -> float:
    predictions = model.predict(X_test)
    mae = mean_absolute_error(y_test, predictions)

    return mae


def run_pipeline(data_path: str):
    df = load_data(data_path)
    df_features = create_features(df)

    X_train, X_test, y_train, y_test = train_test_split_time(df_features)

    model = train_model(X_train, y_train)

    mae = evaluate_model(model, X_test, y_test)

    return model, mae


if __name__ == "__main__":
    model, mae = run_pipeline("data/PJME_prepared.csv")
    print(f"MAE XGBoost pipeline: {mae:.2f}")
