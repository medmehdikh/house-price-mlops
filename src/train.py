import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import joblib
import os

def train_model(data_dir="data/processed", output_dir="models"):

    # Charger les données préprocessed
    df = pd.read_csv(f"{data_dir}/train_processed.csv")

    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Entraîner
    model = LinearRegression()
    model.fit(X_train, y_train)

    # Metrics
    y_pred_test = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred_test))
    r2 = r2_score(y_test, y_pred_test)

    print("=== METRICS ===")
    print("RMSE:", rmse)
    print("R2:", r2)

    # Save model
    os.makedirs(output_dir, exist_ok=True)
    joblib.dump(model, f"{output_dir}/model.pkl")

    print(f"Model saved → {output_dir}/model.pkl")


if __name__ == "__main__":
    train_model()
