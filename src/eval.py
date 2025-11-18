import pandas as pd
import numpy as np
from sklearn.metrics import mean_squared_error, r2_score
import joblib

def evaluate_model(data_dir="data/processed", model_path="models/model.pkl"):

    # Load processed data
    df = pd.read_csv(f"{data_dir}/train_processed.csv")

    X = df.drop(columns=["SalePrice"])
    y = df["SalePrice"]

    # Load model
    model = joblib.load(model_path)

    # Predict
    y_pred = model.predict(X)

    rmse = np.sqrt(mean_squared_error(y, y_pred))
    r2 = r2_score(y, y_pred)

    print("=== EVALUATION ===")
    print("RMSE:", rmse)
    print("R2:", r2)


if __name__ == "__main__":
    evaluate_model()
