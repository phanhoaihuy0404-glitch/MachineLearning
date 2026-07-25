

import json
import numpy as np
import joblib

from pathlib import Path
from sklearn.datasets import fetch_california_housing
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score


ARTIFACT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def main() -> None:
    # 1) Load dữ liệu California Housing
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()

    # target: MedHouseValue
    target_col = 'MedHouseVal'

    # 2) Làm sạch dữ liệu (an toàn)
    # Dataset California Housing thường không có missing, nhưng xử lý để đúng yêu cầu.
    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(axis=0).reset_index(drop=True)

    X = df.drop(columns=[target_col])
    y = df[target_col].astype(float).values

    # 3) Chia train/test 80/20
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # 4) Chuẩn hóa dữ liệu
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 5) Huấn luyện mô hình Linear Regression
    model = LinearRegression()
    model.fit(X_train_scaled, y_train)

    # 6) Đánh giá
    y_pred = model.predict(X_test_scaled)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    metrics = {
        "rmse": float(rmse),
        "r2": float(r2),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    # 7) Lưu model và scaler
    joblib.dump(model, MODEL_PATH)
    joblib.dump(scaler, SCALER_PATH)

    print("Training completed.")
    print(f"RMSE: {rmse:.6f}")
    print(f"R2:   {r2:.6f}")
    print(f"Saved model to: {MODEL_PATH}")
    print(f"Saved scaler to: {SCALER_PATH}")


if __name__ == "__main__":
    main()

