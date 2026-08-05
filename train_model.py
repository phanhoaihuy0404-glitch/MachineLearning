

import argparse
import json
import numpy as np
import pandas as pd
import joblib

from pathlib import Path
from sklearn.compose import ColumnTransformer
from sklearn.impute import SimpleImputer
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler


ARTIFACT_DIR = Path(__file__).resolve().parent
DATASET_PATH = ARTIFACT_DIR / "vietnam_housing_dataset.csv"
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
METRICS_PATH = ARTIFACT_DIR / "metrics.json"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    """Chuẩn hóa tên cột theo dạng snake_case để tương thích với UI và mô hình."""
    rename_map = {
        "address": "address",
        "Area": "area",
        "area": "area",
        "Frontage": "frontage",
        "frontage": "frontage",
        "Access Road": "access_road",
        "access road": "access_road",
        "access_road": "access_road",
        "House Direction": "house_direction",
        "house direction": "house_direction",
        "house_direction": "house_direction",
        "Balcony Direction": "balcony_direction",
        "balcony direction": "balcony_direction",
        "balcony_direction": "balcony_direction",
        "Floors": "floors",
        "floors": "floors",
        "Bedrooms": "bedrooms",
        "bedrooms": "bedrooms",
        "Bathrooms": "bathrooms",
        "bathrooms": "bathrooms",
        "Legal Status": "legal_status",
        "legal status": "legal_status",
        "legal_status": "legal_status",
        "Furniture State": "furniture_state",
        "furniture state": "furniture_state",
        "furniture_state": "furniture_state",
        "Price": "price",
        "price": "price",
    }

    normalized_name = {}
    for col in df.columns:
        raw_key = str(col).strip()
        lowered = raw_key.lower().replace(" ", "_")
        normalized_name[col] = rename_map.get(raw_key, rename_map.get(lowered, lowered))

    return df.rename(columns=normalized_name)


def load_dataset(dataset_path: Path | None = None) -> pd.DataFrame:
    source_path = dataset_path or DATASET_PATH
    if not source_path.exists():
        raise FileNotFoundError(
            "Không tìm thấy file dataset mới. Hãy tải file CSV từ Kaggle vào thư mục dự án với tên "
            "'vietnam_housing_dataset.csv' hoặc truyền đường dẫn bằng --dataset."
        )

    df = pd.read_csv(source_path)
    df = normalize_columns(df)
    return df


def build_pipeline(X: pd.DataFrame) -> Pipeline:
    numeric_cols = X.select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in X.columns if c not in numeric_cols]

    numeric_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="median")),
            ("scaler", StandardScaler()),
        ]
    )

    categorical_transformer = Pipeline(
        steps=[
            ("imputer", SimpleImputer(strategy="most_frequent")),
            ("onehot", OneHotEncoder(handle_unknown="ignore")),
        ]
    )

    preprocessor = ColumnTransformer(
        transformers=[
            ("num", numeric_transformer, numeric_cols),
            ("cat", categorical_transformer, categorical_cols),
        ],
        remainder="drop",
    )

    model = Pipeline(
        steps=[
            ("preprocessor", preprocessor),
            ("model", LinearRegression()),
        ]
    )
    return model


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train Vietnam housing price prediction model")
    parser.add_argument(
        "--dataset",
        type=Path,
        default=DATASET_PATH,
        help="Path to the Vietnam housing CSV dataset.",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    df = load_dataset(args.dataset)

    target_col = "price"
    if target_col not in df.columns:
        raise ValueError("Dataset mới không có cột 'price'. Vui lòng kiểm tra lại tên cột trong file CSV.")

    feature_cols = [c for c in df.columns if c not in {target_col, "address"}]
    if not feature_cols:
        raise ValueError("Không tìm thấy feature phù hợp để huấn luyện mô hình. Kiểm tra file CSV.")

    df = df.replace([np.inf, -np.inf], np.nan)
    df = df.dropna(subset=[target_col]).reset_index(drop=True)

    X = df[feature_cols]
    y = df[target_col].astype(float).values

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    model = build_pipeline(X_train)
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)

    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)
    mae = mean_absolute_error(y_test, y_pred)

    metrics = {
        "rmse": float(rmse),
        "mae": float(mae),
        "r2": float(r2),
        "n_train": int(X_train.shape[0]),
        "n_test": int(X_test.shape[0]),
        "dataset": "vietnam_housing_dataset.csv",
    }
    METRICS_PATH.write_text(json.dumps(metrics, indent=2), encoding="utf-8")

    joblib.dump(model, MODEL_PATH)

    print("Training completed.")
    print(f"RMSE: {rmse:.6f}")
    print(f"MAE:  {mae:.6f}")
    print(f"R2:   {r2:.6f}")
    print(f"Saved model to: {MODEL_PATH}")


if __name__ == "__main__":
    main()

