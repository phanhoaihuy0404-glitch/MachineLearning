"""app.py

Streamlit web app dùng để dự đoán giá nhà Việt Nam từ dataset
vietnam_housing_dataset.csv.

Chạy:
    streamlit run app.py

Yêu cầu file:
    model.pkl (do train_model.py tạo ra)
    vietnam_housing_dataset.csv (được tải xuống từ Kaggle và đặt vào thư mục dự án)
"""

from __future__ import annotations

import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st


ARTIFACT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
DATASET_PATH = ARTIFACT_DIR / "vietnam_housing_dataset.csv"


def normalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    rename_map = {
        "Address": "address",
        "address": "address",
        "Area": "area",
        "area": "area",
        "Frontage": "frontage",
        "frontage": "frontage",
        "Access Road": "access_road",
        "access road": "access_road",
        "House Direction": "house_direction",
        "house direction": "house_direction",
        "Balcony Direction": "balcony_direction",
        "balcony direction": "balcony_direction",
        "Floors": "floors",
        "floors": "floors",
        "Bedrooms": "bedrooms",
        "bedrooms": "bedrooms",
        "Bathrooms": "bathrooms",
        "bathrooms": "bathrooms",
        "Legal Status": "legal_status",
        "legal status": "legal_status",
        "Furniture State": "furniture_state",
        "furniture state": "furniture_state",
        "Price": "price",
        "price": "price",
    }

    normalized = {}
    for col in df.columns:
        raw = str(col).strip()
        lowered = raw.lower().replace(" ", "_")
        normalized[col] = rename_map.get(raw, rename_map.get(lowered, lowered))

    return df.rename(columns=normalized)


@st.cache_data(show_spinner=False)
def load_data(dataset_path: str | None = None) -> pd.DataFrame:
    if dataset_path and Path(dataset_path).exists():
        df = pd.read_csv(dataset_path)
        return normalize_columns(df)

    if DATASET_PATH.exists():
        df = pd.read_csv(DATASET_PATH)
        return normalize_columns(df)

    return pd.DataFrame()


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    return model


def predict_from_features(model, inputs: dict, feature_cols: list[str]) -> float:
    payload = pd.DataFrame([inputs], columns=feature_cols)
    pred = model.predict(payload)[0]
    return float(pred)


def apply_custom_css() -> None:
    st.markdown(
        """
        <style>
            @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700;800&display=swap');
            * { font-family: 'Inter', sans-serif; }
            .stApp {
                background: linear-gradient(135deg, #f8fbff 0%, #eef4ff 100%);
            }
            .main-header {
                background: linear-gradient(135deg, #0f4c81 0%, #1d6fb8 100%);
                padding: 24px 28px;
                border-radius: 16px;
                margin-bottom: 22px;
                box-shadow: 0 10px 30px rgba(15, 76, 129, 0.2);
            }
            .main-header h1 {
                color: white !important;
                margin: 0 0 6px 0;
                font-size: 2.15rem !important;
            }
            .main-header p {
                color: rgba(255,255,255,0.92);
                margin: 0;
                font-size: 0.98rem;
            }
            .stButton > button {
                background: linear-gradient(135deg, #0f4c81, #1d6fb8) !important;
                color: white !important;
                font-weight: 700 !important;
                border: none !important;
                border-radius: 10px !important;
                padding: 10px 24px !important;
            }
            div[data-testid="stSidebar"] {
                background: linear-gradient(180deg, #f4f8ff 0%, #ffffff 100%);
                border-right: 1px solid #dbe8ff;
            }
            .stNumberInput input, .stSelectbox div[data-baseweb="select"] {
                border: 1px solid #bdd6ff !important;
                border-radius: 9px !important;
            }
            hr {
                border-color: #cfe0ff !important;
            }
        </style>
        """,
        unsafe_allow_html=True,
    )


def main() -> None:
    st.set_page_config(
        page_title="Vietnam Housing Price Predictor",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )

    apply_custom_css()

    st.markdown(
        """
        <div class="main-header">
            <h1>🏠 Vietnam Housing Price Predictor</h1>
            <p>Dataset: Vietnam Housing Dataset 2024 • Mô hình Linear Regression + preprocessing cho feature số và categorical</p>
        </div>
        """,
        unsafe_allow_html=True,
    )

    if not MODEL_PATH.exists():
        st.error("Chưa tìm thấy model.pkl. Hãy chạy train_model.py để huấn luyện lại mô hình trước khi chạy UI.")
        st.stop()

    uploaded_file = st.sidebar.file_uploader("Tải dataset CSV của bạn", type=["csv"], help="Bạn có thể upload file dataset Vietnam Housing Dataset 2024 ở đây nếu chưa đặt sẵn trong thư mục dự án.")

    if uploaded_file is not None:
        dataset_bytes = uploaded_file.getvalue()
        temp_csv = ARTIFACT_DIR / uploaded_file.name
        temp_csv.write_bytes(dataset_bytes)
        df = load_data(str(temp_csv))
    else:
        df = load_data()

    if df.empty:
        st.warning("Chưa thấy file vietnam_housing_dataset.csv trong thư mục dự án. Hãy tải dataset Kaggle và đặt vào cùng thư mục với app.py, hoặc upload lên ở sidebar.")
        st.stop()

    target_col = "price"
    if target_col not in df.columns:
        st.error("Dataset chưa khớp với schema mới. Cần file CSV có cột 'price'.")
        st.stop()

    feature_cols = [c for c in df.columns if c not in {target_col, "address"}]
    if not feature_cols:
        st.error("Dataset không có feature đủ để dự đoán.")
        st.stop()

    model = load_artifacts()

    label_map = {
        "area": "Diện tích (m²)",
        "frontage": "Mặt tiền (m)",
        "access_road": "Đường vào (m)",
        "house_direction": "Hướng nhà",
        "balcony_direction": "Hướng ban công",
        "floors": "Số tầng",
        "bedrooms": "Số phòng ngủ",
        "bathrooms": "Số phòng tắm",
        "legal_status": "Tình trạng pháp lý",
        "furniture_state": "Tình trạng nội thất",
    }

    st.sidebar.header("Nhập thông tin bất động sản")
    st.sidebar.caption("Điền các thông số bên dưới để nhận dự đoán giá nhà Việt Nam.")

    inputs = {}
    numeric_cols = df[feature_cols].select_dtypes(include=[np.number]).columns.tolist()
    categorical_cols = [c for c in feature_cols if c not in numeric_cols]

    for col in feature_cols:
        display_name = label_map.get(col, col)

        if col in numeric_cols:
            col_min = float(df[col].min())
            col_max = float(df[col].max())
            col_value = float(df[col].median())
            # Các cột đếm được (số tầng, phòng ngủ, phòng tắm) chỉ nhận số nguyên
            if col in {"floors", "bedrooms", "bathrooms"}:
                col_min = int(col_min)
                col_max = int(col_max)
                col_value = int(col_value)
                inputs[col] = st.sidebar.number_input(
                    label=display_name,
                    min_value=col_min,
                    max_value=col_max,
                    value=col_value,
                    step=1,
                    format="%d",
                )
            else:
                step = (col_max - col_min) / 200.0 if col_max > col_min else 0.1
                step = max(step, 0.1)
                inputs[col] = st.sidebar.number_input(
                    label=display_name,
                    min_value=col_min,
                    max_value=col_max,
                    value=col_value,
                    step=step,
                    format="%.2f",
                )
        else:
            options = sorted(df[col].fillna("Unknown").astype(str).unique().tolist())
            inputs[col] = st.sidebar.selectbox(display_name, options)

    if st.sidebar.button("Dự đoán giá ngay", type="primary"):
        pred = predict_from_features(model, inputs, feature_cols)
        pred_vnd = pred * 1_000_000_000
        st.success("Kết quả dự đoán")

        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                f"""
                <div style="padding: 16px; border-radius: 12px; background: #f2f7ff; border-left: 5px solid #0f4c81;">
                    <div style="font-size: 18px; color: #333; margin-bottom: 6px;">Giá nhà dự đoán</div>
                    <div style="font-size: 38px; font-weight: 800; color: #0f4c81;">{pred_vnd:,.0f} ₫</div>
                    <div style="font-size: 14px; color: #666;">(tương ứng {pred:.4f} tỷ đồng)</div>
                </div>
""",
                unsafe_allow_html=True,
            )
        with col2:
            metrics_path = ARTIFACT_DIR / "metrics.json"
            if metrics_path.exists():
                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                st.metric("RMSE (test)", f"{metrics['rmse']:.4f}")
                st.metric("MAE (test)", f"{metrics.get('mae', 0.0):.4f}")
                st.metric("R² (test)", f"{metrics['r2']:.4f}")
            else:
                st.info("Chưa thấy metrics.json. Chạy lại train_model.py để tạo.")

    st.divider()
    st.subheader("Trực quan hóa dữ liệu")

    numeric_df = df.select_dtypes(include=[np.number])
    corr = numeric_df.corr(numeric_only=True)
    heatmap_fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Ma trận tương quan giữa feature số",
    )
    heatmap_fig.update_layout(height=420)

    hist_fig = px.histogram(
        df,
        x=target_col,
        nbins=50,
        title="Phân phối giá nhà theo dataset",
        labels={target_col: "Giá nhà (tỷ đồng)"},
    )
    hist_fig.update_layout(height=420)

    scatter_fig = None
    if {"area", "price"}.issubset(df.columns):
        scatter_fig = px.scatter(
            df,
            x="area",
            y="price",
            color="bedrooms" if "bedrooms" in df.columns else None,
            title="Giá nhà theo diện tích",
            labels={"area": "Diện tích (m²)", "price": "Giá nhà (tỷ đồng)"},
            opacity=0.75,
        )
        scatter_fig.update_layout(height=420)

    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(heatmap_fig, use_container_width=True)
    with r1c2:
        st.plotly_chart(hist_fig, use_container_width=True)

    if scatter_fig is not None:
        st.plotly_chart(scatter_fig, use_container_width=True)


if __name__ == "__main__":
    main()

