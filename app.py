"""app.py

Streamlit web app chạy trên localhost để dự đoán giá nhà
(đơn vị: $100,000s) dựa trên Linear Regression được huấn luyện
trên California Housing Dataset.

Chạy:
    streamlit run app.py

Yêu cầu file:
    model.pkl, scaler.pkl (do train_model.py tạo ra)

Trực quan hóa:
- Interactive correlation heatmap (Plotly)
- Feature coefficients bar chart
- Histogram phân phối target
- Scatter plot theo vị trí (Latitude, Longitude)
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go

import joblib
from pathlib import Path
from sklearn.datasets import fetch_california_housing


ARTIFACT_DIR = Path(__file__).resolve().parent
MODEL_PATH = ARTIFACT_DIR / "model.pkl"
SCALER_PATH = ARTIFACT_DIR / "scaler.pkl"


@st.cache_data(show_spinner=False)
def load_data() -> pd.DataFrame:
    data = fetch_california_housing(as_frame=True)
    df = data.frame.copy()
    return df


@st.cache_resource(show_spinner=False)
def load_artifacts():
    model = joblib.load(MODEL_PATH)
    scaler = joblib.load(SCALER_PATH)
    return model, scaler


def predict_from_features(model, scaler, feature_values: list[float], feature_names: list[str]) -> float:
    x = np.array(feature_values, dtype=float).reshape(1, -1)
    x_scaled = scaler.transform(x)
    pred = model.predict(x_scaled)[0]
    return float(pred)


def apply_custom_css() -> None:
    """Apply UI/UX Pro Max Data-Dense Dashboard design system"""
    st.markdown("""
    <style>
        /* Data-Dense Dashboard - UI/UX Pro Max Design System */
        @import url('https://fonts.googleapis.com/css2?family=Exo:wght@300;400;500;600;700&family=Roboto+Mono:wght@300;400;500;700&display=swap');
        
        * {
            font-family: 'Roboto Mono', monospace;
        }
        
        h1, h2, h3, h4, h5, h6 {
            font-family: 'Exo', sans-serif !important;
            font-weight: 600 !important;
        }
        
        .stApp {
            background: linear-gradient(135deg, #F0FDF4 0%, #E8F0F1 100%);
        }
        
        .main-header {
            background: linear-gradient(135deg, #15803D 0%, #22C55E 100%);
            padding: 25px 30px;
            border-radius: 16px;
            margin-bottom: 25px;
            box-shadow: 0 8px 32px rgba(21, 128, 61, 0.25);
            border: 1px solid rgba(187, 247, 208, 0.3);
        }
        
        .main-header h1 {
            color: white !important;
            margin: 0;
            font-size: 2.2rem !important;
            letter-spacing: -0.5px;
        }
        
        .main-header p {
            color: rgba(255, 255, 255, 0.9);
            margin: 5px 0 0 0;
            font-size: 1rem;
        }
        
        .prediction-card {
            background: white;
            padding: 20px 25px;
            border-radius: 14px;
            box-shadow: 0 4px 20px rgba(0,0,0,0.08);
            border-left: 5px solid #15803D;
            transition: transform 0.2s ease, box-shadow 0.2s ease;
        }
        
        .prediction-card:hover {
            transform: translateY(-2px);
            box-shadow: 0 6px 25px rgba(0,0,0,0.12);
        }
        
        .metric-card {
            background: white;
            padding: 15px 20px;
            border-radius: 12px;
            box-shadow: 0 2px 10px rgba(0,0,0,0.06);
            border: 1px solid #BBF7D0;
        }
        
        .stButton > button {
            background: linear-gradient(135deg, #15803D, #22C55E) !important;
            color: white !important;
            font-family: 'Exo', sans-serif !important;
            font-weight: 600 !important;
            border: none !important;
            border-radius: 10px !important;
            padding: 10px 25px !important;
            transition: all 0.2s ease !important;
            box-shadow: 0 4px 15px rgba(21, 128, 61, 0.3) !important;
        }
        
        .stButton > button:hover {
            transform: translateY(-2px) !important;
            box-shadow: 0 6px 20px rgba(21, 128, 61, 0.4) !important;
        }
        
        div[data-testid="stSidebar"] {
            background: linear-gradient(180deg, #F0FDF4 0%, #FFFFFF 100%);
            border-right: 1px solid #BBF7D0;
        }
        
        div[data-testid="stSidebar"] h2, 
        div[data-testid="stSidebar"] h3 {
            color: #14532D !important;
        }
        
        .stNumberInput input {
            border: 2px solid #BBF7D0 !important;
            border-radius: 8px !important;
            transition: border-color 0.2s !important;
        }
        
        .stNumberInput input:focus {
            border-color: #15803D !important;
            box-shadow: 0 0 0 3px rgba(21, 128, 61, 0.15) !important;
        }
        
        .stMetric {
            background: white;
            padding: 12px;
            border-radius: 10px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.05);
        }
        
        .stMetric label {
            font-family: 'Exo', sans-serif !important;
            font-weight: 500 !important;
        }
        
        .st-emotion-cache-1kyxreq {
            border: 1px solid #BBF7D0 !important;
        }
        
        hr {
            border-color: #BBF7D0 !important;
            opacity: 0.5;
        }
        
        /* Responsive adjustments */
        @media (max-width: 768px) {
            .main-header h1 { font-size: 1.5rem !important; }
        }
    </style>
    """, unsafe_allow_html=True)


def main() -> None:
    st.set_page_config(
        page_title="California Housing Price Predictor",
        page_icon="🏠",
        layout="wide",
        initial_sidebar_state="expanded",
    )
    
    apply_custom_css()
    
    # Header with UI/UX Pro Max Design
    st.markdown("""
    <div class="main-header">
        <h1>🏠 California Housing Price Predictor</h1>
        <p>Mô hình: Linear Regression trên California Housing Dataset | 
        <strong>Data-Dense Dashboard</strong> style by UI/UX Pro Max</p>
    </div>
    """, unsafe_allow_html=True)

    # Load dữ liệu & model
    df = load_data()
    model, scaler = load_artifacts()

    target_col = "MedHouseVal"
    feature_cols = [c for c in df.columns if c != target_col]

    # Sidebar inputs
    # Từ điển ánh xạ tên cột gốc sang tên hiển thị thân thiện
    label_map = {
        "MedInc": "Thu nhập trung bình (x $10,000)",
        "HouseAge": "Tuổi thọ trung bình của nhà (năm)",
        "AveRooms": "Số phòng trung bình / hộ",
        "AveBedrms": "Số phòng ngủ trung bình / hộ",
        "Population": "Dân số khu vực",
        "AveOccup": "Số người trung bình / hộ",
        "Latitude": "Vĩ độ (Latitude)",
        "Longitude": "Kinh độ (Longitude)"
    }

    # Sidebar inputs
    st.sidebar.header("Nhập thông tin ngôi nhà")
    st.sidebar.caption("Chỉnh sửa các thông số bên dưới:")

    inputs = {}
    for col in feature_cols:
        col_min = float(df[col].min())
        col_max = float(df[col].max())
        col_mean = float(df[col].mean())

        # Tính toán bước nhảy (step) hợp lý cho input
        step = (col_max - col_min) / 200.0 if col_max > col_min else 0.1
        step = max(step, 0.01)

        # Lấy tên tiếng Việt từ label_map, nếu không có thì giữ nguyên tên gốc
        display_name = label_map.get(col, col)

        inputs[col] = st.sidebar.number_input(
            label=display_name,
            min_value=col_min,
            max_value=col_max,
            value=float(col_mean),
            step=float(step),
            format="%.4f",
        )

    feature_values = [inputs[c] for c in feature_cols]

    # Predict
    if st.sidebar.button("Dự đoán giá ngay", type="primary"):
        pred = predict_from_features(model, scaler, feature_values, feature_cols)
        st.success("Kết quả dự đoán")

        # Hiển thị nổi bật
        # MedHouseValue: $100,000s
        pred_dollar = pred * 100_000
        col1, col2 = st.columns([2, 1])
        with col1:
            st.markdown(
                f"""
                <div style="padding: 12px; border-radius: 10px; background: #f0f2f6;">
                  <div style="font-size: 18px; color: #444;">Giá nhà dự đoán</div>
                  <div style="font-size: 40px; font-weight: 800; color: #0b5;">{pred_dollar:,.0f} $</div>
                  <div style="font-size: 14px; color: #666;">(tương ứng {pred:.4f} trong đơn vị $100,000s)</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        with col2:
            # Metrics (nếu có)
            metrics_path = ARTIFACT_DIR / "metrics.json"
            if metrics_path.exists():
                import json

                metrics = json.loads(metrics_path.read_text(encoding="utf-8"))
                st.metric("RMSE (test)", f"{metrics['rmse']:.4f}")
                st.metric("R² (test)", f"{metrics['r2']:.4f}")
            else:
                st.info("Chưa thấy metrics.json. Chạy lại train_model.py để tạo.")

    st.divider()

    # Visualizations
    st.subheader("Trực quan hóa dữ liệu & mô hình")

    # Heatmap correlation
    corr = df[feature_cols + [target_col]].corr(numeric_only=True)
    heatmap_fig = px.imshow(
        corr,
        text_auto=".2f",
        aspect="auto",
        color_continuous_scale="RdBu",
        zmin=-1,
        zmax=1,
        title="Interactive Correlation Heatmap",
    )
    heatmap_fig.update_layout(height=420)

    # Coefficients
    coefs = getattr(model, "coef_", None)
    coef_fig = None
    if coefs is not None:
        coef_series = pd.Series(coefs, index=feature_cols).sort_values(key=lambda s: np.abs(s), ascending=False)
        coef_fig = go.Figure(
            data=[
                go.Bar(
                    x=coef_series.values,
                    y=coef_series.index,
                    orientation="h",
                    marker_color=["#1f77b4" if v >= 0 else "#d62728" for v in coef_series.values],
                )
            ]
        )
        coef_fig.update_layout(
            title="Feature Coefficients (Linear Regression)",
            height=420,
            yaxis_title="Feature",
            xaxis_title="Coefficient (trên dữ liệu đã chuẩn hóa)",
        )

    # Histogram target
    hist_fig = px.histogram(
        df,
        x=target_col,
        nbins=50,
        title="Distribution of MedHouseValue (target)",
        labels={target_col: "MedHouseValue ($100,000s)"},
    )
    hist_fig.update_layout(height=420)

    # Optional scatter map (Latitude/Longitude)
    scatter_fig = None
    if {"Latitude", "Longitude"}.issubset(df.columns):
        scatter_fig = px.scatter(
            df,
            x="Longitude",
            y="Latitude",
            color=target_col,
            color_continuous_scale="Viridis",
            title="House Price by Location (Latitude/Longitude)",
            labels={"Longitude": "Longitude", "Latitude": "Latitude", target_col: "MedHouseValue ($100,000s)"},
            opacity=0.75,
        )
        scatter_fig.update_layout(height=420)

    # Render grid
    r1c1, r1c2 = st.columns(2)
    with r1c1:
        st.plotly_chart(heatmap_fig, use_container_width=True)
    with r1c2:
        if coef_fig is not None:
            st.plotly_chart(coef_fig, use_container_width=True)
        else:
            st.info("Mô hình không có coefficients.")

    r2c1, r2c2 = st.columns(2)
    with r2c1:
        st.plotly_chart(hist_fig, use_container_width=True)
    with r2c2:
        if scatter_fig is not None:
            st.plotly_chart(scatter_fig, use_container_width=True)
        else:
            st.info("Không tìm thấy Latitude/Longitude trong dataset để vẽ scatter plot.")


if __name__ == "__main__":
    main()

