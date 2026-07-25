# TODO

## ✅ Hoàn thành

- [x] Xác nhận tạo mới 4 file: requirements.txt, train_model.py, app.py, README.md
- [x] Tạo `requirements.txt` (dependencies)
- [x] Tạo `train_model.py` (train + đánh giá + lưu model/scaler)
- [x] Tạo `app.py` (Streamlit UI + dự đoán + trực quan hóa)
- [x] Cập nhật `app.py` với UI/UX Pro Max **Data-Dense Dashboard** design system
- [x] Tạo `California_Housing_Analysis.ipynb` (EDA + Model Comparison + Hyperparameter Tuning)
- [x] Tạo `run.py` (launcher để chạy Streamlit app)

## 📂 Cấu trúc project

| File                                | Mô tả                                                    |
| ----------------------------------- | -------------------------------------------------------- |
| `app.py`                            | Streamlit web app (UI/UX Pro Max style)                  |
| `train_model.py`                    | Huấn luyện Linear Regression                             |
| `run.py`                            | Launcher tự động kích hoạt venv + train model + chạy app |
| `California_Housing_Analysis.ipynb` | Jupyter Notebook EDA + So sánh thuật toán                |
| `requirements.txt`                  | Danh sách dependencies                                   |
| `model.pkl`                         | Model đã huấn luyện                                      |
| `scaler.pkl`                        | StandardScaler                                           |
| `metrics.json`                      | Metrics của model                                        |
| `README.md`                         | Hướng dẫn sử dụng                                        |

## 🔧 Cách chạy

```bash
# Cách 1: Dùng run.py
python run.py

# Cách 2: Thủ công
venv\Scripts\activate
python train_model.py
streamlit run app.py

# Notebook
jupyter notebook California_Housing_Analysis.ipynb
```
