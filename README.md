# 🏠 Vietnam Housing Price Predictor

> **Machine Learning Project** — Dự đoán giá nhà Việt Nam từ dataset Vietnam Housing Dataset 2024 bằng Linear Regression và giao diện Web (Streamlit).

---

## ✨ Tính năng

| Tính năng                  | Mô tả                                                                      |
| -------------------------- | -------------------------------------------------------------------------- |
| 🧠 **Huấn luyện model**    | Linear Regression trên dataset Vietnam Housing Dataset 2024                |
| 🌐 **Web UI trực quan**    | Streamlit app cho dự đoán giá nhà Việt Nam theo thông tin bất động sản     |
| 📊 **Dashboard tương tác** | Heatmap tương quan, histogram phân phối, scatter plot theo diện tích       |
| 📦 **Upload dataset**      | Hỗ trợ upload file CSV ngay trong giao diện nếu chưa đặt sẵn trong thư mục |
| 🚀 **1-click launcher**    | `python run.py` — tự động setup & chạy                                     |

---

## 📂 Cấu trúc dự án

```
ML/
├── app.py                                  # Streamlit web app dự đoán giá nhà Việt Nam
├── train_model.py                          # Huấn luyện model trên dataset Việt Nam
├── run.py                                  # Launcher 1 câu lệnh
├── vietnam_housing_dataset.csv             # Dataset mới (nếu đã tải vào project)
├── requirements.txt                        # Dependencies
├── model.pkl                               # Model đã train
├── metrics.json                            # Metrics (RMSE, R²)
├── TODO.md                                 # Kế hoạch công việc
├── .gitignore
└── README.md
```

---

## 🚀 Quick Start (1 câu lệnh)

```bash
python run.py
```

File `run.py` sẽ tự động:

1. Kiểm tra & cài đặt dependencies
2. Huấn luyện model nếu chưa có
3. Chạy Streamlit app trên `http://localhost:8501`

> Lưu ý: bạn cần có file `vietnam_housing_dataset.csv` trong thư mục project hoặc upload file CSV ngay trong sidebar khi chạy app.

---

## 🛠️ Hướng dẫn chi tiết

### 1) Tạo môi trường ảo

```bash
python -m venv venv
```

**Kích hoạt:**

| OS                   | Lệnh                        |
| -------------------- | --------------------------- |
| Windows (cmd)        | `venv\Scripts\activate`     |
| Windows (PowerShell) | `venv\Scripts\Activate.ps1` |
| macOS/Linux          | `source venv/bin/activate`  |

### 2) Cài đặt thư viện

```bash
pip install -r requirements.txt
```

### 3) Huấn luyện mô hình

```bash
python train_model.py --dataset vietnam_housing_dataset.csv
```

Nếu file dataset nằm trong thư mục project, bạn có thể dùng lệnh đơn giản:

```bash
python train_model.py
```

Kết quả: `model.pkl`, `metrics.json`

### 4) Chạy Web UI

```bash
streamlit run app.py
```

Mở trình duyệt → [http://localhost:8501](http://localhost:8501)

### 5) Xem Notebook phân tích

```bash
hist_boxplot_analysis.ipynb
```

---

## 📊 Notebook / phân tích dữ liệu

Notebook hiện đang dùng để thực hiện EDA ban đầu và kiểm tra dữ liệu mới. Nội dung chính có thể bao gồm:

- Load và chuẩn hóa dataset Việt Nam
- Thống kê mô tả các trường như diện tích, số tầng, phòng ngủ, phòng tắm
- Phân tích tương quan giữa feature và giá nhà
- Visualize phân phối target `price`
- So sánh và đánh giá mô hình dự đoán

---

## 🎨 UI/UX Design

Giao diện Streamlit được thiết kế theo **Data-Dense Dashboard** style từ [UI/UX Pro Max](https://github.com/nextlevelbuilder/ui-ux-pro-max-skill):

- **Font:** Exo (heading) + Roboto Mono (body)
- **Color Palette:** Xanh lá chủ đạo (`#15803D`, `#22C55E`)
- **Hiệu ứng:** Gradient header, hover transitions, glassmorphism cards
- **Responsive:** Tương thích mobile & desktop

---

## 📝 Các input trong Sidebar

Các trường đầu vào chính của dataset Việt Nam:

| Feature             | Ý nghĩa             |
| ------------------- | ------------------- |
| `area`              | Diện tích (m²)      |
| `frontage`          | Mặt tiền (m)        |
| `access_road`       | Đường vào (m)       |
| `house_direction`   | Hướng nhà           |
| `balcony_direction` | Hướng ban công      |
| `floors`            | Số tầng             |
| `bedrooms`          | Số phòng ngủ        |
| `bathrooms`         | Số phòng tắm        |
| `legal_status`      | Tình trạng pháp lý  |
| `furniture_state`   | Tình trạng nội thất |

---

## 💰 Ghi chú về đơn vị giá

- Trong dataset Việt Nam: `price` có đơn vị là **tỷ đồng (VND)**
- Ứng dụng hiển thị kết quả dự đoán theo **đồng Việt Nam** và mô tả tương ứng ở dạng **tỷ đồng**

Ví dụ: `price = 2.8` → **2,8 tỷ đồng**

---
