# Dự đoán Giá nhà California (Linear Regression + Streamlit)

Dự án học máy dự đoán **giá nhà** bằng **Linear Regression** trên **California Housing Dataset** và có **web UI chạy localhost** (Streamlit) để nhập dữ liệu và xem kết quả kèm biểu đồ trực quan hóa.

---

## 1) Tạo môi trường ảo (virtual environment)

Tùy chọn dùng `venv` (khuyến nghị):

```bash
python -m venv venv
```

Kích hoạt:

**Windows (cmd):**

```bash
venv\Scripts\activate
```

**Windows (PowerShell):**

```powershell
venv\Scripts\Activate.ps1
```

---

## 2) Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

## 3) Huấn luyện lại mô hình

```bash
python train_model.py
```

Sau khi chạy xong sẽ sinh ra:

- `model.pkl`
- `scaler.pkl`
- `metrics.json` (chứa RMSE & R2 trên tập test)

---

## 4) Chạy ứng dụng web localhost

```bash
streamlit run app.py
```

Mở trình duyệt theo URL Streamlit hiển thị (thường là `http://localhost:8501`).

---

## 5) Các input trong Sidebar

Các ô nhập tương ứng với **8 đặc trưng (features)** của California Housing Dataset, ví dụ:

- `MedInc` (Thu nhập trung bình khu vực)
- `HouseAge` (Tuổi trung bình của nhà)
- `AveRooms` (Số phòng trung bình)
- `AveBedrms` (Số phòng ngủ trung bình)
- `Population` (Dân số)
- `AveOccup` (Số người ở mỗi căn hộ)
- `Latitude`
- `Longitude`

Nhập xong bấm nút **"Dự đoán giá ngay"** để nhận kết quả.

---

## Ghi chú về đơn vị giá

Trong dataset, `MedHouseValue` có đơn vị là **$100,000s**. Ứng dụng sẽ hiển thị kết quả quy đổi ra `$` (nhân thêm 100,000) để dễ hiểu.
