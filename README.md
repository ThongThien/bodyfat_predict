# HƯỚNG DẪN CHẠY DỰ ÁN

## 1. Cài đặt yêu cầu
* Python 3.10+
* Java JDK 17

Kiểm tra:

```bash
python --version
java -version
```

---

## 2. Mở thư mục dự án

```bash
cd Bodyfat_predict
```

---

## 3. Tạo môi trường ảo

```bash
python -m venv venv
```

Kích hoạt:

```bash
venv\Scripts\activate
```

---

## 4. Cài đặt thư viện

```bash
pip install -r requirements.txt
```

---

## 5. Tạo file .env

Tạo file `.env` tại thư mục gốc của dự án:

```env
SUPABASE_URL=YOUR_SUPABASE_URL
SUPABASE_KEY=YOUR_SUPABASE_KEY
SUPABASE_BUCKET_NAME=bodyfat_images
```

---

## 6. Cấu hình Java 17 cho Ontology

Nếu máy có nhiều phiên bản Java, sửa đường dẫn Java trong file ontology tương ứng:

```python
JAVA17_HOME = r"C:\Program Files\Java\jdk-17"
```

hoặc thay bằng đường dẫn JDK 17 trên máy đang sử dụng.

Kiểm tra:

```bash
java -version
```

Phải hiển thị Java 17.

---

## 7. Chạy ứng dụng

```bash
streamlit run app_v5.py
```

Sau khi chạy thành công, truy cập:

```text
http://localhost:8501
```

---

## 8. Một số lỗi thường gặp

### Thiếu thư viện

```bash
pip install -r requirements.txt
```

### Không tìm thấy model

Kiểm tra file:

```text
models/bodyfat_ai_super_clean_v5.pkl
```

### Lỗi Ontology

Kiểm tra:

```bash
java -version
```

Đảm bảo Java 17 đã được cài đặt đúng.

### Lỗi Supabase

Kiểm tra lại file:

```text
.env
```

và các giá trị:

```env
SUPABASE_URL=...
SUPABASE_KEY=...
SUPABASE_BUCKET_NAME=...
```
### Ảnh mẫu trong thư mục assets/anh_chuan
