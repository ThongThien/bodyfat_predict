import streamlit as st

def show_info_page_v5():

    st.title("Body Fat AI v5 - Giải thích hệ thống")

    st.markdown("""
## 1. Tổng quan hệ thống

Ứng dụng sử dụng:
- Computer Vision (Mediapipe)
- Machine Learning (Random Forest)
- Feature Engineering chuyên sâu

---

## 2. Pipeline hoạt động
Image → Segmentation → Landmark → y_map → width scan → ellipse → calib → OUTPUT
### Bước 1: Segmentation
Tách cơ thể khỏi nền bằng AI

### Bước 2: Landmark Detection
Xác định:
- Vai
- Hông
- Chân

### Bước 3: Scan chiều ngang
Quét mask tại 3 vị trí:
- Ngực
- Bụng
- Hông

### Bước 4: Tính chu vi
Dùng ellipse approximation (Ramanujan)

---

## 3. Feature Engineering

Model KHÔNG dùng trực tiếp ảnh  
→ mà dùng các đặc trưng:

- Weight
- Chest
- Abdomen
- Hip

+ 3 biến phái sinh:

### W_per_A
= Abd² / Weight  
→ phản ánh tích mỡ bụng

### WtHR
= Abd / Height  
→ nguy cơ sức khỏe

### WHR
= Abd / Hip  
→ phân bố mỡ

---

## 4. Model AI

Sử dụng:
- RandomForestRegressor
- 1000 trees
- max_depth = 8

Ưu điểm:
- Không overfit
- Robust với dữ liệu nhiễu

---

## 5. Sai số hệ thống

Sai số có thể đến từ:

- Quần áo rộng
- Ánh sáng kém
- Sai tư thế
- Background không rõ

---

## 6. Tối ưu độ chính xác

Để đạt kết quả tốt nhất:
- Chụp đúng guideline
- Không mặc áo
- Quần bó
- Đứng thẳng

---

## 7. Giới hạn

- Không thay thế DEXA scan
- Sai số ~2-5%

---

## 8. Kết luận

Đây là hệ thống:
- Semi-medical AI
- Ưu tiên consistency hơn tuyệt đối

""")