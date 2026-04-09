import streamlit as st

def show_info_page_v5():

    st.title("Body Fat AI v5 - Giải thích hệ thống")

    st.markdown("""
## 1. Tổng quan hệ thống

Ứng dụng sử dụng mô hình lai (Hybrid System):

- Computer Vision (MediaPipe Pose + Segmentation)
- Geometry-based estimation
- Machine Learning (Random Forest - tuned)

Hệ thống thực hiện pipeline:

**Ảnh → Trích xuất số đo cơ thể → Feature Engineering → Dự đoán BodyFat**

---

## 2. Pipeline hoạt động

Image → Segmentation → Landmark → y_map → width/depth scan → ellipse → calibration → ML prediction

---

### Bước 1: Segmentation (Tách cơ thể)
- Sử dụng MediaPipe Selfie Segmentation
- Tạo mask nhị phân (body vs background)

→ Là nền tảng để đo kích thước chính xác

---

### Bước 2: Landmark Detection
Xác định các keypoints chính:
- Vai (shoulder)
- Hông (hip)
- Gót chân (heel)

→ Dùng để:
- Scale chiều cao
- Định vị vùng đo

---

### Bước 3: Xác định vị trí đo (y_map)

Dựa trên tỉ lệ cơ thể:

- Chest = shoulder + 0.27 * torso  
- Abdomen = hip - 0.30 * torso  
- Hip = hip + 0.05 * torso  

→ Giúp chuẩn hóa vị trí đo giữa các body type khác nhau

---

### Bước 4: Scan chiều ngang (Width - Front)

- Quét mask theo từng hàng pixel
- Giới hạn vùng bằng vai (adaptive margin)
- Margin được điều chỉnh theo BMI

→ Thu được **width (cm)**

---

### Bước 5: Scan chiều sâu (Depth - Side)

- Không dùng 1 vị trí cố định
- Scan nhiều điểm quanh vùng:

    y ± [0.01 → 0.03]

- Lấy giá trị lớn nhất

→ Giảm lỗi:
- pose lệch
- landmark sai
- occlusion

---

### Bước 6: Tính chu vi (Ellipse Approximation)

Giả định mặt cắt cơ thể là ellipse:

- a = width / 2  
- b = depth / 2  

Áp dụng công thức Ramanujan:

Circumference ≈ π(a+b)[1 + (3h)/(10 + √(4−3h))]

→ Thu được các số đo:
- Chest
- Abdomen
- Hip

---

### Bước 7: Scale & Calibration

#### Scale:
- Dựa trên chiều cao thực tế
- Chuyển đổi pixel → cm

#### Calibration:
- Điều chỉnh theo BMI để giảm bias hình ảnh
- Tăng độ ổn định giữa các body type

---

## 3. Feature Engineering

Mô hình **không sử dụng ảnh trực tiếp**, mà chỉ sử dụng các số đo:

### Input Features (7 biến):
- Weight
- Chest
- Abdomen
- Hip
- W_per_A
- WtHR
- WHR

---

### Feature phái sinh:

#### W_per_A
= Abdomen² / Weight  
→ Khuếch đại ảnh hưởng mỡ bụng

#### WtHR
= Abdomen / Height  
→ Chỉ số sức khỏe phổ biến trong y học

#### WHR
= Abdomen / Hip  
→ Phản ánh phân bố mỡ (bụng vs hông)

---

## 4. Model AI

### Thuật toán sử dụng:
- RandomForestRegressor (đã tối ưu)

### Tham số cuối:
- n_estimators = 500  
- max_depth = 4  
- max_features = None  
- min_samples_leaf = 3  
- min_samples_split = 2  

### Dataset:
- 195 mẫu
- 7 features

---

### Hiệu năng mô hình:

- R² Score: **0.8249**
- MAE: **~2.38%**
- RMSE: **~2.90%**

→ Sai số thấp, ổn định và phù hợp ứng dụng thực tế

---

### Feature quan trọng nhất:
1. Abdomen
2. WtHR
3. WHR

→ Vùng bụng là yếu tố quyết định chính

---

## 5. Sai số hệ thống

Nguồn sai số chính:

- Quần áo rộng (ảnh hưởng segmentation)
- Pose không chuẩn (ảnh side lệch)
- Ánh sáng kém
- Camera distortion
- Khác biệt sinh học (xương, cơ)

---

## 6. Tối ưu độ chính xác

Khuyến nghị:

- Chụp đủ ảnh front + side
- Đứng thẳng, không nghiêng
- Không mặc áo (hoặc áo bó)
- Quần ôm vùng hông

---

## 7. Giới hạn hệ thống

- Không thay thế DEXA / InBody
- Không đo trực tiếp mỡ nội tạng
- Phụ thuộc vào chất lượng ảnh đầu vào

Sai số thực tế:
- Trung bình: ~2–3%
- Trường hợp xấu: ~5–7%

---

## 8. Kết luận

Đây là hệ thống:

- Kết hợp Computer Vision + Geometry + Machine Learning
- Sử dụng feature engineering thay vì ảnh thô
- Đạt độ chính xác cao với dataset nhỏ

→ Phù hợp cho:
- Ứng dụng fitness
- Theo dõi body tại nhà
- Prototype AI health system

""")