import streamlit as st
import joblib
import pandas as pd
import numpy as np

# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="Dự đoán Body Fat AI", layout="wide")

# --- LOAD MÔ HÌNH ---
@st.cache_resource
def load_model():
    return joblib.load('tuned_xgboost_k7.pkl_final')

model = load_model()
mae_error = 3.85  # Sai số MAE của mô hình Thiên

# --- GIAO DIỆN CHÍNH ---
st.title("🏋️‍♂️ Hệ Thống Dự Đoán Tỷ Lệ Mỡ Cơ Thể (AI Model)")
st.markdown("""
Mô hình sử dụng thuật toán **Tuned XGBoost (k=7)** để dự đoán tỷ lệ mỡ dựa trên các chỉ số nhân trắc học. 
Đơn vị tính: **Kilogram (Kg)** và **Centimet (Cm)**.
""")

st.divider()

# --- CHIA CỘT NHẬP LIỆU ---
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("📍 Chỉ số cơ bản")
    age = st.number_input("1. Tuổi (năm)", min_value=1, max_value=100, value=22)
    weight = st.number_input("2. Cân nặng (kg)", min_value=30.0, max_value=200.0, value=62.0)
    height = st.number_input("3. Chiều cao (cm)", min_value=100.0, max_value=250.0, value=163.0)
    neck = st.number_input("4. Vòng cổ (cm)", value=36.0)

with col2:
    st.subheader("📍 Chỉ số thân trên")
    chest = st.number_input("5. Vòng ngực (cm)", value=90.0)
    abdomen = st.number_input("6. Vòng bụng (cm)", value=78.0)
    biceps = st.number_input("11. Vòng bắp tay (cm)", value=36.0)
    forearm = st.number_input("12. Vòng bắp tay dưới (cm)", value=28.0)
    wrist = st.number_input("13. Vòng cổ tay (cm)", value=16.0)

with col3:
    st.subheader("📍 Chỉ số thân dưới")
    hip = st.number_input("7. Vòng mông (cm)", value=88.0)
    thigh = st.number_input("8. Vòng đùi (cm)", value=52.0)
    knee = st.number_input("9. Vòng đầu gối (cm)", value=34.0)
    ankle = st.number_input("10. Vòng cổ chân (cm)", value=21.0)

st.divider()

# --- XỬ LÝ DỰ ĐOÁN ---
if st.button("📊 TÍNH TOÁN KẾT QUẢ", type="primary", use_container_width=True):
    # Chuẩn bị dữ liệu đúng thứ tự các cột khi train
    input_dict = {
        'Age': age, 'Weight': weight, 'Height': height, 'Neck': neck,
        'Chest': chest, 'Abdomen': abdomen, 'Hip': hip, 'Thigh': thigh,
        'Knee': knee, 'Ankle': ankle, 'Biceps': biceps, 'Forearm': forearm, 'Wrist': wrist
    }
    input_df = pd.DataFrame([input_dict])
    
    # Dự đoán
    prediction = model.predict(input_df)[0]
    
    # Hiển thị kết quả
    st.balloons()
    
    res_col1, res_col2 = st.columns(2)
    
    with res_col1:
        st.metric(label="Tỷ lệ mỡ dự đoán", value=f"{prediction:.2f}%")
        
        # Hiển thị khoảng tin cậy dựa trên MAE
        lower = max(0, prediction - mae_error)
        upper = prediction + mae_error
        st.write(f"⚠️ **Phạm vi thực tế (±3.85% MAE):** {lower:.2f}% - {upper:.2f}%")

    with res_col2:
        # Đánh giá trạng thái
        if prediction < 14:
            status = "Lean (Vận động viên/Khỏe mạnh)"
            color = "green"
        elif prediction < 25:
            status = "Average (Bình thường)"
            color = "blue"
        else:
            status = "Overweight (Cảnh báo thừa mỡ)"
            color = "red"
            
        st.markdown(f"### Trạng thái: :{color}[{status}]")

    # Vẽ biểu đồ đơn giản để tăng tính chuyên nghiệp
    st.progress(min(int(prediction), 100))