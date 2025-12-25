import streamlit as st
import joblib
import pandas as pd
import uuid  # Thêm để fix lỗi render SVG
# --- CẤU HÌNH TRANG ---
st.set_page_config(page_title="ThongThien Fitness AI", layout="wide", page_icon="⚡")

# --- STYLE CSS (Tối ưu layout & làm nổi bật kết quả) ---
st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    
    /* Gọn nhẹ hóa các ô input */
    div[data-testid="stNumberInput"] { margin-bottom: -15px; }
    div[data-testid="stNumberInput"] label { font-size: 13px !important; color: #9CA3AF !important; }
    
    /* Khung kết quả nổi bật */
    .big-value { font-size: 50px !important; font-weight: 900; color: #00FF00; margin: 0; line-height: 1; }
    .metric-item { background: rgba(255,255,255,0.05); padding: 12px; border-radius: 10px; margin-bottom: 8px; border: 1px solid #334155; }
    .metric-label { color: #FFFFFF; font-size: 13px; }
    .metric-val { float: right; font-weight: bold; color: #F3F4F6; }
    </style>
    """, unsafe_allow_html=True)

# --- DỮ LIỆU MẪU ---
PRESETS = {
    "Chỉ số của tôi": [22, 62, 163, 36, 90, 78, 88, 52, 34, 21, 36, 28, 16],
    "Vận động viên (Cực nét)": [25, 70, 175, 37, 100, 72, 90, 55, 36, 22, 40, 31, 17],
    "Người tập Gym (Săn chắc)": [22, 62, 163, 36, 90, 78, 88, 52, 34, 21, 36, 28, 16],
    "Người bình thường": [30, 80, 175, 39, 95, 88, 95, 53, 38, 23, 33, 28, 18],
    "Người thừa mỡ": [35, 95, 175, 41, 105, 105, 110, 62, 42, 25, 31, 26, 19]
}

if 'page' not in st.session_state: 
    st.session_state.page = 'home'
if 'vals' not in st.session_state: 
    st.session_state.vals = PRESETS["Chỉ số của tôi"]

# --- TẢI MÔ HÌNH ---
@st.cache_resource
def load_model():
    return joblib.load('tuned_xgboost_k7_final.pkl')

# Thử load model, nếu file không tồn tại sẽ báo lỗi nhẹ nhàng
try:
    model = load_model()
except:
    st.error("Không tìm thấy file model 'tuned_xgboost_k7_final.pkl'. Vui lòng kiểm tra lại.")

# --- HÀM VẼ BIỂU ĐỒ HÌNH NGƯỜI SVG ---
def get_human_svg(bf):
    unique_id = str(uuid.uuid4())[:8] # Tạo ID duy nhất cho mỗi lần render
    fill_h = max(0, min(100, (bf / 40) * 100))
    y_pos = 210 - (fill_h * 2.1)
    
    svg = f"""
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
        <svg width="140" height="300" viewBox="0 0 100 220">
            <path d="M50,10 C55,10 60,15 60,20 C60,25 55,30 50,30 C45,30 40,25 40,20 C40,15 45,10 50,10 M40,32 L60,32 L65,80 L75,130 L70,210 L55,210 L50,140 L45,210 L30,210 L25,130 L35,80 Z" fill="#2D3748" />
            <defs><clipPath id="cp_{unique_id}"><rect x="0" y="{y_pos}" width="100" height="210" /></clipPath></defs>
            <path d="M50,10 C55,10 60,15 60,20 C60,25 55,30 50,30 C45,30 40,25 40,20 C40,15 45,10 50,10 M40,32 L60,32 L65,80 L75,130 L70,210 L55,210 L50,140 L45,210 L30,210 L25,130 L35,80 Z" fill="#FBBF24" clip-path="url(#cp_{unique_id})" />
            <path d="M50,10 C55,10 60,15 60,20 C60,25 55,30 50,30 C45,30 40,25 40,20 C40,15 45,10 50,10 M40,32 L60,32 L65,80 L75,130 L70,210 L55,210 L50,140 L45,210 L30,210 L25,130 L35,80 Z" fill="none" stroke="#4A5568" stroke-width="2" />
        </svg>
        <p style="color: #FBBF24; font-weight: bold; margin-top: 5px;">Mức phủ mỡ: {bf:.1f}%</p>
    </div>
    """
    return svg

# --- HEADER & NAVIGATION ---
col_h1, col_h2 = st.columns([0.8, 0.2])
with col_h1:
    st.markdown("<h1 style='color:#3B82F6; margin:0;'>THONGTHIEN AI</h1>", unsafe_allow_html=True)
    st.caption("Hệ thống phân tích hình thể đa điểm (MAE 3.85%)")
with col_h2:
    if st.button("ℹ️ Thông tin khoa học", use_container_width=True):
        st.session_state.page = 'info' if st.session_state.page == 'home' else 'home'
        st.rerun()

# ---------------------------------------------------------
# TRANG CHỦ: ĐO CHỈ SỐ
# ---------------------------------------------------------
if st.session_state.page == 'home':
    st.markdown("### Chọn nhanh bộ chỉ số")
    p_col1, p_col2 = st.columns([0.8, 0.2])

    with p_col1:
        choice = st.selectbox("Chọn tạng người mẫu:", list(PRESETS.keys()), label_visibility="collapsed")

    with p_col2:
        if st.button("ÁP DỤNG", use_container_width=True):
            st.session_state.vals = PRESETS[choice]
            st.rerun()

    # 1. NHẬP LIỆU GỌN (5 CỘT)
    v = st.session_state.vals
    c1, c2, c3, c4, c5 = st.columns(5)
    
    with c1:
        age = st.number_input("Tuổi", 1, 100, int(v[0]), help="Tuổi hiện tại")
        weight = st.number_input("Cân nặng (kg)", 30.0, 200.0, float(v[1]), help="Cân lúc bụng đói")
        height = st.number_input("Cao (cm)", 100.0, 250.0, float(v[2]), help="Đo không đi giày")
    with c2:
        neck = st.number_input("Cổ (cm)", 20.0, 60.0, float(v[3]), help="Dưới yết hầu")
        chest = st.number_input("Ngực (cm)", 50.0, 150.0, float(v[4]), help="Ngang núm vú")
        abdomen = st.number_input("Bụng (cm)", 40.0, 150.0, float(v[5]), help="Ngang rốn, thả lỏng")
    with c3:
        hip = st.number_input("Mông (cm)", 50.0, 150.0, float(v[6]), help="Phần nở nhất")
        thigh = st.number_input("Đùi (cm)", 30.0, 100.0, float(v[7]), help="Dưới lằn mông")
        knee = st.number_input("Gối (cm)", 20.0, 60.0, float(v[8]), help="Giữa bánh chè")
    with c4:
        ankle = st.number_input("Cổ chân (cm)", 10.0, 40.0, float(v[9]), help="Trên mắt cá")
        bicep = st.number_input("Bắp tay (cm)", 15.0, 60.0, float(v[10]), help="Gồng vuông góc")
        forearm = st.number_input("Cẳng tay (cm)", 10.0, 50.0, float(v[11]), help="Phần lớn nhất")
    with c5:
        wrist = st.number_input("Cổ tay (cm)", 10.0, 30.0, float(v[12]), help="Trên mắt cá tay")
        st.write("")
        btn = st.button("🔥 PHÂN TÍCH", type="primary", use_container_width=True)

    # --- TẠO TAB ---
    tab1, tab2 = st.tabs(["📊 Dự báo chỉ số", "📷 Hình mẫu tham khảo"])

    with tab1:
        # 2. HIỂN THỊ KẾT QUẢ
        if btn:
            input_data = pd.DataFrame([[age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, bicep, forearm, wrist]],
                                     columns=['Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist'])
            
            prediction = model.predict(input_data)[0]
            
            # --- XỬ LÝ RANGE (Model là trung tâm) ---
            error_margin = 3.85
            low_range = max(3.0, prediction - error_margin)
            high_range = min(45.0, prediction + error_margin)
            
            # Tính toán các chỉ số phụ
            bmi = weight / ((height/100)**2)
            fat_kg = (prediction / 100) * weight
            lbm = weight - fat_kg
            ideal_bf = 15.0 if age < 30 else 18.0

            st.divider()
            res_c1, res_c2, res_c3 = st.columns([0.8, 1.2, 1.2])

            with res_c1:
                st.markdown(get_human_svg(prediction), unsafe_allow_html=True)

            with res_c2:
                st.markdown(f"""
                    <div class='result-box'>
                        <p class='metric-label' style='font-size:15px; color:#9CA3AF;'>TỶ LỆ MỠ</p>
                        <p class='big-value'>{prediction:.1f}%</p>
                        <p class='range-label'>KHOẢNG BIẾN THIÊN THỰC TẾ (±3.85%)</p>
                        <p class='range-text'>{low_range:.1f}% — {high_range:.1f}%</p>
                    </div>
                """, unsafe_allow_html=True)
                # Hiển thị khoảng sai số để tăng tính tin cậy
                st.write("")
                st.markdown(f"<div class='metric-item'><span class='metric-label'>Khối lượng mỡ:</span><span class='metric-val'>{fat_kg:.1f} kg</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-item'><span class='metric-label'>Khối lượng nạc (LBM):</span><span class='metric-val'>{lbm:.1f} kg</span></div>", unsafe_allow_html=True)
                st.caption("*(LBM: Cơ bắp, khung xương, nội tạng và nước)*")
                st.markdown(f"<div class='metric-item'><span class='metric-label'>Mỡ lý tưởng cho tuổi:</span><span class='metric-val'>{ideal_bf}%</span></div>", unsafe_allow_html=True)
                st.markdown(f"<div class='metric-item'><span class='metric-label'>Chỉ số BMI:</span><span class='metric-val'>{bmi:.1f}</span></div>", unsafe_allow_html=True)

            with res_c3:
                st.subheader("💡 Nhận xét chuyên gia")
                if prediction < 14:
                    st.success("Bạn có lượng mỡ cực thấp. Hãy giữ vững kỷ luật nạp đủ Protein.")
                elif prediction < 22:
                    st.info("Cơ thể ở mức lý tưởng để duy trì sức khỏe và thẩm mỹ lâu dài.")
                else:
                    st.warning(" Hãy bắt đầu hành trình giảm mỡ tăng cơ để thấy sự khác biệt.")
                
                st.markdown("""
                <div class='expert-note'>
                <b>Mẹo:</b> Kết quả AI là tham khảo. Hãy quan trọng <b>SỰ THAY ĐỔI</b> qua từng tuần khi đo cùng một thời điểm thay vì quá ám ảnh về con số tuyệt đối hôm nay.
                </div>
                """, unsafe_allow_html=True)
        else:
            st.info("Nhấn nút 'PHÂN TÍCH' để xem kết quả chi tiết.")

    with tab2:
        st.markdown("### 🖼️ Hình mẫu phân loại tỷ lệ mỡ thực tế")
        st.write("Sử dụng các hình ảnh này để đối chiếu trực quan với kết quả dự báo của AI.")
        
        # Chia làm 3 cột để hiển thị 3 ảnh nằm ngang
        col_img1, col_img2, col_img3 = st.columns([1, 1])
        
        with col_img1:
            st.image("anh_1.jpg", caption="Tham khảo 1", use_container_width=True)

        with col_img2:
            st.image("anh_2.jpg", caption="Tham khảo 2", use_container_width=True)

    st.divider()

# ---------------------------------------------------------
# TRANG THÔNG TIN (INFO)
# ---------------------------------------------------------
else:
    st.markdown("## 📋 Tại sao ThongThien Fitness AI vượt trội?")
    
    st.subheader("1. So sánh kỹ thuật đo lường")
    st.markdown("""
    | Đặc điểm | US Navy Formula (Web Online) | Hydrostatic (Cân thủy tĩnh) | **ThongThien AI (XGBoost)** |
    | :--- | :--- | :--- | :--- |
    | **Đầu vào** | 3 chỉ số (Cổ, Eo, Cao) | Tỉ trọng dưới nước | **13 chỉ số toàn diện** |
    | **Độ chính xác** | Thấp (Hay nhầm cơ bụng là mỡ) | Tiêu chuẩn vàng | **Tiệm cận tiêu chuẩn vàng** |
    | **Phân tích** | Công thức toán học cũ (1984) | Vật lý thực tế | **AI học máy Non-linear** |
    """)

    st.subheader("2. Dẫn chứng sức mạnh AI")
    st.write("""
    - **Vượt qua giới hạn BMI/US Navy:** US Navy chỉ nhìn vào vòng eo. AI của chúng tôi phân tích cả vòng bắp tay, ngực và đùi để nhận diện **khối lượng cơ**. Nếu bạn có bắp tay to, AI sẽ biết đó là cơ bắp chứ không phải mỡ.
    - **Thuật toán XGBoost (k=7):** Được huấn luyện trên hàng ngàn mẫu dữ liệu thực tế, xử lý các mối quan hệ phi tuyến tính phức tạp giữa các vòng cơ thể mà công thức truyền thống không làm được.
    - **Định hướng chuyên sâu:** Ứng dụng này sinh ra để phục vụ người tập Gym/Fitness - nơi mà cân nặng không nói lên tất cả.
    """)
    
    if st.button("⬅️ Quay lại trang chính"):
        st.session_state.page = 'home'
        st.rerun()

st.markdown("<br><p style='text-align: center; color: #4B5563;'>© 2025 ThongThien Fitness - Advanced AI Body Analysis</p>", unsafe_allow_html=True)