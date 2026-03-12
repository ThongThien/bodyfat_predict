import streamlit as st
import streamlit.components.v1 as components
from core.predictor import load_model, predict_body_fat
from core.visualizer import get_human_svg, get_custom_css
from core.info_content import show_info_page

# 1. CẤU HÌNH & CSS
st.set_page_config(page_title="ThongThien Fitness AI - Elite", layout="wide", page_icon="⚡")
st.markdown(get_custom_css(), unsafe_allow_html=True)

# 2. DATA PRESETS (Bạn có thể tách tiếp ra file JSON nếu muốn)
PRESETS = {
    "Chỉ số của tôi":[22,64,163,92,86,88,51,36],
    "Vận động viên":[25,70,175,100,72,90,55,40],
    "Gym Lean":[22,62,163,90,78,88,52,36],
    "Người bình thường":[30,80,175,95,88,95,53,33],
    "Người thừa mỡ":[35,95,175,105,105,110,62,31]
}

# 3. INITIALIZATION
if 'page' not in st.session_state: st.session_state.page='home'
if 'vals' not in st.session_state: st.session_state.vals=PRESETS["Chỉ số của tôi"]

model = load_model("models/xgboost_bodyfat_model_k8.pkl")

# 4. HEADER UI
col_h1, col_h2 = st.columns([0.8, 0.2])
with col_h1:
    st.markdown("<h1 style='color:#3B82F6;'>THONGTHIEN FITNESS AI</h1>", unsafe_allow_html=True)
    st.caption("Công nghệ phân tích tỷ lệ mỡ cơ thể bằng Machine Learning")
with col_h2:
    if st.button("ℹ️ THÔNG TIN KHOA HỌC", use_container_width=True):
        st.session_state.page = 'info' if st.session_state.page == 'home' else 'home'
        st.rerun()

# 5. PAGE ROUTING
if st.session_state.page == 'home':
    # --- UI PHẦN NHẬP LIỆU ---
    st.markdown("### 📏 Nhập chỉ số cơ thể")
    choice = st.selectbox("Chọn mẫu:", list(PRESETS.keys()))
    if st.button("ÁP DỤNG MẪU", key="btn_apply_preset"):
        st.session_state.vals = PRESETS[choice]
        st.rerun()

    v = st.session_state.vals
    c1, c2, c3, c4 = st.columns(4)
    age = c1.number_input("Tuổi", 1, 100, int(v[0]))
    weight = c2.number_input("Nặng (kg)", 30.0, 200.0, float(v[1]))
    height = c3.number_input("Cao (cm)", 100.0, 250.0, float(v[2]))
    chest = c4.number_input("Ngực (cm)", 50.0, 150.0, float(v[3]))

    c5, c6, c7, c8 = st.columns(4)
    abdomen = c5.number_input("Bụng (cm)", 40.0, 150.0, float(v[4]))
    hip = c6.number_input("Mông (cm)", 50.0, 150.0, float(v[5]))
    thigh = c7.number_input("Đùi (cm)", 30.0, 100.0, float(v[6]))
    biceps = c8.number_input("Bắp tay (cm)", 15.0, 60.0, float(v[7]))

    btn = st.button("🔥 PHÂN TÍCH BODY FAT", type="primary", use_container_width=True, key="btn_main_analyze")

    # --- KẾT QUẢ ---
    tab1, tab2 = st.tabs(["📊 Kết quả phân tích", "📷 Hình mẫu tham khảo"])
    with tab1:
        if btn:
            if model:
                data = {"Age":age, "Weight":weight, "Height":height, "Chest":chest, 
                        "Abdomen":abdomen, "Hip":hip, "Thigh":thigh, "Biceps":biceps}
                prediction = predict_body_fat(model, data)
                
                # Logic tính toán phụ
                bmi = weight / ((height/100)**2)
                fat_kg = (prediction/100) * weight
                lbm = weight - fat_kg

                res_c1, res_c2, res_c3 = st.columns([0.8, 1.2, 1.2])
                with res_c1:
                    components.html(get_human_svg(prediction), height=340)
                with res_c2:
                    st.markdown(f"<div class='result-box'><p class='big-value'>{prediction:.1f}%</p></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-item'>Fat Mass: {fat_kg:.1f} kg</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-item'>Lean Mass: {lbm:.1f} kg</div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-item'>BMI: {bmi:.1f}</div>", unsafe_allow_html=True)
                with res_c3:
                    st.subheader("💡 Đánh giá AI")
                    status = "VĐV thi đấu" if prediction < 8 else "Lean / Fitness" if prediction < 15 else "Bình thường" if prediction < 22 else "Thừa mỡ"
                    st.success(status)
                    st.markdown("<div class='expert-note'>Tip: Tập trung phát triển cơ bụng để tăng độ nét.</div>", unsafe_allow_html=True)
            else:
                st.error("Model chưa được load!")
    with tab2:
        st.image(["assets/anh_1.jpg", "assets/anh_2.jpg"], use_container_width=True)

else:
    # --- PAGE INFO --- (Giữ nguyên logic của bạn nhưng bọc trong hàm hoặc file riêng nếu cần)
    st.markdown("## 📋 Giải mã Thuật toán & Khoa học")
    # ... Copy phần code info của bạn vào đây ...
    if st.button("⬅️ QUAY LẠI MÁY TÍNH", key="btn_info_back"):
        st.session_state.page = 'home'
        st.rerun()

# 5. PAGE ROUTING (Phần cuối file app.py)
if st.session_state.page == 'home':
    # (Toàn bộ code trang Home đã viết ở phản hồi trước)
    pass 
else:
    # Gọi hàm từ module đã tách
    show_info_page()