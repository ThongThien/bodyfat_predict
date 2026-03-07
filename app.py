import streamlit as st
import joblib
import pandas as pd
import uuid
import random
import streamlit.components.v1 as components

# =========================================================
# 1. CẤU HÌNH & CSS (DARK MODE ELITE)
# =========================================================

st.set_page_config(page_title="ThongThien Fitness AI - Elite", layout="wide", page_icon="⚡")

st.markdown("""
<style>

.stApp {
    background-color: #0E1117;
    color: #FFFFFF;
}

div[data-testid="stNumberInput"] {
    margin-bottom: -15px;
}

div[data-testid="stNumberInput"] label {
    font-size: 13px !important;
    color: #9CA3AF !important;
}

.result-box {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #3B82F6;
    text-align: center;
    margin-bottom: 20px;
}

.big-value {
    font-size: 60px !important;
    font-weight: 900;
    color: #00FF00;
}

.metric-item {
    background: rgba(255,255,255,0.05);
    padding: 12px;
    border-radius: 10px;
    margin-bottom: 8px;
    border: 1px solid #334155;
}

.metric-label {
    color: #FFFFFF;
    font-size: 13px;
}

.metric-val {
    float: right;
    font-weight: bold;
}

.expert-note {
    background-color: rgba(59,130,246,0.1);
    border-left: 4px solid #3B82F6;
    padding: 15px;
    border-radius: 4px;
    margin-top: 10px;
}

.info-card {
    background: #1E293B;
    padding: 20px;
    border-radius: 10px;
    border-left: 5px solid #3B82F6;
    margin-bottom: 20px;
}

</style>
""", unsafe_allow_html=True)

# =========================================================
# 2. DỮ LIỆU MẪU
# =========================================================

PRESETS = {

"Chỉ số của tôi":[22,64,163,92,86,88,51,36],

"Vận động viên":[25,70,175,100,72,90,55,40],

"Gym Lean":[22,62,163,90,78,88,52,36],

"Người bình thường":[30,80,175,95,88,95,53,33],

"Người thừa mỡ":[35,95,175,105,105,110,62,31]

}

if 'page' not in st.session_state:
    st.session_state.page='home'

if 'vals' not in st.session_state:
    st.session_state.vals=PRESETS["Chỉ số của tôi"]

# =========================================================
# 3. LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():

    try:
        return joblib.load("xgboost_bodyfat_model_k8.pkl")

    except:
        return None

model=load_model()

# =========================================================
# 4. ICON BODY SVG (GIỮ NGUYÊN)
# =========================================================
def get_human_svg(bf):

    unique_id = str(uuid.uuid4())[:8]

    fill_h = max(0, min(100, (bf / 45) * 100))
    y_pos = 210 - (fill_h * 2.1)

    svg = f"""
    <div style="display:flex;justify-content:center;align-items:center;flex-direction:column;">
    <svg width="160" height="320" viewBox="0 0 100 220">

    <path d="M50,10 C55,10 60,15 60,20
    C60,25 55,30 50,30
    C45,30 40,25 40,20
    C40,15 45,10 50,10
    M40,32 L60,32 L65,80
    L75,130 L70,210
    L55,210 L50,140
    L45,210 L30,210
    L25,130 L35,80 Z"
    fill="#2D3748"/>

    <defs>
    <clipPath id="cp_{unique_id}">
        <rect x="0" y="{y_pos}" width="100" height="210"/>
    </clipPath>
    </defs>

    <path d="M50,10 C55,10 60,15 60,20
    C60,25 55,30 50,30
    C45,30 40,25 40,20
    C40,15 45,10 50,10
    M40,32 L60,32 L65,80
    L75,130 L70,210
    L55,210 L50,140
    L45,210 L30,210
    L25,130 L35,80 Z"

    fill="#3B82F6"
    clip-path="url(#cp_{unique_id})"/>

    <path d="M50,10 C55,10 60,15 60,20
    C60,25 55,30 50,30
    C45,30 40,25 40,20
    C40,15 45,10 50,10
    M40,32 L60,32 L65,80
    L75,130 L70,210
    L55,210 L50,140
    L45,210 L30,210
    L25,130 L35,80 Z"

    fill="none"
    stroke="#4A5568"
    stroke-width="2"/>

    </svg>

    <p style="color:#3B82F6;font-weight:bold;margin-top:10px;font-size:18px;">
    {bf:.1f}% Body Fat
    </p>

    </div>
    """

    return svg
# =========================================================
# 5. HEADER
# =========================================================

col_h1,col_h2=st.columns([0.8,0.2])

with col_h1:

    st.markdown(
    "<h1 style='color:#3B82F6;'>THONGTHIEN FITNESS AI</h1>",
    unsafe_allow_html=True
    )

    st.caption("Công nghệ phân tích tỷ lệ mỡ cơ thể bằng Machine Learning")

with col_h2:

    if st.button("ℹ️ THÔNG TIN KHOA HỌC",use_container_width=True):

        st.session_state.page='info' if st.session_state.page=='home' else 'home'

        st.rerun()

# =========================================================
# HOME PAGE
# =========================================================

if st.session_state.page=='home':

    st.markdown("### 📏 Nhập chỉ số cơ thể")

    choice=st.selectbox("Chọn mẫu:",list(PRESETS.keys()))

    if st.button("ÁP DỤNG MẪU"):

        st.session_state.vals=PRESETS[choice]

        st.rerun()

    v=st.session_state.vals

    c1,c2,c3,c4=st.columns(4)

    with c1:
        age=st.number_input("Tuổi",1,100,int(v[0]))

    with c2:
        weight=st.number_input("Nặng (kg)",30.0,200.0,float(v[1]))

    with c3:
        height=st.number_input("Cao (cm)",100.0,250.0,float(v[2]))

    with c4:
        chest=st.number_input("Ngực (cm)",50.0,150.0,float(v[3]))

    c5,c6,c7,c8=st.columns(4)

    with c5:
        abdomen=st.number_input("Bụng (cm)",40.0,150.0,float(v[4]))

    with c6:
        hip=st.number_input("Mông (cm)",50.0,150.0,float(v[5]))

    with c7:
        thigh=st.number_input("Đùi (cm)",30.0,100.0,float(v[6]))

    with c8:
        biceps=st.number_input("Bắp tay (cm)",15.0,60.0,float(v[7]))

    btn=st.button("🔥 PHÂN TÍCH BODY FAT",type="primary",use_container_width=True)

# =========================================================
# RESULT TABS
# =========================================================

    tab1,tab2=st.tabs(["📊 Kết quả phân tích","📷 Hình mẫu tham khảo"])

    with tab1:

        if btn:

            if model is None:

                st.error("Không tìm thấy model .pkl")

            else:

                input_df=pd.DataFrame([[

                age,weight,height,chest,abdomen,hip,thigh,biceps

                ]],

                columns=[

                "Age","Weight","Height","Chest","Abdomen","Hip","Thigh","Biceps"

                ])

                prediction=float(model.predict(input_df)[0])

                bmi=weight/((height/100)**2)

                fat_kg=(prediction/100)*weight

                lbm=weight-fat_kg

                st.divider()

                res_c1,res_c2,res_c3=st.columns([0.8,1.2,1.2])

                with res_c1:

                    components.html(get_human_svg(prediction), height=340)

                with res_c2:

                    st.markdown(f"""
                    <div class='result-box'>
                    <p class='big-value'>{prediction:.1f}%</p>
                    </div>
                    """,unsafe_allow_html=True)

                    st.markdown(
                    f"<div class='metric-item'>Fat Mass: {fat_kg:.1f} kg</div>",
                    unsafe_allow_html=True)

                    st.markdown(
                    f"<div class='metric-item'>Lean Mass: {lbm:.1f} kg</div>",
                    unsafe_allow_html=True)

                    st.markdown(
                    f"<div class='metric-item'>BMI: {bmi:.1f}</div>",
                    unsafe_allow_html=True)

                with res_c3:

                    st.subheader("💡 Đánh giá AI")

                    if prediction<8:
                        status="VĐV thi đấu"

                    elif prediction<15:
                        status="Lean / Fitness"

                    elif prediction<22:
                        status="Bình thường"

                    else:
                        status="Thừa mỡ"

                    st.success(status)

                    st.markdown("""
                    <div class='expert-note'>
                    Tip: Tập trung phát triển cơ bụng (abs hypertrophy) để tăng độ nét hình thể.
                    </div>
                    """,unsafe_allow_html=True)

        else:

            st.info("Nhập chỉ số và bấm phân tích.")

    with tab2:

        st.markdown("### 🖼️ Thang Body Fat")

        col_img1,col_img2=st.columns(2)

        with col_img1:

            st.image("anh_1.jpg",use_container_width=True)

        with col_img2:

            st.image("anh_2.jpg",use_container_width=True)

# =========================================================
# 5. TRANG THÔNG TIN (INFO) - NÂNG CẤP CHUYÊN SÂU
# =========================================================

else:

    st.markdown("## 📋 Giải mã Thuật toán & Khoa học")

    col_info1, col_info2 = st.columns(2)

    with col_info1:

        st.markdown("""
        <div class='info-card'>
        <h4>1. Tại sao sử dụng Machine Learning?</h4>

        Các công thức truyền thống (như US Navy hoặc BMI) thường chỉ sử dụng
        2–3 chỉ số cơ thể. Điều này dẫn đến sai số cao đối với người tập Gym,
        vì cơ bắp phát triển có thể bị nhầm thành mỡ.

        <br><br>

        <b>Mô hình của hệ thống:</b>

        <ul>
        <li>Sử dụng thuật toán <b>XGBoost Gradient Boosting</b>.</li>
        <li>Phân tích đồng thời <b>8 chỉ số cơ thể</b>.</li>
        <li>Học mối tương quan phi tuyến giữa các nhóm cơ và phân bố mỡ.</li>
        </ul>

        Điều này giúp mô hình thích nghi tốt hơn với nhiều kiểu hình thể khác nhau.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
        <h4>2. Cơ chế học của XGBoost</h4>

        XGBoost hoạt động bằng cách kết hợp nhiều <b>Decision Tree</b> nhỏ
        để xây dựng một mô hình mạnh hơn.

        <br><br>

        Quy trình:

        <ol>
        <li>Mỗi cây học một phần sai số của cây trước.</li>
        <li>Các cây được cộng lại để tạo thành mô hình cuối.</li>
        <li>Thuật toán tối ưu để giảm overfitting và tăng độ chính xác.</li>
        </ol>

        Phương pháp này thường đạt độ chính xác cao trong các bài toán
        dự đoán sinh học và y học.
        </div>
        """, unsafe_allow_html=True)

    with col_info2:

        st.markdown("""
        <div class='info-card'>
        <h4>3. Ý nghĩa các chỉ số đo</h4>

        <ul>

        <li><b>Weight & Height:</b> xác định BMI và khối lượng cơ thể tổng.</li>

        <li><b>Chest:</b> thể hiện phát triển cơ ngực và lưng trên.</li>

        <li><b>Abdomen:</b> chỉ số quan trọng nhất liên quan đến mỡ nội tạng.</li>

        <li><b>Hip:</b> vùng tích mỡ chính của cơ thể.</li>

        <li><b>Thigh:</b> phản ánh sự phát triển của cơ đùi.</li>

        <li><b>Biceps:</b> chỉ báo mức độ phát triển cơ bắp tay.</li>

        </ul>

        Các thông số này kết hợp giúp AI phân tích phân bố mỡ toàn cơ thể.
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
        <h4>4. Lưu ý để đo chính xác</h4>

        <ol>

        <li><b>Đo vào buổi sáng:</b> khi cơ thể chưa ăn và chưa tập luyện.</li>

        <li><b>Dùng thước dây mềm:</b> đo sát da nhưng không siết quá chặt.</li>

        <li><b>Giữ thước ngang:</b> luôn song song với mặt đất.</li>

        </ol>

        Sai số đo lường có thể ảnh hưởng trực tiếp đến kết quả dự đoán.
        </div>
        """, unsafe_allow_html=True)

    if st.button("⬅️ QUAY LẠI MÁY TÍNH"):

        st.session_state.page = 'home'
        st.rerun()

    st.markdown(
        "<br><p style='text-align: center; color: #4B5563; font-size: 12px;'>ThongThien Fitness AI © 2026 | Machine Learning for Body Composition Analysis</p>",
        unsafe_allow_html=True
    )