import streamlit as st
import joblib
import pandas as pd
import uuid
import statistics

# =========================================================
# 1. CẤU HÌNH & CSS (DARK MODE ELITE)
# =========================================================
st.set_page_config(page_title="ThongThien Fitness AI - Elite", layout="wide", page_icon="⚡")

st.markdown("""
    <style>
    .stApp { background-color: #0E1117; color: #FFFFFF; }
    
    /* Gọn nhẹ hóa các ô input */
    div[data-testid="stNumberInput"] { margin-bottom: -15px; }
    div[data-testid="stNumberInput"] label { font-size: 13px !important; color: #9CA3AF !important; }
    
    /* Khung kết quả nổi bật */
    .result-box {
        background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
        padding: 25px;
        border-radius: 15px;
        border: 1px solid #3B82F6;
        text-align: center;
        margin-bottom: 20px;
    }
    .big-value { font-size: 60px !important; font-weight: 900; color: #00FF00; margin: 0; line-height: 1; }
    .range-label { font-size: 11px; color: #9CA3AF; margin-top: 10px; text-transform: uppercase; letter-spacing: 1px; }
    .range-text { font-size: 18px; font-weight: bold; color: #3B82F6; }
    
    .metric-item { background: rgba(255,255,255,0.05); padding: 12px; border-radius: 10px; margin-bottom: 8px; border: 1px solid #334155; }
    .metric-label { color: #FFFFFF; font-size: 13px; }
    .metric-val { float: right; font-weight: bold; color: #F3F4F6; }

    .expert-note {
        background-color: rgba(59, 130, 246, 0.1);
        border-left: 4px solid #3B82F6;
        padding: 15px;
        font-size: 14px;
        border-radius: 4px;
        margin-top: 10px;
    }
    
    /* Info Page Styling */
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
# 2. DỮ LIỆU MẪU & HẰNG SỐ SURVEY (ĐÃ TINH CHỈNH)
# =========================================================
PRESETS = {
    "Chỉ số của tôi": [22, 64, 163, 36, 90, 82, 88, 52, 34, 21, 36, 28, 16],
    "Vận động viên (Cực nét)": [25, 70, 175, 37, 100, 72, 90, 55, 36, 22, 40, 31, 17],
    "Người tập Gym (Săn chắc)": [22, 62, 163, 36, 90, 78, 88, 52, 34, 21, 36, 28, 16],
    "Người bình thường": [30, 80, 175, 39, 95, 88, 95, 53, 38, 23, 33, 28, 18],
    "Người thừa mỡ": [35, 95, 175, 41, 105, 105, 110, 62, 42, 25, 31, 26, 19]
}

SURVEY_RANGES = {
    "6 múi rõ (không gồng)": (6, 10), "4 múi trên": (11, 13), "Bụng phẳng": (14, 18), "Có nếp gấp": (19, 23), "Bụng tròn": (24, 35),
    "Cơ rất nét": (7, 11), "Cơ thấy nhưng mờ": (12, 16), "Cơ liền khối": (17, 23),
    "Gân máu nhiều": (7, 11), "Gân vừa": (12, 16), "Không thấy gân": (19, 31),
    "Da rất mỏng": (5, 9), "Da trung bình": (10, 20), "Da dày": (21, 36),
    "Đùi tách rõ": (8, 12), "Đùi có cơ": (13, 18), "Đùi trơn": (20, 30),
    "Mông cắt rõ": (8, 12), "Mông tròn": (13, 18), "Mông tích mỡ": (20, 35),
    "Đùi sau có rãnh": (10, 14), "Đùi sau phẳng": (18, 30),
}

if 'page' not in st.session_state: st.session_state.page = 'home'
if 'vals' not in st.session_state: st.session_state.vals = PRESETS["Chỉ số của tôi"]

# =========================================================
# 3. HÀM BỔ TRỢ & LOGIC AI (HYBRID WEIGHTING)
# =========================================================
@st.cache_resource
def load_model():
    try:
        # Tự động thử cả 2 tên file để tránh lỗi
        return joblib.load('tuned_xgboost_k7.pkl')
    except:
        try: return joblib.load('tuned_xgboost_k7_final.pkl')
        except: return None

model = load_model()

def adjust_with_survey(xgb_pred, survey_answers):
    lows, highs = [], []
    for ans in survey_answers:
        if ans in SURVEY_RANGES:
            lo, hi = SURVEY_RANGES[ans]
            lows.append(lo)
            highs.append(hi)

    survey_low = statistics.median(lows)
    survey_high = statistics.median(highs)
    survey_mid = (survey_low + survey_high) / 2

    adjusted = False
    final_pred = xgb_pred

    # Logic Hybrid: Nếu AI lệch khỏi vùng Survey, ưu tiên Survey 70%
    if xgb_pred < survey_low or xgb_pred > survey_high:
        final_pred = (xgb_pred * 0.4) + (survey_mid * 0.6)
        adjusted = True
    
    return final_pred, adjusted, survey_low, survey_high

def get_human_svg(bf):
    unique_id = str(uuid.uuid4())[:8]
    fill_h = max(0, min(100, (bf / 45) * 100))
    y_pos = 210 - (fill_h * 2.1)
    
    svg = f"""
    <div style="display: flex; justify-content: center; align-items: center; flex-direction: column;">
        <svg width="160" height="320" viewBox="0 0 100 220">
            <path d="M50,10 C55,10 60,15 60,20 C60,25 55,30 50,30 C45,30 40,25 40,20 C40,15 45,10 50,10 M40,32 L60,32 L65,80 L75,130 L70,210 L55,210 L50,140 L45,210 L30,210 L25,130 L35,80 Z" fill="#2D3748" />
            <defs><clipPath id="cp_{unique_id}"><rect x="0" y="{y_pos}" width="100" height="210" /></clipPath></defs>
            <path d="M50,10 C55,10 60,15 60,20 C60,25 55,30 50,30 C45,30 40,25 40,20 C40,15 45,10 50,10 M40,32 L60,32 L65,80 L75,130 L70,210 L55,210 L50,140 L45,210 L30,210 L25,130 L35,80 Z" fill="#3B82F6" clip-path="url(#cp_{unique_id})" />
            <path d="M50,10 C55,10 60,15 60,20 C60,25 55,30 50,30 C45,30 40,25 40,20 C40,15 45,10 50,10 M40,32 L60,32 L65,80 L75,130 L70,210 L55,210 L50,140 L45,210 L30,210 L25,130 L35,80 Z" fill="none" stroke="#4A5568" stroke-width="2" />
        </svg>
        <p style="color: #3B82F6; font-weight: bold; margin-top: 10px; font-size:18px;">{bf:.1f}% Body Fat</p>
    </div>
    """
    return svg

# =========================================================
# 4. GIAO DIỆN CHÍNH (MAIN UI)
# =========================================================
col_h1, col_h2 = st.columns([0.8, 0.2])
with col_h1:
    st.markdown("<h1 style='color:#3B82F6; margin:0;'>THONGTHIEN FITNESS AI <span style='font-size:16px; color:#9CA3AF;'>v2.1 Hybrid</span></h1>", unsafe_allow_html=True)
    st.caption("Công nghệ phân tích tỷ lệ mỡ đa tầng (XGBoost + Visual Appraisal)")
with col_h2:
    if st.button("ℹ️ THÔNG TIN KHOA HỌC", use_container_width=True):
        st.session_state.page = 'info' if st.session_state.page == 'home' else 'home'
        st.rerun()

if st.session_state.page == 'home':
    # --- PHẦN 1: NHẬP CHỈ SỐ ---
    st.markdown("### 📏 1. Chỉ số đo lường thực tế")
    choice = st.selectbox("Chọn nhanh tạng người mẫu:", list(PRESETS.keys()))
    if st.button("ÁP DỤNG MẪU"):
        st.session_state.vals = PRESETS[choice]
        st.rerun()

    v = st.session_state.vals
    c1, c2, c3, c4, c5 = st.columns(5)
    with c1:
        age = st.number_input("Tuổi", 1, 100, int(v[0]))
        weight = st.number_input("Nặng (kg)", 30.0, 200.0, float(v[1]))
        height = st.number_input("Cao (cm)", 100.0, 250.0, float(v[2]))
    with c2:
        neck = st.number_input("Cổ (cm)", 20.0, 60.0, float(v[3]))
        chest = st.number_input("Ngực (cm)", 50.0, 150.0, float(v[4]))
        abdomen = st.number_input("Bụng (cm)", 40.0, 150.0, float(v[5]))
    with c3:
        hip = st.number_input("Mông (cm)", 50.0, 150.0, float(v[6]))
        thigh = st.number_input("Đùi (cm)", 30.0, 100.0, float(v[7]))
        knee = st.number_input("Gối (cm)", 20.0, 60.0, float(v[8]))
    with c4:
        ankle = st.number_input("Cổ chân (cm)", 10.0, 40.0, float(v[9]))
        bicep = st.number_input("Bắp tay (cm)", 15.0, 60.0, float(v[10]))
        forearm = st.number_input("Cẳng tay (cm)", 10.0, 50.0, float(v[11]))
    with c5:
        wrist = st.number_input("Cổ tay (cm)", 10.0, 30.0, float(v[12]))

    # --- PHẦN 2: SURVEY HÌNH THỂ ---
    st.divider()
    st.markdown("### 👁️ 2. Thẩm định hình thể trực quan")
    st.caption("Hãy chọn trạng thái gần nhất với cơ thể bạn hiện tại (khi gồng nhẹ)")
    sc1, sc2, sc3 = st.columns(3)
    with sc1:
        s1 = st.selectbox("Cơ bụng:", ["6 múi rõ (không gồng)", "4 múi trên", "Bụng phẳng", "Có nếp gấp", "Bụng tròn"])
        s2 = st.selectbox("Độ nét cơ:", ["Cơ rất nét", "Cơ thấy nhưng mờ", "Cơ liền khối"])
    with sc2:
        s3 = st.selectbox("Gân máu:", ["Gân máu nhiều", "Gân vừa", "Không thấy gân"])
        s4 = st.selectbox("Độ dày da:", ["Da rất mỏng", "Da trung bình", "Da dày"])
        s5 = st.selectbox("Đùi trước:", ["Đùi tách rõ", "Đùi có cơ", "Đùi trơn"])
    with sc3:
        s6 = st.selectbox("Mông:", ["Mông cắt rõ", "Mông tròn", "Mông tích mỡ"])
        s7 = st.selectbox("Đùi sau:", ["Đùi sau có rãnh", "Đùi sau phẳng"])

    btn = st.button("🔥 BẮT ĐẦU PHÂN TÍCH TỔNG HỢP", type="primary", use_container_width=True)

    # --- PHẦN 3: KẾT QUẢ ---
    tab1, tab2 = st.tabs(["📊 Kết quả phân tích", "📷 Hình mẫu tham khảo"])

    with tab1:
        if btn:
            if model is None:
                st.error("Lỗi: Không thể tải mô hình trí tuệ nhân tạo (.pkl). Vui lòng kiểm tra file nguồn.")
            else:
                input_df = pd.DataFrame([[age, weight, height, neck, chest, abdomen, hip, thigh, knee, ankle, bicep, forearm, wrist]],
                                       columns=['Age', 'Weight', 'Height', 'Neck', 'Chest', 'Abdomen', 'Hip', 'Thigh', 'Knee', 'Ankle', 'Biceps', 'Forearm', 'Wrist'])
                xgb_prediction = model.predict(input_df)[0]
                
                survey_ans = [s1, s2, s3, s4, s5, s6, s7]
                final_pred, is_adj, s_low, s_high = adjust_with_survey(xgb_prediction, survey_ans)

                bmi = weight / ((height/100)**2)
                fat_kg = (final_pred / 100) * weight
                lbm = weight - fat_kg

                st.divider()
                res_c1, res_c2, res_c3 = st.columns([0.8, 1.2, 1.2])

                with res_c1:
                    st.markdown(get_human_svg(final_pred), unsafe_allow_html=True)

                with res_c2:
                    st.markdown(f"""
                        <div class='result-box'>
                            <p class='range-label'>TỶ LỆ MỠ CUỐI CÙNG</p>
                            <p class='big-value'>{final_pred:.1f}%</p>
                            <p class='range-label'>Khoảng quan sát thị giác</p>
                            <p class='range-text'>{s_low:.1f}% — {s_high:.1f}%</p>
                        </div>
                    """, unsafe_allow_html=True)
                    
                    st.markdown(f"<div class='metric-item'><span class='metric-label'>Khối lượng mỡ:</span><span class='metric-val'>{fat_kg:.1f} kg</span></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-item'><span class='metric-label'>Cân nặng nạc (LBM):</span><span class='metric-val'>{lbm:.1f} kg</span></div>", unsafe_allow_html=True)
                    st.markdown(f"<div class='metric-item'><span class='metric-label'>Chỉ số BMI:</span><span class='metric-val'>{bmi:.1f}</span></div>", unsafe_allow_html=True)

                with res_c3:
                    st.subheader("💡 Đánh giá của AI")
                    if is_adj:
                        st.warning(f"**Hiệu chỉnh:** AI tính toán {xgb_prediction:.1f}%, nhưng quan sát hình thể cho thấy mức {s_low}-{s_high}%. Kết quả đã được cân bằng lại để sát thực tế nhất.")
                    else:
                        st.success("✅ **Độ chính xác cao:** Các chỉ số đo lường và quan sát hình thể của bạn rất đồng nhất.")
                    
                    # Phân loại
                    if final_pred < 8: status, desc = "VĐV Thi đấu", "Mức mỡ cực thấp, chỉ dành cho thi đấu ngắn hạn."
                    elif final_pred < 15: status, desc = "Săn chắc (Lean)", "Lý tưởng cho thẩm mỹ và hiệu suất tập luyện."
                    elif final_pred < 22: status, desc = "Bình thường", "Sức khỏe tốt nhưng độ nét cơ chưa cao."
                    else: status, desc = "Thừa mỡ", "Cần tập trung vào thâm hụt calo và Cardio."
                    
                    st.markdown(f"**Trạng thái:** `{status}`")
                    st.caption(desc)

                    st.markdown(f"""
                    <div class='expert-note'>
                    <b>Lời khuyên Elite:</b> Để đạt độ nét cao hơn, hãy tập trung vào độ dày của cơ bụng (Abs hypertrophy) thay vì chỉ giảm mỡ. Điều này giúp cơ bụng "xuyên thấu" lớp mỡ dày hơn.
                    </div>
                    """, unsafe_allow_html=True)
        else:
            st.info("Nhập chỉ số và thực hiện Survey để xem kết quả phân tích.")

    with tab2:
        st.markdown("### 🖼️ Thang tham chiếu Body Fat")
        col_img1, col_img2 = st.columns(2)
        with col_img1: 
            st.image("anh_1.jpg", caption="Thang đo phổ thông", use_container_width=True)
            st.caption("Lưu ý: Cùng một tỷ lệ mỡ nhưng người có nhiều cơ bắp trông sẽ sắc nét hơn.")
        with col_img2: 
            st.image("anh_2.jpg", caption="Thang đo chuyên sâu (Athlete)", use_container_width=True)

# =========================================================
# 5. TRANG THÔNG TIN (INFO) - NÂNG CẤP CHUYÊN SÂU
# =========================================================
else:
    st.markdown("## 📋 Giải mã Thuật toán & Khoa học")
    
    col_info1, col_info2 = st.columns(2)
    
    with col_info1:
        st.markdown("""
        <div class='info-card'>
        <h4>1. Tại sao cần XGBoost AI?</h4>
        Các công thức truyền thống (như US Navy) chỉ dùng 2-3 chỉ số, dẫn đến sai số rất lớn cho người tập Gym (vì cơ cổ hoặc mông lớn thường bị nhầm là mỡ).<br><br>
        <b>Mô hình của chúng tôi:</b>
        <ul>
            <li>Sử dụng thuật toán <b>Gradient Boosting</b> xử lý 13 biến số đồng thời.</li>
            <li>Phân tích sự tương quan giữa các nhóm cơ (Bắp tay vs Cổ tay, Bụng vs Ngực).</li>
            <li>Giảm thiểu sai số do cấu trúc xương (với thông số Gối, Cổ chân, Cổ tay).</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
        <h4>2. Cơ chế "Hybrid Adjustment"</h4>
        AI có thể "mù" trước chất lượng cơ bắp. Đó là lý do Survey Visual ra đời:<br><br>
        Final = (AI x 0.3) + (Survey x 0.7)<br><br>
        Trọng số này được áp dụng khi có sự lệch biệt lớn (Outlier), giúp kết quả không bị lố khi bạn có khung xương to nhưng mỡ cực thấp hoặc ngược lại.
        </div>
        """, unsafe_allow_html=True)

    with col_info2:
        st.markdown("""
        <div class='info-card'>
        <h4>3. Ý nghĩa các chỉ số đo</h4>
        <ul>
            <li><b>Cổ tay/Gối/Cổ chân:</b> Điểm mốc khung xương (Bone Structure). Giúp AI biết bạn là tạng người xương to hay nhỏ.</li>
            <li><b>Cẳng tay/Bắp tay:</b> Chỉ số phát triển cơ bắp.</li>
            <li><b>Bụng/Mông:</b> Các kho mỡ chính của cơ thể.</li>
        </ul>
        </div>
        """, unsafe_allow_html=True)

        st.markdown("""
        <div class='info-card'>
        <h4>4. Lưu ý để có kết quả chính xác nhất</h4>
        <ol>
            <li><b>Đo vào buổi sáng:</b> Khi bụng rỗng và chưa tập luyện (cơ chưa bị pump máu).</li>
            <li><b>Thước dây:</b> Luôn giữ thước song song với mặt đất, áp sát da nhưng không thắt chặt làm lún da.</li>
            <li><b>Survey:</b> Hãy thành thật. Nếu bạn thấy mờ, đừng chọn "rất nét".</li>
        </ol>
        </div>
        """, unsafe_allow_html=True)

    if st.button("⬅️ QUAY LẠI MÁY TÍNH"):
        st.session_state.page = 'home'
        st.rerun()

st.markdown("<br><p style='text-align: center; color: #4B5563; font-size: 12px;'>Elite Hybrid Analysis Engine © 2025 | Tối ưu cho cộng đồng Fitness Việt Nam</p>", unsafe_allow_html=True)