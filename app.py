import streamlit as st
import joblib
import pandas as pd
import uuid

# =========================================================
# 1. PAGE CONFIG
# =========================================================

st.set_page_config(
    page_title="Body Fat Prediction AI",
    layout="wide",
    page_icon="⚡"
)

# =========================================================
# 2. DARK MODE CSS
# =========================================================

st.markdown("""
<style>
.stApp {
    background-color: #0E1117;
    color: white;
}

.result-box {
    background: linear-gradient(135deg, #1E293B 0%, #0F172A 100%);
    padding: 25px;
    border-radius: 15px;
    border: 1px solid #3B82F6;
    text-align: center;
}

.big-value {
    font-size: 60px;
    font-weight: 900;
    color: #00FF00;
}

.metric-item {
    background: rgba(255,255,255,0.05);
    padding: 10px;
    border-radius: 10px;
    margin-bottom: 8px;
}
</style>
""", unsafe_allow_html=True)

# =========================================================
# 3. LOAD MODEL
# =========================================================

@st.cache_resource
def load_model():
    try:
        model = joblib.load("xgboost_bodyfat_model_k8.pkl")
        return model
    except:
        return None

model = load_model()

# =========================================================
# 4. SVG BODY VISUALIZATION
# =========================================================

def get_human_svg(bf):

    unique_id = str(uuid.uuid4())[:8]

    fill_h = max(0, min(100, (bf / 45) * 100))
    y_pos = 210 - (fill_h * 2.1)

    svg = f"""
    <div style="display:flex;justify-content:center;">
    <svg width="160" height="320" viewBox="0 0 100 220">

    <path d="M50,10 C55,10 60,15 60,20 C60,25 55,30 50,30
    C45,30 40,25 40,20 C40,15 45,10 50,10
    M40,32 L60,32 L65,80 L75,130
    L70,210 L55,210 L50,140
    L45,210 L30,210 L25,130 L35,80 Z"
    fill="#2D3748"/>

    <defs>
    <clipPath id="cp_{unique_id}">
    <rect x="0" y="{y_pos}" width="100" height="210"/>
    </clipPath>
    </defs>

    <path d="M50,10 C55,10 60,15 60,20 C60,25 55,30 50,30
    C45,30 40,25 40,20 C40,15 45,10 50,10
    M40,32 L60,32 L65,80 L75,130
    L70,210 L55,210 L50,140
    L45,210 L30,210 L25,130 L35,80 Z"
    fill="#3B82F6"
    clip-path="url(#cp_{unique_id})"/>

    </svg>
    </div>
    """

    return svg

# =========================================================
# 5. HEADER
# =========================================================

st.title("⚡ Body Fat Prediction AI")
st.caption("Machine Learning based Body Fat Estimation using Anthropometric Data")

# =========================================================
# 6. INPUT SECTION
# =========================================================

st.subheader("📏 Enter Body Measurements")

col1, col2, col3, col4 = st.columns(4)

with col1:
    age = st.number_input("Age", 10, 80, 22)

with col2:
    weight = st.number_input("Weight (kg)", 30.0, 200.0, 70.0)

with col3:
    height = st.number_input("Height (cm)", 120.0, 220.0, 170.0)

with col4:
    chest = st.number_input("Chest (cm)", 60.0, 150.0, 95.0)

col5, col6, col7, col8 = st.columns(4)

with col5:
    abdomen = st.number_input("Abdomen (cm)", 50.0, 150.0, 85.0)

with col6:
    hip = st.number_input("Hip (cm)", 60.0, 150.0, 95.0)

with col7:
    thigh = st.number_input("Thigh (cm)", 30.0, 100.0, 55.0)

with col8:
    biceps = st.number_input("Biceps (cm)", 15.0, 60.0, 32.0)

# =========================================================
# 7. PREDICT BUTTON
# =========================================================

predict_btn = st.button("🔥 Predict Body Fat", use_container_width=True)

# =========================================================
# 8. PREDICTION
# =========================================================

if predict_btn:

    if model is None:
        st.error("❌ Model not found. Please check .pkl file.")
    else:

        input_df = pd.DataFrame(
            [[age, weight, height, chest, abdomen, hip, thigh, biceps]],
            columns=[
                "Age",
                "Weight",
                "Height",
                "Chest",
                "Abdomen",
                "Hip",
                "Thigh",
                "Biceps"
            ]
        )

        prediction = model.predict(input_df)[0]

        # ======================
        # BODY METRICS
        # ======================

        bmi = weight / ((height/100)**2)

        fat_mass = weight * prediction / 100

        lean_mass = weight - fat_mass

        st.divider()

        colA, colB, colC = st.columns([1,1,1])

        with colA:

            st.markdown(get_human_svg(prediction), unsafe_allow_html=True)

        with colB:

            st.markdown(f"""
            <div class="result-box">
            <p>Body Fat Percentage</p>
            <p class="big-value">{prediction:.2f}%</p>
            </div>
            """, unsafe_allow_html=True)

        with colC:

            st.markdown(f"""
            <div class="metric-item">BMI: {bmi:.2f}</div>
            <div class="metric-item">Fat Mass: {fat_mass:.2f} kg</div>
            <div class="metric-item">Lean Mass: {lean_mass:.2f} kg</div>
            """, unsafe_allow_html=True)

        # ======================
        # BODY CATEGORY
        # ======================

        if prediction < 8:
            status = "Athlete Level"
        elif prediction < 15:
            status = "Lean / Fitness"
        elif prediction < 22:
            status = "Normal"
        else:
            status = "Overfat"

        st.success(f"Body Type: {status}")

# =========================================================
# FOOTER
# =========================================================

st.markdown("""
<br>
<center>
Body Fat Prediction System | AI Graduation Project
</center>
""", unsafe_allow_html=True)