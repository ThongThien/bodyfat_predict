import streamlit as st
import streamlit.components.v1 as components
import cv2
import numpy as np
import time
import os

# Core imports
from core.predictor import load_model, predict_body_fat
from core.visualizer import get_human_svg
from core.info_content import show_info_page
from core.cv_engine import process_body_measurements 

# --- 1. CONFIG & STYLE ---
st.set_page_config(page_title="Predict Body Fat Free", layout="wide")

st.markdown("""
    <style>
    .block-container {padding-top: 1rem;}
    .stButton>button {width: 100%; border-radius: 5px;}
    .copyright {position: fixed; bottom: 10px; left: 10px; font-size: 12px; color: gray; z-index: 100;}
    * {font-family: 'Segoe UI', sans-serif;}
    </style>
""", unsafe_allow_html=True)

# --- 2. SESSION STATE ---
if 'logged_in' not in st.session_state: st.session_state.logged_in = False
if 'vals' not in st.session_state: st.session_state.vals = [None]*7 # age, w, h, chest, abd, hip, thigh
if 'step' not in st.session_state: st.session_state.step = 1 
if 'is_editing' not in st.session_state: st.session_state.is_editing = False
if 'analysis_done' not in st.session_state: st.session_state.analysis_done = False
if 'prediction' not in st.session_state: st.session_state.prediction = 0.0
if 'pipeline_front' not in st.session_state: st.session_state.pipeline_front = None
if 'pipeline_side' not in st.session_state: st.session_state.pipeline_side = None

model = load_model("models/bodyfat_xgboost_model_final.pkl")

# --- 3. SIDEBAR MENU ---
with st.sidebar:
    st.title("Predict Body Fat Free")
    st.markdown("---")
    selection = st.radio("MENU", ["Measure Body Fat", "Scientific Info", "Body Fat Samples", "Your History", "Settings"])
    
    st.markdown("---")
    # Login/Logout
    if st.session_state.logged_in:
        st.write("Status: Logged In")
        if st.button("LOGOUT"):
            st.session_state.logged_in = False
            st.rerun()
    else:
        st.write("Status: Guest")
        if st.button("LOGIN"):
            st.session_state.logged_in = True
            st.rerun()

    st.markdown("---")
    # LOAD SAMPLE DATA with REAL CV PROCESSING
    if st.button("LOAD SAMPLE DATA"):
        with st.spinner('AI processing sample images...'):
            # Load sample files
            img_f = cv2.imread("assets/front_Lap.jpg")
            img_s = cv2.imread("assets/side_Lap.jpg")
            
            if img_f is not None and img_s is not None:
                # Set initial known metrics
                age_s, w_s, h_s = 22, 59.0, 169.0
                # Process via your CV engine
                res_cv, pipe_f, pipe_s = process_body_measurements(img_f, img_s, h_s, age_s, w_s)
                
                if res_cv:
                    st.session_state.vals = [age_s, w_s, h_s, res_cv['Chest'], res_cv['Abdomen'], res_cv['Hip'], res_cv['Thigh']]
                    st.session_state.pipeline_front = pipe_f
                    st.session_state.pipeline_side = pipe_s
                    st.session_state.step = 2
                    st.rerun()
            else:
                st.error("Sample images not found in assets folder.")
    
    if st.button("RESET ALL"):
        for key in list(st.session_state.keys()): del st.session_state[key]
        st.rerun()

# --- 4. MAIN CONTENT ---

if selection == "Measure Body Fat":
    is_locked = st.session_state.analysis_done
    
    col_inputs, col_display = st.columns([1, 1.2])
    
    with col_inputs:
        st.subheader("1. Fundamental Metrics & AI Scan")
        age = st.number_input("Age", 10, 100, st.session_state.vals[0], disabled=is_locked)
        weight = st.number_input("Weight (kg)", 30.0, 200.0, st.session_state.vals[1], disabled=is_locked)
        height = st.number_input("Height (cm)", 120.0, 230.0, st.session_state.vals[2], disabled=is_locked)
        up_f = st.file_uploader("Front View Image", type=['jpg', 'png'], disabled=is_locked)
        up_s = st.file_uploader("Side View Image", type=['jpg', 'png'], disabled=is_locked)
        is_loose_clothing = st.checkbox("Mặc quần đùi/đồ rộng", value=False, help="Tích chọn nếu bạn mặc quần đùi hoặc đồ không bó sát để AI trừ hao chính xác hơn.")
        if st.button("RUN AI SCAN", disabled=is_locked):
            if up_f and up_s and height:
                with st.spinner('AI Scanning Body...'):
                    # Convert uploaded files to OpenCV
                    file_bytes_f = np.asarray(bytearray(up_f.read()), dtype=np.uint8)
                    file_bytes_s = np.asarray(bytearray(up_s.read()), dtype=np.uint8)
                    img_f = cv2.imdecode(file_bytes_f, 1)
                    img_s = cv2.imdecode(file_bytes_s, 1)
                    
                    # Call your CV engine
                    res_cv, pipe_f, pipe_s = process_body_measurements(img_f, img_s, height, age, weight, is_loose_clothing)
                    
                    if res_cv:
                        st.session_state.vals = [age, weight, height, res_cv['Chest'], res_cv['Abdomen'], res_cv['Hip'], res_cv['Thigh']]
                        st.session_state.pipeline_front = pipe_f
                        st.session_state.pipeline_side = pipe_s
                        st.session_state.step = 2
                        st.rerun()
            else:
                st.warning("Please fill height and upload both photos.")
    with col_display:
        if st.session_state.pipeline_front is not None:
            st.write("AI ANALYSIS PIPELINE")
            tab1, tab2 = st.tabs(["FRONT VIEW SCAN", "SIDE VIEW SCAN"])
            with tab1:
                # Streamlit tự động hỗ trợ click để zoom (Fullscreen)
                st.image(st.session_state.pipeline_front, use_container_width=True, caption="Front: Original - Mask - Skeleton - Scan")
            
            with tab2:
                st.image(st.session_state.pipeline_side, use_container_width=True, caption="Side: Original - Mask - Skeleton - Scan")
        else:
            st.info("AI Analysis result will appear here after scanning.")

    # SECTION 2: CONFIRMATION
    if st.session_state.step >= 2:
        st.markdown("---")
        st.subheader("2. Confirm Girth Measurements (cm)")
        
        # Edit Toggle Logic
        btn_label = "Done" if st.session_state.is_editing else "Edit"
        if st.button(btn_label):
            st.session_state.is_editing = not st.session_state.is_editing
            st.rerun()

        input_disabled = not st.session_state.is_editing or is_locked
        v = st.session_state.vals
        
        g1, g2, g3, g4 = st.columns(4)
        c_v = g1.number_input("Chest", 40.0, 200.0, v[3], disabled=input_disabled)
        a_v = g2.number_input("Abdomen", 40.0, 200.0, v[4], disabled=input_disabled)
        h_v = g3.number_input("Hip", 40.0, 200.0, v[5], disabled=input_disabled)
        t_v = g4.number_input("Thigh", 20.0, 150.0, v[6], disabled=input_disabled)
        
        # Save values back to state if edited
        st.session_state.vals[3:7] = [c_v, a_v, h_v, t_v]

        if st.button("ANALYZE", type="primary", disabled=is_locked):
            with st.spinner('Calculating Body Fat...'):
                data = {"Age": age, "Weight": weight, "Height": height, 
                        "Chest": c_v, "Abdomen": a_v, "Hip": h_v, "Thigh": t_v}
                st.session_state.prediction = predict_body_fat(model, data)
                st.session_state.analysis_done = True
                st.session_state.is_editing = False
                st.rerun()

    # SECTION 3: RESULTS
    if st.session_state.analysis_done:
        st.markdown("---")
        st.subheader("3. Results")
        pred = st.session_state.prediction
        r1, r2 = st.columns([1, 1])
        with r1:
            st.metric("Body Fat Percentage", f"{pred:.1f}%")
            components.html(get_human_svg(pred), height=350)
        with r2:
            st.write("Health Category")
            if pred < 14: st.success("Athletic / Underfat")
            elif pred < 18: st.success("Fitness")
            elif pred < 25: st.warning("Average")
            else: st.error("Overweight")
            
            if st.session_state.logged_in:
                if st.button("Save Progress"): st.toast("Saved!")

elif selection == "Scientific Info":
    show_info_page()

elif selection == "Body Fat Samples":
    st.subheader("Compare your body fat visually")
    st.image("assets/anh_1.jpg", use_container_width=True)

elif selection == "Settings":
    st.subheader("Settings")
    st.selectbox("App Language", ["English", "Vietnamese"])
    st.radio("Interface Theme", ["Light", "Dark"])

st.markdown("""
    <style>
    /* Đẩy nội dung xuống để không bị Menu che và tạo khoảng trống phía dưới */
    .block-container {
        padding-top: 3.5rem;
        padding-bottom: 5rem;
    }
    /* Ẩn dòng chữ "Made with Streamlit" ở dưới cùng nếu muốn */
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    .stButton>button {width: 100%; border-radius: 5px;}
    .copyright {
        position: fixed; 
        bottom: 0; 
        left: 0; 
        width: 100%; 
        background-color: white; 
        text-align: center; 
        padding: 5px; 
        font-size: 12px; 
        color: gray; 
        z-index: 1000;
        border-top: 1px solid #ddd;
    }
    * {font-family: 'Segoe UI', sans-serif;}
    </style>
""", unsafe_allow_html=True)