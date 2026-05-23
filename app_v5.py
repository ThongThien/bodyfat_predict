import streamlit as st
from ontology.ontology_engine_v2 import run_ontology
from streamlit_modal import Modal
import plotly.express as px
import os
import cv2
import numpy as np
import pandas as pd
import time

# --- IMPORT MODULES V5 ---
from core.database import (
    get_current_user, get_user_history, save_complete_measurement, 
    sign_up, sign_in, supabase 
)
from core.visualizer import get_custom_css
from core.info_content_v5 import show_info_page_v5

# Sử dụng các bản nâng cấp v5
from core.predictor_v5 import load_model_v5, predict_body_fat_v5
from core.cv_engine_v5 import process_body_measurements_v5

# --- 1. CONFIG & INITIALIZATION ---
st.set_page_config(page_title="Predict Body Fat AI", layout="wide")

# Load model v5 (7 features)
model_v5 = load_model_v5("models/bodyfat_ai_super_clean_v5.pkl")

# CSS Setup
st.markdown(f"{get_custom_css()}", unsafe_allow_html=True)
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #0E1117; }
    .stMetric { background-color: #1E293B; padding: 15px; border-radius: 10px; border: 1px solid #3B82F6; }
    .stButton>button { border-radius: 8px; height: 3em; transition: 0.3s; }
    .stButton>button:hover { border: 1px solid #3B82F6; color: #3B82F6; }
    </style>
""", unsafe_allow_html=True)

ontology_modal = Modal(
    "Ontology Semantic Dashboard",
    key="ontology_dashboard",
    max_width=1000
)

# Session State cho v5
for key, default in {
    'active_mode': None, 
    'vals': [25, 82.0, 172.0], 
    'res_tab1': None, 
    'res_scan_v5': None, 'res_final_v5': None, 'pipe_v5': (None, None)
}.items():
    if key not in st.session_state: 
        st.session_state[key] = default

@st.dialog("Ontology Semantic Dashboard", width="large")
def show_ontology_dashboard(onto):
    st.markdown(
        """
        <style>
        .block-container { padding-top: 1rem; padding-bottom: 1rem; }
        div[data-testid="stExpander"] div { padding: 8px; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.caption(
        "This dashboard explains the runtime reasoning graph of the Hybrid AI + Ontology system."
    )
    st.divider()

    session_id = onto.get("Session_ID", "Unknown")
    validation_status = onto.get("Validation_Status", "Unknown")
    confidence_level = onto.get("Confidence_Level", "Unknown")
    warning_level = onto.get("Warning_Level", "None")
    anomaly_type = onto.get("Anomaly_Type", [])
    image_quality = onto.get("Image_Quality", "Unknown")

    st.markdown(f"### Prediction Session: `{session_id}`")

    top1, top2, top3, top4 = st.columns(4)

    top1.metric("Validation", validation_status)
    top2.metric("Confidence Level", confidence_level)
    top3.metric("Warning Level", warning_level)
    top4.metric("Image Quality", image_quality)

    st.divider()

    col_left, col_mid, col_right = st.columns([1, 1, 1.1])

    # =========================================================
    # 1. INPUT + FEATURE + PREDICTION
    # =========================================================
    with col_left:
        st.markdown("### 1. Input → Feature → Prediction")

        c1, c2 = st.columns(2)
        c1.metric("BMI", f"{onto.get('BMI', 0):.2f}", onto.get("BMI_Class", "Unknown"))
        c2.metric("Body Fat %", f"{onto.get('BodyFat', 0):.2f}", onto.get("Fat_Level", "Unknown"))

        c3, c4 = st.columns(2)
        c3.metric("WHR", f"{onto.get('WHR', 0):.2f}")
        c4.metric("WtHR", f"{onto.get('WtHR', 0):.2f}")

        st.markdown("##### Image Quality Metrics")

        q1, q2 = st.columns(2)
        q1.metric("Pose Visibility", f"{onto.get('Pose_Visibility', 0):.2f}")
        q2.metric("Mask Confidence", f"{onto.get('Mask_Confidence', 0):.2f}")

        q3, q4 = st.columns(2)
        q3.metric("Confidence Score", f"{onto.get('Confidence_Score', 0):.2f}")
        q4.metric("Missing Landmarks", onto.get("Missing_Landmark_Count", 0))

        st.markdown("##### Reasoner Status")

        if onto.get("Reasoning_Status") == "Success":
            st.success("Pellet Reasoner: Success")
        else:
            st.error("Pellet Reasoner: Failed")

        reasoning_error = onto.get("Reasoning_Error")
        if reasoning_error:
            st.error(reasoning_error)

        with st.expander("Semantic Pipeline Trace", expanded=False):
            pipeline = onto.get("Semantic_Pipeline", [])
            if pipeline:
                st.caption(" ➜ ".join(pipeline))
            else:
                st.caption("Pipeline trace unavailable.")

    # =========================================================
    # 2. SEMANTIC RULES + ANOMALY
    # =========================================================
    with col_mid:
        st.markdown("### 2. Semantic Rules & Anomaly")

        st.markdown("##### Semantic Flags")
        semantic_flags = onto.get("Semantic_Flags", [])
        if semantic_flags:
            for flag in semantic_flags:
                st.info(flag)
        else:
            st.success("No semantic flag detected.")

        st.markdown("##### Anomaly Type")
        if anomaly_type:
            for anomaly in anomaly_type:
                st.error(anomaly)
        else:
            st.success("No anomaly detected.")

        st.markdown("##### Triggered SWRL Rules")
        triggered_rules = onto.get("Triggered_Rules", [])
        if triggered_rules:
            for idx, rule in enumerate(triggered_rules, start=1):
                st.markdown(f"**Rule #{idx}**")
                st.code(rule, language="text")
        else:
            st.caption("No ontology rule was triggered.")

        with st.expander("Inferred Ontology Trace", expanded=False):
            inferred_trace = onto.get("Inferred_Class_Trace", {})
            if inferred_trace:
                for k, v in inferred_trace.items():
                    st.markdown(f"**{k}**")
                    if v:
                        for item in v:
                            st.code(item, language="text")
                    else:
                        st.caption("No inferred value.")
            else:
                st.caption("No inferred trace available.")

    # =========================================================
    # 3. VALIDATION + EXPLANATION + RECOMMENDATION
    # =========================================================
    with col_right:
        st.markdown("### 3. Validation → Explanation → Action")

        if validation_status == "Valid":
            st.success("Semantic Validation: PASSED")
        elif validation_status == "Invalid":
            st.error("Semantic Validation: FAILED")
        else:
            st.warning(f"Semantic Validation: {validation_status}")

        validation_errors = onto.get("Validation_Errors", [])
        if validation_errors:
            for err in validation_errors:
                st.error(err)

        st.markdown("##### Semantic Explanations")
        explanations = onto.get("Explanations", [])
        if explanations:
            for e in explanations:
                st.warning(f"• {e}")
        else:
            st.caption("No semantic explanation generated.")

        st.markdown("##### Ontology-based Recommendations")
        recommendations = onto.get("Recommendations", [])
        if recommendations:
            for r in recommendations:
                st.success(f"• {r}")
        else:
            st.info("Maintain current healthy lifestyle.")

        st.markdown("##### Ontology Latency")
        st.metric("Latency", f"{onto.get('Ontology_Latency_Ms', 0):.2f} ms")

    st.divider()

    with st.expander("Runtime Reasoning Graph Trace", expanded=False):
        graph_trace = onto.get("Reasoning_Graph_Trace", [])

        if graph_trace:
            for edge in graph_trace:
                st.code(edge, language="text")
        else:
            st.caption("Reasoning graph trace unavailable.")
            
# --- 2. HELPERS ---
def get_status_color(bf_value):
    if bf_value < 13: return "#34D399", "Athletic"
    if bf_value < 22: return "#60A5FA", "Fitness/Normal"
    if bf_value < 28: return "#FBBF24", "Average "
    return "#F87171", "High Body Fat (Thừa mỡ)"

def parse_filename(file_name):
    base = os.path.basename(file_name)
    name = base.split("_")[1]
    data = base.split("_")[2].split(".")[0]

    age, h, w, c, a, hip, *_ = map(float, data.split("-"))

    return {
        "Name": name,
        "Age": age,
        "Height": h,
        "Weight": w,
        "Chest": c,
        "Abdomen": a,
        "Hip": hip
    }

def handle_save_logic(age, weight, height, scan_res, final_bf, pipe_images, method_name):
    with st.spinner("Data is being synchronized..."):
        img_f, img_s = pipe_images
        buf_f = cv2.imencode('.jpg', img_f)[1].tobytes() if img_f is not None else None
        buf_s = cv2.imencode('.jpg', img_s)[1].tobytes() if img_s is not None else None
        
        success = save_complete_measurement(
            age=age, weight=weight, height=height,
            results_dict={**scan_res, "body_fat": final_bf},
            img_front_bytes=buf_f, img_side_bytes=buf_s,
            method=method_name
        )

        if success.get("success"):
            st.balloons()
            st.success("Data has been saved successfully!")
            time.sleep(1)
            st.rerun()
        else:
            st.error(f"Lỗi: {success.get('error')}")

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title(" PREDICT BODYFAT AI ")
    
    st.markdown("---")
    selection = st.radio("MENU", ["Measure Body Fat", "Scientific Info","History",])
    if st.button("Reset Input Data"):
        st.session_state.clear()
        st.rerun()
        
    is_logged_in = False
    try:
        user_res = get_current_user()
        is_logged_in = True if user_res and user_res.user else False
    except: is_logged_in = False

    if not is_logged_in:
        auth_mode = st.radio("Account", ["Login", "Sign up"], horizontal=True)
        email = st.text_input("Email")
        pw = st.text_input("Password", type="password")
        if auth_mode == "Sign up":
            fname = st.text_input("Full name")
            if st.button("Create account"):
                res = sign_up(email, pw, fname)
                st.success("Check your email!") if hasattr(res, 'user') else st.error("Registration error")
        else:
            if st.button("Login"):
                if hasattr(sign_in(email, pw), 'user'): st.rerun()
                else: st.error("Invalid credentials!")
    else:
        st.success(f"Hi, {user_res.user.email}")
        if st.button("Log out"):
            supabase.auth.sign_out()
            st.rerun()

# --- 4. MAIN CONTENT ---
if selection == "Measure Body Fat":
    tab1, tab2 = st.tabs(["Manual Input", "AI Scan"])

    # --- TAB 1: MANUAL ---
    with tab1:
        c1, c2 = st.columns([1, 1.2])
        with c1:
            st.subheader("Input Raw Metrics")
            age1 = st.number_input("Age", 10, 100, st.session_state.vals[0])
            w1 = st.number_input("Weight (kg)", 30.0, 200.0, st.session_state.vals[1])
            h1 = st.number_input("Height (cm)", 120.0, 230.0, st.session_state.vals[2])
            
            st.markdown("---")
            st.markdown("**Circumference Measurements (cm)**")
            chest1 = st.number_input("Chest", 50.0, 180.0, 95.0)
            abd1 = st.number_input("Abdomen (Navel level)", 50.0, 180.0, 85.0)
            hip1 = st.number_input("Hip (Buttocks)", 50.0, 180.0, 95.0)

            if st.button("ANALYZE", use_container_width=True):
                data_v5 = {
                    "Name": "Manual_User", "Age": age1, "Weight": w1, "Height": h1,
                    "Chest": chest1, "Abdomen": abd1, "Hip": hip1
                }
                st.session_state.res_tab1 = predict_body_fat_v5(model_v5, data_v5)
                st.session_state.active_mode = "Manual"

        with c2:
            if st.session_state.res_tab1:
                res_v1 = st.session_state.res_tab1
                color, status = get_status_color(res_v1)
                st.metric("Prediction Result", f"{res_v1}%")
                st.markdown(f"<h3 style='color:{color}; text-align:center;'>{status}</h3>", unsafe_allow_html=True)
            else:
                st.image("assets/hd.jpg", caption="Standard Measurement Guide", use_container_width=True)

    # --- TAB 2: AI SCAN ---
    with tab2:
        col_in, col_disp = st.columns([1, 1.2])
        with col_in:
            st.subheader("AI Vision Scanning")
            age_v = st.number_input("Age", 10, 100, st.session_state.vals[0], key="age_v5")
            w_v = st.number_input("Weight (kg)", 30.0, 200.0, st.session_state.vals[1], key="w_v5")
            h_v = st.number_input("Height (cm)", 120.0, 230.0, st.session_state.vals[2], key="h_v5")
            
            u_f = st.file_uploader("Front Image", type=['jpg', 'png'])
            u_s = st.file_uploader("Side Image", type=['jpg', 'png'])
            use_long_pants = st.checkbox("Wearing long pants (Hip adjustment)")
            if st.button("RUN", use_container_width=True) and u_f and u_s:
                with st.spinner("Analyzing 7 parameters..."):
                    img_f = cv2.imdecode(np.frombuffer(u_f.read(), np.uint8), 1)
                    img_s = cv2.imdecode(np.frombuffer(u_s.read(), np.uint8), 1)
                    
                    res_scan, viz_f, viz_s, debug_pack, quality_pack  = process_body_measurements_v5(
                        img_f, img_s, h_v, w_v, use_long_pants=use_long_pants
                    )
                    if res_scan is not None:
                        st.success("Measurement extraction successful!")
                    else:
                        st.error("Measurement extraction failed! Please ensure the photos are clear and follow the guidelines.")
                    if res_scan:
                        st.session_state.res_scan_v5 = res_scan
                        st.session_state.pipe_v5 = (viz_f, viz_s)
                        st.session_state.debug_pack = debug_pack
                        # Immediate prediction after scan
                        input_v5 = {
                            "Name": "Scan_User",
                            "Age": age_v,
                            "Weight": w_v,
                            "Height": h_v,
                            **res_scan
                        }

                        # ML PREDICTION
                        predicted_bf = predict_body_fat_v5(model_v5, input_v5)

                        st.session_state.res_final_v5 = predicted_bf

                        # ONTOLOGY REASONING
                        ontology_result = run_ontology(
                        height=h_v,
                        weight=w_v,
                        chest=res_scan["Chest"],
                        abdomen=res_scan["Abdomen"],
                        hip=res_scan["Hip"],
                        predicted_bf=predicted_bf,
                        pose_visibility=quality_pack.get("pose_visibility", 1.0),
                        mask_confidence=quality_pack.get("mask_confidence", 1.0),
                        missing_landmark_count=quality_pack.get("missing_landmark_count", 0),
                        source_type="AI Scan",
                        image_name=f"{u_f.name} | {u_s.name}",
                        image_path=None
                    )

                        st.session_state.ontology_result = ontology_result

            if st.session_state.res_scan_v5:
                r = st.session_state.res_scan_v5
                st.success(f"Extraction successful: Chest: {r['Chest']} | Abdomen: {r['Abdomen']} | Hip: {r['Hip']}")
                with st.expander("View measurement details", expanded=False):
                    input_debug = {
                        "Weight": w_v,
                        "Chest": r["Chest"],
                        "Abdomen": r["Abdomen"],
                        "Hip": r["Hip"],
                    }

                    # Additional calculations
                    abd = r.get("Abdomen")
                    hip = r.get("Hip")

                    wpa = (abd**2) / w_v if abd and w_v else None
                    wthr = abd / h_v if abd and h_v else None
                    whr = abd / hip if abd and hip else None

                    debug_df = pd.DataFrame([{
                        "Weight": w_v,
                        "Chest": r["Chest"],
                        "Abdomen": r["Abdomen"],
                        "Hip": r["Hip"],
                        "W_per_A": round(wpa,2) if wpa else None,
                        "WtHR": round(wthr,3) if wthr else None,
                        "WHR": round(whr,3) if whr else None,
                    }])

                    st.dataframe(debug_df)
            
            with st.expander("Index explanations", expanded=False):
                st.markdown("""
                **W_per_A (Waist Power Index)**  
                = Abdomen² / Weight  
                → Assesses abdominal fat accumulation relative to weight

                **WtHR (Waist to Height Ratio)**  
                = Abdomen / Height  
                → Cardiovascular risk indicator

                **WHR (Waist to Hip Ratio)**  
                = Abdomen / Hip  
                → Fat distribution (abdomen vs hip)

                **General Meaning:**
                - High W_per_A → abnormal abdominal size
                - WtHR > 0.5 → visceral fat risk
                - High WHR → "apple shape" body type
                """)
        with col_disp:
            # -------- PHOTO GUIDE (ALWAYS DISPLAYED) --------
            with st.expander("Photo guide for accurate measurements", expanded=False):
                st.markdown("""
                Standard photography conditions:
                
                **1. Distance:** 
                - Place the camera at torso level and about **2m – 2.5m** away  
                (ensure the entire body is within the frame from **heels → top of head**)

                **2. Lighting:** 
                - Sufficient lighting, avoid dark or backlit settings  
                - Clear distinction between body and background  
                - Avoid white backgrounds or colors that match your skin/clothing

                **3. Pose:**
                - **Front photo:** Stand straight, arms out to the sides forming a **T** shape  
                - **Side photo:** Stand sideways, raise both arms high

                **4. Clothing:**  
                - Do not wear a shirt  
                - Wear tight shorts or thin leggings to clearly show the thigh area  
                - Form-fitting attire is preferred for precise measurement

                **5. Background:**
                - Simple, uncluttered background  
                - Contrast background color with your body
                """)

            # -------- SAMPLE IMAGES --------
            with st.expander("Standard Sample Images", expanded=False):

                sample_f = "assets/anh_chuan/front_Thien_22-163-60-89-80-86-48.jpg"
                sample_s = "assets/anh_chuan/side_Thien_22-163-60-89-80-86-48.jpg"

                if os.path.exists(sample_f) and os.path.exists(sample_s):
                    c1, c2 = st.columns(2)
                    c1.image(sample_f, caption="Sample Front")
                    c2.image(sample_s, caption="Sample Side")

                st.markdown("---")

            # -------- AI RESULTS --------
            if st.session_state.res_final_v5:

                res_v5 = st.session_state.res_final_v5
                color_v5, status_v5 = get_status_color(res_v5)

                # SCAN IMAGE
                viz_f, viz_s = st.session_state.pipe_v5
                v1, v2 = st.columns(2)
                v1.image(viz_f, caption="Front Scan")
                v2.image(viz_s, caption="Side Scan")
                
                # RESULT
                st.metric("Prediction", f"{res_v5}%")
                st.markdown(f"**Status:** {status_v5}")

                if st.button("View Full Ontology Dashboard", use_container_width=True):
                    show_ontology_dashboard(st.session_state.ontology_result)
                
                # -------- SAVE --------
                if is_logged_in:
                    if st.button("SAVE RESULT"):
                        scan_res = st.session_state.res_scan_v5

                        handle_save_logic(
                            age=age_v, weight=w_v, height=h_v,
                            scan_res=scan_res,
                            final_bf=res_v5,
                            pipe_images=(viz_f, viz_s),
                            method_name="AI Scan v5"
                        )
                else:
                    st.info("Log in to save results.")
            else:
                st.info("Upload 2 images for the AI to start scanning measurements.")
    # with tab4:
    #     st.subheader("Batch Test Folder (Compare Model vs AI Scan)")

    #     folder_path = "assets/anh_chuan"

    #     # -------- VALIDATE FILENAME --------
    #     def is_valid_filename(file):
    #         try:
    #             parts = file.split("_")
    #             if len(parts) < 3:
    #                 return False
    #             data = parts[2].split(".")[0]
    #             nums = data.split("-")
    #             return len(nums) >= 6
    #         except:
    #             return False

    #     if st.button("SCAN TOÀN BỘ FOLDER"):

    #         import gc

    #         results_measure = []
    #         results_bf = []

    #         files = [f for f in os.listdir(folder_path) if f.startswith("front")]

    #         #  LIMIT để tránh OOM
    #         max_files = st.slider("Số ảnh test", 1, 50, 10)
    #         files = files[:max_files]

    #         progress = st.progress(0)

    #         for idx, file in enumerate(files):

    #             progress.progress((idx + 1) / len(files))

    #             if not is_valid_filename(file):
    #                 print(f" Skip file sai format: {file}")
    #                 continue

    #             try:
    #                 info = parse_filename(file)
    #             except:
    #                 print(f" Lỗi parse: {file}")
    #                 continue

    #             path_f = os.path.join(folder_path, file)
    #             path_s = path_f.replace("front", "side")

    #             if not os.path.exists(path_s):
    #                 print(f" Thiếu side image: {file}")
    #                 continue

    #             try:
    #                 img_f = cv2.imread(path_f)
    #                 img_s = cv2.imread(path_s)

    #                 if img_f is None or img_s is None:
    #                     print(f" Lỗi đọc ảnh: {file}")
    #                     continue

    #                 #  resize giảm RAM
    #                 def resize_img(img, max_w=640):
    #                     h, w = img.shape[:2]
    #                     if w > max_w:
    #                         scale = max_w / w
    #                         img = cv2.resize(img, (int(w*scale), int(h*scale)))
    #                     return img

    #                 img_f = resize_img(img_f)
    #                 img_s = resize_img(img_s)

    #                 # -------- RAW ----------
    #                 raw_pred = predict_body_fat_v5(model_v5, info)

    #                 # -------- AI SCAN ----------
    #                 res_scan1, _, _, _ = process_body_measurements_v5(
    #                     img_f, img_s, info["Height"], info["Weight"], False
    #                 )

    #                 res_scan2, _, _, _ = process_body_measurements_v5(
    #                     img_f, img_s, info["Height"], info["Weight"], True
    #                 )

    #                 if not res_scan1 or not res_scan2:
    #                     print(f" Scan lỗi: {file}")
    #                     continue

    #                 pred_scan1 = predict_body_fat_v5(model_v5, {**info, **res_scan1})
    #                 pred_scan2 = predict_body_fat_v5(model_v5, {**info, **res_scan2})

    #                 # ===== TABLE 1 =====
    #                 results_measure.append({
    #                     "Name": info["Name"],

    #                     "Chest_raw": info["Chest"],
    #                     "Abd_raw": info["Abdomen"],
    #                     "Hip_raw": info["Hip"],

    #                     "Chest_AI": res_scan1["Chest"],
    #                     "Abd_AI": res_scan1["Abdomen"],
    #                     "Hip_AI": res_scan1["Hip"],

    #                     #  chỉ lưu path (KHÔNG lưu ảnh)
    #                     "img_path_f": path_f,
    #                     "img_path_s": path_s
    #                 })

    #                 # ===== TABLE 2 =====
    #                 results_bf.append({
    #                     "Name": info["Name"],
    #                     "BF_Raw": raw_pred,
    #                     "BF_AI": pred_scan1,
    #                     "BF_AI_Loose": pred_scan2,
    #                     "Delta_AI": round(pred_scan1 - raw_pred, 2),
    #                     "Delta_Loose": round(pred_scan2 - raw_pred, 2)
    #                 })

    #             except Exception as e:
    #                 print(f" Crash file {file}: {e}")
    #                 continue

    #         #  FREE RAM mỗi vòng
    #         del img_f, img_s
    #         gc.collect()

    #         # ===== DATAFRAME =====
    #         df_measure = pd.DataFrame(results_measure)
    #         df_bf = pd.DataFrame(results_bf)

    #         if df_measure.empty:
    #             st.warning("Không scan được ảnh nào hợp lệ !!")
    #         else:
    #             st.session_state.df_measure = df_measure
    #             st.session_state.df_bf = df_bf

    #             # -------- DISPLAY --------
    #             st.markdown("## Bảng 1: So sánh số đo")
    #             st.dataframe(df_measure.drop(columns=["img_path_f", "img_path_s"], errors="ignore"))

    #             st.markdown("## Bảng 2: So sánh Body Fat")
    #             st.markdown("DATA GỐC:")
    #             st.markdown("Dao - 20-24%, Hieu - 18-20%, Thien - 15-19%, Lap - 12-16%, Lo - 25-30%")
    #             st.dataframe(df_bf)

    #             # -------- SELECT IMAGE --------
    #             selected_name = st.selectbox("Chọn sample", df_measure["Name"])

    #             row = df_measure[df_measure["Name"] == selected_name].iloc[0]

    #             c1, c2 = st.columns(2)

    #             img_f = cv2.imread(row["img_path_f"])
    #             img_s = cv2.imread(row["img_path_s"])

    #             c1.image(img_f, caption="Front")
    #             c2.image(img_s, caption="Side")

    #             # -------- EXPORT --------
    #             csv1 = df_measure.drop(columns=["img_path_f", "img_path_s"], errors="ignore") \
    #                             .to_csv(index=False).encode("utf-8")

    #             csv2 = df_bf.to_csv(index=False).encode("utf-8")

    #             st.download_button("Download Measurements CSV", csv1, "measurements.csv")
    #             st.download_button("Download BodyFat CSV", csv2, "bodyfat.csv")
elif selection == "Scientific Info":
    show_info_page_v5()
elif selection == "History":
    st.subheader("Body Transformation History")

    if is_logged_in:

        history = get_user_history()

        if history:

            df = pd.DataFrame(history)

            # ===== FORMAT =====
            df["created_at"] = pd.to_datetime(df["created_at"])
            df = df.sort_values("created_at", ascending=False)

            df["Date"] = df["created_at"].dt.strftime("%Y-%m-%d %H:%M")

            # ===== ROUND =====
            round_cols = [
                "weight", "height", "chest", "abdomen",
                "hip", "body_fat", "wpa", "wthr", "whr"
            ]

            for c in round_cols:
                if c in df.columns:
                    df[c] = df[c].round(2)

            # ===== SIMPLE STATUS =====
            def get_bf_status(v):
                if v < 13:
                    return "Athletic"
                elif v < 22:
                    return "Fit"
                elif v < 28:
                    return "Average"
                return "High Fat"

            df["Status"] = df["body_fat"].apply(get_bf_status)

            # ===== CHART =====
            st.markdown("### Body Fat Trend")

            chart_df = (
                df.sort_values("created_at")
                .set_index("Date")[["body_fat"]]
            )

            st.line_chart(chart_df)

            # ===== CLEAN TABLE =====
            st.markdown("### Measurement Records")

            table_df = df[[
                "Date",
                "body_fat",
                "Status",
                "weight",
                "height",
                "method"
            ]].rename(columns={
                "body_fat": "Body Fat %",
                "weight": "Weight (kg)",
                "height": "Height (cm)",
                "method": "Method"
            })

            st.dataframe(
                table_df,
                width="stretch",
                hide_index=True
            )

            # ===== SELECT =====
            st.markdown("### Detailed View")

            df["label"] = (
                df["Date"]
                + "  |  BF: "
                + df["body_fat"].astype(str)
                + "%"
            )

            selected = st.selectbox(
                "Select a record",
                df["label"]
            )

            row = df[df["label"] == selected].iloc[0]

            # ===== SUMMARY CARDS =====
            c1, c2, c3 = st.columns(3)

            c1.metric(
                "Body Fat",
                f"{row['body_fat']}%"
            )

            c2.metric(
                "Weight",
                f"{row['weight']} kg"
            )

            c3.metric(
                "Status",
                row["Status"]
            )

            # ===== MAIN INFO =====
            left, right = st.columns(2)

            with left:

                st.markdown("### General Information")

                st.info(f"""
        Date: {row['Date']}

        Method: {row.get('method', '-')}

        Height: {row.get('height', '-')} cm

        Weight: {row.get('weight', '-')} kg
        """)

                with right:

                    st.markdown("### Body Measurements")

                    st.info(f"""
        Chest: {row.get('chest', '-')}

        Abdomen: {row.get('abdomen', '-')}

        Hip: {row.get('hip', '-')}
        """)

            # ===== BODY INDEX =====
            st.markdown("### Body Index Analysis")

            i1, i2, i3 = st.columns(3)

            i1.metric(
                "WPA",
                row.get("wpa", "-")
            )

            i2.metric(
                "WtHR",
                row.get("wthr", "-")
            )

            i3.metric(
                "WHR",
                row.get("whr", "-")
            )

            # ===== INTERPRETATION =====
            st.markdown("### Interpretation")

            bf = row["body_fat"]
            wthr = row.get("wthr", 0)
            whr = row.get("whr", 0)

            if bf < 13:
                st.success(
                    "Low body fat level with athletic body composition."
                )

            elif bf < 22:
                st.info(
                    "Healthy and balanced body fat range."
                )

            elif bf < 28:
                st.warning(
                    "Moderate fat accumulation detected."
                )

            else:
                st.error(
                    "High body fat percentage detected."
                )

            if wthr and wthr > 0.5:
                st.warning(
                    "WtHR suggests elevated abdominal fat risk."
                )

            if whr and whr > 0.9:
                st.warning(
                    "WHR indicates central fat distribution."
                )

            # ===== IMAGES =====
            st.markdown("### Scan Images")

            img_f = row.get("image_url_front")
            img_s = row.get("image_url_side")

            c1, c2 = st.columns(2)

            if img_f:
                c1.image(
                    img_f,
                    caption="Front Image",
                    width="stretch"
                )
            else:
                c1.info("No front image")

            if img_s:
                c2.image(
                    img_s,
                    caption="Side Image",
                    width="stretch"
                )
            else:
                c2.info("No side image")

        else:
            st.info("No history data found.")

    else:
        st.warning("Please log in to view your history.")