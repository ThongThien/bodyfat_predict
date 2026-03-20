import streamlit as st
import streamlit.components.v1 as components
import os
import sys
import time
import pandas as pd
import cv2
import numpy as np

from core.database import (
    get_current_user, get_user_history, save_complete_measurement, 
    sign_up, sign_in, supabase 
)
from core.visualizer import get_custom_css, get_human_svg
from core.predictor import load_model, predict_body_fat
from core.info_content import show_info_page
from core.cv_engine import process_body_measurements

# --- 1. CONFIG & INITIALIZATION ---
st.set_page_config(page_title="Predict Body Fat Free", layout="wide")
model = load_model("models/bodyfat_xgboost_model_final.pkl")


# CSS Setup
st.markdown(f"<style>{get_custom_css()}", unsafe_allow_html=True)
st.markdown("""
    <style>
    [data-testid="stSidebar"] { background-color: #111; }
    .block-container { padding-top: 2rem; padding-bottom: 5rem; }
    .stButton>button { width: 100%; border-radius: 5px; height: 3em; background-color: #3B82F6; color: white; }
""", unsafe_allow_html=True)
    
# Session State Initialization
for key, default in {
    'active_mode': None, 'vals': [22, 60.0, 167.0],  # Sample Age, Weight, Height
    'res_tab1': None, 'res_tab2_scan': None, 'res_tab2_final': None,
    'res_tab3_scan': None, 'res_tab3_final': None,
    'pipe2': (None, None), 'pipe3': (None, None)
}.items():
    if key not in st.session_state: st.session_state[key] = default

# --- 2. HELPERS ---
def get_status_color(bf_value):
    if bf_value < 15: return "#00CC66", "Athletic/Low"
    if bf_value < 25: return "#FFA500", "Normal/Average"
    return "#FF4B4B", "High/Consider Diet"

def handle_save_logic(age, weight, height, scan_res, final_bf, pipe_images, method_name):
    """Xử lý logic lưu dữ liệu dùng chung cho các Tab AI"""
    if st.button(f"SAVE {method_name.upper()} TO CLOUD"):
        with st.spinner("Đang đồng bộ dữ liệu..."):
            img_f, img_s = pipe_images
            buf_f = cv2.imencode('.jpg', img_f)[1].tobytes() if img_f is not None else None
            buf_s = cv2.imencode('.jpg', img_s)[1].tobytes() if img_s is not None else None
            
            success = save_complete_measurement(
                age=age, weight=weight, height=height,
                results_dict={**scan_res, "body_fat": final_bf},
                img_front_bytes=buf_f, img_side_bytes=buf_s,
                method=method_name
            )
            if success:
                st.balloons()
                st.success("Dữ liệu đã được lưu thành công!")
                time.sleep(1)
                st.rerun()

# --- 3. SIDEBAR ---
with st.sidebar:
    st.title("BODYFAT PREDICT")
    
    # Auth System
    user_res = None
    is_logged_in = False
    try:
        user_res = get_current_user()
        is_logged_in = True if user_res and user_res.user else False
    except: is_logged_in = False

    if not is_logged_in:
        auth_mode = st.radio("Tài khoản", ["Đăng nhập", "Đăng ký"])
        email = st.text_input("Email")
        pw = st.text_input("Mật khẩu", type="password")
        if auth_mode == "Đăng ký":
            fname = st.text_input("Họ tên")
            if st.button("Tạo tài khoản"):
                res = sign_up(email, pw, fname)
                st.success("Kiểm tra email xác nhận!") if hasattr(res, 'user') else st.error(f"Lỗi: {res}")
        else:
            if st.button("Vào hệ thống"):
                if hasattr(sign_in(email, pw), 'user'): st.rerun()
                else: st.error("Sai thông tin!")
    else:
        st.success(f"{user_res.user.email}")
        if st.button("Đăng xuất"):
            supabase.auth.sign_out()
            st.cache_data.clear()
            st.rerun()

    st.markdown("---")
    selection = st.radio("MENU", ["Measure Body Fat", "Scientific Info", "Body Fat Samples", "Settings"])
    
    if st.button("LOAD SAMPLE DATA"):
        img_f, img_s = cv2.imread("assets/front_Lap.jpg"), cv2.imread("assets/side_Lap.jpg")
        if img_f is not None:
            s_age, s_w, s_h = 22, 59.0, 167.0
            st.session_state.vals = [s_age, s_w, s_h]
            r2, p2f, p2s = process_body_measurements(img_f, img_s, s_h, s_age, s_w, is_loose=False)
            r3, p3f, p3s = process_body_measurements(img_f, img_s, s_h, s_age, s_w, is_loose=True)
            st.session_state.res_tab2_scan, st.session_state.pipe2 = r2, (p2f, p2s)
            st.session_state.res_tab3_scan, st.session_state.pipe3 = r3, (p3f, p3s)
            st.session_state.active_mode = "AI"
            st.rerun()

    if st.button("RESET ALL"):
        st.session_state.clear()
        st.rerun()

#--- 4. MAIN CONTENT ---
if selection == "Measure Body Fat":
    tab1, tab2, tab3, tab4 = st.tabs(["Manual Input", "AI Scan", "AI Scan + Heuristic", "History"])

    # --- TAB 1: MANUAL ---
    with tab1:
        if st.session_state.active_mode in [None, "Manual"]:
            c1, c2 = st.columns(2)
            with c1:
                age1 = st.number_input("Age", 10, 100, st.session_state.vals[0], key="age1")
                w1 = st.number_input("Weight (kg)", 30.0, 200.0, st.session_state.vals[1], key="w1")
                h1 = st.number_input("Height (cm)", 120.0, 230.0, st.session_state.vals[2], key="h1")
                st.subheader("Body Measurements (cm)")
                data1 = {
                    "Age": age1, "Weight": w1, "Height": h1,
                    "Chest": st.number_input("Chest", 40.0, 200.0, 90.0),
                    "Abdomen": st.number_input("Abdomen", 40.0, 200.0, 80.0),
                    "Hip": st.number_input("Hip", 40.0, 200.0, 95.0),
                    "Thigh": st.number_input("Thigh", 20.0, 150.0, 55.0)
                }
                if st.button("Analyze Manual"):
                    st.session_state.res_tab1 = predict_body_fat(model, data1)
                    st.session_state.active_mode = "Manual"
                    st.rerun()
            with c2:
                if st.session_state.res_tab1:
                    st.metric("Body Fat", f"{st.session_state.res_tab1:.1f}%")
                    components.html(get_human_svg(st.session_state.res_tab1), height=400)
        else: st.warning("Locked: AI Mode active.")

    # --- TAB 2 & 3: AI MODES ---
    ai_tabs = [(tab2, "res_tab2_scan", "res_tab2_final", "pipe2", False, "AI Scan"),
               (tab3, "res_tab3_scan", "res_tab3_final", "pipe3", True, "AI Heuristic")]

    for t, scan_key, final_key, pipe_key, heuristic, label in ai_tabs:
        with t:
            if st.session_state.active_mode in [None, "AI"]:
                col_in, col_disp = st.columns([1, 1.2])
                with col_in:
                    st.subheader(f"{label} Settings")
                    age = st.number_input("Age", 10, 100, st.session_state.vals[0], key=f"age_{label}")
                    w = st.number_input("Weight (kg)", 30.0, 200.0, st.session_state.vals[1], key=f"w_{label}")
                    h = st.number_input("Height (cm)", 120.0, 230.0, st.session_state.vals[2], key=f"h_{label}")
                    u_f = st.file_uploader("Front View", type=['jpg', 'png'], key=f"uf_{label}")
                    u_s = st.file_uploader("Side View", type=['jpg', 'png'], key=f"us_{label}")
                    
                    if st.button(f"RUN {label.upper()}") and u_f and u_s:
                        img_f = cv2.imdecode(np.frombuffer(u_f.read(), np.uint8), 1)
                        img_s = cv2.imdecode(np.frombuffer(u_s.read(), np.uint8), 1)
                        res, pf, ps = process_body_measurements(img_f, img_s, h, age, w, is_loose=heuristic)
                        st.session_state[scan_key], st.session_state[pipe_key] = res, (pf, ps)
                        st.session_state.active_mode = "AI"; st.rerun()
                    
                    if st.session_state[scan_key]:
                        r = st.session_state[scan_key]
                        st.info(f"Scan: C:{r['Chest']:.1f} | A:{r['Abdomen']:.1f} | H:{r['Hip']:.1f} | T:{r['Thigh']:.1f}")
                        with st.expander("Check Details"): st.json({**r, "Age": age, "Height": h, "Weight": w, "Heuristic": heuristic})
                        if st.button(f"Analyze {label} Results"):
                            st.session_state[final_key] = predict_body_fat(model, {"Age": age, "Weight": w, "Height": h, **r})
                            st.rerun()

                with col_disp:
                    st.subheader("Processing Pipeline")
                    pipe_f, pipe_s = st.session_state[pipe_key]
                    gt1, gt2 = st.tabs(["Front", "Side"])
                    with gt1: st.image(pipe_f if pipe_f is not None else "assets/example_front.jpg", width=350)
                    with gt2: st.image(pipe_s if pipe_s is not None else "assets/example_side.jpg", width=350)
                    
                    if st.session_state[final_key]:
                        res_val = st.session_state[final_key]
                        color, status = get_status_color(res_val)
                        st.markdown(f"**Status:** {status}")
                        components.html(get_human_svg(res_val, color=color), height=350)
                        
                        if is_logged_in:
                            handle_save_logic(age, w, h, st.session_state[scan_key], res_val, st.session_state[pipe_key], label)
                        else:
                            st.warning("Đăng nhập để lưu kết quả!")
            else: st.warning("Locked: Manual Mode active.")

    # Thêm vào Tab 4
    with tab4:
        st.header("Personal Progress")
        if is_logged_in:
            history = get_user_history()
            if history:
                last_bf = history[0]['body_fat']
                prev_bf = history[1]['body_fat'] if len(history) > 1 else last_bf
                st.metric("Kết quả gần nhất", f"{last_bf}%", delta=f"{round(last_bf - prev_bf, 1)}%", delta_color="inverse")
                
                df = pd.DataFrame(history)
                df['created_at'] = pd.to_datetime(df['created_at'])
                st.line_chart(df.set_index('created_at')['body_fat'])
                
                for item in history:
                    with st.container(border=True):
                        c_t, c_i = st.columns([2, 1])
                        c_t.write(f"**Ngày:** {item['created_at'][:10]} | **BF:** {item['body_fat']}% ({item['method']})")
                        c_t.write(f"Cân nặng: {item['weight']}kg | Bụng: {item['abdomen']}cm")
                        if item['image_url_front']: c_i.image(item['image_url_front'], width=100)
            else: st.info("Chưa có dữ liệu.")
        else: st.warning("Vui lòng đăng nhập.")

elif selection == "Scientific Info": show_info_page()
elif selection == "Body Fat Samples": st.image("assets/anh_1.jpg")
