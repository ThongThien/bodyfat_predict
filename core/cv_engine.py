import numpy as np
import cv2
import streamlit as st
import importlib

def get_data(img_bgr):
    if img_bgr is None: return None, None
    
    # --- PHẦN NẠP MODULE AN TOÀN ---
    mp_pose_mod = None
    mp_seg_mod = None

    try:
        # Cách 1: Thử nạp trực tiếp qua importlib (dành cho môi trường Cloud khắt khe)
        spec_pose = importlib.util.find_spec("mediapipe.solutions.pose")
        if spec_pose:
            mp_pose_mod = importlib.util.module_from_spec(spec_pose)
            spec_pose.loader.exec_module(mp_pose_mod)
        
        spec_seg = importlib.util.find_spec("mediapipe.solutions.selfie_segmentation")
        if spec_seg:
            mp_seg_mod = importlib.util.module_from_spec(spec_seg)
            spec_seg.loader.exec_module(mp_seg_mod)
    except:
        pass

    # Cách 2: Dự phòng nếu Cách 1 thất bại
    if mp_pose_mod is None or mp_seg_mod is None:
        try:
            import mediapipe.solutions.pose as mp_pose_mod
            import mediapipe.solutions.selfie_segmentation as mp_seg_mod
        except:
            st.error("Không thể nạp MediaPipe. Hãy kiểm tra requirements.txt")
            return None, None

    # --- XỬ LÝ ẢNH ---
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Khởi tạo model với module đã nạp
    mask = None
    res_pose = None
    
    try:
        with mp_seg_mod.SelfieSegmentation(model_selection=1) as seg:
            res_seg = seg.process(img_rgb)
            mask = res_seg.segmentation_mask > 0.5
            
        with mp_pose_mod.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            res_pose = pose.process(img_rgb)
    except Exception as e:
        st.error(f"Lỗi xử lý AI: {e}")
        
    return mask, res_pose

def process_body_measurements(front_img, side_img, real_h, age, weight, is_loose=False):
    # Nạp drawing utils cục bộ để vẽ skeleton
    mp_drawing = None
    mp_pose_vars = None
    try:
        import mediapipe.solutions.drawing_utils as mp_drawing
        import mediapipe.solutions.pose as mp_pose_vars
    except:
        pass

    # 1. Thu thập dữ liệu Mask và Pose
    mask_f_raw, res_f = get_data(front_img)
    mask_s_raw, res_s = get_data(side_img)

    if res_f is None or res_f.pose_landmarks is None or res_s is None or res_s.pose_landmarks is None:
        st.warning("AI không nhận diện được vóc dáng. Hãy đứng rõ trước camera và chụp lại.")
        return None, None, None

    # --- BƯỚC 1: LÀM SẠCH MASK ---
    iter_val = 2 if is_loose else 0 
    kernel = np.ones((3, 3), np.uint8)
    mask_f = cv2.erode(mask_f_raw.astype(np.uint8), kernel, iterations=iter_val).astype(bool) if iter_val > 0 else mask_f_raw
    mask_s = cv2.erode(mask_s_raw.astype(np.uint8), kernel, iterations=iter_val).astype(bool) if iter_val > 0 else mask_s_raw

    lm_f = res_f.pose_landmarks.landmark
    lm_s = res_s.pose_landmarks.landmark
    h_img, w_img, _ = front_img.shape

    # --- BƯỚC 2: RATIO (PIXEL TO CM) ---
    y_nose = lm_f[0].y * h_img
    y_heel = ((lm_f[29].y + lm_f[30].y) / 2) * h_img
    head_top_offset = abs(y_nose - (lm_f[1].y * h_img)) * 2.5 
    ratio = real_h / abs(y_heel - (y_nose - head_top_offset))

    # --- BƯỚC 3: HÀM QUÉT BIÊN ---
    def get_dim_v9(mask, y_norm, lm_list, part_name, view='front'):
        curr_h, curr_w = mask.shape
        y_pixel = int(y_norm * curr_h)
        y_pixel = np.clip(y_pixel, 0, curr_h - 1)
        
        shoulder_w = abs(lm_list[12].x - lm_list[11].x) * curr_w
        x_limit_min = int(min(lm_list[11].x, lm_list[12].x) * curr_w - shoulder_w * 0.2)
        x_limit_max = int(max(lm_list[11].x, lm_list[12].x) * curr_w + shoulder_w * 0.2)
        
        x_limit_min, x_limit_max = max(0, x_limit_min), min(curr_w-1, x_limit_max)
        row = mask[y_pixel, x_limit_min:x_limit_max]
        white_pixels = np.where(row)[0]
        
        if len(white_pixels) < 2: return 0, 0, 0

        x1_local = np.min(white_pixels)
        x2_local = np.max(white_pixels)

        if part_name == 'Thigh' and view == 'front':
            diff = np.diff(white_pixels)
            gap = np.where(diff > 7)[0]
            if len(gap) > 0: x2_local = white_pixels[gap[0]]
            else:
                max_th = shoulder_w * (0.45 if is_loose else 0.55)
                if (x2_local - x1_local) > max_th: x2_local = x1_local + max_th

        width_cm = (x2_local - x1_local) * ratio
        return width_cm, x_limit_min + x1_local, x_limit_min + x2_local

    # --- BƯỚC 4: TỌA ĐỘ Y ---
    y_map = {
        'Chest': lm_f[11].y + (lm_f[23].y - lm_f[11].y) * 0.3, 
        'Abdomen': lm_f[23].y - 0.02, 
        'Hip': lm_f[23].y + 0.08,     
        'Thigh': lm_f[23].y + (lm_f[25].y - lm_f[23].y) * 0.3
    }

    raw_results = {}
    viz_f = front_img.copy()
    
    for part, y in y_map.items():
        w_v, x1f, x2f = get_dim_v9(mask_f, y, lm_f, part, 'front')
        d_v, _, _ = get_dim_v9(mask_s, y, lm_s, part, 'side')
        
        a, b = w_v / 2, d_v / 2
        if a > 0 and b > 0:
            h_el = ((a - b)**2) / ((a + b)**2)
            circum = np.pi * (a + b) * (1 + (3 * h_el) / (10 + np.sqrt(4 - 3 * h_el)))
            raw_results[part] = circum * 1.08 # Hệ số hiệu chỉnh người Việt
        else: raw_results[part] = 0
            
        cv2.line(viz_f, (int(x1f), int(y*h_img)), (int(x2f), int(y*h_img)), (0, 255, 0), 3)

    # --- BƯỚC 5: FILTERING ---
    final_results = {}
    adj = 0.95 if is_loose else 1.0
    max_reasonable = {
        'Chest': weight * 1.75, 'Abdomen': weight * 1.35, 
        'Hip': weight * 1.60 * adj, 'Thigh': weight * 0.95 * adj
    }

    for part, val in raw_results.items():
        limit = max_reasonable[part]
        if is_loose and part in ['Hip', 'Thigh']: val -= 2.0
        val = max(val, 20.0)
        final_results[part] = round(min(val, limit), 2)

    # --- BƯỚC 6: VISUALIZATION ---
    viz_s = side_img.copy()
    h_s, w_s, _ = side_img.shape
    for part, y in y_map.items():
        _, x1s, x2s = get_dim_v9(mask_s, y, lm_list=lm_s, part_name=part, view='side')
        cv2.line(viz_s, (int(x1s), int(y*h_s)), (int(x2s), int(y*h_s)), (0, 255, 0), 3)

    skel_f, skel_s = front_img.copy(), side_img.copy()
    if mp_drawing and mp_pose_vars:
        mp_drawing.draw_landmarks(skel_f, res_f.pose_landmarks, mp_pose_vars.POSE_CONNECTIONS)
        mp_drawing.draw_landmarks(skel_s, res_s.pose_landmarks, mp_pose_vars.POSE_CONNECTIONS)

    mask_f_viz = cv2.cvtColor((mask_f.astype(np.uint8)*255), cv2.COLOR_GRAY2BGR)
    mask_s_viz = cv2.cvtColor((mask_s.astype(np.uint8)*255), cv2.COLOR_GRAY2BGR)

    def res(img): return cv2.resize(img, (300, 450))

    pipeline_front = np.vstack([
        np.hstack([res(front_img), res(mask_f_viz)]),
        np.hstack([res(skel_f), res(viz_f)])
    ])
    pipeline_side = np.vstack([
        np.hstack([res(side_img), res(mask_s_viz)]),
        np.hstack([res(skel_s), res(viz_s)])
    ])

    return final_results, pipeline_front, pipeline_side