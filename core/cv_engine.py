import cv2
import numpy as np
import mediapipe as mp

# Cách gọi an toàn tuyệt đối trên Streamlit Cloud
try:
    from mediapipe.python.solutions import pose as mp_pose
    from mediapipe.python.solutions import selfie_segmentation as mp_segmentation
    from mediapipe.python.solutions import drawing_utils as mp_drawing
except ImportError:
    import mediapipe.solutions.pose as mp_pose
    import mediapipe.solutions.selfie_segmentation as mp_segmentation
    import mediapipe.solutions.drawing_utils as mp_drawing

def process_body_measurements(front_img, side_img, real_h, age, weight, is_loose=False):
    """
    Xử lý đo cơ thể: Sửa lỗi thiếu hụt vòng ngực, chống dính tay và hiệu chỉnh đồ rộng.
    """
    
    def get_data(img_bgr):
        if img_bgr is None: return None, None
        img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
        with mp_segmentation.SelfieSegmentation(model_selection=1) as seg:
            res_seg = seg.process(img_rgb)
            mask = res_seg.segmentation_mask > 0.5
        with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
            res_pose = pose.process(img_rgb)
        return mask, res_pose

    # 1. Thu thập dữ liệu Mask và Pose
    mask_f_raw, res_f = get_data(front_img)
    mask_s_raw, res_s = get_data(side_img)

    if res_f.pose_landmarks is None or res_s.pose_landmarks is None:
        return None, None, None

    # --- BƯỚC 1: LÀM SẠCH MASK ---
    # Giảm Erosion khi không mặc đồ rộng để giữ biên da thật (tránh nerf ngực)
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

    # --- BƯỚC 3: HÀM QUÉT BIÊN TUYỆT ĐỐI (SỬA LỖI THIẾU 10CM) ---
    def get_dim_v9(mask, y_norm, lm_list, part_name, view='front'):
        curr_h, curr_w = mask.shape
        y_pixel = int(y_norm * curr_h)
        
        # Lấy mốc vai để giới hạn vùng tìm kiếm (tránh dính rác trong ảnh)
        shoulder_w = abs(lm_list[12].x - lm_list[11].x) * curr_w
        # Mở rộng vùng quét ra ngoài vai 20% để lấy hết phần xô/ngực
        x_limit_min = int(min(lm_list[11].x, lm_list[12].x) * curr_w - shoulder_w * 0.2)
        x_limit_max = int(max(lm_list[11].x, lm_list[12].x) * curr_w + shoulder_w * 0.2)
        
        x_limit_min, x_limit_max = max(0, x_limit_min), min(curr_w-1, x_limit_max)
        row = mask[y_pixel, x_limit_min:x_limit_max]
        white_pixels = np.where(row)[0]
        
        if len(white_pixels) < 2: return 0, 0, 0

        # Tìm biên trái/phải xa nhất (np.min/max đảm bảo kéo line ra hết biên da)
        x1_local = np.min(white_pixels)
        x2_local = np.max(white_pixels)

        # Tách đùi nếu là mặt trước
        if part_name == 'Thigh' and view == 'front':
            diff = np.diff(white_pixels)
            gap = np.where(diff > 7)[0]
            if len(gap) > 0: x2_local = white_pixels[gap[0]]
            else:
                max_th = shoulder_w * (0.45 if is_loose else 0.55)
                if (x2_local - x1_local) > max_th: x2_local = x1_local + max_th

        width_cm = (x2_local - x1_local) * ratio
        return width_cm, x_limit_min + x1_local, x_limit_min + x2_local

    # --- BƯỚC 4: THIẾT LẬP TỌA ĐỘ Y (HẠ THẤP NGỰC ĐỂ QUA ĐỈNH TI) ---
    y_map = {
        'Chest': lm_f[11].y + (lm_f[23].y - lm_f[11].y) * 0.3, 
        'Abdomen': lm_f[23].y - 0.06, # Rốn
        'Hip': lm_f[23].y + 0.05,     # Mông
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
            raw_results[part] = circum
        else: raw_results[part] = 0
            
        cv2.line(viz_f, (int(x1f), int(y*h_img)), (int(x2f), int(y*h_img)), (0, 255, 0), 3)

    # --- BƯỚC 5: HEURISTIC FILTERING (TĂNG NGƯỠNG NGỰC) ---
    final_results = {}
    adj = 0.95 if is_loose else 1.0
    
    max_reasonable = {
        'Chest': weight * 1.75,  
        'Abdomen': weight * 1.32, 
        'Hip': weight * 1.58 * adj,
        'Thigh': weight * 0.90 * adj
    }

    for part, val in raw_results.items():
        limit = max_reasonable[part]
        if is_loose and part in ['Hip', 'Thigh']:
            val -= 2.5 # Trừ hao vải quần đùi
        val = max(val, 20.0)
        final_results[part] = round(min(val, limit), 2)

    # --- BƯỚC 6: VISUALIZATION (Tách riêng Front & Side) ---
    
    # Chuẩn bị ảnh cho Side View
    viz_s = side_img.copy()
    h_s, w_s, _ = side_img.shape
    for part, y in y_map.items():
        _, x1s, x2s = get_dim_v9(mask_s, y, lm_s, part, 'side')
        cv2.line(viz_s, (int(x1s), int(y*h_s)), (int(x2s), int(y*h_s)), (0, 255, 0), 3)

    # Chuẩn bị Skeleton cho cả 2 mặt
    skel_f = front_img.copy()
    skel_s = side_img.copy()
    mp_drawing.draw_landmarks(skel_f, res_f.pose_landmarks, mp_pose.POSE_CONNECTIONS)
    mp_drawing.draw_landmarks(skel_s, res_s.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Chuẩn bị Mask (Chuyển sang BGR để ghép với ảnh màu)
    mask_f_viz = cv2.cvtColor((mask_f.astype(np.uint8)*255), cv2.COLOR_GRAY2BGR)
    mask_s_viz = cv2.cvtColor((mask_s.astype(np.uint8)*255), cv2.COLOR_GRAY2BGR)

    def res(img): return cv2.resize(img, (300, 450))

    # Ghép 4 ảnh cho Front Section (2x2)
    pipeline_front = np.vstack([
        np.hstack([res(front_img), res(mask_f_viz)]), # Gốc & Mask
        np.hstack([res(skel_f), res(viz_f)])         # Khung xương & Quét biên
    ])

    # Ghép 4 ảnh cho Side Section (2x2)
    pipeline_side = np.vstack([
        np.hstack([res(side_img), res(mask_s_viz)]), # Gốc & Mask
        np.hstack([res(skel_s), res(viz_s)])         # Khung xương & Quét biên
    ])

    return final_results, pipeline_front, pipeline_side