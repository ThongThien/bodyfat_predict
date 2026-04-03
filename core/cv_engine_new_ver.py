import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

def get_body_data(img_bgr: np.ndarray):
    if img_bgr is None: return None, None
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp_segmentation.SelfieSegmentation(model_selection=1) as seg, \
         mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        res_seg = seg.process(img_rgb)
        res_pose = pose.process(img_rgb)
        mask = res_seg.segmentation_mask > 0.5
        return mask, res_pose

def refine_body_mask(mask_raw, landmarks, h_img, iterations_lower):
    y_hip_px = int(((landmarks[23].y + landmarks[24].y) / 2) * h_img)
    mask_uint8 = mask_raw.astype(np.uint8)
    mask_upper = cv2.medianBlur(mask_uint8.copy(), 5)
    mask_upper[y_hip_px:, :] = 0
    mask_lower = mask_uint8.copy()
    mask_lower[:y_hip_px, :] = 0
    if iterations_lower > 0:
        kernel = np.ones((3, 3), np.uint8)
        mask_lower = cv2.erode(mask_lower, kernel, iterations=iterations_lower)
    return cv2.bitwise_or(mask_upper, mask_lower).astype(bool)

def get_dimension_at_y(mask, y_norm, lm_list, part_name, view, ratio, is_loose):
    h_img, w_img = mask.shape
    y_pixel = min(int(y_norm * h_img), h_img - 1)
    shoulder_w = abs(lm_list[12].x - lm_list[11].x) * w_img
    x_min = max(0, int(min(lm_list[11].x, lm_list[12].x) * w_img - shoulder_w * 0.3))
    x_max = min(w_img - 1, int(max(lm_list[11].x, lm_list[12].x) * w_img + shoulder_w * 0.3))
    row = mask[y_pixel, x_min:x_max]
    white_pixels = np.where(row)[0]
    if len(white_pixels) < 2: return 0, 0, 0, 0
    x1, x2 = np.min(white_pixels), np.max(white_pixels)
    if part_name == 'Thigh' and view == 'front':
        diff = np.diff(white_pixels)
        gap = np.where(diff > 7)[0]
        if len(gap) > 0: x2 = white_pixels[gap[0]]
        else: x2 = min(x2, x1 + shoulder_w * (0.45 if is_loose else 0.55))
    
    width_px = x2 - x1
    width_cm = width_px * ratio
    return width_cm, x_min + x1, x_min + x2, width_px

def calculate_ramanujan_circumference(w_cm, d_cm):
    a, b = w_cm / 2, d_cm / 2
    if a <= 0 or b <= 0: return 0.0
    h_el = ((a - b)**2) / ((a + b)**2)
    return np.pi * (a + b) * (1 + (3 * h_el) / (10 + np.sqrt(4 - 3 * h_el)))

def get_pixel_ratio(landmarks, h_img, real_h_cm):
    y_nose = landmarks[0].y * h_img
    y_heel = ((landmarks[29].y + landmarks[30].y) / 2) * h_img
    head_top_offset = abs(y_nose - (landmarks[1].y * h_img)) * 2.5
    pixel_height = abs(y_heel - (y_nose - head_top_offset))
    return real_h_cm / pixel_height

def process_body_measurements(front_img, side_img, real_h, age, weight, is_loose=False, is_raw=False):
    mask_f_raw, res_f = get_body_data(front_img)
    mask_s_raw, res_s = get_body_data(side_img)
    if not all([res_f, res_f.pose_landmarks, res_s, res_s.pose_landmarks]): return None, None, None

    lm_f, lm_s = res_f.pose_landmarks.landmark, res_s.pose_landmarks.landmark
    h_f, h_s = front_img.shape[0], side_img.shape[0]
    ratio_f, ratio_s = get_pixel_ratio(lm_f, h_f, real_h), get_pixel_ratio(lm_s, h_s, real_h)

    it_lower = (2 if is_loose else 1) if not is_raw else 0
    mask_f = refine_body_mask(mask_f_raw, lm_f, h_f, it_lower)
    mask_s = refine_body_mask(mask_s_raw, lm_s, h_s, it_lower)

    y_hip_f, y_shoulder_f = (lm_f[23].y + lm_f[24].y)/2, (lm_f[11].y + lm_f[12].y)/2
    torso_h = y_hip_f - y_shoulder_f
    y_map = {'Chest': y_shoulder_f + torso_h*0.25, 'Abdomen': y_shoulder_f + torso_h*0.75, 
             'Hip': y_hip_f + 0.08, 'Thigh': y_hip_f + (lm_f[25].y - y_hip_f)*0.3}

    bmi = weight / ((real_h / 100) ** 2)
    f_cal = 1.204 # Mặc định
    if not is_raw:
        if bmi < 18.5: f_cal = 1.12
        elif bmi < 25: f_cal = 1.204
        elif bmi < 30: f_cal = 1.24
        else: f_cal = 1.28
    
    s_cloth = (0.96 if is_loose else 1.0) if not is_raw else 1.0
    final_results, raw_dimensions = {}, {}
    viz_f, viz_s = front_img.copy(), side_img.copy()

    for part, y_coord in y_map.items():
        w_cm, x1f, x2f, w_px = get_dimension_at_y(mask_f, y_coord, lm_f, part, 'front', ratio_f, is_loose)
        d_cm, x1s, x2s, d_px = get_dimension_at_y(mask_s, y_coord, lm_s, part, 'side', ratio_s, is_loose)
        
        f_part = f_cal * (s_cloth if part in ['Hip', 'Thigh'] else 1.0)
        circum = calculate_ramanujan_circumference(w_cm, d_cm) * f_part
        final_results[part] = round(circum, 2)
        raw_dimensions[part] = {"w_cm": w_cm, "d_cm": d_cm, "w_px": w_px, "d_px": d_px, "ratio_f": ratio_f, "ratio_s": ratio_s}

        # Vẽ Line hiển thị vị trí đo
        yf_px, ys_px = int(y_coord * h_f), int(y_coord * h_s)
        cv2.line(viz_f, (int(x1f), yf_px), (int(x2f), yf_px), (0, 255, 0), 3) # Front: Xanh lá
        cv2.line(viz_s, (int(x1s), ys_px), (int(x2s), ys_px), (0, 0, 255), 3) # Side: Đỏ

    def create_pipe(img, mask, res, viz):
        r = lambda i: cv2.resize(i, (300, 450))
        mask_v = cv2.cvtColor((mask.astype(np.uint8)*255), cv2.COLOR_GRAY2BGR)
        top_row = np.hstack([r(img), r(mask_v)])
        bottom_row = cv2.resize(viz, (600, 900))
        return np.vstack([top_row, bottom_row])

    # Trả về thêm raw_dimensions để phục vụ calibrate.py
    return final_results, create_pipe(front_img, mask_f, res_f, viz_f), create_pipe(side_img, mask_s, res_s, viz_s), raw_dimensions