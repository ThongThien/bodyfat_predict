import numpy as np
import cv2
import mediapipe as mp
import math
import pandas as pd

# --- KHỞI TẠO MODEL CV CỦA ÔNG ---
mp_pose = mp.solutions.pose
mp_segmentation = mp.solutions.selfie_segmentation

def get_body_data_research(img_bgr):
    """Sử dụng logic Model CV của ông để lấy Mask sạch"""
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    with mp_segmentation.SelfieSegmentation(model_selection=1) as seg, \
         mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        res_seg = seg.process(img_rgb)
        res_pose = pose.process(img_rgb)
        mask = res_seg.segmentation_mask > 0.5
        return mask, res_pose

def refine_body_mask_research(mask_raw, landmarks, h_img, iterations_lower):
    """Hàm phân vùng quét biên của anh em mình"""
    y_hip_px = int(((landmarks[23].y + landmarks[24].y) / 2) * h_img)
    kernel = np.ones((3, 3), np.uint8)
    
    mask_upper = mask_raw.copy().astype(np.uint8)
    mask_upper[y_hip_px:, :] = 0
    mask_upper_refined = cv2.erode(mask_upper, kernel, iterations=1)

    mask_lower = mask_raw.copy().astype(np.uint8)
    mask_lower[:y_hip_px, :] = 0
    mask_lower_refined = cv2.erode(mask_lower, kernel, iterations=iterations_lower)

    return cv2.bitwise_or(mask_upper_refined, mask_lower_refined)

def calculate_ramanujan(w_cm, d_cm):
    a, b = w_cm / 2, d_cm / 2
    if a <= 0 or b <= 0: return 0
    h_el = ((a - b)**2) / ((a + b)**2)
    return np.pi * (a + b) * (1 + (3 * h_el) / (10 + np.sqrt(4 - 3 * h_el)))

# --- CẤU HÌNH DỮ LIỆU ---
SAMPLES = {
    "D": {"f": "front_D_new.jpg", "s": "side_D_new.jpg", "h": 167.0, "real_hip": 93.0},
    "H": {"f": "front_H_new.jpg", "s": "side_H_new.jpg", "h": 158.0, "real_hip": 86.0},
    "L": {"f": "front_L_new.jpg", "s": "side_L_new.jpg", "h": 169.0, "real_hip": 90.0},
    "T": {"f": "front_T_new.jpg", "s": "side_T_new.jpg", "h": 163.0, "real_hip": 86.0},
    "K": {"f": "front_K_new.jpg", "s": "side_K_new.jpg", "h": 165.0, "real_hip": 93.0},
}

K_FACTOR = 2.5
ITER_LIST = [0, 2, 4, 6]
results_table = []

print("Đang xử lý mẫu dữ liệu bằng Model CV...")

for name, data in SAMPLES.items():
    img_f = cv2.imread(f"assets/{data['f']}")
    img_s = cv2.imread(f"assets/{data['s']}")
    if img_f is None or img_s is None: continue

    h_img, w_img, _ = img_f.shape
    mask_f_raw, res_f = get_body_data_research(img_f)
    mask_s_raw, res_s = get_body_data_research(img_s)

    if not res_f.pose_landmarks: continue
    lm_f = res_f.pose_landmarks.landmark
    lm_s = res_s.pose_landmarks.landmark

    # --- TÍNH RATIO VỚI K=2.5 ---
    y_nose = lm_f[0].y * h_img
    y_heel = ((lm_f[29].y + lm_f[30].y) / 2) * h_img
    head_top_offset = abs(y_nose - (lm_f[1].y * h_img)) * K_FACTOR
    ratio = data['h'] / abs(y_heel - (y_nose - head_top_offset))

    # --- TỌA ĐỘ Y CỦA MÔNG ---
    y_hip_norm = lm_f[23].y + 0.08
    y_hip_px = int(y_hip_norm * h_img)

    row_data = {"Mẫu": name, "Real Hip": data['real_hip']}
    
    for i in ITER_LIST:
        m_f = refine_body_mask_research(mask_f_raw, lm_f, h_img, i)
        m_s = refine_body_mask_research(mask_s_raw, lm_s, h_img, i)

        # Quét chiều rộng pixel tại Y Hip
        px_f = np.sum(m_f[y_hip_px, :] > 0)
        px_s = np.sum(m_s[y_hip_px, :] > 0)

        calc_hip = calculate_ramanujan(px_f * ratio, px_s * ratio)
        row_data[f"Iter_{i}"] = round(calc_hip, 2)
    
    results_table.append(row_data)

# --- XUẤT BẢNG SO SÁNH ---
df = pd.DataFrame(results_table)
print("\nBẢNG SO SÁNH CHU VI MÔNG (HIP) THEO HỆ SỐ QUÉT BIÊN")
print(df.to_string(index=False))
