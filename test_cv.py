import cv2
import mediapipe as mp
import numpy as np
import time

# Khởi tạo công cụ
mp_pose = mp.solutions.pose
mp_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

def process_body_measurements(front_path, side_path, real_h, age, weight):
    print(f"🚀 [INIT] Age: {age}, W: {weight}kg, H: {real_h}cm")

    # --- BƯỚC 1: ĐỌC VÀ TÁCH NỀN ---
    def get_data(path):
        img = cv2.imread(path)
        if img is None: return None, None, None
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        with mp_segmentation.SelfieSegmentation(model_selection=1) as seg:
            res_seg = seg.process(img_rgb)
            mask = res_seg.segmentation_mask > 0.5
        with mp_pose.Pose(static_image_mode=True) as pose:
            res_pose = pose.process(img_rgb)
        return img, mask, res_pose

    img_f, mask_f, res_f = get_data(front_path)
    img_s, mask_s, res_s = get_data(side_path)

    if img_f is None or img_s is None:
        print("❌ Lỗi: Không tìm thấy ảnh!")
        return

    # --- BƯỚC 2: TÍNH TỈ LỆ PIXEL/CM ---
    lm_f = res_f.pose_landmarks.landmark
    lm_s = res_s.pose_landmarks.landmark
    h, w, _ = img_f.shape
    
    # Tính toán chiều cao pixel từ đỉnh đầu đến gót chân để lấy tỉ lệ
    y_nose = lm_f[0].y * h
    y_heel = ((lm_f[29].y + lm_f[30].y) / 2) * h
    ratio = (real_h * 0.94) / abs(y_heel - y_nose) 

    # --- BƯỚC 3: HÀM QUÉT PIXEL THÔNG MINH (ROI + THIGH GAP) ---
    def get_dim_smart_v2(mask, y_norm, lm_list, part_name, view='front'):
        curr_h, curr_w = mask.shape
        y_pixel = int(y_norm * curr_h)
        
        # Xác định vùng giới hạn (ROI) để tránh lấy nhầm cánh tay
        if view == 'front':
            x_left = lm_list[12].x * curr_w
            x_right = lm_list[11].x * curr_w
            margin = abs(x_left - x_right) * 0.2
            x_start, x_end = int(min(x_left, x_right) - margin), int(max(x_left, x_right) + margin)
        else:
            center_x = lm_list[24].x * curr_w
            x_start, x_end = int(center_x - curr_w*0.15), int(center_x + curr_w*0.15)

        x_start, x_end = max(0, x_start), min(curr_w-1, x_end)
        row = mask[y_pixel, x_start:x_end]
        pixels = np.where(row)[0]
        
        if len(pixels) < 2: return 0, 0, 0

        # LOGIC ĐẶC BIỆT CHO ĐÙI (Tránh đo dính 2 chân hoặc khoảng trống giữa đùi)
        if part_name == 'Thigh' and view == 'front':
            diff = np.diff(pixels)
            splits = np.where(diff > 5)[0] # Nếu có khoảng trắng > 5 pixel
            if len(splits) > 0:
                # Chỉ lấy cụm pixel đầu tiên (đùi bên trái của ảnh)
                pixels = pixels[:splits[0]]

        width_cm = (pixels[-1] - pixels[0]) * ratio

        # TRỪ HAO QUẦN ÁO (Mông & Đùi)
        if part_name == 'Hip':
            width_cm *= 0.96 # Trừ khoảng 3-4% do quần rộng
        if part_name == 'Thigh':
            width_cm *= 0.97

        return width_cm, x_start + pixels[0], x_start + pixels[-1]

    # --- BƯỚC 4: VỊ TRÍ ĐO CHUẨN (TI, RỐN, MÔNG, ĐÙI) ---
    y_map = {
        'Chest': lm_f[11].y + (lm_f[23].y - lm_f[11].y) * 0.32, # Ngang ti (khoảng 1/3 thân trên)
        'Abdomen': lm_f[23].y - 0.04,                          # Ngang rốn (ngay trên xương hông)
        'Hip': lm_f[23].y + 0.05,                               # Điểm lớn nhất của mông/quần
        'Thigh': lm_f[23].y + (lm_f[25].y - lm_f[23].y) * 0.3    # 1/3 đùi tính từ trên xuống
    }

    final_results = {}
    viz_f, viz_s = img_f.copy(), img_s.copy()

    for part, y in y_map.items():
        w_val, x1f, x2f = get_dim_smart_v2(mask_f, y, lm_f, part, 'front')
        d_val, x1s, x2s = get_dim_smart_v2(mask_s, y, lm_s, part, 'side')
        
        # Công thức Elip chuẩn
        a, b = w_val/2, d_val/2
        if a > 0 and b > 0:
            h_el = ((a - b)**2) / ((a + b)**2)
            circum = np.pi * (a + b) * (1 + (3 * h_el) / (10 + np.sqrt(4 - 3 * h_el)))
        else: circum = 0
            
        final_results[part] = circum
        
        # Vẽ đường đo để debug
        cv2.line(viz_f, (x1f, int(y*h)), (x2f, int(y*h)), (0, 255, 0), 3)
        cv2.line(viz_s, (x1s, int(y*img_s.shape[0])), (x2s, int(y*img_s.shape[0])), (0, 255, 0), 3)
        print(f"📏 {part:7}: CV={circum:6.2f}cm (W={w_val:.1f}, D={d_val:.1f})")

    # --- BƯỚC 5: HIỂN THỊ PIPELINE TỔNG HỢP ---
    # Tạo mask 3 kênh để ghép ảnh
    mask_viz = cv2.cvtColor((mask_f * 255).astype(np.uint8), cv2.COLOR_GRAY2BGR)
    # Vẽ skeleton lên một bản sao
    skeleton_viz = img_f.copy()
    mp_drawing.draw_landmarks(skeleton_viz, res_f.pose_landmarks, mp_pose.POSE_CONNECTIONS)

    # Ghép 4 bước: Gốc | Mask | Khung xương | Kết quả đo
    top_row = np.hstack([cv2.resize(img_f, (300, 450)), cv2.resize(mask_viz, (300, 450))])
    bottom_row = np.hstack([cv2.resize(skeleton_viz, (300, 450)), cv2.resize(viz_f, (300, 450))])
    pipeline = np.vstack([top_row, bottom_row])

    cv2.imshow("BODY ANALYSIS PIPELINE", pipeline)
    cv2.imshow("SIDE VIEW ANALYSIS", cv2.resize(viz_s, (400, 600)))
    
    print("\n✅ KIỂM TRA: Đường màu xanh trên ảnh có khớp với vị trí Ti, Rốn và đùi đơn lẻ không?")
    cv2.waitKey(0)
    cv2.destroyAllWindows()
    return final_results

if __name__ == "__main__":
    process_body_measurements("assets/front_T2.jpg", "assets/side_T2.jpg", 163.0, 22, 61.0)