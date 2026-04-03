import numpy as np
import cv2
import streamlit as st
import mediapipe as mp

# Khởi tạo Mediapipe sẵn ở ngoài để tối ưu hiệu suất
mp_pose = mp.solutions.pose
mp_segmentation = mp.solutions.selfie_segmentation
mp_drawing = mp.solutions.drawing_utils

def get_body_data(img_bgr: np.ndarray):
    """Trích xuất Mask và Pose Landmarks từ ảnh."""
    if img_bgr is None:
        return None, None
    
    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)
    
    # Thực hiện Segmentation và Pose đồng thời
    with mp_segmentation.SelfieSegmentation(model_selection=1) as seg, \
        mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        # Confidence 0.5 giúp lọc các phát hiện yếu, chỉ giữ lại những phát hiện có độ tin cậy cao hơn, giảm nhiễu từ các phát hiện sai hoặc không rõ ràng.
        res_seg = seg.process(img_rgb)
        res_pose = pose.process(img_rgb)
        
        mask = res_seg.segmentation_mask > 0.5
        return mask, res_pose
    # Đoạn code này dùng để trích xuất dữ liệu từ ảnh, bao gồm cả mask và pose landmarks. 
    # Việc sử dụng context manager (with) giúp đảm bảo rằng các tài nguyên được giải phóng đúng cách sau khi sử dụng, đồng thời giữ cho code gọn gàng và dễ đọc hơn. 
    # Nếu có lỗi trong quá trình xử lý, nó sẽ trả về None, giúp tránh lỗi tiếp theo khi cố gắng truy cập vào kết quả không hợp lệ.

def refine_body_mask(mask_raw, landmarks, h_img, w_img, iter_upper, iterations_lower):
    """
    Tách Mask thành 2 phần: Thân trên (giữ nguyên) và Thân dưới (bào mòn theo iterations).
    """
    # 1. Xác định tọa độ Y của Hông (Landmark 23: Left Hip, 24: Right Hip)
    y_hip_ratio = (landmarks[23].y + landmarks[24].y) / 2
    y_hip_px = int(y_hip_ratio * h_img)

    # 2. Tạo Kernel cho phép bào mòn
    kernel = np.ones((3, 3), np.uint8)

    # --- XỬ LÝ THÂN TRÊN (Từ đỉnh đầu đến Hông) ---
    mask_upper = mask_raw.copy()
    mask_upper[y_hip_px:, :] = 0  # Xóa bỏ phần dưới hông
    # Thân trên cởi trần nên chỉ quét 1 lần để khử nhiễu răng cưa biên
    mask_upper_refined = cv2.erode(mask_upper.astype(np.uint8), kernel, iterations=iter_upper)

    # --- XỬ LÝ THÂN DƯỚI (Từ Hông xuống Gót chân) ---
    mask_lower = mask_raw.copy()
    mask_lower[:y_hip_px, :] = 0  # Xóa bỏ phần trên hông
    # Thân dưới mặc quần nên bào mòn theo tham số iterations truyền vào
    mask_lower_refined = cv2.erode(mask_lower.astype(np.uint8), kernel, iterations=iterations_lower)

    # 3. Kết hợp 2 vùng Mask lại
    combined_mask = cv2.bitwise_or(mask_upper_refined, mask_lower_refined)
    
    return combined_mask.astype(bool)
    
def get_dimension_at_y(mask, y_norm, lm_list, part_name, view, ratio, is_loose):
    # Giải thích các tham số:
    # mask: Mặt nạ nhị phân của cơ thể đã được xử lý.
    # y_norm: Tỷ lệ Y (0-1) để xác định vị trí quét trên ảnh.
    # lm_list: Danh sách các landmarks của pose để xác định vị trí vai và điều chỉnh vùng quét.
    # part_name: Tên bộ phận đang đo (Chest, Abdomen, Hip, Thigh).
    # view: Góc nhìn (front hoặc side) để áp dụng logic xử lý riêng cho đùi.
    # ratio: Tỷ lệ chuyển đổi từ pixel sang cm.
    # is_loose: Biến boolean để điều chỉnh logic xử lý cho loose fit, giúp tránh việc đo được kích thước quá lớn do quần áo rộng.
    """Quét biên để lấy chiều rộng (cm) tại tọa độ Y xác định."""
    h_img, w_img = mask.shape
    y_pixel = min(int(y_norm * h_img), h_img - 1)
    # Giải thích y_norm và y_pixel:
    # y_norm là một giá trị tỷ lệ từ 0 đến 1, đại diện cho vị trí dọc trên ảnh, với 0 là đỉnh đầu và 1 là gót chân. 
    # Khi nhân y_norm với chiều cao của ảnh (h_img), ta sẽ có được vị trí pixel tương ứng trên trục Y. 
    # Hàm min được sử dụng để đảm bảo rằng y_pixel không vượt quá giới hạn của ảnh, tránh lỗi truy cập ngoài phạm vi mảng.
    # Việc sử dụng y_norm giúp cho code linh hoạt hơn, có thể áp dụng cho các ảnh có kích thước khác nhau mà không cần phải điều chỉnh lại các giá trị pixel cụ thể.

    # Giới hạn vùng quét X dựa trên vai để tránh nhiễu tay
    shoulder_w = abs(lm_list[12].x - lm_list[11].x) * w_img
    x_min = max(0, int(min(lm_list[11].x, lm_list[12].x) * w_img - shoulder_w * 0.2))
    x_max = min(w_img - 1, int(max(lm_list[11].x, lm_list[12].x) * w_img + shoulder_w * 0.2))
    # Giải thích logic giới hạn vùng quét X:
    # Để tránh nhiễu từ tay khi đo các bộ phận như ngực hoặc bụng, chúng ta giới hạn vùng quét trên trục X dựa vào vị trí của vai.
    row = mask[y_pixel, x_min:x_max]
    white_pixels = np.where(row)[0]
    
    if len(white_pixels) < 2:
        return 0, 0, 0

    x1, x2 = np.min(white_pixels), np.max(white_pixels)

    # Logic xử lý riêng cho đùi (tránh dính 2 chân)
    if part_name == 'Thigh' and view == 'front':
        diff = np.diff(white_pixels)
        gap = np.where(diff > 7)[0]
        if len(gap) > 0:
            x2 = white_pixels[gap[0]]
        else:
            max_th = shoulder_w * (0.45 if is_loose else 0.55)
            x2 = min(x2, x1 + max_th)

    width_cm = (x2 - x1) * ratio
    return width_cm, x_min + x1, x_min + x2

def calculate_ramanujan_circumference(w_cm: float, d_cm: float) -> float:
    """Tính chu vi hình elip bằng công thức Ramanujan."""
    a, b = w_cm / 2, d_cm / 2
    if a <= 0 or b <= 0:
        return 0.0
    h_el = ((a - b)**2) / ((a + b)**2)
    circum = np.pi * (a + b) * (1 + (3 * h_el) / (10 + np.sqrt(4 - 3 * h_el)))
    return circum
    # Đoạn code này tính chu vi của một hình elip dựa trên chiều rộng và chiều sâu đã được chuyển đổi sang cm.

def process_body_measurements(front_img, side_img, real_h, age, weight, is_loose=False, is_raw=False):
    # 1. Thu thập dữ liệu
    mask_f_raw, res_f = get_body_data(front_img)
    mask_s_raw, res_s = get_body_data(side_img)

    if not all([res_f, res_f.pose_landmarks, res_s, res_s.pose_landmarks]):
        st.warning("Không tìm thấy người rõ ràng trong ảnh.")
        return None, None, None

    # 1.5 Lấy thông số ảnh và Landmarks trước để làm căn cứ phân vùng ---
    h_img, w_img, _ = front_img.shape
    lm_f = res_f.pose_landmarks.landmark 
    lm_s = res_s.pose_landmarks.landmark

    # 2. Tiền xử lý Mask phân vùng (Regional Refinement) ---
    # Gọi hàm xử lý riêng cho thân trên (giữ eo) và thân dưới (gọt quần)
    if is_raw:
        # Chế độ THÔ: Không bào mòn để xem AI quét rộng đến mức nào
        mask_f = mask_f_raw.astype(bool)
        mask_s = mask_s_raw.astype(bool)
    else:
        # Chế độ Hiệu chỉnh: Áp dụng logic Thân trên/Thân dưới như cũ
        mask_f = refine_body_mask(mask_f_raw, lm_f, h_img, w_img, iter_upper=1, iterations_lower=(2 if is_loose else 1))
        mask_s = refine_body_mask(mask_s_raw, lm_s, h_img, w_img, iter_upper=1, iterations_lower=(2 if is_loose else 1))
    # Giải thích cho báo cáo:
    # Tọa độ y của Landmark (0-1) được nhân với h_img để xác định ranh giới Hông.
    # Thân trên (cởi trần): Áp dụng i=1 để bảo toàn chỉ số eo/ngực.
    # Thân dưới (mặc quần): Áp dụng i=iter_val để triệt tiêu độ phồng trang phục.

    # 3. Tính toán tỉ lệ (Pixel to CM)
    y_nose = lm_f[0].y * h_img
    y_heel = ((lm_f[29].y + lm_f[30].y) / 2) * h_img
    head_top_offset = abs(y_nose - (lm_f[1].y * h_img)) * 2.5
    ratio = real_h / abs(y_heel - (y_nose - head_top_offset))
    
    # 3.5 Tính toán BMI để thích nghi hệ số bù
    bmi = weight / ((real_h / 100) ** 2)

    # THIẾT LẬP HỆ SỐ BÙ THEO BMI (Adaptive F-Factor)
    # Con số 1.204 là trung bình hệ thống, ta lấy nó làm mốc cho nhóm Cân đối
    if bmi < 18.5:
        f_calibration = 1.12  # Người gầy: Biên sắc nét, AI ít lẹm
    elif bmi < 25:
        f_calibration = 1.204 # Mốc chuẩn từ thực nghiệm 6 mẫu của ông
    elif bmi < 30:
        f_calibration = 1.24  # Tiền béo phì: Mô mềm, AI dễ cắt lẹm biên mỡ
    else:
        f_calibration = 1.28  # Béo phì: Cần bù mạnh nhất

    # 4. Cấu hình các điểm đo Y
    y_map = {
        # 1. NGỰC (Chest): Tính từ VAI (11) xuống HÔNG (23)
        # Lấy đoạn Thân trên, chia làm 10 phần, lấy từ trên xuống 2.3 phần (mốc 23%).
        # Đây là vị trí đi ngang qua bả vai và bầu ngực.
        'Chest': lm_f[11].y + (lm_f[23].y - lm_f[11].y) * 0.23, 

        # 2. BỤNG (Abdomen): Tính từ XƯƠNG CHẬU (23) ngược lên trên
        # Trừ đi 0.05 (tương đương ~5-8cm tùy chiều cao) để lên vùng bụng trên/ngang rốn.
        # Lưu ý: Đây là nơi đo "vòng bụng" để may áo/quần cho thoải mái.
        'Abdomen': lm_f[23].y - 0.05,

        # 3. MÔNG/HÔNG (Hip): Tính từ XƯƠNG CHẬU (23) đi xuống dưới
        # Cộng thêm 0.08 để dịch thước đo xuống đúng phần "nở" nhất của mông.
        # Trong may mặc, đây là "Vòng 3" - quyết định size quần.
        'Hip': lm_f[23].y + 0.08,

        # 4. ĐÙI (Thigh): Tính từ HÔNG (23) xuống ĐẦU GỐI (25)
        # Lấy đoạn đùi, chia làm 10 phần, lấy từ trên xuống 3 phần (mốc 30%).
        # Đây là vị trí đùi to nhất (sát bẹn), giúp đo vòng đùi chính xác.
        'Thigh': lm_f[23].y + (lm_f[25].y - lm_f[23].y) * 0.3
    }

    # 5. Tính toán chu vi và Filter
    final_results = {}
    viz_f, viz_s = front_img.copy(), side_img.copy()

    if is_raw:
        # Chế độ Raw: Không bù, không gọt
        f_final = 1.0
        s_cloth = 1.0
    else:
        # Chế độ AI Scan/Heuristic: Có bù lẹm BMI
        f_final = f_calibration 
        # Chỉ gọt vải (s_cloth < 1.0) khi ở Tab Heuristic (is_loose=True)
        s_cloth = 0.95 if is_loose else 1.0

    # ÁP DỤNG HỆ SỐ VÀO TỪNG BỘ PHẬN
    # Lưu ý: Chest và Abdomen thường ít bị ảnh hưởng bởi quần rộng (Heuristic) 
    # nên s_cloth chủ yếu tác động vào Hip và Thigh.
    f_map = {
        'Chest':   f_final, 
        'Abdomen': f_final,
        'Hip':     f_final * s_cloth, # Nhân thêm s_cloth để gọt quần rộng
        'Thigh':   f_final * s_cloth
    }

    for part, y_coord in y_map.items():
        w_v, x1f, x2f = get_dimension_at_y(mask_f, y_coord, lm_f, part, 'front', ratio, is_loose)
        d_v, x1s, x2s = get_dimension_at_y(mask_s, y_coord, lm_s, part, 'side', ratio, is_loose)
        
        # 2. Tính toán dựa trên f_map đã định nghĩa
        circum_raw = calculate_ramanujan_circumference(w_v, d_v)
        val = circum_raw * f_map[part] # Nhân trực tiếp hệ số tương ứng
        
        # 3. Ép kết quả cuối cùng vào khoảng hợp lý (20-180cm) và làm tròn 2 chữ số thập phân
        final_results[part] = round(max(min(val, 180.0), 20.0), 2)
        
        # Vẽ trực quan
        y_pixel = int(y_coord * h_img)

        # Vẽ đường đo Width trên ảnh Front (Màu Đỏ)
        cv2.line(viz_f, (int(x1f), y_pixel), (int(x2f), y_pixel), (255, 255, 255), 3)
        # Thêm 2 điểm mút để nhìn rõ biên quét
        cv2.circle(viz_f, (int(x1f), y_pixel), 4, (255, 0, 0), -1)
        cv2.circle(viz_f, (int(x2f), y_pixel), 4, (255, 0, 0), -1)

        # Vẽ đường đo Depth trên ảnh Side (Màu Đỏ)
        cv2.line(viz_s, (int(x1s), y_pixel), (int(x2s), y_pixel), (0, 0, 255), 3)
        cv2.circle(viz_s, (int(x1s), y_pixel), 4, (255, 0, 0), -1)
        cv2.circle(viz_s, (int(x2s), y_pixel), 4, (255, 0, 0), -1)  

    # 6. Tạo Pipeline Visuals (Giữ nguyên logic của bạn nhưng gọn hơn)
    def create_pipeline(img, mask, res_pose, viz_img):
        mask_viz = cv2.cvtColor((mask.astype(np.uint8)*255), cv2.COLOR_GRAY2BGR)
        skel = img.copy()
        mp_drawing.draw_landmarks(skel, res_pose.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        def r(i): return cv2.resize(i, (300, 450))
        return np.vstack([np.hstack([r(img), r(mask_viz)]), np.hstack([r(skel), r(viz_img)])])

    pipeline_f = create_pipeline(front_img, mask_f, res_f, viz_f)
    pipeline_s = create_pipeline(side_img, mask_s, res_s, viz_s)

    return final_results, pipeline_f, pipeline_s