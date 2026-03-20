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

def process_body_measurements(front_img, side_img, real_h, age, weight, is_loose=False):
    # 1. Thu thập dữ liệu
    mask_f_raw, res_f = get_body_data(front_img)
    mask_s_raw, res_s = get_body_data(side_img)

    if not all([res_f, res_f.pose_landmarks, res_s, res_s.pose_landmarks]):
        st.warning("Không tìm thấy người rõ ràng trong ảnh.")
        return None, None, None

    # 2. Tiền xử lý Mask
    kernel = np.ones((3, 3), np.uint8)
    iterations = 6 if is_loose else 1
    mask_f = cv2.erode(mask_f_raw.astype(np.uint8), kernel, iterations=iterations).astype(bool)
    mask_s = cv2.erode(mask_s_raw.astype(np.uint8), kernel, iterations=iterations).astype(bool)

    lm_f = res_f.pose_landmarks.landmark 
    lm_s = res_s.pose_landmarks.landmark
    # Lấy danh sách 33 điểm mốc (từ mũi, vai đến gót chân) của cả 2 ảnh (front_img và side_img).
    # Các tọa độ x, y trong landmark lúc này đang ở dạng tỉ lệ (từ 0 đến 1). 
    # Ví dụ y=0.5 nghĩa là điểm đó nằm ở giữa chiều cao ảnh, chưa phải là số pixel thực tế.
    h_img, w_img, _ = front_img.shape
    # h_img (Height): Chiều cao ảnh (ví dụ 1920 pixel).
    # w_img (Width): Chiều rộng ảnh (ví dụ 1080 pixel).
    # _: Bỏ qua thông số về kênh màu (RGB).
    # Mục đích: Cần hai con số này để nhân với tọa độ tỉ lệ ở trên. 
    # Nếu điểm vai có y=0.2 và ảnh cao 1000px, thì vị trí vai thực tế trên ảnh là 0.2 x 1000 = 200px.

    # 3. Tính toán tỉ lệ (Pixel to CM)
    y_nose = lm_f[0].y * h_img
    y_heel = ((lm_f[29].y + lm_f[30].y) / 2) * h_img
    head_top_offset = abs(y_nose - (lm_f[1].y * h_img)) * 2.5 
    ratio = real_h / abs(y_heel - (y_nose - head_top_offset))

    # 4. Cấu hình các điểm đo Y
    y_map = {
        # 1. NGỰC (Chest): Tính từ VAI (11) xuống HÔNG (23)
        # Lấy đoạn Thân trên, chia làm 10 phần, lấy từ trên xuống 3 phần (mốc 30%).
        # Đây là vị trí đi ngang qua bả vai và bầu ngực.
        'Chest': lm_f[11].y + (lm_f[23].y - lm_f[11].y) * 0.3, 

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
    
    # Hệ số hiệu chỉnh
    # Nếu là loose fit thì sẽ dùng adj_factor và loose_factor để điều chỉnh kết quả, giúp tránh việc đo được kích thước quá lớn do quần áo rộng.
    adj_factor = 0.88 if is_loose else 1.0
    loose_factor = 0.93 if is_loose else 1.08
    # adj_factor giúp điều chỉnh kết quả cho loose fit, giảm kích thước 12% để phù hợp với quần áo rộng. Tránh việc tăng quá nhiều sẽ dẫn đến kết quả không thực tế.
    # loose_factor giúp điều chỉnh chu vi, giảm 7% cho loose fit để tránh kết quả quá lớn do quần áo rộng. Nếu đồ sát cơ, tăng 8% để bù lại sự chặt chẽ.
    
    max_limits = {
        'Chest': weight * 1.65 * adj_factor, 
        'Abdomen': weight * 1.25 * adj_factor, 
        'Hip': weight * 1.50 * adj_factor, 
        'Thigh': weight * 0.85 * adj_factor
        # max_limits giúp tránh kết quả phi lý, dựa trên cân nặng và hệ số điều chỉnh cho loose fit. 
        # Nhưng nếu dữ liệu vào không cần điều chỉnh, thì các hệ số sẽ không được áp dụng.
        # Nếu đối tượng đo là vận động viên thể hình hoặc người béo phì, có thể cần điều chỉnh lại các hệ số này hoặc loại bỏ hoàn toàn để tránh giới hạn quá mức.
    }

    for part, y_coord in y_map.items():
        w_v, x1f, x2f = get_dimension_at_y(mask_f, y_coord, lm_f, part, 'front', ratio, is_loose)
        d_v, x1s, x2s = get_dimension_at_y(mask_s, y_coord, lm_s, part, 'side', ratio, is_loose)
        
        # Tính toán hình học
        circum = calculate_ramanujan_circumference(w_v, d_v) * loose_factor
        
        # Hậu xử lý kết quả
        if is_loose and part in ['Hip', 'Thigh']: 
            circum *= 0.98
            
        val = max(circum, 20.0) # Không có bộ phận nào < 20cm
        final_results[part] = round(min(val, max_limits[part]), 2)
        
        # Vẽ trực quan hóa
        cv2.line(viz_f, (int(x1f), int(y_coord*h_img)), (int(x2f), int(y_coord*h_img)), (0, 255, 0), 3)
        cv2.line(viz_s, (int(x1s), int(y_coord*h_img)), (int(x2s), int(y_coord*h_img)), (0, 255, 0), 3)

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