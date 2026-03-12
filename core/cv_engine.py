import cv2
import numpy as np
import mediapipe as mp

# Khởi tạo MediaPipe Pose
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

def estimate_measurements(image_path, real_height_cm):
    # Đọc ảnh từ đường dẫn
    image = cv2.imread(image_path)
    if image is None:
        print("Không thể đọc ảnh!")
        return None

    # Chuyển đổi màu sang RGB vì MediaPipe làm việc trên không gian màu này
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    with mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:
        results = pose.process(image_rgb)
        
        if not results.pose_landmarks:
            print("Không tìm thấy người trong ảnh.")
            return None
        
        # Lấy danh sách các điểm mốc (landmarks)
        landmarks = results.pose_landmarks.landmark
        h, w, _ = image.shape 

        # --- 1. XỬ LÝ ĐỈNH ĐẦU VÀ GÓT CHÂN ---
        # Lấy tọa độ Y của mũi và gót chân
        nose_y = landmarks[mp_pose.PoseLandmark.NOSE].y
        l_heel_y = landmarks[mp_pose.PoseLandmark.LEFT_HEEL].y
        r_heel_y = landmarks[mp_pose.PoseLandmark.RIGHT_HEEL].y
        foot_y = (l_heel_y + r_heel_y) / 2 # Trung bình cộng 2 gót chân

        # TÍNH TOÁN ĐỈNH ĐẦU: 
        # Vì MediaPipe không có điểm đỉnh đầu, ta dựa vào khoảng cách từ mắt đến mũi
        # Thông thường đỉnh đầu cách mũi một khoảng bằng ~7% tổng chiều cao cơ thể
        body_height_nose_to_foot = abs(foot_y - nose_y)
        head_top_y = nose_y - (body_height_nose_to_foot * 0.07) # Trừ Y để đi ngược lên trên ảnh

        # Chiều cao pixel thực tế sau khi đã bù đỉnh đầu
        pixel_height = abs(foot_y - head_top_y)

        # --- 2. TÍNH TỶ LỆ QUY ĐỔI ---
        # Tỷ lệ này dùng để nhân với các khoảng cách pixel khác để ra cm
        ratio = real_height_cm / pixel_height

        # --- 3. TÍNH CHIỀU RỘNG VAI ---
        l_sh = landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER]
        r_sh = landmarks[mp_pose.PoseLandmark.RIGHT_SHOULDER]
        
        # Công thức tính khoảng cách 2D (Euclidean distance) giữa 2 vai
        shoulder_width_px = np.sqrt((l_sh.x - r_sh.x)**2 + (l_sh.y - r_sh.y)**2)
        shoulder_cm = shoulder_width_px * ratio

        # --- 4. ƯỚC TÍNH CÁC SỐ ĐO VÒNG (Dựa trên nhân trắc học) ---
        # Hệ số 2.5 cho ngực dựa trên thực tế vòng ngực bao quanh cả trước và sau
        chest_cm = shoulder_cm * 2.5   
        abdomen_cm = chest_cm * 0.85   # Bụng thường nhỏ hơn ngực ở người cân đối
        hip_cm = chest_cm * 0.95       # Mông

        # --- 5. VẼ LÊN ẢNH ĐỂ TRỰC QUAN HÓA ---
        annotated_image = image.copy()
        # Vẽ các điểm mốc cơ bản
        mp_drawing.draw_landmarks(annotated_image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        # Vẽ thêm một đường kẻ ngang tại vị trí "đỉnh đầu ảo" để kiểm tra
        head_y_px = int(head_top_y * h)
        cv2.line(annotated_image, (0, head_y_px), (w, head_y_px), (0, 255, 0), 2)

        return {
            "shoulder": round(shoulder_cm, 2),
            "chest": round(chest_cm, 2),
            "abdomen": round(abdomen_cm, 2),
            "hip": round(hip_cm, 2),
            "image": annotated_image
        }

# --- CHẠY THỬ ---
height_input = 170 # Chiều cao thật của bạn
results = estimate_measurements('person.jpg', height_input)

if results:
    print(f"--- Kết quả sau khi bù đỉnh đầu ({height_input}cm) ---")
    print(f"Rộng vai: {results['shoulder']} cm")
    print(f"Vòng ngực (ước tính): {results['chest']} cm")
    print(f"Vòng bụng (ước tính): {results['abdomen']} cm")
    print(f"Vòng mông (ước tính): {results['hip']} cm")
    
    # Hiển thị ảnh
    cv2.imshow("Kiem tra dinh dau (duong mau xanh)", results['image'])
    cv2.waitKey(0)
    cv2.destroyAllWindows()