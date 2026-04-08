import numpy as np
import cv2
import mediapipe as mp

mp_pose = mp.solutions.pose
mp_segmentation = mp.solutions.selfie_segmentation


def get_body_data_v5(img_bgr, debug=False):
    if img_bgr is None:
        return None, None, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    with mp_segmentation.SelfieSegmentation(model_selection=1) as seg, \
         mp_pose.Pose(static_image_mode=True, min_detection_confidence=0.5) as pose:

        res_seg = seg.process(img_rgb)
        res_pose = pose.process(img_rgb)

        mask_raw = res_seg.segmentation_mask
        mask = mask_raw > 0.5

        if debug:
            return mask, mask_raw, res_pose

        return mask, mask_raw, res_pose


def get_dimension_at_y_v5(mask, y_norm, lm_list, part_name, ratio, iterator=1):
    h_img, w_img = mask.shape
    y_pixel = min(int(y_norm * h_img), h_img - 1)

    shoulder_w = abs(lm_list[12].x - lm_list[11].x) * w_img

    base_margin = 0.2
    if part_name == 'Hip':
        base_margin = 0.25

    margin = base_margin * iterator

    x_min = max(0, int(min(lm_list[11].x, lm_list[12].x) * w_img - shoulder_w * margin))
    x_max = min(w_img - 1, int(max(lm_list[11].x, lm_list[12].x) * w_img + shoulder_w * margin))

    row = mask[y_pixel, x_min:x_max]
    white_pixels = np.where(row)[0]

    if len(white_pixels) < 2:
        return 0, 0, 0

    x1, x2 = np.min(white_pixels), np.max(white_pixels)
    width_cm = (x2 - x1) * ratio

    return width_cm, x_min + x1, x_min + x2


def process_body_measurements_v5(front_img, side_img, real_h, weight, use_long_pants=False):
    mask_f, mask_raw_f, res_f = get_body_data_v5(front_img, debug=True)
    mask_s, mask_raw_s, res_s = get_body_data_v5(side_img, debug=True)

    if not all([res_f, res_f.pose_landmarks, res_s, res_s.pose_landmarks]):
        return None, None, None, None

    h_img, w_img, _ = front_img.shape
    lm_f = res_f.pose_landmarks.landmark
    lm_s = res_s.pose_landmarks.landmark

    # ===== SCALE =====
    y_nose = lm_f[0].y * h_img
    y_heel = ((lm_f[29].y + lm_f[30].y) / 2) * h_img
    head_offset = abs(y_nose - (lm_f[1].y * h_img)) * 2.5
    ratio = real_h / abs(y_heel - (y_nose - head_offset))

    # ===== BMI CALIB =====
    bmi = weight / ((real_h / 100) ** 2)
    f_calib = 1.12 if bmi < 18.5 else (1.204 if bmi < 25 else 1.25)

    # ===== BASE =====
    shoulder = lm_f[11].y
    hip = lm_f[23].y
    torso = hip - shoulder

    # ===== FRONT (GIỮ NGUYÊN) =====
    y_map_front = {
        'Chest': shoulder + torso * 0.27,
        'Abdomen': hip - torso * 0.30,
        'Hip': hip + torso * 0.05
    }

    # ===== SIDE (FIX LỖI 5.06 → 0.06) =====
    y_map_side = {
        'Chest': shoulder + torso * 0.15,
        'Abdomen': hip - torso * 0.32,
        'Hip': hip + torso * 0.06
    }

    results = {}
    viz_f, viz_s = front_img.copy(), side_img.copy()

    print("\n" + "="*50)
    print(" DEBUG MEASUREMENTS ".center(50, "="))

    for part in ['Chest', 'Abdomen', 'Hip']:

        y_f = y_map_front[part]
        y_s = y_map_side[part]

        # iterator
        if part == 'Hip':
            iterator = 2.2 if use_long_pants else 1.8
        else:
            iterator = 1

        # scan
        w_v, x1f, x2f = get_dimension_at_y_v5(
            mask_f, y_f, lm_f, part, ratio, iterator
        )

        d_v, x1s, x2s = get_dimension_at_y_v5(
            mask_s, y_s, lm_s, part, ratio, iterator
        )

        # ellipse
        a, b = w_v / 2, d_v / 2
        h_el = ((a - b)**2) / ((a + b)**2) if (a + b) != 0 else 0

        circum_raw = np.pi * (a + b) * (
            1 + (3 * h_el) / (10 + np.sqrt(4 - 3 * h_el))
        ) if (a + b) != 0 else 0

        circum_final = round(circum_raw * f_calib, 2)
        results[part] = circum_final

        # ===== LOG CHUẨN =====
        print(f"\n--- {part} ---")
        print(f"y_front: {y_f:.3f} | y_side: {y_s:.3f}")
        print(f"Width (front): {w_v:.2f}")
        print(f"Depth (side): {d_v:.2f}")
        print(f"Circum raw: {circum_raw:.2f}")
        print(f"Circum final: {circum_final:.2f}")

        # draw
        y_px_f = int(y_f * h_img)
        y_px_s = int(y_s * h_img)

        cv2.line(viz_f, (int(x1f), y_px_f), (int(x2f), y_px_f), (0, 255, 0), 3)
        cv2.line(viz_s, (int(x1s), y_px_s), (int(x2s), y_px_s), (0, 255, 0), 3)

    print("="*50)

    debug_pack = {
        "mask_f": mask_f,
        "mask_s": mask_s,
        "mask_raw_f": mask_raw_f,
        "mask_raw_s": mask_raw_s
    }

    return results, viz_f, viz_s, debug_pack