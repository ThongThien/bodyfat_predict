import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
mp_segmentation = mp.solutions.selfie_segmentation

SEG_MODEL = mp_segmentation.SelfieSegmentation(model_selection=1)
POSE_MODEL = mp_pose.Pose(
    static_image_mode=True,
    min_detection_confidence=0.5,
)


def get_body_data_v5(img_bgr, debug=False):
    """Extract body segmentation mask and pose landmarks from an input image."""
    if img_bgr is None:
        return None, None, None

    img_rgb = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2RGB)

    seg_result = SEG_MODEL.process(img_rgb)
    pose_result = POSE_MODEL.process(img_rgb)

    mask_raw = seg_result.segmentation_mask
    mask = mask_raw > 0.5

    return mask, mask_raw, pose_result


def get_dimension_at_y_v5(mask, y_norm, lm_list, part_name, ratio, iterator=1):
    """Estimate body width at a specific normalized Y position."""
    img_h, img_w = mask.shape
    y_pixel = min(int(y_norm * img_h), img_h - 1)

    shoulder_width = abs(lm_list[12].x - lm_list[11].x) * img_w

    base_margin = 0.25 if part_name == "Hip" else 0.2
    margin = base_margin * iterator

    x_min = max(
        0,
        int(min(lm_list[11].x, lm_list[12].x) * img_w - shoulder_width * margin),
    )
    x_max = min(
        img_w - 1,
        int(max(lm_list[11].x, lm_list[12].x) * img_w + shoulder_width * margin),
    )

    row = mask[y_pixel, x_min:x_max]
    white_pixels = np.where(row)[0]

    if len(white_pixels) < 2:
        return 0, 0, 0

    x1 = np.min(white_pixels)
    x2 = np.max(white_pixels)

    width_cm = (x2 - x1) * ratio

    return width_cm, x_min + x1, x_min + x2


def find_best_depth(mask, y_center, lm_list, part_name, ratio, iterator):
    """Search nearby Y positions to find the best side-view body depth."""
    offsets = [-0.03, -0.02, -0.01, 0, 0.01, 0.02, 0.03]

    best_depth = 0
    best_y = y_center
    best_x1 = 0
    best_x2 = 0

    for offset in offsets:
        y_try = y_center + offset

        depth, x1, x2 = get_dimension_at_y_v5(
            mask,
            y_try,
            lm_list,
            part_name,
            ratio,
            iterator,
        )

        if depth > best_depth:
            best_depth = depth
            best_y = y_try
            best_x1 = x1
            best_x2 = x2

    return best_depth, best_y, best_x1, best_x2


def get_iterator(bmi, part_name, use_long_pants=False):
    """Adjust search range based on BMI and body part."""
    if part_name == "Hip":
        base = 1.7 if use_long_pants else 1.6
    else:
        base = 1.0

    scale = 1 + (bmi - 22) * 0.02

    return np.clip(base * scale, 0.8, 2.2)


def process_body_measurements_v5(
    front_img,
    side_img,
    real_h,
    weight,
    use_long_pants=False,
):
    """Estimate chest, abdomen and hip circumferences from front and side images."""
    mask_f, mask_raw_f, pose_f = get_body_data_v5(front_img, debug=True)
    mask_s, mask_raw_s, pose_s = get_body_data_v5(side_img, debug=True)

    if not all([pose_f, pose_f.pose_landmarks, pose_s, pose_s.pose_landmarks]):
        return None, None, None, None, None

    img_h, img_w, _ = front_img.shape

    lm_f = pose_f.pose_landmarks.landmark
    lm_s = pose_s.pose_landmarks.landmark

    pose_visibility_score = np.mean([lm.visibility for lm in lm_f])
    missing_landmark_count = sum(1 for lm in lm_f if lm.visibility < 0.5)
    mask_confidence = float(np.mean(mask_raw_f))

    # Calculate pixel-to-centimeter scale using height estimation.
    y_nose = lm_f[0].y * img_h
    y_heel = ((lm_f[29].y + lm_f[30].y) / 2) * img_h
    head_offset = abs(y_nose - (lm_f[1].y * img_h)) * 2.5
    ratio = real_h / abs(y_heel - (y_nose - head_offset))

    # BMI-based calibration factor.
    bmi = weight / ((real_h / 100) ** 2)

    if bmi < 18.5:
        calibration_factor = 1.138
    elif bmi < 25:
        calibration_factor = 1.222
    else:
        calibration_factor = 1.258

    # Define target Y positions for each measurement area.
    shoulder_y = lm_f[11].y
    hip_y = lm_f[23].y
    torso_height = hip_y - shoulder_y

    y_map_front = {
        "Chest": shoulder_y + torso_height * 0.27,
        "Abdomen": hip_y - torso_height * 0.30,
        "Hip": hip_y + torso_height * 0.05,
    }

    results = {}
    viz_f = front_img.copy()
    viz_s = side_img.copy()

    for part_name in ["Chest", "Abdomen", "Hip"]:
        y_front = y_map_front[part_name]

        iterator = get_iterator(bmi, part_name, use_long_pants)

        width, x1_front, x2_front = get_dimension_at_y_v5(
            mask_f,
            y_front,
            lm_f,
            part_name,
            ratio,
            iterator,
        )

        depth, y_side, x1_side, x2_side = find_best_depth(
            mask_s,
            y_front,
            lm_s,
            part_name,
            ratio,
            iterator,
        )

        if width == 0 or depth == 0:
            continue

        # Estimate circumference using ellipse approximation.
        a = width / 2
        b = depth / 2

        if a + b == 0:
            circumference_raw = 0
        else:
            h_el = ((a - b) ** 2) / ((a + b) ** 2)
            circumference_raw = np.pi * (a + b) * (
                1 + (3 * h_el) / (10 + np.sqrt(4 - 3 * h_el))
            )

        circumference_final = round(circumference_raw * calibration_factor, 2)

        if use_long_pants and part_name == "Hip":
            circumference_final *= 0.9

        results[part_name] = circumference_final

        # Draw measurement lines for visual inspection.
        y_px_front = int(y_front * img_h)
        y_px_side = int(y_side * img_h)

        cv2.line(viz_f, (int(x1_front), y_px_front), (int(x2_front), y_px_front), (0, 255, 0), 3)
        cv2.line(viz_s, (int(x1_side), y_px_side), (int(x2_side), y_px_side), (0, 255, 0), 3)

        cv2.circle(viz_f, (int(img_w / 2), y_px_front), 5, (0, 0, 255), -1)
        cv2.circle(viz_s, (int(img_w / 2), y_px_side), 5, (0, 0, 255), -1)

    debug_pack = {
        "mask_f": mask_f,
        "mask_s": mask_s,
        "mask_raw_f": mask_raw_f,
        "mask_raw_s": mask_raw_s,
    }

    quality_pack = {
        "pose_visibility": pose_visibility_score,
        "mask_confidence": mask_confidence,
        "missing_landmarks": missing_landmark_count,
    }

    return results, viz_f, viz_s, debug_pack, quality_pack