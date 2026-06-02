import os

import cv2
import numpy as np

Y_MAP_CONFIGS = [
    {"name": "A_base", "Chest": 0.25, "Abdomen": 0.25, "Hip": 0.00},
    {"name": "B_current", "Chest": 0.27, "Abdomen": 0.30, "Hip": 0.05},
    {"name": "C_high", "Chest": 0.30, "Abdomen": 0.35, "Hip": 0.08},
    {"name": "D_low", "Chest": 0.20, "Abdomen": 0.20, "Hip": 0.00},
]

IMG_DIR = "assets/anh_chuan"

REAL_HEIGHT = 170
WEIGHT = 65


def process_with_custom_y_map(front_img, side_img, real_h, weight, y_map_config):
    """Process body measurements with a custom Y-position configuration."""
    import core.cv_engine_v5 as engine

    def patched_process(front_img, side_img, real_h, weight, use_long_pants=False):
        """Run the measurement pipeline using the selected custom Y-map."""
        mask_f, _, pose_f = engine.get_body_data_v5(front_img)
        mask_s, _, pose_s = engine.get_body_data_v5(side_img)

        if not all([pose_f, pose_f.pose_landmarks, pose_s, pose_s.pose_landmarks]):
            return None

        img_h, _, _ = front_img.shape

        lm_f = pose_f.pose_landmarks.landmark
        lm_s = pose_s.pose_landmarks.landmark

        # Calculate pixel-to-centimeter scale from estimated body height.
        nose_y = lm_f[0].y * img_h
        heel_y = ((lm_f[29].y + lm_f[30].y) / 2) * img_h
        head_offset = abs(nose_y - (lm_f[1].y * img_h)) * 2.5
        ratio = real_h / abs(heel_y - (nose_y - head_offset))

        # Calculate BMI-based calibration factor.
        bmi = weight / ((real_h / 100) ** 2)

        if bmi < 18.5:
            calibration_factor = 1.12
        elif bmi < 25:
            calibration_factor = 1.204
        else:
            calibration_factor = 1.25

        shoulder_y = lm_f[11].y
        hip_y = lm_f[23].y
        torso_height = hip_y - shoulder_y

        # Apply the custom Y-map configuration.
        y_map_front = {
            "Chest": shoulder_y + torso_height * y_map_config["Chest"],
            "Abdomen": hip_y - torso_height * y_map_config["Abdomen"],
            "Hip": hip_y + torso_height * y_map_config["Hip"],
        }

        results = {}

        for part_name in ["Chest", "Abdomen", "Hip"]:
            y_front = y_map_front[part_name]
            iterator = engine.get_iterator(bmi, part_name)

            width, _, _ = engine.get_dimension_at_y_v5(
                mask_f,
                y_front,
                lm_f,
                part_name,
                ratio,
                iterator,
            )

            depth, _, _, _ = engine.find_best_depth(
                mask_s,
                y_front,
                lm_s,
                part_name,
                ratio,
                iterator,
            )

            if width == 0 or depth == 0:
                continue

            a = width / 2
            b = depth / 2

            if a + b == 0:
                circumference = 0
            else:
                h_el = ((a - b) ** 2) / ((a + b) ** 2)
                circumference = np.pi * (a + b) * (
                    1 + (3 * h_el) / (10 + np.sqrt(4 - 3 * h_el))
                )

            results[part_name] = circumference * calibration_factor

        return results

    return patched_process(front_img, side_img, real_h, weight)


def run_test():
    """Run Y-map configuration tests on all images in the target folder."""
    images = [
        file_name
        for file_name in os.listdir(IMG_DIR)
        if file_name.endswith((".jpg", ".png"))
    ]

    print("\n" + "=" * 80)
    print(" TEST Y_MAP CONFIGS ".center(80, "="))

    for image_name in images:
        image_path = os.path.join(IMG_DIR, image_name)

        front_img = cv2.imread(image_path)
        side_img = front_img.copy()

        print(f"\nIMAGE: {image_name}")

        results_table = []

        for config in Y_MAP_CONFIGS:
            result = process_with_custom_y_map(
                front_img,
                side_img,
                REAL_HEIGHT,
                WEIGHT,
                config,
            )

            if result is None:
                continue

            chest = result.get("Chest", 0)
            abdomen = result.get("Abdomen", 0)
            hip = result.get("Hip", 0)

            score = chest + abdomen + hip

            results_table.append(
                {
                    "name": config["name"],
                    "Chest": chest,
                    "Abdomen": abdomen,
                    "Hip": hip,
                    "Score": score,
                }
            )

        results_table = sorted(
            results_table,
            key=lambda item: item["Score"],
            reverse=True,
        )

        print("\nCONFIG COMPARISON:")
        print(f"{'Name':<12} | {'Chest':<8} | {'Abdomen':<10} | {'Hip':<8} | Score")
        print("-" * 60)

        for row in results_table:
            print(
                f"{row['name']:<12} | "
                f"{row['Chest']:<8.1f} | "
                f"{row['Abdomen']:<10.1f} | "
                f"{row['Hip']:<8.1f} | "
                f"{row['Score']:.1f}"
            )

        if results_table:
            best_config = results_table[0]["name"]
            print(f"\nBEST CONFIG: {best_config}")

        cv2.imshow("Test Image", front_img)
        cv2.waitKey(0)

    cv2.destroyAllWindows()


if __name__ == "__main__":
    run_test()