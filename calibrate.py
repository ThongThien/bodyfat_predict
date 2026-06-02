import os

import cv2
import pandas as pd

from core.cv_engine_new_ver import process_body_measurements


FOLDER_PATH = "assets/anh_chuan"
OUTPUT_REPORT = "calibration_detailed_report.csv"


def parse_filename(filename):
    """
    Parse calibration metadata from the image filename.
    Expected format:
    front_name_age-height-weight-chest-abdomen-hip-thigh.jpg
    """

    try:
        clean_name = os.path.splitext(filename)[0]
        parts = clean_name.split("_")

        if len(parts) < 3:
            return None

        name = parts[1]
        stats = [float(value) for value in parts[2].split("-")]

        if len(stats) < 7:
            return None

        return {
            "name": name,
            "age": int(stats[0]),
            "h": stats[1],
            "w": stats[2],
            "real_c": stats[3],
            "real_a": stats[4],
            "real_h": stats[5],
            "real_t": stats[6],
        }

    except (ValueError, IndexError):
        return None


def run_calibration():
    """
    Run calibration on paired front and side images, then export a detailed CSV report.
    """

    if not os.path.exists(FOLDER_PATH):
        print(f"Folder not found: {FOLDER_PATH}")
        return

    all_files = os.listdir(FOLDER_PATH)
    front_images = [file for file in all_files if file.startswith("front_")]
    data_list = []

    for front_file in front_images:
        side_file = front_file.replace("front_", "side_")

        if side_file not in all_files:
            continue

        info = parse_filename(front_file)

        if not info:
            continue

        front_image_path = os.path.join(FOLDER_PATH, front_file)
        side_image_path = os.path.join(FOLDER_PATH, side_file)

        front_image = cv2.imread(front_image_path)
        side_image = cv2.imread(side_image_path)

        if front_image is None or side_image is None:
            print(f"Cannot read image pair: {front_file}, {side_file}")
            continue

        raw_results, _, _, raw_dimensions = process_body_measurements(
            front_image,
            side_image,
            info["h"],
            info["age"],
            info["w"],
            is_raw=True,
        )

        if not raw_results:
            continue

        row = {
            "Name": info["name"],
            "BMI": round(info["w"] / ((info["h"] / 100) ** 2), 2),
            "Ratio_F": round(raw_dimensions["Abdomen"]["ratio_f"], 4),
            "Ratio_S": round(raw_dimensions["Abdomen"]["ratio_s"], 4),
        }

        body_parts = [
            ("Chest", "C"),
            ("Abdomen", "A"),
            ("Hip", "H"),
            ("Thigh", "T"),
        ]

        for part, short_name in body_parts:
            dimensions = raw_dimensions[part]
            real_value = info[f"real_{short_name.lower()}"]
            raw_value = raw_results[part]

            row.update(
                {
                    f"Real_{short_name}": real_value,
                    f"Raw_{short_name}": raw_value,
                    f"W_px_{part}": dimensions["w_px"],
                    f"D_px_{part}": dimensions["d_px"],
                    f"W_cm_{part}": round(dimensions["w_cm"], 2),
                    f"D_cm_{part}": round(dimensions["d_cm"], 2),
                    f"f_{part}": round(real_value / raw_value, 3)
                    if raw_value
                    else None,
                }
            )

        row["f_Avg_Person"] = round(
            (
                row["f_Chest"]
                + row["f_Abdomen"]
                + row["f_Hip"]
                + row["f_Thigh"]
            )
            / 4,
            3,
        )

        data_list.append(row)

    if not data_list:
        print("No valid calibration data found.")
        return

    report_df = pd.DataFrame(data_list)
    report_df.to_csv(OUTPUT_REPORT, index=False)

    average_system_factor = round(report_df["f_Avg_Person"].mean(), 4)

    print(f"Detailed calibration report exported: {OUTPUT_REPORT}")
    print(f"Average system factor: {average_system_factor}")


if __name__ == "__main__":
    run_calibration()