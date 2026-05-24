import os
import gc
import cv2
import time
import traceback
from datetime import datetime
from typing import Any, Dict, List, Optional

import pandas as pd

from core.predictor_v5 import load_model_v5, predict_body_fat_v5
from core.cv_engine_v5 import process_body_measurements_v5

# Use the new ontology engine
from ontology.ontology_engine_v2 import run_ontology


# =====================================================
# CONFIG
# =====================================================

FOLDER_PATH = "assets/anh_chuan"
MODEL_PATH = "models/bodyfat_ai_super_clean_v5.pkl"
OUTPUT_DIR = "ontology"
OUTPUT_FILE = os.path.join(
    OUTPUT_DIR,
    f"ontology_evaluation_long_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx"
)

# None = scan all front images
MAX_FILES: Optional[int] = None

# Reduce RAM usage when reading images
MAX_IMAGE_WIDTH = 640


# =====================================================
# HELPERS
# =====================================================

def safe_join(values: Any, sep: str = ", ") -> str:
    if not values:
        return ""
    if isinstance(values, str):
        return values
    return sep.join(map(str, values))


def safe_round(value: Any, digits: int = 2) -> Any:
    try:
        if value is None:
            return None
        return round(float(value), digits)
    except Exception:
        return value


def resize_img(img, max_w: int = MAX_IMAGE_WIDTH):
    h, w = img.shape[:2]
    if w > max_w:
        scale = max_w / w
        img = cv2.resize(img, (int(w * scale), int(h * scale)))
    return img


def is_valid_filename(file: str) -> bool:
    """
    Expected filename example:
    front_Thien_22-163-60-89-80-86-48.jpg
    """
    try:
        parts = file.split("_")
        if len(parts) < 3:
            return False
        nums = parts[2].split(".")[0].split("-")
        return len(nums) >= 6
    except Exception:
        return False


def parse_filename(file_name: str) -> Dict[str, Any]:
    base = os.path.basename(file_name)
    name = base.split("_")[1]
    data = base.split("_")[2].split(".")[0]

    age, height, weight, chest, abdomen, hip, *_ = map(float, data.split("-"))

    return {
        "Name": name,
        "Age": age,
        "Height": height,
        "Weight": weight,
        "Chest": chest,
        "Abdomen_GT": abdomen,
        "Hip_GT": hip,
    }


def add_long_rows(
    rows: List[Dict[str, Any]],
    sample_id: str,
    sample_name: str,
    group: str,
    metrics: Dict[str, Any],
) -> None:
    """
    Long-format export:
    each sample has multiple rows, grouped by information type.
    This is easier to read than a very wide Excel table.
    """
    for metric_name, metric_value in metrics.items():
        rows.append({
            "Sample_ID": sample_id,
            "Sample_Name": sample_name,
            "Group": group,
            "Metric": metric_name,
            "Value": metric_value,
        })


def build_summary_row(
    sample_id,
    info,
    res_scan,
    raw_pred,
    pred_ai,
    onto,
    pose_visibility,
    mask_confidence,
    elapsed_ms,
):
    semantic_flags = onto.get("Semantic_Flags", [])
    triggered_rules = onto.get("Triggered_Rules", [])
    explanations = onto.get("Explanations", [])
    recommendations = onto.get("Recommendations", [])
    anomaly_type = onto.get("Anomaly_Type", [])

    return {
        "Sample_ID": sample_id,
        "Sample_Name": info["Name"],
        "Age": info["Age"],

        "GT_Height_cm": info["Height"],
        "GT_Weight_kg": info["Weight"],
        "GT_Chest_cm": info["Chest"],
        "GT_Abdomen_cm": info["Abdomen_GT"],
        "GT_Hip_cm": info["Hip_GT"],

        "AI_Chest_cm": round(res_scan["Chest"], 2),
        "AI_Abdomen_cm": round(res_scan["Abdomen"], 2),
        "AI_Hip_cm": round(res_scan["Hip"], 2),

        "Delta_Chest_cm": round(res_scan["Chest"] - info["Chest"], 2),
        "Delta_Abdomen_cm": round(res_scan["Abdomen"] - info["Abdomen_GT"], 2),
        "Delta_Hip_cm": round(res_scan["Hip"] - info["Hip_GT"], 2),

        "BF_Raw_from_filename_measurement": round(raw_pred, 2),
        "BF_AI_from_CV_measurement": round(pred_ai, 2),
        "Delta_BF_AI_minus_Raw": round(pred_ai - raw_pred, 2),

        "BMI": onto.get("BMI"),
        "WHR": onto.get("WHR"),
        "WtHR": onto.get("WtHR"),
        "BMI_Class": onto.get("BMI_Class"),
        "Fat_Level": onto.get("Fat_Level"),

        "Pose_Visibility": round(pose_visibility, 3),
        "Mask_Confidence": round(mask_confidence, 3),
        "Confidence_Score": onto.get("Confidence_Score"),
        "Confidence_Level": onto.get("Confidence_Level"),
        "Image_Quality": onto.get("Image_Quality"),

        "Validation_Status": onto.get("Validation_Status"),
        "Warning_Level": onto.get("Warning_Level"),
        "Anomaly_Type": safe_join(anomaly_type),
        "Semantic_Flags": safe_join(semantic_flags),

        "Rule_Count": len(triggered_rules),
        "Triggered_Rules": "\n".join(triggered_rules),

        "Explanation_Count": len(explanations),
        "Explanations": "\n".join(explanations),

        "Recommendation_Count": len(recommendations),
        "Recommendations": "\n".join(recommendations),

        "Reasoning_Status": onto.get("Reasoning_Status"),
        "Ontology_Latency_ms": round(elapsed_ms, 2),
    }


def append_sample_to_long_rows(
    rows: List[Dict[str, Any]],
    sample_id: str,
    info: Dict[str, Any],
    res_scan: Dict[str, Any],
    raw_pred: float,
    pred_ai: float,
    onto: Dict[str, Any],
    pose_visibility: float,
    mask_confidence: float,
    elapsed_ms: float,
    front_path: str,
    side_path: str,
) -> None:
    name = info.get("Name", sample_id)

    add_long_rows(rows, sample_id, name, "01_Sample", {
        "Name": info.get("Name"),
        "Age": info.get("Age"),
        "Front_Image": front_path,
        "Side_Image": side_path,
    })

    add_long_rows(rows, sample_id, name, "02_GroundTruth", {
        "GT_Height_cm": info.get("Height"),
        "GT_Weight_kg": info.get("Weight"),
        "GT_Chest_cm": info.get("Chest"),
        "GT_Abdomen_cm": info.get("Abdomen_GT"),
        "GT_Hip_cm": info.get("Hip_GT"),
    })

    add_long_rows(rows, sample_id, name, "03_CV_Measurement", {
        "AI_Chest_cm": safe_round(res_scan.get("Chest")),
        "AI_Abdomen_cm": safe_round(res_scan.get("Abdomen")),
        "AI_Hip_cm": safe_round(res_scan.get("Hip")),
        "Delta_Chest_cm": safe_round(res_scan.get("Chest", 0) - info.get("Chest", 0)),
        "Delta_Abdomen_cm": safe_round(res_scan.get("Abdomen", 0) - info.get("Abdomen_GT", 0)),
        "Delta_Hip_cm": safe_round(res_scan.get("Hip", 0) - info.get("Hip_GT", 0)),
    })

    add_long_rows(rows, sample_id, name, "04_Model_Prediction", {
        "Model_Name": "bodyfat_ai_super_clean_v5.pkl",
        "BF_Raw_from_filename_measurement": safe_round(raw_pred),
        "BF_AI_from_CV_measurement": safe_round(pred_ai),
        "Delta_BF_AI_minus_Raw": safe_round(pred_ai - raw_pred),
    })

    add_long_rows(rows, sample_id, name, "05_Derived_Features", {
        "BMI": onto.get("BMI"),
        "WHR": onto.get("WHR"),
        "WtHR": onto.get("WtHR"),
        "BMI_Class": onto.get("BMI_Class", "Unknown"),
        "Fat_Level": onto.get("Fat_Level", "Unknown"),
    })

    add_long_rows(rows, sample_id, name, "06_Image_Quality", {
        "Pose_Visibility": safe_round(pose_visibility, 3),
        "Mask_Confidence": safe_round(mask_confidence, 3),
        "Confidence_Score": onto.get("Confidence_Score"),
        "Confidence_Level": onto.get("Confidence_Level", "Unknown"),
        "Image_Quality": onto.get("Image_Quality", "Unknown"),
        "Missing_Landmark_Count": onto.get("Missing_Landmark_Count", 0),
    })

    add_long_rows(rows, sample_id, name, "07_Semantic_Validation", {
        "Session_ID": onto.get("Session_ID"),
        "Validation_Status": onto.get("Validation_Status", "Unknown"),
        "Warning_Level": onto.get("Warning_Level", "None"),
        "Anomaly_Type": safe_join(onto.get("Anomaly_Type", [])),
        "Semantic_Flags": safe_join(onto.get("Semantic_Flags", [])),
        "Reasoning_Status": onto.get("Reasoning_Status", "Unknown"),
    })

    add_long_rows(rows, sample_id, name, "08_Rules", {
        "Rule_Count": len(onto.get("Triggered_Rules", [])),
        "Triggered_Rules": "\n".join(onto.get("Triggered_Rules", [])),
    })

    add_long_rows(rows, sample_id, name, "09_Explanation", {
        "Explanation_Count": onto.get("Explanation_Count", len(onto.get("Explanations", []))),
        "Explanations": "\n".join(onto.get("Explanations", [])),
    })

    add_long_rows(rows, sample_id, name, "10_Recommendation", {
        "Recommendation_Count": onto.get("Recommendation_Count", len(onto.get("Recommendations", []))),
        "Recommendations": "\n".join(onto.get("Recommendations", [])),
    })

    add_long_rows(rows, sample_id, name, "11_System", {
        "Ontology_Latency_ms": safe_round(onto.get("Ontology_Latency_ms", elapsed_ms), 2),
        "Batch_Measured_Latency_ms": safe_round(elapsed_ms, 2),
        "Ontology_Warnings": "\n".join(onto.get("Ontology_Warnings", [])),
        "Reasoning_Error": onto.get("Reasoning_Error"),
    })


# =====================================================
# MAIN
# =====================================================
def build_metric_comparison_sheet(summary_rows):
    """
    Convert summary rows into vertical metric comparison format.

    Output:
    Metric | Sample_1 | Sample_2 | Sample_3
    """

    sample_columns = {}

    for row in summary_rows:
        sample_name = row.get("Sample_Name", row.get("Name", "Unknown"))
        sample_id = row.get("Sample_ID", sample_name)

        column_name = f"{sample_name}"

        sample_columns[column_name] = {
            "Sample_ID": sample_id,
            "Age": row.get("Age"),

            "GT_Height_cm": row.get("GT_Height_cm"),
            "GT_Weight_kg": row.get("GT_Weight_kg"),
            "GT_Chest_cm": row.get("GT_Chest_cm"),
            "GT_Abdomen_cm": row.get("GT_Abdomen_cm"),
            "GT_Hip_cm": row.get("GT_Hip_cm"),

            "AI_Chest_cm": row.get("AI_Chest_cm"),
            "AI_Abdomen_cm": row.get("AI_Abdomen_cm"),
            "AI_Hip_cm": row.get("AI_Hip_cm"),

            "Delta_Chest_cm": row.get("Delta_Chest_cm"),
            "Delta_Abdomen_cm": row.get("Delta_Abdomen_cm"),
            "Delta_Hip_cm": row.get("Delta_Hip_cm"),

            "BF_Raw_from_filename_measurement": row.get("BF_Raw_from_filename_measurement"),
            "BF_AI_from_CV_measurement": row.get("BF_AI_from_CV_measurement"),
            "Delta_BF_AI_minus_Raw": row.get("Delta_BF_AI_minus_Raw"),

            "BMI": row.get("BMI"),
            "WHR": row.get("WHR"),
            "WtHR": row.get("WtHR"),
            "BMI_Class": row.get("BMI_Class"),
            "Fat_Level": row.get("Fat_Level"),

            "Pose_Visibility": row.get("Pose_Visibility"),
            "Mask_Confidence": row.get("Mask_Confidence"),
            "Confidence_Score": row.get("Confidence_Score"),
            "Confidence_Level": row.get("Confidence_Level"),
            "Image_Quality": row.get("Image_Quality"),

            "Validation_Status": row.get("Validation_Status"),
            "Warning_Level": row.get("Warning_Level"),
            "Anomaly_Type": row.get("Anomaly_Type"),
            "Semantic_Flags": row.get("Semantic_Flags"),

            "Rule_Count": row.get("Rule_Count"),
            "Triggered_Rules": row.get("Triggered_Rules"),

            "Explanation_Count": row.get("Explanation_Count"),
            "Explanations": row.get("Explanations"),

            "Recommendation_Count": row.get("Recommendation_Count"),
            "Recommendations": row.get("Recommendations"),

            "Reasoning_Status": row.get("Reasoning_Status"),
            "Ontology_Latency_ms": row.get("Ontology_Latency_ms"),
        }

    df_compare = pd.DataFrame(sample_columns)
    df_compare.insert(0, "Metric", df_compare.index)
    df_compare.reset_index(drop=True, inplace=True)

    return df_compare

def main() -> None:
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    print("\nLoading model...")
    model_v5 = load_model_v5(MODEL_PATH)
    print("Model loaded.")

    files = sorted([
        f for f in os.listdir(FOLDER_PATH)
        if f.startswith("front")
    ])

    if MAX_FILES:
        files = files[:MAX_FILES]

    print("\n" + "=" * 70)
    print(" ONTOLOGY BATCH TEST V2 - LONG FORMAT ".center(70, "="))
    print("=" * 70)

    long_rows: List[Dict[str, Any]] = []
    summary_rows: List[Dict[str, Any]] = []
    error_rows: List[Dict[str, Any]] = []

    for idx, file in enumerate(files):
        print(f"\n[{idx + 1}/{len(files)}] {file}")

        sample_id = os.path.splitext(file)[0].replace("front_", "")

        try:
            if not is_valid_filename(file):
                print(" Invalid filename")
                error_rows.append({"Sample_ID": sample_id, "File": file, "Error": "Invalid filename"})
                continue

            info = parse_filename(file)

            front_path = os.path.join(FOLDER_PATH, file)
            side_path = front_path.replace("front", "side")

            if not os.path.exists(side_path):
                print(" Missing side image")
                error_rows.append({"Sample_ID": sample_id, "File": file, "Error": "Missing side image"})
                continue

            img_f = cv2.imread(front_path)
            img_s = cv2.imread(side_path)

            if img_f is None or img_s is None:
                print(" Cannot read image")
                error_rows.append({"Sample_ID": sample_id, "File": file, "Error": "Cannot read image"})
                continue

            img_f = resize_img(img_f)
            img_s = resize_img(img_s)

            # Raw baseline from filename/manual measurements
            raw_pred = predict_body_fat_v5(model_v5, info)

            # CV engine
            scan_result = process_body_measurements_v5(
                img_f,
                img_s,
                info["Height"],
                info["Weight"],
                False,
            )

            if scan_result is None:
                print(" Scan result None")
                error_rows.append({"Sample_ID": sample_id, "File": file, "Error": "Scan result None"})
                continue

            if len(scan_result) != 5:
                print(" Invalid scan result")
                error_rows.append({"Sample_ID": sample_id, "File": file, "Error": "Invalid scan result length"})
                continue

            res_scan, _, _, debug_pack, quality_pack = scan_result

            if not res_scan:
                print(" Scan failed")
                error_rows.append({"Sample_ID": sample_id, "File": file, "Error": "Scan failed"})
                continue

            pred_ai = predict_body_fat_v5(model_v5, {**info, **res_scan})

            pose_visibility = quality_pack.get("pose_visibility", 0.0)
            mask_confidence = quality_pack.get("mask_confidence", 0.0)
            missing_landmark_count = quality_pack.get("missing_landmark_count", 0)

            start_time = time.perf_counter()
            onto = run_ontology(
                height=info["Height"],
                weight=info["Weight"],
                chest=res_scan.get("Chest"),
                abdomen=res_scan["Abdomen"],
                hip=res_scan["Hip"],
                predicted_bf=pred_ai,
                pose_visibility=pose_visibility,
                mask_confidence=mask_confidence,
                missing_landmark_count=missing_landmark_count,
                source_type="AI Scan Batch",
                image_name=f"{file} | {os.path.basename(side_path)}",
                image_path=f"{front_path} | {side_path}",
            )
            elapsed_ms = (time.perf_counter() - start_time) * 1000

            append_sample_to_long_rows(
                rows=long_rows,
                sample_id=sample_id,
                info=info,
                res_scan=res_scan,
                raw_pred=raw_pred,
                pred_ai=pred_ai,
                onto=onto,
                pose_visibility=pose_visibility,
                mask_confidence=mask_confidence,
                elapsed_ms=elapsed_ms,
                front_path=front_path,
                side_path=side_path,
            )

            summary_rows.append(
                build_summary_row(
                    sample_id=sample_id,
                    info=info,
                    res_scan=res_scan,
                    raw_pred=raw_pred,
                    pred_ai=pred_ai,
                    onto=onto,
                    pose_visibility=pose_visibility,
                    mask_confidence=mask_confidence,
                    elapsed_ms=elapsed_ms,
                )
            )

            print(f" BF AI       : {pred_ai:.2f}")
            print(f" BMI         : {onto.get('BMI')}")
            print(f" BMI Class   : {onto.get('BMI_Class')}")
            print(f" Fat Level   : {onto.get('Fat_Level')}")
            print(f" Warning     : {onto.get('Warning_Level')}")
            print(f" Validation  : {onto.get('Validation_Status')}")
            print(f" Rules       : {len(onto.get('Triggered_Rules', []))}")

        except Exception as e:
            print("\n CRASH DETECTED")
            print(str(e))
            traceback.print_exc()
            error_rows.append({
                "Sample_ID": sample_id,
                "File": file,
                "Error": str(e),
                "Traceback": traceback.format_exc(),
            })

        finally:
            try:
                del img_f, img_s
            except Exception:
                pass
            gc.collect()

    print("\nExporting Excel...")

    df_long = pd.DataFrame(long_rows)
    df_summary = pd.DataFrame(summary_rows)
    df_errors = pd.DataFrame(error_rows)

    df_compare = build_metric_comparison_sheet(summary_rows)

    with pd.ExcelWriter(OUTPUT_FILE, engine="openpyxl") as writer:
        df_compare.to_excel(writer, sheet_name="Metric_Comparison", index=False)
        df_summary.to_excel(writer, sheet_name="Summary", index=False)
        df_long.to_excel(writer, sheet_name="Long_Format", index=False)
        df_errors.to_excel(writer, sheet_name="Errors", index=False)

        # Basic formatting
        workbook = writer.book
        for sheet_name in writer.sheets:
            ws = writer.sheets[sheet_name]
            ws.freeze_panes = "A2"
            ws.auto_filter.ref = ws.dimensions

            for col in ws.columns:
                max_len = 0
                col_letter = col[0].column_letter
                for cell in col:
                    value = str(cell.value) if cell.value is not None else ""
                    max_len = max(max_len, min(len(value), 80))
                ws.column_dimensions[col_letter].width = max(12, min(max_len + 2, 60))

    print("\n" + "=" * 70)
    print(" EXPORT SUCCESS ")
    print(f" Saved: {OUTPUT_FILE}")
    print(" Sheets: Long_Format, Summary, Errors")
    print("=" * 70)


if __name__ == "__main__":
    main()
