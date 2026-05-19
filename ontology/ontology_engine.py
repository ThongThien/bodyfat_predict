import os
import uuid
import owlready2

JAVA17_HOME = r"C:\Program Files\Java\jdk-17"

os.environ["JAVA_HOME"] = JAVA17_HOME
os.environ["PATH"] = JAVA17_HOME + r"\bin;" + os.environ["PATH"]

owlready2.JAVA_EXE = JAVA17_HOME + r"\bin\java.exe"

os.system("java -version")

from owlready2 import *

# FORCE JAVA EXEC
owlready2.JAVA_EXE = "java"

def owl_name(obj):
    if obj is None:
        return None
    return getattr(obj, "name", str(obj))

def owl_list(obj_list):
    if not obj_list:
        return []
    return [owl_name(x) for x in obj_list]

def to_name(obj):
    if obj is None:
        return None
    return getattr(obj, "name", str(obj)).split(".")[-1]

def owl_contains(obj_list, target):
    return target in owl_list(obj_list)

def run_ontology(
    height,
    weight,
    abdomen,
    hip,
    predicted_bf,
    pose_visibility=1.0,
    mask_confidence=1.0
):
    # 1. ĐƯỜNG DẪN FILE GỐC VÀ FILE KẾT QUẢ TÁCH BIỆT RẠCH RÒI
    onto_path = os.path.abspath("ontology/BodyFatOntology.owl")
    output_path = os.path.abspath("ontology/output.owl")

    # 2. LOAD FILE GỐC (TUYỆT ĐỐI KHÔNG DÙNG reload=True Ở ĐÂY)
    onto = get_ontology(f"file://{onto_path}").load()

    # 3. TẠO MỘT UID ĐỘC NHẤT CHO LƯỢT QUÉT NÀY
    uid = uuid.uuid4().hex[:8]
    with onto:
        # CREATE INSTANCES
        with onto:
            person, measurement, feature, prediction, quality = [
                cls(f"{name}_{uid}")
                for cls, name in [
                    (onto.People, "person"),
                    (onto.Measurement, "measurement"),
                    (onto.Feature, "feature"),
                    (onto.Prediction, "prediction"),
                    (onto.QualityAssessment, "quality")
                ]
            ]

        # LINK OBJECT PROPERTIES
        for prop, obj in [
            (person.hasMeasurement, measurement),
            (person.hasFeature, feature),
            (person.hasPrediction, prediction),
            (person.hasQuality, quality),
        ]:
            prop.append(obj)

        # VALIDATION STATUS
        validation_status = "Valid"
        validation_errors = []

        # SAFE FLOAT CONVERSION
        fields = {
            "height": height,
            "weight": weight,
            "abdomen": abdomen,
            "hip": hip,
            "predicted_bf": predicted_bf,
            "pose_visibility": pose_visibility,
            "mask_confidence": mask_confidence
        }

        try:
            fields = {
                k: float(v)
                for k, v in fields.items()
            }

        except Exception:
            return {
                "Validation_Status": "Invalid",
                "Validation_Errors": [
                    "Input datatype conversion failed"
                ]
            }

        # EXTRACT VALUES
        height = fields["height"]
        weight = fields["weight"]
        abdomen = fields["abdomen"]
        hip = fields["hip"]

        predicted_bf = fields["predicted_bf"]
        pose_visibility = fields["pose_visibility"]
        mask_confidence = fields["mask_confidence"]

        # RANGE VALIDATION
        for name, value in {
            "Height": height,
            "Weight": weight,
            "Hip": hip,
            "Abdomen": abdomen
        }.items():

            if value <= 0:
                validation_errors.append(
                    f"{name} must be > 0"
                )

        if validation_errors:
            return {
                "Validation_Status": "Invalid",
                "Validation_Errors": validation_errors
            }

        # RAW INPUT DATA
        raw_data_mapping = [
            (measurement.heightCm, round(float(height), 2)),
            (measurement.weightKg, round(float(weight), 2)),
            (measurement.abdomenCm, round(float(abdomen), 2)),
            (measurement.hipCm, round(float(hip), 2)),
            (prediction.predictedBodyFat, round(float(predicted_bf), 2)),
            (quality.poseVisibility, round(float(pose_visibility), 2)),
            (quality.maskConfidence, round(float(mask_confidence), 2))
        ]

        for prop, value in raw_data_mapping:
            prop.clear()
            prop.append(value)

        # FEATURE ENGINEERING
        bmi = weight / ((height / 100) ** 2)
        whr = abdomen / hip
        wthr = abdomen / height

        # STORE FEATURES
        feature_mapping = [
            (feature.bmiValue, round(float(bmi), 2)),
            (feature.whrValue, round(float(whr), 2)),
            (feature.wthrValue, round(float(wthr), 2))
        ]

        for prop, value in feature_mapping:
            prop.clear()
            prop.append(value)

        # CONFIDENCE SCORE
        confidence_score = round(
            (
                pose_visibility * 0.6 +
                mask_confidence * 0.4
            ),
            3
        )

        # SEMANTIC PIPELINE TRACE
        semantic_pipeline = [
            "Input Data Loaded",
            "Feature Engineering Completed",
            "Ontology Instances Created"
        ]

        # RUN REASONER
        reasoning_status = "Success"
        reasoning_error = None

        try:

            sync_reasoner_pellet(
                infer_property_values=True,
                infer_data_property_values=True
            )

            semantic_pipeline.append(
                "Pellet Reasoner Executed"
            )
            
            print("========== QUALITY DEBUG ==========")

            print("POSE VISIBILITY:")
            print(quality.poseVisibility)

            print("QUALITY FLAGS:")
            if hasattr(quality, "hasQualityAssessment"):
                print(quality.hasQualityAssessment)

            print("PERSON FLAGS:")
            if hasattr(person, "hasSemanticFlag"):
                print(person.hasSemanticFlag)

            print("===================================")
            
            print("\n========== [SIÊU DEBUG] KIỂM TRA THUỘC TÍNH TRONG ONTOLOGY ==========")
            print("FEATURE TYPES:", feature.is_a)
            print("PREDICTION TYPES:", prediction.is_a)
            print("QUALITY TYPES:", quality.is_a)
            print("-" * 50)
            
            # Kiểm tra xem các thuộc tính chuẩn của Protégé có dữ liệu không
            print("Dữ liệu trong feature.bmiValue :", getattr(feature, "bmiValue", "❌ Không tồn tại thuộc tính này!"))
            print("Dữ liệu trong feature.whrValue :", getattr(feature, "whrValue", "❌ Không tồn tại thuộc tính này!"))
            print("Dữ liệu trong feature.wthrValue:", getattr(feature, "wthrValue", "❌ Không tồn tại thuộc tính này!"))
            print("Dữ liệu trong prediction.predictedBodyFat:", getattr(prediction, "predictedBodyFat", "❌ Không tồn tại thuộc tính này!"))
            print("-" * 50)
            
            # Kiểm tra xem có bị gán nhầm sang tên viết tắt không
            print("Dữ liệu trong feature.BMI       :", getattr(feature, "BMI", "Không có"))
            print("Dữ liệu trong feature.WHR       :", getattr(feature, "WHR", "Không có"))
            print("Dữ liệu trong feature.WtHR      :", getattr(feature, "WtHR", "Không có"))
            print("Dữ liệu trong prediction.BF     :", getattr(prediction, "BF", "Không có"))
            print("======================================================================\n")
            
        except Exception as e:
            reasoning_status = "Failed"
            reasoning_error = str(e)
            semantic_pipeline.append(
                "Pellet Reasoner Failed"
            )

    # SAVE OUTPUT ONTOLOGY
    try:
        print("\n========== [DEBUG] CHẠY PELLET REASONER ==========")
        
        # LƯU RA FILE OUTPUT RIÊNG BIỆT ĐỂ STREAMLIT ĐỌC HACK CACHE
        onto.save(file=output_path)
        print("✅ Đã lưu kết quả suy luận ra file:", output_path)
        
        semantic_pipeline.append("Pellet Reasoner Executed")

    except Exception as e:
        reasoning_status = "Failed"
        reasoning_error = str(e)
        semantic_pipeline.append("Pellet Reasoner Failed")

    # EXTRACT INFERRED CLASSES
    def extract_classes(instance):

        return [
            cls.name
            for cls in instance.INDIRECT_is_a
            if hasattr(cls, "name")
        ]

    inferred_trace = {
        "Feature_Classes":
            extract_classes(feature),

        "Prediction_Classes":
            extract_classes(prediction),

        "Quality_Classes":
            extract_classes(quality)
    }

    # DEFAULT VALUES
    bmi_class = "Unknown"
    fat_level = "Unknown"

    semantic_flags = []
    triggered_rules = []
    
    bmi_map_output = {
        "Lean_Instance": "Lean",
        "Normal_Instance": "Normal",
        "Overweight_Instance": "Overweight",
        "Obese_Instance": "Obese"
    }
    
    bmi_mapping = {
        "Lean_Instance": ("Lean", "RULE_BMI_LEAN"),
        "Normal_Instance": ("Normal", "RULE_BMI_NORMAL"),
        "Overweight_Instance": ("Overweight", "RULE_BMI_OVERWEIGHT"),
        "Obese_Instance": ("Obese", "RULE_BMI_OBESE"),
        "AbdominalObesity_Instance": (
            "AbdominalObesity",
            "RULE_ABDOMINAL_OBESITY"
        ),
        "HiddenObesity_Instance": (
            "HiddenObesity",
            "RULE_HIDDEN_OBESITY"
        )
    }
    # normalize inferred ontology values
    bmi_classes = {
        to_name(v)
        for v in getattr(feature, "hasBMIClass", [])
        if v is not None
    }
    print("========== BMI DEBUG ==========")
    print("RAW:", feature.hasBMIClass)
    print("NORMALIZED:", bmi_classes)

    for instance_name, (label, rule) in bmi_mapping.items():

        print("CHECK:", instance_name)

        if instance_name in bmi_classes:

            print("MATCHED:", label)

            # BMI CATEGORY
            if label in [
                "Lean",
                "Normal",
                "Overweight",
                "Obese"
            ]:
                bmi_class = label

            # SEMANTIC FLAGS
            else:
                if label not in semantic_flags:
                    semantic_flags.append(label)

            # RULE TRACE
            if rule not in triggered_rules:
                triggered_rules.append(rule)

    print("FINAL BMI:", bmi_class)
    print("==============================")
    # 2. ĐỌC KẾT QUẢ FAT LEVEL TỪ THUỘC TÍNH hasFatLevel CỦA PREDICTION
    if hasattr(prediction, "hasFatLevel"):

        fat_set = {to_name(v) for v in prediction.hasFatLevel if v}

        if "HighFat_Instance" in fat_set:
            fat_level = "HighFat"

        elif "NormalFat_Instance" in fat_set:
            fat_level = "NormalFat"

        elif "LowFat_Instance" in fat_set:
            fat_level = "LowFat"

    if fat_level != "Unknown":
        triggered_rules.append(f"RULE_BODYFAT_{fat_level.upper()}")

    # 3. ĐỌC KẾT QUẢ CHẤT LƯỢNG ẢNH TỪ THUỘC TÍNH hasBMIClass CỦA QUALITY ASSESSMENT
    if hasattr(quality, "hasQualityAssessment"):

        for v in getattr(quality, "hasQualityAssessment", []):

            name = getattr(v, "name", str(v))

            if "LowImageQuality" in name:
                if "LowImageQuality" not in semantic_flags:
                    semantic_flags.append("LowImageQuality")

                triggered_rules.append("RULE_LOW_IMAGE_QUALITY")

    # =========================================================================
    # KẾT THÚC ĐOẠN SỬA ĐỔI - GIỮ NGUYÊN TOÀN BỘ LOGIC BÊN DƯỚI
    # =========================================================================

    # IMAGE QUALITY
    image_quality = "GoodImage"

    if hasattr(quality, "hasQualityAssessment"):

        for v in quality.hasQualityAssessment:

            if to_name(v) == "LowImageQuality_Instance":
                image_quality = "LowImageQuality"
                break

    # EXPLANATIONS
    explanations = []

    # BMI explanation
    if bmi_class != "Unknown":
        explanations.append(
            f"BMI = {bmi:.2f} → Body state: {bmi_class}"
        )

    # Fat explanation
    if fat_level != "Unknown":
        explanations.append(
            f"Body Fat = {predicted_bf:.2f}% → Fat level: {fat_level}"
        )
    
    # Semantic flags (IMPORTANT)
    for flag in semantic_flags:
        if flag == "AbdominalObesity_Instance":
            explanations.append(
                "High waist ratio detected → abdominal obesity risk"
            )

        if flag == "HiddenObesity_Instance":
            explanations.append(
                "Normal BMI but high WHR → hidden obesity risk"
            )

        if flag == "LowImageQuality":
            explanations.append(
                "Image quality too low → reduced confidence"
            )

    if "AbdominalObesity" in semantic_flags:

        explanations.append(
            f"WtHR = {wthr:.2f} suggests abdominal obesity"
        )

    if "HiddenObesity" in semantic_flags:

        explanations.append(
            "Normal BMI but high waist ratio detected"
        )

    if image_quality == "LowImageQuality":

        explanations.append(
            "Low image quality detected"
        )

    # RECOMMENDATIONS
    recommendations = []

    recommendation_rules = {
        "AbdominalObesity":
            "Reduce abdominal fat through calorie control",

        "LowImageQuality":
            "Retake image under better lighting"
    }

    for flag, recommendation in recommendation_rules.items():

        if flag in semantic_flags:

            recommendations.append(
                recommendation
            )

    fat_recommendations = {
        "HighFat":
            "Increase physical activity and maintain calorie deficit",

        "LowFat":
            "Maintain balanced nutrition and resistance training"
    }

    if fat_level in fat_recommendations:

        recommendations.append(
            fat_recommendations[fat_level]
        )

    if not recommendations:

        recommendations.append(
            "Maintain current healthy lifestyle"
        )

    # FINAL OUTPUT
    output = {

        # VALIDATION
        "Validation_Status":
            validation_status,

        "Validation_Errors":
            validation_errors,

        # REASONER
        "Reasoning_Status":
            reasoning_status,

        "Reasoning_Error":
            reasoning_error,

        # FEATURES
        "BMI":
            round(bmi, 2),

        "WHR":
            round(whr, 2),

        "WtHR":
            round(wthr, 2),

        # PREDICTION
        "BodyFat":
            round(predicted_bf, 2),

        # CONFIDENCE
        "Confidence_Score":
            confidence_score,

        # INFERRED CLASSES
        "BMI_Class":
            bmi_class,

        "Fat_Level":
            fat_level,

        # SEMANTIC FLAGS
        "Semantic_Flags":
            semantic_flags,

        # RULE TRACE
        "Triggered_Rules":
            triggered_rules,

        # INFERRED TRACE
        "Inferred_Class_Trace":
            inferred_trace,

        # PIPELINE TRACE
        "Semantic_Pipeline":
            semantic_pipeline,

        # EXPLANATIONS
        "Explanations":
            explanations,

        # RECOMMENDATIONS
        "Recommendations":
            recommendations,

        # IMAGE QUALITY
        "Image_Quality":
            image_quality,
        
        "Pose_Visibility":
            round(pose_visibility, 2),

        "Mask_Confidence":
            round(mask_confidence, 2)
    }

    return output