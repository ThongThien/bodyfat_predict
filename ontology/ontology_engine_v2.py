import os
import time
import uuid
from datetime import datetime
from typing import Any, Dict, List, Optional

import owlready2
from owlready2 import get_ontology, sync_reasoner_pellet


# ============================================================
# JAVA / OWLREADY2 CONFIG
# ============================================================

JAVA17_HOME = r"C:\Program Files\Java\jdk-17"

if os.path.exists(JAVA17_HOME):
    os.environ["JAVA_HOME"] = JAVA17_HOME
    os.environ["PATH"] = JAVA17_HOME + r"\bin;" + os.environ.get("PATH", "")
    owlready2.JAVA_EXE = JAVA17_HOME + r"\bin\java.exe"
else:
    # Fallback: use java from system PATH.
    owlready2.JAVA_EXE = "java"


# ============================================================
# HELPER FUNCTIONS
# ============================================================

def owl_name(obj: Any) -> Optional[str]:
    if obj is None:
        return None
    return getattr(obj, "name", str(obj)).split(".")[-1]


def owl_names(values: Any) -> List[str]:
    if not values:
        return []
    return [owl_name(v) for v in values if v is not None]


def unique_keep_order(values: List[str]) -> List[str]:
    seen = set()
    output = []
    for v in values:
        if v and v not in seen:
            seen.add(v)
            output.append(v)
    return output


def get_entity(onto: Any, entity_name: str, warnings: List[str]) -> Optional[Any]:
    # 1. Try normal Owlready attribute access
    entity = getattr(onto, entity_name, None)
    if entity is not None:
        return entity

    # 2. Try IRI search
    try:
        entity = onto.search_one(iri=f"*#{entity_name}")
        if entity is not None:
            return entity
    except Exception:
        pass

    # 3. Fallback: scan classes, properties, individuals by .name
    try:
        for collection in [
            onto.classes(),
            onto.object_properties(),
            onto.data_properties(),
            onto.individuals(),
        ]:
            for item in collection:
                if getattr(item, "name", None) == entity_name:
                    return item
    except Exception:
        pass

    warnings.append(f"Missing ontology entity: {entity_name}")
    return None


def create_instance(onto: Any, class_name: str, instance_name: str, warnings: List[str]) -> Optional[Any]:
    cls = get_entity(onto, class_name, warnings)
    if cls is None:
        return None
    return cls(instance_name)


def set_data_property(instance: Any, prop_name: str, value: Any, warnings: List[str]) -> None:
    if instance is None:
        warnings.append(f"Cannot set {prop_name}: instance is None")
        return

    prop = getattr(instance, prop_name, None)
    if prop is None:
        warnings.append(f"Missing data property on {owl_name(instance)}: {prop_name}")
        return

    try:
        prop.clear()
        prop.append(value)
    except Exception as e:
        warnings.append(f"Failed to set {prop_name} on {owl_name(instance)}: {e}")


def append_object_property(subject: Any, prop_name: str, obj: Any, warnings: List[str]) -> None:
    if subject is None or obj is None:
        warnings.append(f"Cannot append object property {prop_name}: subject or object is None")
        return

    prop = getattr(subject, prop_name, None)
    if prop is None:
        warnings.append(f"Missing object property on {owl_name(subject)}: {prop_name}")
        return

    try:
        if obj not in prop:
            prop.append(obj)
    except Exception as e:
        warnings.append(f"Failed to append {prop_name} on {owl_name(subject)}: {e}")


def first_name_from_property(instance: Any, prop_name: str) -> Optional[str]:
    if instance is None:
        return None
    values = getattr(instance, prop_name, [])
    if not values:
        return None
    return owl_name(values[0])


def names_from_property(instance: Any, prop_name: str) -> List[str]:
    if instance is None:
        return []
    return owl_names(getattr(instance, prop_name, []))


def extract_classes(instance: Any) -> List[str]:
    if instance is None:
        return []
    return unique_keep_order([
        cls.name
        for cls in getattr(instance, "INDIRECT_is_a", [])
        if hasattr(cls, "name")
    ])


def safe_float_map(fields: Dict[str, Any]) -> Dict[str, float]:
    output = {}
    for key, value in fields.items():
        if value is None:
            output[key] = 0.0
        else:
            output[key] = float(value)
    return output


def rule_individual(onto: Any, rule_name: str) -> Optional[Any]:
    target_name = f"{rule_name}_Instance"

    inst = getattr(onto, target_name, None)
    if inst is not None:
        return inst

    try:
        inst = onto.search_one(iri=f"*#{target_name}")
        if inst is not None:
            return inst
    except Exception:
        pass

    try:
        for item in onto.individuals():
            if getattr(item, "name", None) == target_name:
                return item
    except Exception:
        pass

    return None


def add_rule_trace(
    onto: Any,
    target_instance: Any,
    rule_name: str,
    triggered_rules: List[str],
    warnings: List[str],
) -> None:
    if rule_name not in triggered_rules:
        triggered_rules.append(rule_name)

    rule_inst = rule_individual(onto, rule_name)
    if rule_inst is not None and target_instance is not None:
        append_object_property(target_instance, "triggeredByRule", rule_inst, warnings)


# ============================================================
# MAIN ONTOLOGY FUNCTION
# ============================================================

def run_ontology(
    height: float,
    weight: float,
    abdomen: float,
    hip: float,
    predicted_bf: float,
    pose_visibility: float = 1.0,
    mask_confidence: float = 1.0,
    chest: Optional[float] = None,
    missing_landmark_count: int = 0,
    source_type: str = "AI Scan",
    image_name: Optional[str] = None,
    image_path: Optional[str] = None,
    ontology_path: str = "ontology/BodyFatOntology.owl",
    output_path: str = "ontology/output.owl",
) -> Dict[str, Any]:
    """
    Hybrid AI + Ontology semantic reasoning layer.

    This function does NOT replace the ML model.
    It receives the ML/CV output and creates one runtime PredictionSession graph:

    PredictionSession
        -> Measurement
        -> Feature
        -> Prediction
        -> QualityAssessment
        -> InputSource / ImageInput
        -> inferred ValidationResult / AnomalyType / WarningLevel
        -> dynamic SemanticExplanation / Recommendation

    Notes:
    - Protege stores static ontology schema, semantic labels, and SWRL rules.
    - Python creates dynamic instances for each prediction session.
    """

    start_time = time.perf_counter()
    uid = uuid.uuid4().hex[:8]
    session_id = f"session_{uid}"

    ontology_warnings: List[str] = []
    validation_errors: List[str] = []
    semantic_pipeline: List[str] = []

    # ------------------------------------------------------------
    # 1. Load ontology
    # ------------------------------------------------------------
    onto_file = os.path.abspath(ontology_path)
    out_file = os.path.abspath(output_path)

    try:
        onto = get_ontology(f"file://{onto_file}").load()
        semantic_pipeline.append("Ontology loaded")
        print("ONTOLOGY FILE:", onto_file)
        print("HAS PredictionSession:", hasattr(onto, "PredictionSession"))
        print("HAS ImageInput:", hasattr(onto, "ImageInput"))
        print("CLASSES:", [c.name for c in onto.classes()])
    except Exception as e:
        return {
            "Validation_Status": "Invalid",
            "Reasoning_Status": "Failed",
            "Reasoning_Error": f"Failed to load ontology: {e}",
            "Ontology_Warnings": ontology_warnings,
            "Validation_Errors": [str(e)],
        }

    # ------------------------------------------------------------
    # 2. Validate and convert input
    # ------------------------------------------------------------
    try:
        numeric_fields = safe_float_map({
            "height": height,
            "weight": weight,
            "abdomen": abdomen,
            "hip": hip,
            "predicted_bf": predicted_bf,
            "pose_visibility": pose_visibility,
            "mask_confidence": mask_confidence,
            "chest": chest if chest is not None else 0.0,
            "missing_landmark_count": missing_landmark_count,
        })
    except Exception:
        return {
            "Validation_Status": "Invalid",
            "Reasoning_Status": "Skipped",
            "Validation_Errors": ["Input datatype conversion failed"],
            "Ontology_Warnings": ontology_warnings,
        }

    height = numeric_fields["height"]
    weight = numeric_fields["weight"]
    abdomen = numeric_fields["abdomen"]
    hip = numeric_fields["hip"]
    predicted_bf = numeric_fields["predicted_bf"]
    pose_visibility = numeric_fields["pose_visibility"]
    mask_confidence = numeric_fields["mask_confidence"]
    chest_value = numeric_fields["chest"]
    missing_landmark_count = int(numeric_fields["missing_landmark_count"])

    for name, value in {
        "Height": height,
        "Weight": weight,
        "Abdomen": abdomen,
        "Hip": hip,
    }.items():
        if value <= 0:
            validation_errors.append(f"{name} must be > 0")

    if predicted_bf < 0:
        validation_errors.append("Predicted body fat must be >= 0")

    if validation_errors:
        return {
            "Session_ID": session_id,
            "Validation_Status": "Invalid",
            "Reasoning_Status": "Skipped",
            "Validation_Errors": validation_errors,
            "Ontology_Warnings": ontology_warnings,
        }

    semantic_pipeline.append("Input validation completed")

    # ------------------------------------------------------------
    # 3. Feature engineering outside ontology
    # ------------------------------------------------------------
    bmi = weight / ((height / 100.0) ** 2)
    whr = abdomen / hip
    wthr = abdomen / height
    confidence_score = round((pose_visibility * 0.6 + mask_confidence * 0.4), 3)

    semantic_pipeline.append("Feature engineering completed")

    # ------------------------------------------------------------
    # 4. Create runtime ontology graph
    # ------------------------------------------------------------
    with onto:
        session = create_instance(onto, "PredictionSession", session_id, ontology_warnings)
        measurement = create_instance(onto, "Measurement", f"measurement_{uid}", ontology_warnings)
        feature = create_instance(onto, "Feature", f"feature_{uid}", ontology_warnings)
        prediction = create_instance(onto, "Prediction", f"prediction_{uid}", ontology_warnings)
        quality = create_instance(onto, "QualityAssessment", f"quality_{uid}", ontology_warnings)

        # Input source: ImageInput for AI scan; MeasurementSource for manual input.
        if str(source_type).lower().replace("_", " ") in ["manual", "manual input"]:
            input_source = create_instance(onto, "MeasurementSource", f"measurement_source_{uid}", ontology_warnings)
            image_input = None
        else:
            input_source = create_instance(onto, "ImageInput", f"image_input_{uid}", ontology_warnings)
            image_input = input_source

        # Session links
        append_object_property(session, "hasMeasurement", measurement, ontology_warnings)
        append_object_property(session, "hasFeature", feature, ontology_warnings)
        append_object_property(session, "hasPrediction", prediction, ontology_warnings)
        append_object_property(session, "hasQuality", quality, ontology_warnings)
        append_object_property(session, "usesModelOutput", prediction, ontology_warnings)
        append_object_property(session, "hasInputSource", input_source, ontology_warnings)

        if image_input is not None:
            append_object_property(session, "hasImageInput", image_input, ontology_warnings)
            append_object_property(feature, "derivedFromImage", image_input, ontology_warnings)

        append_object_property(feature, "derivedFromMeasurement", measurement, ontology_warnings)

        # Measurement data
        set_data_property(measurement, "heightCm", round(height, 2), ontology_warnings)
        set_data_property(measurement, "weightKg", round(weight, 2), ontology_warnings)
        set_data_property(measurement, "abdomenCm", round(abdomen, 2), ontology_warnings)
        set_data_property(measurement, "hipCm", round(hip, 2), ontology_warnings)

        if chest is not None:
            set_data_property(measurement, "chestCm", round(chest_value, 2), ontology_warnings)

        # Feature data
        set_data_property(feature, "bmiValue", round(bmi, 2), ontology_warnings)
        set_data_property(feature, "whrValue", round(whr, 2), ontology_warnings)
        set_data_property(feature, "wthrValue", round(wthr, 2), ontology_warnings)

        # Prediction data
        set_data_property(prediction, "predictedBodyFat", round(predicted_bf, 2), ontology_warnings)
        set_data_property(prediction, "predictedBodyFat_raw", round(predicted_bf, 2), ontology_warnings)
        set_data_property(prediction, "predictedBodyFat_validated", round(predicted_bf, 2), ontology_warnings)

        # Quality data
        set_data_property(quality, "poseVisibility", round(pose_visibility, 2), ontology_warnings)
        set_data_property(quality, "maskConfidence", round(mask_confidence, 2), ontology_warnings)
        set_data_property(quality, "confidenceScore", round(confidence_score, 3), ontology_warnings)
        set_data_property(quality, "missingLandmarkCount", missing_landmark_count, ontology_warnings)

        # Session metadata
        set_data_property(session, "sessionId", session_id, ontology_warnings)
        set_data_property(session, "timestamp", datetime.now().isoformat(timespec="seconds"), ontology_warnings)

        # Input source metadata
        set_data_property(input_source, "sourceType", source_type, ontology_warnings)

        if image_input is not None:
            if image_name:
                set_data_property(image_input, "imageName", image_name, ontology_warnings)
            if image_path:
                set_data_property(image_input, "imagePath", image_path, ontology_warnings)

        semantic_pipeline.append("Runtime PredictionSession graph created")

        # ------------------------------------------------------------
        # 5. Run reasoner
        # ------------------------------------------------------------
        reasoning_status = "Success"
        reasoning_error = None

        try:
            sync_reasoner_pellet(
                infer_property_values=True,
                infer_data_property_values=True,
            )

            semantic_pipeline.append("Pellet reasoner executed")

            print("\n========== AFTER REASONER DEBUG ==========")
            print("session obj:", session)
            print("session name:", owl_name(session) if session else "SESSION_IS_NONE")

            if session:
                print("hasQuality:", [owl_name(x) for x in getattr(session, "hasQuality", [])])
                print("hasFeature:", [owl_name(x) for x in getattr(session, "hasFeature", [])])
                print("hasWarningLevel RAW:", [owl_name(x) for x in getattr(session, "hasWarningLevel", [])])
                print("hasAnomalyFlag RAW:", [owl_name(x) for x in getattr(session, "hasAnomalyFlag", [])])
                print("hasValidationResult RAW:", [owl_name(x) for x in getattr(session, "hasValidationResult", [])])

            print("quality:", owl_name(quality) if quality else "QUALITY_IS_NONE")
            if quality:
                print("hasQualityAssessment:", [owl_name(x) for x in getattr(quality, "hasQualityAssessment", [])])
                print("hasConfidenceLevel:", [owl_name(x) for x in getattr(quality, "hasConfidenceLevel", [])])

            print("feature:", owl_name(feature) if feature else "FEATURE_IS_NONE")
            if feature:
                print("hasBMIClass:", [owl_name(x) for x in getattr(feature, "hasBMIClass", [])])
                print("hasSemanticFlag:", [owl_name(x) for x in getattr(feature, "hasSemanticFlag", [])])

            print("ontology warnings:", ontology_warnings)
            print("========== END AFTER DEBUG ==========\n")

        except Exception as e:
            reasoning_status = "Failed"
            reasoning_error = str(e)
            semantic_pipeline.append("Pellet reasoner failed")

        # ------------------------------------------------------------
        # 6. Read inferred ontology results
        # ------------------------------------------------------------
   
        bmi_class = "Unknown"
        bmi_class_values = names_from_property(feature, "hasBMIClass")

        for instance_name, label in {
            "Lean_Instance": "Lean",
            "Normal_Instance": "Normal",
            "Overweight_Instance": "Overweight",
            "Obese_Instance": "Obese",
        }.items():
            if instance_name in bmi_class_values:
                bmi_class = label
                break

        fat_level = "Unknown"
        fat_values = names_from_property(prediction, "hasFatLevel")

        for instance_name, label in {
            "LowFat_Instance": "Low",
            "NormalFat_Instance": "Normal",
            "HighFat_Instance": "High",
        }.items():
            if instance_name in fat_values:
                fat_level = label
                break

        semantic_flags: List[str] = []
        semantic_flag_values = names_from_property(feature, "hasSemanticFlag")

        # Backward compatibility: some older rules may still write flags into hasBMIClass.
        semantic_flag_values += [
            v for v in bmi_class_values
            if v in ["AbdominalObesity_Instance", "HiddenObesity_Instance"]
        ]

        semantic_flag_mapping = {
            "AbdominalObesity_Instance": "AbdominalObesity",
            "HiddenObesity_Instance": "HiddenObesity",
        }

        for inst_name, label in semantic_flag_mapping.items():
            if inst_name in semantic_flag_values:
                semantic_flags.append(label)

        semantic_flags = unique_keep_order(semantic_flags)

        image_quality = "GoodImage"
        quality_assessment_values = names_from_property(quality, "hasQualityAssessment")
        if "LowImageQuality_Instance" in quality_assessment_values:
            image_quality = "LowImageQuality"

        confidence_level = "Unknown"
        confidence_values = names_from_property(quality, "hasConfidenceLevel")
        for instance_name, label in {
            "LowConfidence_Instance": "LowConfidence",
            "MediumConfidence_Instance": "MediumConfidence",
            "HighConfidence_Instance": "HighConfidence",
        }.items():
            if instance_name in confidence_values:
                confidence_level = label
                break

        anomaly_types: List[str] = []
        anomaly_values = names_from_property(session, "hasAnomalyFlag")
        for instance_name, label in {
            "QualityAnomaly_Instance": "QualityAnomaly",
            "SemanticInconsistency_Instance": "SemanticInconsistency",
            "MeasurementInconsistency_Instance": "MeasurementInconsistency",
        }.items():
            if instance_name in anomaly_values:
                anomaly_types.append(label)

        anomaly_types = unique_keep_order(anomaly_types)

        warning_levels: List[str] = []
        warning_values = names_from_property(session, "hasWarningLevel")
        for instance_name, label in {
            "LowWarning_Instance": "LowWarning",
            "MediumWarning_Instance": "MediumWarning",
            "HighWarning_Instance": "HighWarning",
        }.items():
            if instance_name in warning_values:
                warning_levels.append(label)

        warning_levels = []

        if hasattr(session, "hasWarningLevel"):
            warning_levels = [
                owl_name(v).replace("_Instance", "")
                for v in session.hasWarningLevel
                if v is not None
            ]
            
        warning_level = "None"
        if "HighWarning" in warning_levels:
            warning_level = "HighWarning"
        elif "MediumWarning" in warning_levels:
            warning_level = "MediumWarning"
        elif "LowWarning" in warning_levels:
            warning_level = "LowWarning"

        validation_values = names_from_property(session, "hasValidationResult")

        # Invalid has priority over valid.
        if "InvalidResult_Instance" in validation_values:
            validation_status = "Invalid"
        elif "ValidResult_Instance" in validation_values:
            validation_status = "Valid"
        else:
            validation_status = "Valid"

        triggered_rules: List[str] = []

        # Rule trace from inferred semantic labels.
        if bmi_class != "Unknown":
            triggered_rules.append(f"RULE_BMI_{bmi_class.upper()}")

        if fat_level != "Unknown":
            triggered_rules.append(f"RULE_BODYFAT_{fat_level.upper()}")

        if "AbdominalObesity" in semantic_flags:
            triggered_rules.append("RULE_ABDOMINAL_OBESITY")

        if "HiddenObesity" in semantic_flags:
            triggered_rules.append("RULE_HIDDEN_OBESITY")

        if pose_visibility < 0.5:
            triggered_rules.append("RULE_LOW_IMAGE_QUALITY_POSE")

        if mask_confidence < 0.2:
            triggered_rules.append("RULE_LOW_IMAGE_QUALITY_MASK")

        confidence_rule_map = {
            "LowConfidence": "RULE_LOW_CONFIDENCE",
            "MediumConfidence": "RULE_MEDIUM_CONFIDENCE",
            "HighConfidence": "RULE_HIGH_CONFIDENCE",
        }

        if confidence_level != "Unknown":
            triggered_rules.append(
                confidence_rule_map.get(confidence_level)
            )

        for anomaly in anomaly_types:
            if anomaly == "QualityAnomaly":
                triggered_rules.append("RULE_QUALITY_ANOMALY")
            elif anomaly == "SemanticInconsistency":
                triggered_rules.append("RULE_SEMANTIC_INCONSISTENCY")
            elif anomaly == "MeasurementInconsistency":
                triggered_rules.append("RULE_MEASUREMENT_INCONSISTENCY")

        for warning in warning_levels:
            if warning == "HighWarning":
                if "SemanticInconsistency" in anomaly_types:
                    triggered_rules.append("RULE_HIGH_WARNING_SEMANTIC")
                if "MeasurementInconsistency" in anomaly_types:
                    triggered_rules.append("RULE_HIGH_WARNING_MEASUREMENT")

            elif warning == "MediumWarning":
                if "AbdominalObesity" in semantic_flags:
                    triggered_rules.append("RULE_MEDIUM_WARNING_ABDOMINAL_OBESITY")

                if image_quality == "LowImageQuality":
                    triggered_rules.append("RULE_MEDIUM_WARNING_LOW_IMAGE_QUALITY")

                if "QualityAnomaly" in anomaly_types:
                    triggered_rules.append("RULE_MEDIUM_WARNING_QUALITY")

            elif warning == "LowWarning":
                if "HiddenObesity" in semantic_flags:
                    triggered_rules.append("RULE_LOW_WARNING_HIDDEN_OBESITY")

        if validation_status == "Invalid":
            if "SemanticInconsistency" in anomaly_types:
                triggered_rules.append("RULE_INVALID_SEMANTIC")
            if "MeasurementInconsistency" in anomaly_types:
                triggered_rules.append("RULE_INVALID_MEASUREMENT")
        elif validation_status == "Valid" and confidence_level == "HighConfidence":
            triggered_rules.append("RULE_VALID_HIGH_CONFIDENCE")

        triggered_rules = unique_keep_order(triggered_rules)

        # ------------------------------------------------------------
        # 7. Create dynamic explanations and recommendations
        # ------------------------------------------------------------

        explanations: List[str] = []
        recommendations: List[str] = []

        def create_explanation(text: str, rule_name: Optional[str] = None) -> None:
            explanations.append(text)
            exp_inst = create_instance(
                onto,
                "SemanticExplanation",
                f"explanation_{uid}_{len(explanations)}",
                ontology_warnings,
            )
            if exp_inst is not None:
                set_data_property(exp_inst, "explanationText", text, ontology_warnings)
                append_object_property(session, "hasExplanation", exp_inst, ontology_warnings)
                if rule_name:
                    add_rule_trace(onto, exp_inst, rule_name, triggered_rules, ontology_warnings)

        def create_recommendation(text: str, rule_name: Optional[str] = None) -> None:
            recommendations.append(text)
            rec_inst = create_instance(
                onto,
                "Recommendation",
                f"recommendation_{uid}_{len(recommendations)}",
                ontology_warnings,
            )
            if rec_inst is not None:
                set_data_property(rec_inst, "recommendationText", text, ontology_warnings)
                append_object_property(session, "hasRecommendation", rec_inst, ontology_warnings)
                if rule_name:
                    add_rule_trace(onto, rec_inst, rule_name, triggered_rules, ontology_warnings)

        if bmi_class != "Unknown":
            create_explanation(
                f"BMI = {bmi:.2f} indicates {bmi_class} body state.",
                f"RULE_BMI_{bmi_class.upper()}",
            )

        if fat_level != "Unknown":
            create_explanation(
                f"Predicted body fat = {predicted_bf:.2f}% indicates {fat_level}.",
                f"RULE_BODYFAT_{fat_level.upper()}",
            )

        if "AbdominalObesity" in semantic_flags:
            create_explanation(
                f"WtHR = {wthr:.2f} suggests central abdominal fat risk.",
                "RULE_ABDOMINAL_OBESITY",
            )
            create_recommendation(
                "Reduce central fat risk through calorie control, regular cardio, and resistance training.",
                "RULE_ABDOMINAL_OBESITY",
            )

        if "HiddenObesity" in semantic_flags:
            create_explanation(
                f"BMI is normal but WHR = {whr:.2f} is high, suggesting hidden obesity risk.",
                "RULE_HIDDEN_OBESITY",
            )
            create_recommendation(
                "Monitor waist ratio and prioritize reducing abdominal fat despite normal BMI.",
                "RULE_HIDDEN_OBESITY",
            )

        if image_quality == "LowImageQuality":
            create_explanation(
                "Image quality is low, so semantic confidence is reduced.",
                "RULE_LOW_IMAGE_QUALITY_POSE",
            )
            create_recommendation(
                "Retake the image with better lighting, clearer pose, and full body visibility.",
                "RULE_LOW_IMAGE_QUALITY_POSE",
            )

        confidence_rule_map = {
            "LowConfidence": "RULE_LOW_CONFIDENCE",
            "MediumConfidence": "RULE_MEDIUM_CONFIDENCE",
            "HighConfidence": "RULE_HIGH_CONFIDENCE",
        }

        if confidence_level != "Unknown":
            create_explanation(
                f"Ontology confidence level is {confidence_level} based on pose visibility and mask confidence.",
                confidence_rule_map.get(confidence_level),
            )

        if "SemanticInconsistency" in anomaly_types:
            create_explanation(
                "Semantic inconsistency detected: normal BMI combined with high WHR and high predicted body fat.",
                "RULE_SEMANTIC_INCONSISTENCY",
            )
            create_recommendation(
                "Review central fat risk and verify the prediction with higher quality images or manual measurements.",
                "RULE_SEMANTIC_INCONSISTENCY",
            )

        if "MeasurementInconsistency" in anomaly_types:
            create_explanation(
                "Measurement inconsistency detected: large abdomen value conflicts with a low predicted fat level.",
                "RULE_MEASUREMENT_INCONSISTENCY",
            )
            create_recommendation(
                "Recheck abdomen measurement and rerun AI scan before trusting the result.",
                "RULE_MEASUREMENT_INCONSISTENCY",
            )

        if "QualityAnomaly" in anomaly_types:
            create_explanation(
                "Quality anomaly detected because input quality produced low semantic confidence.",
                "RULE_QUALITY_ANOMALY",
            )
            create_recommendation(
                "Improve image quality before using the prediction for decision support.",
                "RULE_QUALITY_ANOMALY",
            )

        if fat_level == "HighFat":
            create_recommendation(
                "Increase physical activity and maintain a moderate calorie deficit.",
                "RULE_BODYFAT_HIGH",
            )
        elif fat_level == "LowFat":
            create_recommendation(
                "Maintain balanced nutrition and resistance training.",
                "RULE_BODYFAT_LOW",
            )

        if not recommendations:
            create_recommendation(
                "Maintain current healthy lifestyle and monitor body measurements periodically.",
                None,
            )

        # ------------------------------------------------------------
        # 8. Store latency and save output ontology
        # ------------------------------------------------------------

        latency_ms = round((time.perf_counter() - start_time) * 1000, 2)
        set_data_property(session, "latencyMs", latency_ms, ontology_warnings)

        # Optional warning text on static warning individual is not ideal because it is shared.
        # Instead, the warning explanation is represented through dynamic SemanticExplanation nodes.

        try:
            onto.save(file=out_file)
            semantic_pipeline.append(f"Output ontology saved: {out_file}")
        except Exception as e:
            reasoning_status = "Failed"
            reasoning_error = f"{reasoning_error}; Save failed: {e}" if reasoning_error else f"Save failed: {e}"
            semantic_pipeline.append("Failed to save output ontology")

    # ------------------------------------------------------------
    # 9. Build output for UI/dashboard/batch evaluation
    # ------------------------------------------------------------

    reasoning_graph_trace = {
        "Session": session_id,
        "InputSource": owl_name(input_source) if "input_source" in locals() else None,
        "ImageInput": owl_name(image_input) if "image_input" in locals() and image_input is not None else None,
        "Measurement": owl_name(measurement) if "measurement" in locals() else None,
        "Feature": owl_name(feature) if "feature" in locals() else None,
        "Prediction": owl_name(prediction) if "prediction" in locals() else None,
        "QualityAssessment": owl_name(quality) if "quality" in locals() else None,
        "ValidationResult": validation_status if "validation_status" in locals() else None,
        "AnomalyType": anomaly_types if "anomaly_types" in locals() else [],
        "WarningLevel": warning_level if "warning_level" in locals() else "None",
        "ExplanationCount": len(explanations) if "explanations" in locals() else 0,
        "RecommendationCount": len(recommendations) if "recommendations" in locals() else 0,
    }

    inferred_trace = {
        "Session_Classes": extract_classes(session) if "session" in locals() else [],
        "Feature_Classes": extract_classes(feature) if "feature" in locals() else [],
        "Prediction_Classes": extract_classes(prediction) if "prediction" in locals() else [],
        "Quality_Classes": extract_classes(quality) if "quality" in locals() else [],
    }

    return {
        "Session_ID": session_id,

        # Validation / Reasoner
        "Validation_Status": validation_status if "validation_status" in locals() else "Unknown",
        "Validation_Errors": validation_errors,
        "Reasoning_Status": reasoning_status if "reasoning_status" in locals() else "Unknown",
        "Reasoning_Error": reasoning_error if "reasoning_error" in locals() else None,
        "Ontology_Warnings": ontology_warnings,

        # Measurement / Feature / Prediction
        "Height": round(height, 2),
        "Weight": round(weight, 2),
        "Chest": round(chest_value, 2) if chest is not None else None,
        "Abdomen": round(abdomen, 2),
        "Hip": round(hip, 2),
        "BMI": round(bmi, 2),
        "WHR": round(whr, 2),
        "WtHR": round(wthr, 2),
        "BodyFat": round(predicted_bf, 2),

        # Semantic classification
        "BMI_Class": bmi_class if "bmi_class" in locals() else "Unknown",
        "Fat_Level": fat_level if "fat_level" in locals() else "Unknown",
        "Semantic_Flags": semantic_flags if "semantic_flags" in locals() else [],
        "Image_Quality": image_quality if "image_quality" in locals() else "Unknown",
        "Confidence_Level": confidence_level if "confidence_level" in locals() else "Unknown",
        "Confidence_Score": confidence_score,
        "Anomaly_Type": anomaly_types if "anomaly_types" in locals() else [],
        "Warning_Level": warning_level if "warning_level" in locals() else "None",

        # Traceability
        "Triggered_Rules": triggered_rules if "triggered_rules" in locals() else [],
        "Explanations": explanations if "explanations" in locals() else [],
        "Recommendations": recommendations if "recommendations" in locals() else [],
        "Explanation_Count": len(explanations) if "explanations" in locals() else 0,
        "Recommendation_Count": len(recommendations) if "recommendations" in locals() else 0,
        "Reasoning_Graph_Trace": reasoning_graph_trace,
        "Inferred_Class_Trace": inferred_trace,
        "Semantic_Pipeline": semantic_pipeline,

        # Quality / metadata
        "Pose_Visibility": round(pose_visibility, 2),
        "Mask_Confidence": round(mask_confidence, 2),
        "Missing_Landmark_Count": missing_landmark_count,
        "Source_Type": source_type,
        "Image_Name": image_name,
        "Image_Path": image_path,
        "Ontology_Latency_ms": latency_ms if "latency_ms" in locals() else None,
    }


# ============================================================
# QUICK LOCAL TEST
# ============================================================

if __name__ == "__main__":
    result = run_ontology(
        height=170,
        weight=72,
        chest=95,
        abdomen=96,
        hip=100,
        predicted_bf=23,
        pose_visibility=0.42,
        mask_confidence=0.45,
        missing_landmark_count=3,
        source_type="AI Scan",
        image_name="sample_front_side.jpg",
        image_path="data/sample_front_side.jpg",
    )

    for key, value in result.items():
        print(f"{key}: {value}")
