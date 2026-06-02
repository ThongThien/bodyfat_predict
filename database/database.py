import os
import time
import uuid
from typing import Any

from dotenv import load_dotenv
from supabase import Client, create_client


# Configuration
load_dotenv()

SUPABASE_URL = os.getenv("SUPABASE_URL")
SUPABASE_KEY = os.getenv("SUPABASE_KEY")
BUCKET_NAME = os.getenv("SUPABASE_BUCKET_NAME", "bodyfat_images")

if not SUPABASE_URL or not SUPABASE_KEY:
    raise ValueError("Missing SUPABASE_URL or SUPABASE_KEY in .env file.")

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)


# Helpers
def safe_float(value: Any):
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None


# Storage functions
def upload_image_and_get_url(image_bytes, user_id, prefix="img"):
    try:
        if not image_bytes:
            return None

        unique_id = str(uuid.uuid4())[:8]
        file_path = f"{user_id}/{prefix}_{int(time.time())}_{unique_id}.jpg"

        supabase.storage.from_(BUCKET_NAME).upload(
            path=file_path,
            file=image_bytes,
            file_options={
                "content-type": "image/jpeg",
                "upsert": "true",
            },
        )

        url_response = supabase.storage.from_(BUCKET_NAME).get_public_url(file_path)
        return url_response if isinstance(url_response, str) else url_response.public_url

    except Exception as error:
        print(f"Storage upload error ({prefix}): {error}")
        return None


# Database functions
def save_complete_measurement(
    age,
    weight,
    height,
    results_dict,
    img_front_bytes,
    img_side_bytes,
    method="AI Scan",
):
    try:
        user_response = supabase.auth.get_user()

        if not (user_response and user_response.user):
            return {"success": False, "error": "User is not authenticated."}

        user_id = user_response.user.id

        front_image_url = upload_image_and_get_url(img_front_bytes, user_id, "front")
        side_image_url = upload_image_and_get_url(img_side_bytes, user_id, "side")

        chest = safe_float(results_dict.get("Chest"))
        abdomen = safe_float(results_dict.get("Abdomen"))
        hip = safe_float(results_dict.get("Hip"))
        thigh = safe_float(results_dict.get("Thigh"))
        body_fat = safe_float(results_dict.get("body_fat"))

        weight_value = safe_float(weight)
        height_value = safe_float(height)

        wpa = (abdomen**2) / weight_value if abdomen and weight_value else None
        wthr = abdomen / height_value if abdomen and height_value else None
        whr = abdomen / hip if abdomen and hip else None

        record = {
            "user_id": user_id,
            "age": int(age) if age is not None else None,
            "weight": weight_value,
            "height": height_value,
            "chest": chest,
            "abdomen": abdomen,
            "hip": hip,
            "thigh": thigh,
            "wpa": safe_float(wpa),
            "wthr": safe_float(wthr),
            "whr": safe_float(whr),
            "body_fat": body_fat,
            "image_url_front": front_image_url,
            "image_url_side": side_image_url,
            "method": method,
        }

        supabase.table("measurements").insert(record).execute()
        return {"success": True}

    except Exception as error:
        print(f"Save measurement error: {error}")
        return {"success": False, "error": str(error)}


# History functions
def get_measurement_history(limit=20):
    try:
        response = (
            supabase.table("measurements")
            .select("*")
            .order("created_at", desc=True)
            .limit(limit)
            .execute()
        )
        return response.data

    except Exception as error:
        print(f"Fetch measurement history error: {error}")
        return []


def get_user_history():
    try:
        response = supabase.auth.get_user()

        if response and hasattr(response, "user") and response.user is not None:
            user_id = response.user.id

            history_response = (
                supabase.table("measurements")
                .select("*")
                .eq("user_id", user_id)
                .order("created_at", desc=True)
                .execute()
            )

            return history_response.data

        return []

    except Exception as error:
        print(f"Fetch user history error: {error}")
        return []


# Delete functions
def delete_measurement(record_id):
    try:
        supabase.table("measurements").delete().eq("id", record_id).execute()
        return True

    except Exception as error:
        print(f"Delete measurement error: {error}")
        return False


# Profile functions
def create_or_update_profile(user_id, fullname):
    try:
        data = {
            "id": user_id,
            "fullname": fullname,
            "updated_at": "now()",
        }

        return supabase.table("profiles").upsert(data).execute()

    except Exception as error:
        print(f"Profile upsert error: {error}")
        return None


def get_profile(user_id):
    try:
        response = (
            supabase.table("profiles")
            .select("*")
            .eq("id", user_id)
            .single()
            .execute()
        )

        return response.data

    except Exception as error:
        print(f"Fetch profile error: {error}")
        return None


# Authentication functions
def sign_up(email, password, fullname):
    try:
        response = supabase.auth.sign_up(
            {
                "email": email,
                "password": password,
            }
        )

        if response.user:
            profile_data = {
                "id": response.user.id,
                "fullname": fullname,
            }

            supabase.table("profiles").upsert(profile_data).execute()

        return response

    except Exception as error:
        print(f"Sign up error: {error}")
        return error


def sign_in(email, password):
    try:
        return supabase.auth.sign_in_with_password(
            {
                "email": email,
                "password": password,
            }
        )

    except Exception as error:
        print(f"Sign in error: {error}")
        return error


def get_current_user():
    return supabase.auth.get_user()