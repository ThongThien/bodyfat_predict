import os
import time
import uuid
from supabase import create_client, Client

# --- 1. CONFIGURATION ---
SUPABASE_URL = "https://niwzlqcufmiiuitfxcjm.supabase.co"
SUPABASE_KEY = "eyJhbGciOiJIUzI1NiIsInR5cCI6IkpXVCJ9.eyJpc3MiOiJzdXBhYmFzZSIsInJlZiI6Im5pd3pscWN1Zm1paXVpdGZ4Y2ptIiwicm9sZSI6ImFub24iLCJpYXQiOjE3NzM4MDQwMzcsImV4cCI6MjA4OTM4MDAzN30.Mrux0DVyVp3lqgO4wBt0UTzFjm-hANOLcoHjRJK0Vmc" 
BUCKET_NAME = "bodyfat_images"

supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 2. HELPERS ---

def safe_float(val):
    try:
        return float(val) if val is not None else None
    except:
        return None

# --- 3. STORAGE FUNCTIONS ---

def upload_image_and_get_url(image_bytes, user_id, prefix="img"):
    try:
        if not image_bytes:
            return None
            
        unique_id = str(uuid.uuid4())[:8]
        file_path = f"{user_id}/{prefix}_{int(time.time())}_{unique_id}.jpg"
        
        supabase.storage.from_(BUCKET_NAME).upload(
            path=file_path,
            file=image_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"}
        )
        
        url_res = supabase.storage.from_(BUCKET_NAME).get_public_url(file_path)
        return url_res if isinstance(url_res, str) else url_res.public_url
        
    except Exception as e:
        print(f"❌ Lỗi Upload Storage ({prefix}): {e}")
        return None

# --- 4. DATABASE FUNCTIONS ---

def save_complete_measurement(age, weight, height, results_dict, img_front_bytes, img_side_bytes, method="AI Scan"):
    try:
        # --- AUTH ---
        user_res = supabase.auth.get_user()
        if not (user_res and user_res.user):
            return {"success": False, "error": "Cần đăng nhập"}
        
        u_id = user_res.user.id

        # --- UPLOAD IMAGE ---
        url_f = upload_image_and_get_url(img_front_bytes, u_id, "front")
        url_s = upload_image_and_get_url(img_side_bytes, u_id, "side")

        # --- EXTRACT RAW VALUES ---
        chest = results_dict.get("Chest")
        abd = results_dict.get("Abdomen")
        hip = results_dict.get("Hip")
        thigh = results_dict.get("Thigh")
        bf = results_dict.get("body_fat")

        # --- DERIVED FEATURES (NEW v5) ---
        wpa = (abd**2) / weight if abd and weight else None
        wthr = abd / height if abd and height else None
        whr = abd / hip if abd and hip else None

        # --- RECORD ---
        record = {
            "user_id": u_id,

            # BASIC
            "age": int(age) if age is not None else None,
            "weight": safe_float(weight),
            "height": safe_float(height),

            # BODY (ALL NULLABLE)
            "chest": safe_float(chest),
            "abdomen": safe_float(abd),
            "hip": safe_float(hip),
            "thigh": safe_float(thigh),  # ✅ NULL nếu không có

            # NEW FEATURES
            "wpa": safe_float(wpa),
            "wthr": safe_float(wthr),
            "whr": safe_float(whr),

            # RESULT
            "body_fat": safe_float(bf),

            # MEDIA
            "image_url_front": url_f,
            "image_url_side": url_s,

            # META
            "method": method
        }

        supabase.table("measurements").insert(record).execute()
        return {"success": True}
        
    except Exception as e:
        print(f"❌ Lỗi Save Record: {e}")
        return {"success": False, "error": str(e)}

# --- HISTORY ---

def get_measurement_history(limit=20):
    try:
        response = supabase.table("measurements") \
            .select("*") \
            .order("created_at", desc=True) \
            .limit(limit) \
            .execute()
        return response.data
    except Exception as e:
        print(f"❌ Lỗi Fetch History: {e}")
        return []

def get_user_history():
    try:
        res = supabase.auth.get_user()
        
        if res and hasattr(res, 'user') and res.user is not None:
            user_id = res.user.id
            response = supabase.table("measurements") \
                .select("*") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .execute()
            return response.data
        return []
    except Exception as e:
        print(f"Lỗi lấy lịch sử: {e}")
        return []

# --- DELETE ---

def delete_measurement(record_id):
    try:
        supabase.table("measurements").delete().eq("id", record_id).execute()
        return True
    except Exception as e:
        print(f"❌ Lỗi Delete: {e}")
        return False

# --- PROFILE ---

def create_or_update_profile(user_id, fullname):
    try:
        data = {
            "id": user_id,
            "fullname": fullname,
            "updated_at": "now()"
        }
        return supabase.table("profiles").upsert(data).execute()
    except Exception as e:
        print(f"❌ Lỗi Profile: {e}")
        return None

def get_profile(user_id):
    try:
        response = supabase.table("profiles") \
            .select("*") \
            .eq("id", user_id) \
            .single() \
            .execute()
        return response.data
    except Exception as e:
        print(f"❌ Lỗi lấy Profile: {e}")
        return None

# --- AUTH ---

def sign_up(email, password, fullname):
    try:
        res = supabase.auth.sign_up({"email": email, "password": password})
        
        if res.user:
            user_id = res.user.id
            profile_data = {
                "id": user_id,
                "fullname": fullname
            }
            supabase.table("profiles").upsert(profile_data).execute()
        return res
    except Exception as e:
        print(f"Lỗi signup: {e}")
        return e

def sign_in(email, password):
    try:
        return supabase.auth.sign_in_with_password({"email": email, "password": password})
    except Exception as e:
        return e

def get_current_user():
    return supabase.auth.get_user()