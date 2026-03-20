import os
import time
import uuid
from supabase import create_client, Client

# --- 1. CONFIGURATION ---
# Thay thế bằng thông tin từ Supabase Dashboard của bạn
SUPABASE_URL = "https://niwzlqcufmiiuitfxcjm.supabase.co"
SUPABASE_KEY = "sb_publishable_XqTrliFP9a2WBNQ2QSntFQ_hLaDjXSr" 
BUCKET_NAME = "bodyfat_images"

# Khởi tạo Client
supabase: Client = create_client(SUPABASE_URL, SUPABASE_KEY)

# --- 2. STORAGE FUNCTIONS (Xử lý ảnh) ---

def upload_image_and_get_url(image_bytes, user_id, prefix="img"):
    """
    Tối ưu: Thêm user_id vào đường dẫn để phân loại ảnh theo người dùng.
    """
    try:
        if not image_bytes:
            return None
            
        unique_id = str(uuid.uuid4())[:8]
        # Tạo đường dẫn: user_id/prefix_timestamp.jpg
        file_path = f"{user_id}/{prefix}_{int(time.time())}_{unique_id}.jpg"
        
        # Upload lên Storage
        supabase.storage.from_(BUCKET_NAME).upload(
            path=file_path,
            file=image_bytes,
            file_options={"content-type": "image/jpeg", "upsert": "true"}
        )
        
        # Lấy URL công khai
        url_res = supabase.storage.from_(BUCKET_NAME).get_public_url(file_path)
        return url_res if isinstance(url_res, str) else url_res.public_url
        
    except Exception as e:
        print(f"❌ Lỗi Upload Storage ({prefix}): {e}")
        return None

# --- 3. DATABASE FUNCTIONS (Xử lý bảng measurements) ---

def save_complete_measurement(age, weight, height, results_dict, img_front_bytes, img_side_bytes, method="AI Scan"):
    try:
        # Lấy User hiện tại
        user_res = supabase.auth.get_user()
        if not (user_res and user_res.user):
            return {"success": False, "error": "Cần đăng nhập"}
        
        u_id = user_res.user.id

        # 1. Upload ảnh với cấu trúc thư mục mới
        url_f = upload_image_and_get_url(img_front_bytes, u_id, "front")
        url_s = upload_image_and_get_url(img_side_bytes, u_id, "side")

        # 2. Tạo bản ghi (Bỏ updated_at để DB tự lo)
        record = {
            "user_id": u_id,
            "age": int(age),
            "weight": float(weight),
            "height": float(height),
            "chest": float(results_dict.get('Chest', 0)),
            "abdomen": float(results_dict.get('Abdomen', 0)),
            "hip": float(results_dict.get('Hip', 0)),
            "thigh": float(results_dict.get('Thigh', 0)),
            "body_fat": float(results_dict.get('body_fat', 0)),
            "image_url_front": url_f,
            "image_url_side": url_s,
            "method": method
        }

        supabase.table("measurements").insert(record).execute()
        return {"success": True}
        
    except Exception as e:
        print(f"❌ Lỗi Save Record: {e}")
        return {"success": False, "error": str(e)}

def get_measurement_history(limit=20):
    """
    Truy vấn danh sách lịch sử đo đạc mới nhất.
    """
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

def delete_measurement(record_id):
    """
    Xóa một bản ghi theo ID.
    """
    try:
        supabase.table("measurements").delete().eq("id", record_id).execute()
        return True
    except Exception as e:
        print(f"❌ Lỗi Delete: {e}")
        return False

def create_or_update_profile(user_id, fullname):
    """
    Tạo mới hoặc cập nhật thông tin fullname cho người dùng.
    """
    try:
        data = {
            "id": user_id,
            "fullname": fullname,
            "updated_at": "now()" # Supabase sẽ tự hiểu để lấy giờ hiện tại
        }
        # Sử dụng upsert: Nếu ID đã tồn tại thì cập nhật, chưa có thì thêm mới
        response = supabase.table("profiles").upsert(data).execute()
        return response
    except Exception as e:
        print(f"❌ Lỗi Profile: {e}")
        return None

def get_profile(user_id):
    """
    Lấy thông tin profile của một người dùng cụ thể.
    """
    try:
        response = supabase.table("profiles").select("*").eq("id", user_id).single().execute()
        return response.data
    except Exception as e:
        print(f"❌ Lỗi lấy Profile: {e}")
        return None

# --- Bổ sung vào core/database.py ---

def sign_up(email, password, fullname):
    try:
        # 1. Tạo user trong hệ thống Auth
        res = supabase.auth.sign_up({"email": email, "password": password})
        
        # 2. Kiểm tra nếu res có user (đăng ký thành công)
        if res.user:
            user_id = res.user.id
            # 3. Ghi vào bảng profiles
            profile_data = {
                "id": user_id, 
                "fullname": fullname
                # update_at để mặc định hoặc dùng datetime.now()
            }
            # Sử dụng table().upsert() để tránh lỗi trùng lặp hoặc RLS chặt chẽ
            supabase.table("profiles").upsert(profile_data).execute()
            return res
        return res
    except Exception as e:
        print(f"Lỗi signup: {e}")
        return e

def sign_in(email, password):
    """Đăng nhập người dùng"""
    try:
        return supabase.auth.sign_in_with_password({"email": email, "password": password})
    except Exception as e:
        return e

def get_current_user():
    """Lấy thông tin người dùng đang đăng nhập"""
    return supabase.auth.get_user()

def get_user_history():
    try:
        # Lấy session hiện tại từ Supabase
        res = supabase.auth.get_user()
        
        # Kiểm tra an toàn: res không None và có thuộc tính user
        if res and hasattr(res, 'user') and res.user is not None:
            user_id = res.user.id
            response = supabase.table("measurements") \
                .select("*") \
                .eq("user_id", user_id) \
                .order("created_at", desc=True) \
                .execute()
            return response.data
        return [] # Trả về danh sách trống nếu chưa đăng nhập
    except Exception as e:
        print(f"Lỗi lấy lịch sử: {e}")
        return []