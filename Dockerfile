# Sử dụng đúng phiên bản Python bạn đang dùng ở Local
FROM python:3.11.0-slim

# Cài đặt các thư viện hệ thống cần thiết để MediaPipe và OpenCV chạy trên Linux
RUN apt-get update && apt-get install -y \
    libgl1-mesa-glx \
    libglib2.0-0 \
    libsm6 \
    libxext6 \
    libxrender1 \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /app

# Copy và cài đặt thư viện
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy toàn bộ code vào
COPY . .

# Hugging Face Spaces chạy trên cổng 7860
EXPOSE 7860

# Lệnh chạy Streamlit
CMD ["streamlit", "run", "app.py", "--server.port=7860", "--server.address=0.0.0.0"]