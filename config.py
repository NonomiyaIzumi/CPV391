"""
Centralized configuration for Smart Classroom Attendance System.
"""
import os

# ── RTSP / Camera ──────────────────────────────────────────────
RTSP_URL = os.getenv("RTSP_URL", "0")  # "0" = webcam fallback

# ── Database ───────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "attendance.db")

# ── Pre-processing ─────────────────────────────────────────────
RESIZE_WIDTH = 640           # Giảm về 640 (VGA). Đây là điểm ngọt để YuNet chạy được 30 FPS trên CPU mà vẫn nhìn tới 5m.
USE_CLAHE = False            # TẮT CLAHE vì hàm này ngốn rất nhiều CPU gây lag
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
USE_GAUSSIAN_BLUR = False    # Gaussian blur nhẹ
GAUSSIAN_KERNEL = (3, 3)

# ── Face Detection & Recognition (OpenCV / SFace) ────────────────────
# OpenCV YuNet siêu nhẹ. Để kết hợp bắt 5m xa + góc quay ngược + siêu mượt, ta code Stateful Tracker trong detect.py
YUNET_MODEL = "models/face_detection_yunet_2023mar.onnx"
SFACE_MODEL = "models/face_recognition_sface_2021dec.onnx"
DET_SCORE_THRESHOLD = 0.40       # Hạ xuống 0.4 để cố gắng bắt các khuôn mặt bị cắt nửa ở mép khung hình
MIN_FACE_SIZE = 15               # Hạ cực hạn xuống 15 pixel để quét khuôn mặt chiếm tỷ lệ rất nhỏ ở cự ly 2-5m

# ── Face Recognition ──────────────────────────────────────────
TOLERANCE = 0.35             # cosine distance threshold (SFace dùng cosine distance 1.0 - cosine_similarity; < 0.36 là cùng một người)

# ── Enrollment ─────────────────────────────────────────────────
REQUIRED_SAMPLES = 20        # số mẫu embedding cần thu
SAVE_ALL_SAMPLES = True      # True = lưu tất cả; False = chỉ lưu mean
BLUR_THRESHOLD = 50.0        # variance of Laplacian; thấp hơn = mờ
OUTLIER_REMOVE_RATIO = 0.15  # loại bỏ 15% embedding xa nhất

# ── Attendance ─────────────────────────────────────────────────
FRAME_SKIP = 5               # Xử lý 1 frame mỗi 5 frame (chạy siêu mượt)
LATE_MINUTES = 10            # cửa sổ phút đầu: Present, sau đó Late
ABSENCE_THRESHOLD_SEC = 120  # giây vắng mặt → ghi time_out

# ── Display ────────────────────────────────────────────────────
SHOW_VIDEO = True            # hiển thị OpenCV window
BBOX_COLOR = (0, 255, 0)     # xanh lá
UNKNOWN_BBOX_COLOR = (0, 0, 255)  # đỏ
FONT_SCALE = 0.6
FONT_THICKNESS = 2
