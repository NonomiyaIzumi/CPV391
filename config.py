"""
Centralized configuration for Smart Classroom Attendance System.
"""
import os

# ── RTSP / Camera ──────────────────────────────────────────────
RTSP_URL = os.getenv("RTSP_URL", "0")  # "0" = webcam fallback

# ── Database ───────────────────────────────────────────────────
DB_PATH = os.getenv("DB_PATH", "attendance.db")

# ── Pre-processing ─────────────────────────────────────────────
RESIZE_WIDTH = 1280          # giữ lớn cho camera xa/góc rộng
USE_CLAHE = True             # cân bằng sáng cục bộ
CLAHE_CLIP_LIMIT = 2.0
CLAHE_TILE_SIZE = (8, 8)
USE_GAUSSIAN_BLUR = False    # Gaussian blur nhẹ
GAUSSIAN_KERNEL = (3, 3)

# ── InsightFace ────────────────────────────────────────────────
INSIGHTFACE_MODEL = "buffalo_l"  # buffalo_l (best), buffalo_s (fast), buffalo_sc (fastest)
GPU_ID = 0                       # CUDA device index (0 = first GPU)
DET_SCORE_THRESHOLD = 0.5       # detection confidence threshold
MIN_FACE_SIZE = 40               # pixel – lọc mặt quá nhỏ

# ── Face Recognition ──────────────────────────────────────────
TOLERANCE = 0.45             # cosine distance threshold (InsightFace dùng cosine, thấp hơn dlib)

# ── Enrollment ─────────────────────────────────────────────────
REQUIRED_SAMPLES = 20        # số mẫu embedding cần thu
SAVE_ALL_SAMPLES = True      # True = lưu tất cả; False = chỉ lưu mean
BLUR_THRESHOLD = 50.0        # variance of Laplacian; thấp hơn = mờ
OUTLIER_REMOVE_RATIO = 0.15  # loại bỏ 15% embedding xa nhất

# ── Attendance ─────────────────────────────────────────────────
FRAME_SKIP = 2               # xử lý 1 frame mỗi N frame (tăng FPS)
LATE_MINUTES = 10            # cửa sổ phút đầu: Present, sau đó Late
ABSENCE_THRESHOLD_SEC = 120  # giây vắng mặt → ghi time_out

# ── Display ────────────────────────────────────────────────────
SHOW_VIDEO = True            # hiển thị OpenCV window
BBOX_COLOR = (0, 255, 0)     # xanh lá
UNKNOWN_BBOX_COLOR = (0, 0, 255)  # đỏ
FONT_SCALE = 0.6
FONT_THICKNESS = 2
