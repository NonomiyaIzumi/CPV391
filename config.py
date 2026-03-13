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

# ── Face Detection & Recognition (YOLOv26 & SFace) ────────────────────
# YOLOv26 dùng OpenVINO để siêu bức tốc trên Intel CPU. Tích hợp SAHI để cắt ảnh thành nhiều patch dò mặt nhỏ đầu cực đỉnh.
YOLO_FACE_MODEL = "yolov8n-face.pt" # Bạn dùng thư viện face cho yolov8 vì yolov26 là yolov8 đổi tên
SFACE_MODEL = "models/face_recognition_sface_2021dec.onnx"

# ── Balanced Attendance Pipeline (GPU-first) ─────────────────
PIPELINE_MODE = os.getenv("PIPELINE_MODE", "balanced")  # balanced | legacy
PERSON_TRACK_MODEL = os.getenv("PERSON_TRACK_MODEL", "rtdetr-l.pt")
PERSON_TRACK_IMG_SIZE = int(os.getenv("PERSON_TRACK_IMG_SIZE", "960"))
PERSON_CONF_THRESHOLD = float(os.getenv("PERSON_CONF_THRESHOLD", "0.35"))
TRACKER_CONFIG = os.getenv("TRACKER_CONFIG", "botsort.yaml")
DETECT_DEVICE = os.getenv("DETECT_DEVICE", "cuda:0")
TRACK_RECOG_INTERVAL = int(os.getenv("TRACK_RECOG_INTERVAL", "3"))
TRACK_KEEP_ID_MISSES = int(os.getenv("TRACK_KEEP_ID_MISSES", "10"))

# ── SAHI & OpenVINO ───────────────────────────────────────────
SAHI_SLICE_SIZE = 480            # Tăng lên 480 (bằng đúng chiều dọc Camera 640x480). Nó sẽ giúp chẻ ảnh thành 2 mảnh thay vì 6 mảnh. Super FPS!
SAHI_OVERLAP = 0.15              # Giảm overlap để bớt tải CPU nhưng vẫn đủ chống rơi mặt ở rìa tile
USE_OPENVINO = False             # Ưu tiên GPU cho pipeline cân bằng; bật lại True nếu chỉ chạy CPU

DET_SCORE_THRESHOLD = 0.30       # Tăng nhẹ ngưỡng để giảm false-positive và giảm tải hậu xử lý
MIN_FACE_SIZE = 12               # Bắt được các khuôn mặt nhỏ hơn ở khoảng cách xa

# ── Face Recognition ──────────────────────────────────────────
TOLERANCE = 0.45             # ArcFace cosine distance threshold (thấp hơn = chặt hơn)

# ── Enrollment ─────────────────────────────────────────────────
REQUIRED_SAMPLES = 20        # số mẫu embedding cần thu
SAVE_ALL_SAMPLES = True      # True = lưu tất cả; False = chỉ lưu mean
BLUR_THRESHOLD = 50.0        # variance of Laplacian; thấp hơn = mờ
OUTLIER_REMOVE_RATIO = 0.15  # loại bỏ 15% embedding xa nhất

# ── Attendance ─────────────────────────────────────────────────
FRAME_SKIP = 4               # Tăng FPS hiển thị; vẫn bám ổn nhờ cơ chế grace + adaptive SAHI
LATE_MINUTES = 10            # cửa sổ phút đầu: Present, sau đó Late
ABSENCE_THRESHOLD_SEC = 120  # giây vắng mặt → ghi time_out
ATTENDANCE_NOFACE_GRACE_INFERENCES = 3  # số lần suy luận không thấy mặt liên tiếp trước khi tính vắng

# ── Tracking / BBox Smoothing ────────────────────────────────
BBOX_SMOOTH_ALPHA = 0.65     # 0..1, cao hơn = bám nhanh hơn nhưng ít mượt hơn
TRACK_IOU_THRESHOLD = 0.20   # ngưỡng ghép detection vào track cũ
TRACK_MAX_MISSES = 8         # số vòng suy luận được phép mất dấu trước khi xóa track

# ── Display ────────────────────────────────────────────────────
SHOW_VIDEO = True            # hiển thị OpenCV window
BBOX_COLOR = (0, 255, 0)     # xanh lá
UNKNOWN_BBOX_COLOR = (0, 0, 255)  # đỏ
FONT_SCALE = 0.6
FONT_THICKNESS = 2
