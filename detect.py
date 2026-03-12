"""
Face detection using InsightFace SCRFD pretrained model (GPU accelerated).
"""
import numpy as np
import insightface
from insightface.app import FaceAnalysis

from config import INSIGHTFACE_MODEL, GPU_ID, DET_SCORE_THRESHOLD, MIN_FACE_SIZE

# ── Singleton model instance ───────────────────────────────────
_app: FaceAnalysis | None = None


def get_face_app() -> FaceAnalysis:
    """Initialize InsightFace model (singleton, loaded once)."""
    global _app
    if _app is None:
        print(f"[Detect] Loading InsightFace model: {INSIGHTFACE_MODEL} on GPU:{GPU_ID}")
        _app = FaceAnalysis(
            name=INSIGHTFACE_MODEL,
            providers=["CUDAExecutionProvider", "CPUExecutionProvider"],
        )
        _app.prepare(ctx_id=GPU_ID, det_thresh=DET_SCORE_THRESHOLD, det_size=(640, 640))
        print("[Detect] Model loaded successfully!")
    return _app


def detect_and_encode(rgb: np.ndarray) -> list:
    """
    Detect faces AND compute embeddings in one pass.

    InsightFace does detection + alignment + recognition together,
    which is faster than doing them separately.

    Returns list of insightface.app.common.Face objects:
        - face.bbox: (x1, y1, x2, y2)
        - face.embedding: 512-D numpy array
        - face.det_score: detection confidence
        - face.age, face.sex (optional)
    """
    app = get_face_app()
    faces = app.get(rgb)

    # Filter by MIN_FACE_SIZE
    filtered = []
    for face in faces:
        x1, y1, x2, y2 = face.bbox
        w = x2 - x1
        h = y2 - y1
        if w >= MIN_FACE_SIZE and h >= MIN_FACE_SIZE:
            filtered.append(face)

    return filtered


def detect_faces(rgb: np.ndarray) -> list[tuple[int, int, int, int]]:
    """
    Detect face bounding boxes only (backward-compatible API).

    Returns list of (x1, y1, x2, y2) as integers.
    """
    faces = detect_and_encode(rgb)
    return [
        (int(f.bbox[0]), int(f.bbox[1]), int(f.bbox[2]), int(f.bbox[3]))
        for f in faces
    ]


def crop_face(rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a face region from the image."""
    x1, y1, x2, y2 = bbox
    h, w = rgb.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return rgb[y1:y2, x1:x2]
