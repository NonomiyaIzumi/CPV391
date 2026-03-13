"""
Face recognition using InsightFace ArcFace embeddings.
Embeddings are L2-normalized vectors and compared by cosine distance.
"""
import numpy as np
from insightface.utils import face_align
import insightface

from config import TOLERANCE

# ── Singleton model instances ──────────────────────────────────
_rec_model = None


def get_face_recognizer():
    """Initialize InsightFace ArcFace recognizer (singleton)."""
    global _rec_model
    if _rec_model is None:
        print("[Recognize] Loading InsightFace ArcFace model...")
        # model='buffalo_l' includes robust ArcFace recognition model.
        try:
            app = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=['CUDAExecutionProvider', 'CPUExecutionProvider']
            )
            app.prepare(ctx_id=0, det_size=(640, 640))
        except Exception:
            app = insightface.app.FaceAnalysis(
                name='buffalo_l',
                providers=['CPUExecutionProvider']
            )
            app.prepare(ctx_id=-1, det_size=(640, 640))
        _rec_model = app.models.get('recognition')
        if _rec_model is None:
            raise RuntimeError("InsightFace recognition model not available in buffalo_l package")
        print("[Recognize] InsightFace ArcFace loaded successfully!")
    return _rec_model


def extract_features(rgb: np.ndarray, face_info: dict) -> np.ndarray:
    """
    Hàm cực kỳ quan trọng:
    1. Lấy tọa độ 5 điểm mốc (landmarks) từ YuNet.
    2. Căn chỉnh khuôn mặt cho thẳng lại, crop thật sát quanh mắt mũi miệng (loại bỏ tóc, gọng kính rầm rà, và background).
    3. Mã hóa thành vector đặc trưng 128-D.
    """
    rec_model = get_face_recognizer()

    if "landmarks" not in face_info:
        return None
    lmk = np.asarray(face_info["landmarks"], dtype=np.float32)
    if lmk.shape != (5, 2):
        return None

    # InsightFace expects BGR image for norm_crop + get_feat.
    if rgb.shape[-1] == 3:
        img_bgr = rgb[:, :, ::-1]
    else:
        img_bgr = rgb

    aligned = face_align.norm_crop(img_bgr, landmark=lmk, image_size=112)
    feat = rec_model.get_feat(aligned).flatten().astype(np.float32)
    norm = np.linalg.norm(feat)
    if norm < 1e-8:
        return None
    return feat / norm


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance on normalized ArcFace vectors."""
    a = np.asarray(a, dtype=np.float32)
    b = np.asarray(b, dtype=np.float32)
    an = a / (np.linalg.norm(a) + 1e-8)
    bn = b / (np.linalg.norm(b) + 1e-8)
    sim = float(np.dot(an, bn))
    return float(1.0 - sim)


def distance_to_confidence(dist: float, tolerance: float = TOLERANCE) -> float:
    """Map SFace cosine distance to 0-1 confidence score."""
    if dist >= tolerance * 2:
        return 0.0
    return max(0.0, 1.0 - dist / (tolerance * 2))


def match_embedding(
    query_vec: np.ndarray,
    known_db: dict[str, list[np.ndarray]],
    tolerance: float = TOLERANCE,
) -> tuple[str | None, float, float]:
    """
    Match a 128-D query embedding against the known database using cosine distance.
    Returns (student_id, distance, confidence) or (None, best_dist, 0.0).
    """
    best_student = None
    best_dist = float("inf")

    for student_id, ref_vecs in known_db.items():
        for ref_vec in ref_vecs:
            dist = cosine_distance(query_vec, ref_vec)
            if dist < best_dist:
                best_dist = dist
                best_student = student_id

    if best_dist <= tolerance and best_student is not None:
        conf = distance_to_confidence(best_dist, tolerance)
        return best_student, best_dist, conf

    return None, best_dist, 0.0


def match_embedding_mean(
    query_vec: np.ndarray,
    known_db: dict[str, list[np.ndarray]],
    tolerance: float = TOLERANCE,
) -> tuple[str | None, float, float]:
    """
    Match using mean embedding per student.
    """
    best_student = None
    best_dist = float("inf")

    for student_id, ref_vecs in known_db.items():
        mean_vec = np.mean(ref_vecs, axis=0)
        dist = cosine_distance(query_vec, mean_vec)
        if dist < best_dist:
            best_dist = dist
            best_student = student_id

    if best_dist <= tolerance and best_student is not None:
        conf = distance_to_confidence(best_dist, tolerance)
        return best_student, best_dist, conf

    return None, best_dist, 0.0
