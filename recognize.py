"""
Face recognition using OpenCV's built-in SFace model.
Embeddings are 128-D vectors. Highly robust to lighting and background.
"""
import cv2
import numpy as np

from config import SFACE_MODEL, TOLERANCE

# ── Singleton model instance ───────────────────────────────────
_recognizer = None

def get_face_recognizer():
    """Initialize SFace model (singleton)."""
    global _recognizer
    if _recognizer is None:
        print(f"[Recognize] Loading SFace model: {SFACE_MODEL}")
        _recognizer = cv2.FaceRecognizerSF.create(
            model=SFACE_MODEL,
            config="",
            backend_id=cv2.dnn.DNN_BACKEND_OPENCV,
            target_id=cv2.dnn.DNN_TARGET_CPU
        )
        print("[Recognize] SFace loaded successfully!")
    return _recognizer


def extract_features(rgb: np.ndarray, face_info: dict) -> np.ndarray:
    """
    Hàm cực kỳ quan trọng:
    1. Lấy tọa độ 5 điểm mốc (landmarks) từ YuNet.
    2. Căn chỉnh khuôn mặt cho thẳng lại, crop thật sát quanh mắt mũi miệng (loại bỏ tóc, gọng kính rầm rà, và background).
    3. Mã hóa thành vector đặc trưng 128-D.
    """
    recognizer = get_face_recognizer()
    
    # SFace and YuNet officially work best under BGR
    if rgb.shape[-1] == 3:
        img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = rgb

    # Format face tensor for SFace alignCrop
    # SFace expects a numpy array shaped (15,) containing [x, y, w, h, x_re, y_re, x_le, y_le, x_n, y_n, x_rm, y_rm, x_lm, y_lm, det_score]
    bbox = face_info["bbox"] # (x1, y1, x2, y2)
    x, y, w, h = bbox[0], bbox[1], bbox[2] - bbox[0], bbox[3] - bbox[1]
    
    landmarks = face_info["landmarks"].flatten() # 10 values
    score = face_info["det_score"]
    
    face_tensor = np.zeros(15, dtype=np.float32)
    face_tensor[0:4] = [x, y, w, h]
    face_tensor[4:14] = landmarks
    face_tensor[14] = score

    # Căn chỉnh mặt để loại bỏ yếu tố môi trường và tư thế
    aligned_face = recognizer.alignCrop(img_bgr, face_tensor)
    
    # Trích xuất 128D
    feature = recognizer.feature(aligned_face)
    return feature[0] # Return 1D array


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance according to SFace standard."""
    recognizer = get_face_recognizer()
    # match function returns Cosine Similarity by default if distance_type=0
    # similarity: higher is better, 1.0 is identical
    # distance = 1.0 - similarity: lower is better, 0.0 is identical
    # Reshape arrays for match function
    a_2d = a.reshape(1, -1)
    b_2d = b.reshape(1, -1)
    similarity = recognizer.match(a_2d, b_2d, cv2.FaceRecognizerSF_FR_COSINE)
    return float(1.0 - similarity)


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
