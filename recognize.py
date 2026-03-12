"""
Face recognition using InsightFace ArcFace pretrained model.
Embeddings are 512-D vectors, matching uses cosine similarity.
"""
import numpy as np

from config import TOLERANCE


def cosine_distance(a: np.ndarray, b: np.ndarray) -> float:
    """Compute cosine distance between two vectors (0 = identical, 2 = opposite)."""
    dot = np.dot(a, b)
    norm_a = np.linalg.norm(a)
    norm_b = np.linalg.norm(b)
    if norm_a == 0 or norm_b == 0:
        return 2.0
    similarity = dot / (norm_a * norm_b)
    return float(1.0 - similarity)


def distance_to_confidence(dist: float, tolerance: float = TOLERANCE) -> float:
    """Map cosine distance to 0-1 confidence score."""
    if dist >= tolerance * 2:
        return 0.0
    return max(0.0, 1.0 - dist / (tolerance * 2))


def match_embedding(
    query_vec: np.ndarray,
    known_db: dict[str, list[np.ndarray]],
    tolerance: float = TOLERANCE,
) -> tuple[str | None, float, float]:
    """
    Match a 512-D query embedding against the known database using cosine distance.

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
    Match using mean embedding per student (faster for large DBs).
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
