"""
Enrollment workflow: register a student's face embeddings via camera.
Uses InsightFace pretrained model (SCRFD + ArcFace) with GPU acceleration.
"""
import argparse
import sys
import time

import cv2
import numpy as np

from config import (
    RTSP_URL,
    REQUIRED_SAMPLES,
    SAVE_ALL_SAMPLES,
    BLUR_THRESHOLD,
    MIN_FACE_SIZE,
    OUTLIER_REMOVE_RATIO,
    SHOW_VIDEO,
    BBOX_COLOR,
    FONT_SCALE,
    FONT_THICKNESS,
)
from camera import connect_camera, read_frame, release_camera
from preprocess import preprocess_frame, get_display_frame
from detect import detect_faces, crop_face
from recognize import extract_features
from database import init_db, upsert_student, insert_face_encoding, delete_encodings_for_student


def estimate_blur(face_rgb: np.ndarray) -> float:
    """Variance of Laplacian — higher = sharper."""
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    return cv2.Laplacian(gray, cv2.CV_64F).var()


def estimate_brightness(face_rgb: np.ndarray) -> float:
    """Mean brightness of face region."""
    gray = cv2.cvtColor(face_rgb, cv2.COLOR_RGB2GRAY)
    return float(np.mean(gray))


def remove_outliers(embeddings: list[np.ndarray], ratio: float = OUTLIER_REMOVE_RATIO) -> list[np.ndarray]:
    """Remove the farthest embeddings from the centroid."""
    if len(embeddings) <= 3:
        return embeddings

    center = np.mean(embeddings, axis=0)
    dists = [np.linalg.norm(e - center) for e in embeddings]

    paired = sorted(zip(dists, embeddings), key=lambda x: x[0])
    keep_count = max(3, int(len(paired) * (1 - ratio)))
    return [e for _, e in paired[:keep_count]]


def enrollment(student_id: str, name: str, source, samples_needed: int = REQUIRED_SAMPLES):
    """
    Capture face samples from camera and store embeddings to DB.
    Uses InsightFace for detection + embedding in one pass.
    """
    conn = init_db()
    upsert_student(conn, student_id, name)
    delete_encodings_for_student(conn, student_id)

    cap = connect_camera(source)
    samples: list[np.ndarray] = []
    frame_count = 0
    skipped_reasons: dict[str, int] = {
        "no_face": 0, "multi_face": 0, "too_small": 0, "too_blur": 0, "too_dark": 0
    }

    print(f"\n[Enrollment] Registering: {name} ({student_id})")
    print(f"[Enrollment] Need {samples_needed} samples. Look at the camera from different angles.")
    print("[Enrollment] Press 'q' to cancel.\n")

    try:
        while len(samples) < samples_needed:
            frame = read_frame(cap)
            if frame is None:
                time.sleep(0.01)
                continue

            frame_count += 1
            rgb = preprocess_frame(frame)

            # YuNet: detect bounding box and landmarks
            faces = detect_faces(rgb)

            display = get_display_frame(frame)
            status_text = f"Samples: {len(samples)}/{samples_needed}"

            if len(faces) == 0:
                skipped_reasons["no_face"] += 1
                status_text += " | No face detected"
            elif len(faces) > 1:
                skipped_reasons["multi_face"] += 1
                status_text += " | Multiple faces! Only 1 allowed"
            else:
                face = faces[0]
                x1, y1, x2, y2 = face["bbox"]
                face_crop = crop_face(rgb, (x1, y1, x2, y2))
                face_h, face_w = face_crop.shape[:2]

                if face_h < MIN_FACE_SIZE or face_w < MIN_FACE_SIZE:
                    skipped_reasons["too_small"] += 1
                    status_text += " | Face too small, move closer"
                elif estimate_blur(face_crop) < BLUR_THRESHOLD:
                    skipped_reasons["too_blur"] += 1
                    status_text += " | Too blurry, hold still"
                elif estimate_brightness(face_crop) < 40:
                    skipped_reasons["too_dark"] += 1
                    status_text += " | Too dark"
                else:
                    # SFace: extract 128-D embedding from aligned crop
                    embedding = extract_features(rgb, face)
                    if embedding is not None:
                        samples.append(embedding)
                        status_text += f" | ✓ Captured! (score: {face['det_score']:.2f})"

                # Draw bbox on display
                scale_x = display.shape[1] / rgb.shape[1]
                scale_y = display.shape[0] / rgb.shape[0]
                d_x1 = int(x1 * scale_x)
                d_y1 = int(y1 * scale_y)
                d_x2 = int(x2 * scale_x)
                d_y2 = int(y2 * scale_y)
                cv2.rectangle(display, (d_x1, d_y1), (d_x2, d_y2), BBOX_COLOR, 2)

            # Draw status
            cv2.putText(
                display, status_text, (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS,
            )
            progress_pct = len(samples) / samples_needed
            bar_w = int(display.shape[1] * 0.6)
            bar_x = 10
            bar_y = display.shape[0] - 30
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + bar_w, bar_y + 20), (50, 50, 50), -1)
            cv2.rectangle(display, (bar_x, bar_y), (bar_x + int(bar_w * progress_pct), bar_y + 20), BBOX_COLOR, -1)

            if SHOW_VIDEO:
                cv2.imshow("Enrollment", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    print("[Enrollment] Cancelled by user.")
                    release_camera(cap)
                    conn.close()
                    return False

            time.sleep(0.05)

    except KeyboardInterrupt:
        print("\n[Enrollment] Interrupted.")
        release_camera(cap)
        conn.close()
        return False

    release_camera(cap)

    # Outlier removal
    print(f"[Enrollment] Collected {len(samples)} raw samples.")
    filtered = remove_outliers(samples)
    print(f"[Enrollment] After outlier removal: {len(filtered)} samples.")

    # Save to DB
    if SAVE_ALL_SAMPLES:
        for emb in filtered:
            insert_face_encoding(conn, student_id, emb, quality=1.0)
        print(f"[Enrollment] Saved {len(filtered)} embeddings (128-D) to DB.")
    else:
        mean_emb = np.mean(filtered, axis=0)
        insert_face_encoding(conn, student_id, mean_emb, quality=1.0)
        print("[Enrollment] Saved 1 mean embedding (128-D) to DB.")

    print(f"[Enrollment] Skip reasons: {skipped_reasons}")
    print(f"[Enrollment] ✓ {name} ({student_id}) registered successfully!\n")
    conn.close()
    return True


def main():
    parser = argparse.ArgumentParser(description="Enroll a student's face")
    parser.add_argument("--source", default=RTSP_URL, help="RTSP URL or camera index (default: config)")
    parser.add_argument("--student_id", required=True, help="Student ID (e.g. SE12345)")
    parser.add_argument("--name", required=True, help="Student name")
    parser.add_argument("--samples", type=int, default=REQUIRED_SAMPLES, help="Number of samples")
    args = parser.parse_args()

    success = enrollment(args.student_id, args.name, args.source, args.samples)
    sys.exit(0 if success else 1)


if __name__ == "__main__":
    main()
