"""
Face detection using OpenCV's built-in YuNet model (extremely fast on CPU).
Tối ưu hóa siêu tốc bằng Stateful Rotation Tracker: Ghi nhớ góc camera để không quét lại 4 góc.
"""
import cv2
import numpy as np

from config import YUNET_MODEL, DET_SCORE_THRESHOLD, MIN_FACE_SIZE

# ── Global State ───────────────────────────────────────────────
_detector = None
_last_successful_angle = 0  # Ghi nhớ góc xoay cam hiện hành để tiết kiệm 75% CPU mỗi khung hình


def get_face_detector(input_size=(320, 320)):
    """Initialize YuNet model (singleton)."""
    global _detector
    if _detector is None:
        print(f"[Detect] Loading YuNet model: {YUNET_MODEL}")
        _detector = cv2.FaceDetectorYN.create(
            model=YUNET_MODEL,
            config="",
            input_size=input_size,
            score_threshold=DET_SCORE_THRESHOLD,
            nms_threshold=0.3, # Bắt mờ ảo ở xa thì để NMS vừa phải chống trùng
            top_k=5000,
        )
        print("[Detect] YuNet loaded successfully!")
    else:
        _detector.setInputSize(input_size)
    return _detector


def transform_points(points: np.ndarray, M: np.ndarray) -> np.ndarray:
    """Áp dụng ma trận biến đổi Affine cho danh sách các điểm (x, y)."""
    ones = np.ones((points.shape[0], 1))
    pts_homogeneous = np.hstack([points, ones])
    transformed = M.dot(pts_homogeneous.T).T
    return transformed


def _detect_single_angle(detector, img_bgr, center, angle, w, h):
    """Thực thi YuNet trên 1 góc duy nhất."""
    if angle == 0:
        return detector.detect(img_bgr), None

    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M_inv = cv2.getRotationMatrix2D(center, -angle, 1.0)
    
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])
    new_w = int((h * sin) + (w * cos))
    new_h = int((h * cos) + (w * sin))
    M[0, 2] += (new_w / 2) - center[0]
    M[1, 2] += (new_h / 2) - center[1]
    M_inv[0, 2] += center[0] - (new_w / 2)
    M_inv[1, 2] += center[1] - (new_h / 2)
    
    rotated = cv2.warpAffine(img_bgr, M, (new_w, new_h))
    detector.setInputSize((new_w, new_h))
    status, faces = detector.detect(rotated)
    detector.setInputSize((w, h))
    
    return (status, faces), M_inv

def _detect_angles(detector, img_bgr: np.ndarray) -> tuple[list, float]:
    """
    Stateful Rotation Tracker (Theo dõi trạng thái góc quay của AI):
    Thay vì quét luôn 4 góc. Hệ thống sẽ:
    1. Kiểm tra góc thành công ở Frame ngay trước đó. Nếu vẫn CÒN MẶT, thì chốt luôn, KHÔNG QUÉT NỮA. Tốc độ nhảy vọt lên 30 FPS.
    2. Nếu mất hình, khi đó mới rảnh rang đi quét chậm lại các góc nghiêng để tìm xem camera có bị ai đó vặn/úp ngược hay không.
    """
    global _last_successful_angle
    h, w = img_bgr.shape[:2]
    center = (w // 2, h // 2)

    # BƯỚC 1: Quét lại đúng cái góc xịn xò của Frame trước
    (status, faces), M_inv = _detect_single_angle(detector, img_bgr, center, _last_successful_angle, w, h)
    
    if faces is not None and len(faces) > 0:
        # Nếu AI còn tự tin cao (> 0.6) thì cứ chốt luôn góc này sống lâu dài
        max_conf = np.max(faces[:, 14])
        if max_conf > 0.6:
            return faces, M_inv

    # BƯỚC 2: Rơi vào đây tức là Camera hỏng hình, úp ngược hoặc mất dấu người cũ
    best_faces = None
    best_M_inv = None
    best_total_score = -1
    best_angle = _last_successful_angle
    
    angles = [0, 90, 180, 270]
    angles.remove(_last_successful_angle) # Đỡ phải test lại góc cũ tốn thời gian
    
    # Nhanh chóng duyệt thử xem 3 góc còn lại có tìm ra cái gì không (thời gian chậm 1 chút xíu ở Frame này)
    for angle in angles:
        (status, faces), current_M_inv = _detect_single_angle(detector, img_bgr, center, angle, w, h)
        if faces is not None and len(faces) > 0:
            total_score = np.sum(faces[:, 14])
            if total_score > best_total_score:
                best_total_score = total_score
                best_faces = faces
                best_M_inv = current_M_inv
                best_angle = angle

    # BƯỚC 3: Nếu mà dò ra được 1 góc nào đó có điểm tự tin ngon, HỌC LUÔN góc đó làm trạng thái Stateful cho các Frame sau
    if best_faces is not None and best_total_score > 0:
        _last_successful_angle = best_angle
        # Ghi đè file log ngầm vào terminal cho bạn theo dõi
        # print(f"📍 TRẠNG THÁI CAMERA THAY ĐỔI: Khóa cứng ở góc lật {best_angle} độ.")
        return best_faces, best_M_inv
        
    # Không tìm thấy cái gì cả (Không có người)
    return None, None


def _compute_iou(boxA, boxB):
    """Tính Intersection over Union (IoU) giữa 2 Bounding Box."""
    xA = max(boxA[0], boxB[0])
    yA = max(boxA[1], boxB[1])
    xB = min(boxA[2], boxB[2])
    yB = min(boxA[3], boxB[3])

    interArea = max(0, xB - xA) * max(0, yB - yA)
    boxAArea = (boxA[2] - boxA[0]) * (boxA[3] - boxA[1])
    boxBArea = (boxB[2] - boxB[0]) * (boxB[3] - boxB[1])
    
    if float(boxAArea + boxBArea - interArea) == 0:
       return 0.0
    iou = interArea / float(boxAArea + boxBArea - interArea)
    return iou


def _nms_ensemble(faces_list, iou_threshold=0.4):
    """
    Ensemble NMS (Thuật toán Non-Maximum Suppression).
    Loại bỏ các Bounding Box kém chất lượng trùng lặp. Củng cố Box có độ tự tin cao do TTA.
    """
    if not faces_list:
        return []

    # Sắp xếp các khuôn mặt theo độ tin cậy tự tin (score) giảm dần
    faces_list = sorted(faces_list, key=lambda x: x["det_score"], reverse=True)
    keep = []

    for f in faces_list:
        f_box = f["bbox"]
        overlap = False
        for k in keep:
            k_box = k["bbox"]
            if _compute_iou(f_box, k_box) > iou_threshold:
                overlap = True
                break
        if not overlap:
            keep.append(f)

    return keep


def _process_faces_output(faces, M_inv, w, h):
    """Tiền xử lý output thô của YuNet ra format x, y."""
    parsed = []
    if faces is None: return parsed
    
    for face in faces:
        bbox_raw = face[0:4] # x, y, fw, fh
        score = face[14]
        landmarks_raw = face[4:14].reshape(5, 2)

        if M_inv is not None:
            # Xoay ngược toạ độ Landmarks
            landmarks = transform_points(landmarks_raw, M_inv)
            x, y, fw, fh = bbox_raw
            corners = np.array([
                [x, y], [x + fw, y], [x + fw, y + fh], [x, y + fh]
            ])
            corners_rot = transform_points(corners, M_inv)
            # Box sau khi quay
            min_x, min_y = np.min(corners_rot[:, 0]), np.min(corners_rot[:, 1])
            max_x, max_y = np.max(corners_rot[:, 0]), np.max(corners_rot[:, 1])
            bbox = (int(min_x), int(min_y), int(max_x), int(max_y))
            fw, fh = int(max_x - min_x), int(max_y - min_y)
        else:
            landmarks = landmarks_raw
            x, y, fw, fh = map(int, bbox_raw)
            bbox = (max(0, x), max(0, y), min(w, x + fw), min(h, y + fh))

        if fw >= MIN_FACE_SIZE and fh >= MIN_FACE_SIZE:
            parsed.append({
                "bbox": bbox,
                "landmarks": landmarks,
                "det_score": score
            })
    return parsed

def detect_faces(rgb: np.ndarray) -> list:
    """
    Detect faces usando Stateful Rotation Tracking YuNet.
    Vẫn đảm bảo quét đa góc nhưng FPS cao kịch trần 30-60.
    """
    h, w = rgb.shape[:2]
    # Lấy thể hình cho YuNet là 640 là cực kỳ tiêu chuẩn để cân đối MƯỢT + XA TÍT TẮP
    detector = get_face_detector((w, h))

    if rgb.shape[-1] == 3:
        img_bgr = cv2.cvtColor(rgb, cv2.COLOR_RGB2BGR)
    else:
        img_bgr = rgb

    # Luồng xử lý MỚI: Chỉ vứt cho bộ Tracker.
    faces, M_inv = _detect_angles(detector, img_bgr)
    results = _process_faces_output(faces, M_inv, w, h)
    
    return results


def crop_face(rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a face region from the image."""
    x1, y1, x2, y2 = bbox
    h, w = rgb.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return rgb[y1:y2, x1:x2]
