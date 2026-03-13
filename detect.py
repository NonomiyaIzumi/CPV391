"""
Face detection using YOLOv26 (via Ultralytics) integrated with SAHI for High-Resolution Detection (P2 software emulation).
Exports and uses OpenVINO for maximum CPU inference speed.
"""
import os
import cv2
import numpy as np

from ultralytics import YOLO
from sahi import AutoDetectionModel
from sahi.predict import get_sliced_prediction, get_prediction

from config import (
    YOLO_FACE_MODEL, DET_SCORE_THRESHOLD, MIN_FACE_SIZE, 
    SAHI_SLICE_SIZE, SAHI_OVERLAP, USE_OPENVINO,
    PERSON_TRACK_MODEL, PERSON_TRACK_IMG_SIZE, PERSON_CONF_THRESHOLD,
    TRACKER_CONFIG, DETECT_DEVICE,
)

# ── Global State ───────────────────────────────────────────────
_sahi_model = None
_yolo_model = None
_person_tracker_model = None
_detect_frame_idx = 0


def get_face_detector():
    """
    Initialize YOLOv26 model and wrap it in SAHI AutoDetectionModel.
    Automatically exports to OpenVINO if requested and not already present.
    """
    global _sahi_model, _yolo_model
    if _sahi_model is None:
        print(f"[Detect] Loading YOLO Face model: {YOLO_FACE_MODEL}")
        
        model_path = YOLO_FACE_MODEL
        
        # Determine if we should use OpenVINO export
        if USE_OPENVINO:
            # YOLOv8 export folder format usually drops the .pt extension and adds _openvino_model
            base_name = os.path.splitext(model_path)[0]
            ov_model_dir = f"{base_name}_openvino_model"
            
            if not os.path.exists(ov_model_dir):
                print(f"[Detect] OpenVINO model not found at {ov_model_dir}. Exporting now (this takes a moment)...")
                # Load PyTorch model to export
                pt_model = YOLO(model_path)
                # Export to OpenVINO format (half=True for FP16 speedup if supported, though CPU might prefer FP32; we'll stick to default format)
                pt_model.export(format="openvino", imgsz=640)
                print("[Detect] OpenVINO export complete!")
            
            model_path = ov_model_dir
            print(f"[Detect] Using OpenVINO model: {model_path}")
        
        # Load the PyTorch or OpenVINO model via Ultralytics natively
        # Then inject it into SAHI's AutoDetectionModel to bypass its rigid path-checking logic
        pt_model = YOLO(model_path, task='detect')
        _yolo_model = pt_model
        
        from sahi.models.ultralytics import UltralyticsDetectionModel
        
        class OpenVINODetectionModel(UltralyticsDetectionModel):
            def load_model(self):
                # Bypass default loading and directly use the object we created
                self.model = pt_model
                self.set_device()
        
        _sahi_model = OpenVINODetectionModel(
            model_path=model_path,
            confidence_threshold=DET_SCORE_THRESHOLD,
            device="cpu", # Force CPU as per requirement
            category_mapping={"0": "face"}
        )
        
        print("[Detect] SAHI + YOLO loaded successfully!")
        
    return _sahi_model


def get_person_tracker_model():
    """Initialize RT-DETR/Ultralytics model for person detection+tracking."""
    global _person_tracker_model
    if _person_tracker_model is None:
        print(f"[Track] Loading person track model: {PERSON_TRACK_MODEL}")
        _person_tracker_model = YOLO(PERSON_TRACK_MODEL, task='detect')
    return _person_tracker_model


def track_people(frame_bgr: np.ndarray) -> list[dict]:
    """Track multiple people and return [{'track_id', 'bbox', 'score'}]."""
    model = get_person_tracker_model()
    results = model.track(
        source=frame_bgr,
        persist=True,
        tracker=TRACKER_CONFIG,
        classes=[0],  # person class
        conf=PERSON_CONF_THRESHOLD,
        iou=0.5,
        imgsz=PERSON_TRACK_IMG_SIZE,
        device=DETECT_DEVICE,
        verbose=False,
    )

    if not results:
        return []

    r0 = results[0]
    if r0.boxes is None or r0.boxes.xyxy is None:
        return []

    boxes = r0.boxes.xyxy.cpu().numpy() if hasattr(r0.boxes.xyxy, "cpu") else np.asarray(r0.boxes.xyxy)
    confs = r0.boxes.conf.cpu().numpy() if hasattr(r0.boxes.conf, "cpu") else np.asarray(r0.boxes.conf)
    ids = None
    if getattr(r0.boxes, "id", None) is not None:
        ids = r0.boxes.id.cpu().numpy().astype(int) if hasattr(r0.boxes.id, "cpu") else np.asarray(r0.boxes.id, dtype=int)

    tracked = []
    for i, box in enumerate(boxes):
        x1, y1, x2, y2 = map(int, box)
        if (x2 - x1) < MIN_FACE_SIZE or (y2 - y1) < MIN_FACE_SIZE:
            continue
        tid = int(ids[i]) if ids is not None and i < len(ids) else i
        tracked.append({
            "track_id": tid,
            "bbox": (x1, y1, x2, y2),
            "score": float(confs[i]) if i < len(confs) else 0.0,
        })
    return tracked


def _compute_iou(box_a, box_b):
    xA = max(box_a[0], box_b[0])
    yA = max(box_a[1], box_b[1])
    xB = min(box_a[2], box_b[2])
    yB = min(box_a[3], box_b[3])

    inter_w = max(0, xB - xA)
    inter_h = max(0, yB - yA)
    inter = inter_w * inter_h
    if inter == 0:
        return 0.0

    area_a = max(1, (box_a[2] - box_a[0]) * (box_a[3] - box_a[1]))
    area_b = max(1, (box_b[2] - box_b[0]) * (box_b[3] - box_b[1]))
    return inter / float(area_a + area_b - inter)


def _merge_detections(dets: list[dict], iou_threshold: float = 0.5) -> list[dict]:
    """Keep best-scored face among overlapping boxes."""
    if not dets:
        return []

    dets = sorted(dets, key=lambda d: d["det_score"], reverse=True)
    kept = []
    for det in dets:
        overlap = False
        for k in kept:
            if _compute_iou(det["bbox"], k["bbox"]) >= iou_threshold:
                overlap = True
                break
        if not overlap:
            kept.append(det)
    return kept


def _estimate_landmarks_from_bbox(bbox):
    """
    YOLOv8-face provides landmarks, but SAHI standard predictions might drop custom tensors.
    If landmarks are lost during SAHI standard slicing, we estimate 5 dummy landmarks 
    (eyes, nose, mouth corners) based on the bounding box to prevent SFace from crashing.
    SFace expectation: [x_re, y_re, x_le, y_le, x_n, y_n, x_rm, y_rm, x_lm, y_lm] (right eye, left eye, nose, right mouth, left mouth)
    Note: Right/Left is from the observer's perspective in many face datasets, but here we just need structurally valid points.
    """
    x1, y1, x2, y2 = bbox
    w = x2 - x1
    h = y2 - y1
    
    # Rough anatomical proportions
    re_x, re_y = x1 + w * 0.3, y1 + h * 0.4
    le_x, le_y = x1 + w * 0.7, y1 + h * 0.4
    n_x, n_y   = x1 + w * 0.5, y1 + h * 0.6
    rm_x, rm_y = x1 + w * 0.35, y1 + h * 0.8
    lm_x, lm_y = x1 + w * 0.65, y1 + h * 0.8
    
    return np.array([
        [re_x, re_y],
        [le_x, le_y],
        [n_x, n_y],
        [rm_x, rm_y],
        [lm_x, lm_y]
    ], dtype=np.float32)


def _process_sahi_output(prediction_result, w, h):
    """Tiền xử lý kết quả băm ảnh của SAHI ra format x, y và landmarks."""
    parsed = []
    
    for obj_prediction in prediction_result.object_prediction_list:
        score = obj_prediction.score.value
        bbox_raw = obj_prediction.bbox.to_xyxy() # [x1, y1, x2, y2]
        x1, y1, x2, y2 = map(int, bbox_raw)
        
        # Clip to image boundaries
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        
        fw = x2 - x1
        fh = y2 - y1
        
        if fw >= MIN_FACE_SIZE and fh >= MIN_FACE_SIZE:
            bbox = (x1, y1, x2, y2)
            
            # TODO: Extract real landmarks if SAHI preserves custom YOLO outputs in extra_data.
            # For now, generate estimated landmarks to keep pipeline running.
            landmarks = _estimate_landmarks_from_bbox(bbox)
            
            parsed.append({
                "bbox": bbox,
                "landmarks": landmarks,
                "det_score": score
            })
            
    return parsed


def _process_yolo_result(result, w, h, flipped=False):
    """Parse Ultralytics result with real keypoints when available."""
    parsed = []
    boxes = result.boxes
    if boxes is None or boxes.xyxy is None:
        return parsed

    xyxy = boxes.xyxy.cpu().numpy() if hasattr(boxes.xyxy, "cpu") else np.asarray(boxes.xyxy)
    confs = boxes.conf.cpu().numpy() if hasattr(boxes.conf, "cpu") else np.asarray(boxes.conf)

    keypoints_xy = None
    if hasattr(result, "keypoints") and result.keypoints is not None and result.keypoints.xy is not None:
        keypoints_xy = result.keypoints.xy.cpu().numpy() if hasattr(result.keypoints.xy, "cpu") else np.asarray(result.keypoints.xy)

    for i, box in enumerate(xyxy):
        x1, y1, x2, y2 = map(int, box)
        x1 = max(0, x1)
        y1 = max(0, y1)
        x2 = min(w, x2)
        y2 = min(h, y2)
        fw = x2 - x1
        fh = y2 - y1
        if fw < MIN_FACE_SIZE or fh < MIN_FACE_SIZE:
            continue

        bbox = (x1, y1, x2, y2)
        landmarks = None
        if keypoints_xy is not None and i < len(keypoints_xy):
            lmk = np.asarray(keypoints_xy[i], dtype=np.float32)
            if flipped:
                lmk[:, 0] = (w - 1) - lmk[:, 0]
            landmarks = lmk

        if landmarks is None or landmarks.shape != (5, 2):
            landmarks = _estimate_landmarks_from_bbox(bbox)

        parsed.append({
            "bbox": bbox,
            "landmarks": landmarks,
            "det_score": float(confs[i]) if i < len(confs) else float(DET_SCORE_THRESHOLD),
        })

    return parsed


def _detect_with_raw_yolo(rgb: np.ndarray, conf: float, with_flip: bool = True) -> list[dict]:
    """Direct YOLO inference keeps keypoints, which improves profile-face robustness for SFace."""
    h, w = rgb.shape[:2]
    if _yolo_model is None:
        return []

    pred = _yolo_model(
        rgb,
        conf=conf,
        iou=0.5,
        imgsz=640,
        device="cpu",
        verbose=False,
    )
    raw_dets = _process_yolo_result(pred[0], w, h, flipped=False) if pred else []

    if with_flip:
        flipped = cv2.flip(rgb, 1)
        pred_flip = _yolo_model(
            flipped,
            conf=conf,
            iou=0.5,
            imgsz=640,
            device="cpu",
            verbose=False,
        )
        flip_dets = _process_yolo_result(pred_flip[0], w, h, flipped=True) if pred_flip else []
        # Remap flipped bboxes to original image coordinates.
        for det in flip_dets:
            x1, y1, x2, y2 = det["bbox"]
            det["bbox"] = (w - x2, y1, w - x1, y2)
        raw_dets.extend(flip_dets)

    return _merge_detections(raw_dets, iou_threshold=0.45)


def detect_faces(rgb: np.ndarray, use_sahi: bool = True) -> list:
    """
    Detect faces using SAHI + YOLOv26.
    Slices the image to find tiny faces at the back of the classroom.
    """
    global _detect_frame_idx
    _detect_frame_idx += 1

    h, w = rgb.shape[:2]
    detection_model = get_face_detector()

    # Fast path: direct YOLO with real keypoints for better recognition quality.
    conf_low = max(0.22, DET_SCORE_THRESHOLD - 0.05)
    direct = _detect_with_raw_yolo(rgb, conf=conf_low, with_flip=False)

    # In fast mode (enrollment), return immediately for maximum FPS.
    if not use_sahi:
        return direct

    sahi_faces = []
    need_sahi = not direct

    if need_sahi:
        # SAHI runs only when direct YOLO misses, so FPS stays stable in normal cases.
        result = get_sliced_prediction(
            rgb,
            detection_model,
            slice_height=SAHI_SLICE_SIZE,
            slice_width=SAHI_SLICE_SIZE,
            overlap_height_ratio=SAHI_OVERLAP,
            overlap_width_ratio=SAHI_OVERLAP,
            perform_standard_pred=False,
            postprocess_type="NMM",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=0.45,
        )
        sahi_faces = _process_sahi_output(result, w, h)

    merged = _merge_detections(direct + sahi_faces, iou_threshold=0.45)

    # Final safety fallback for very hard frames.
    if not merged:
        heavy = get_sliced_prediction(
            rgb,
            detection_model,
            slice_height=SAHI_SLICE_SIZE,
            slice_width=SAHI_SLICE_SIZE,
            overlap_height_ratio=SAHI_OVERLAP,
            overlap_width_ratio=SAHI_OVERLAP,
            perform_standard_pred=True,
            postprocess_type="NMM",
            postprocess_match_metric="IOU",
            postprocess_match_threshold=0.45,
        )
        merged = _process_sahi_output(heavy, w, h)
        if not merged:
            merged = _detect_with_raw_yolo(rgb, conf=0.15, with_flip=True)

    return merged


def crop_face(rgb: np.ndarray, bbox: tuple[int, int, int, int]) -> np.ndarray:
    """Crop a face region from the image."""
    x1, y1, x2, y2 = bbox
    h, w = rgb.shape[:2]
    x1 = max(0, x1)
    y1 = max(0, y1)
    x2 = min(w, x2)
    y2 = min(h, y2)
    return rgb[y1:y2, x1:x2]
