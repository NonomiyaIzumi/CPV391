"""
Flask web UI for Smart Classroom Attendance System.
"""
import os
import threading
import time
from datetime import datetime

import cv2
cv2.setNumThreads(1) # Ngăn OpenCV giành CPU với OpenVINO
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_file

from config import (
    RTSP_URL, REQUIRED_SAMPLES, LATE_MINUTES,
    FRAME_SKIP, ABSENCE_THRESHOLD_SEC, ATTENDANCE_NOFACE_GRACE_INFERENCES,
    BBOX_SMOOTH_ALPHA, TRACK_IOU_THRESHOLD, TRACK_MAX_MISSES,
    PIPELINE_MODE, TRACK_RECOG_INTERVAL, TRACK_KEEP_ID_MISSES,
    BBOX_COLOR, UNKNOWN_BBOX_COLOR, FONT_SCALE, FONT_THICKNESS,
    MIN_FACE_SIZE, BLUR_THRESHOLD, SAVE_ALL_SAMPLES, OUTLIER_REMOVE_RATIO,
)
from database import (
    init_db, get_all_students, get_all_sessions,
    get_attendance_for_session, get_session, get_student,
    load_all_encodings, upsert_student, insert_face_encoding,
    delete_encodings_for_student, create_session, finalize_session,
    insert_attendance_if_new, update_last_seen, set_checkout_time,
    open_movement_interval, close_last_movement_interval,
    add_student_to_course, remove_student_from_course, get_student_courses,
    get_students_by_course, delete_student, load_encodings_by_course,
)
from camera import connect_camera, read_frame, release_camera
from preprocess import preprocess_frame, get_display_frame
from detect import detect_faces, crop_face, track_people
from recognize import extract_features, match_embedding
from enrollment import estimate_blur, estimate_brightness, remove_outliers
from report import export_report

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────
camera_lock = threading.Lock()
current_frame = None        # latest display frame (BGR)
frame_lock = threading.Lock()

ENROLL_POSE_SEQUENCE = ["front", "left", "right", "up", "down"]
ENROLL_POSE_LABELS = {
    "front": "Chinh dien",
    "left": "Quay trai",
    "right": "Quay phai",
    "up": "Nhin len",
    "down": "Nhin xuong",
}

enrollment_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "message": "",
    "done": False,
    "success": False,
    "capture_requested": 0,
    "capture_processed": 0,
    "pose_order": ENROLL_POSE_SEQUENCE,
    "current_pose_index": 0,
    "current_pose": "front",
    "current_pose_label": ENROLL_POSE_LABELS["front"],
    "captured_poses": [],
}

attendance_status = {
    "running": False,
    "session_id": None,
    "class_name": "",
    "start_time": None,
    "students_detected": 0,
    "fps": 0.0,
    "recognized": [],   # latest recognized list
    "message": "",
}


# ── Pages ─────────────────────────────────────────────────────
@app.route("/")
def dashboard():
    conn = init_db()
    students = get_all_students(conn)
    sessions = get_all_sessions(conn)
    # Count face encodings per student
    for s in students:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM face_encodings WHERE student_id = ?",
            (s["student_id"],)
        ).fetchone()
        s["encoding_count"] = row["cnt"] if row else 0
        s["courses"] = get_student_courses(conn, s["student_id"])

    # Recent attendance
    recent_records = []
    if sessions:
        latest = sessions[0]
        recent_records = get_attendance_for_session(conn, latest["session_id"])

    # Stats
    total_students = len(students)
    total_sessions = len(sessions)
    today = datetime.now().strftime("%Y-%m-%d")
    today_sessions = [s for s in sessions if s.get("start_time", "").startswith(today)]

    conn.close()
    return render_template("dashboard.html",
        students=students,
        sessions=sessions,
        recent_records=recent_records,
        total_students=total_students,
        total_sessions=total_sessions,
        today_sessions=len(today_sessions),
    )


@app.route("/enrollment")
def enrollment_page():
    conn = init_db()
    students = get_all_students(conn)
    for s in students:
        row = conn.execute(
            "SELECT COUNT(*) as cnt FROM face_encodings WHERE student_id = ?",
            (s["student_id"],)
        ).fetchone()
        s["encoding_count"] = row["cnt"] if row else 0
        s["courses"] = get_student_courses(conn, s["student_id"])
    conn.close()
    return render_template("enrollment.html", students=students)


@app.route("/attendance")
def attendance_page():
    conn = init_db()
    students = get_all_students(conn)
    conn.close()
    return render_template("attendance.html",
        students=students,
        status=attendance_status,
    )


@app.route("/reports")
def reports_page():
    conn = init_db()
    sessions = get_all_sessions(conn)
    conn.close()
    return render_template("reports.html", sessions=sessions)


@app.route("/reports/<session_id>")
def report_detail(session_id):
    conn = init_db()
    session = get_session(conn, session_id)
    records = get_attendance_for_session(conn, session_id)
    conn.close()
    return render_template("report_detail.html",
        session=session, records=records)


# ── API ───────────────────────────────────────────────────────
@app.route("/api/students")
def api_students():
    conn = init_db()
    students = get_all_students(conn)
    for s in students:
        s["courses"] = get_student_courses(conn, s["student_id"])
    conn.close()
    return jsonify(students)


@app.route("/api/students/<student_id>", methods=["DELETE"])
def api_delete_student(student_id):
    conn = init_db()
    delete_student(conn, student_id)
    conn.close()
    return jsonify({"status": "deleted"})


@app.route("/api/students/<student_id>/courses", methods=["POST"])
def api_add_student_course(student_id):
    data = request.json
    course_name = data.get("course_name", "").strip()
    if not course_name:
        return jsonify({"error": "course_name required"}), 400
    conn = init_db()
    add_student_to_course(conn, student_id, course_name)
    conn.close()
    return jsonify({"status": "added", "course_name": course_name})


@app.route("/api/students/<student_id>/courses/<course_name>", methods=["DELETE"])
def api_remove_student_course(student_id, course_name):
    conn = init_db()
    remove_student_from_course(conn, student_id, course_name)
    conn.close()
    return jsonify({"status": "removed"})


@app.route("/api/enrollment/start", methods=["POST"])
def api_enrollment_start():
    global enrollment_status
    if enrollment_status["running"]:
        return jsonify({"error": "Enrollment already running"}), 400

    data = request.json
    student_id = data.get("student_id", "").strip()
    name = data.get("name", "").strip()
    samples = len(ENROLL_POSE_SEQUENCE)
    source = data.get("source", RTSP_URL)

    if not student_id or not name:
        return jsonify({"error": "student_id and name are required"}), 400

    enrollment_status = {
        "running": True, "progress": 0, "total": samples,
        "message": "Starting...", "done": False, "success": False,
        "capture_requested": 0, "capture_processed": 0,
        "pose_order": ENROLL_POSE_SEQUENCE,
        "current_pose_index": 0,
        "current_pose": ENROLL_POSE_SEQUENCE[0],
        "current_pose_label": ENROLL_POSE_LABELS[ENROLL_POSE_SEQUENCE[0]],
        "captured_poses": [],
    }

    thread = threading.Thread(
        target=_enrollment_worker,
        args=(student_id, name, source, samples),
        daemon=True,
    )
    thread.start()
    return jsonify({"status": "started"})


@app.route("/api/enrollment/capture", methods=["POST"])
def api_enrollment_capture():
    global enrollment_status
    if not enrollment_status.get("running"):
        return jsonify({"error": "Enrollment is not running"}), 400

    enrollment_status["capture_requested"] = enrollment_status.get("capture_requested", 0) + 1
    return jsonify({
        "status": "queued",
        "requested": enrollment_status["capture_requested"],
        "processed": enrollment_status.get("capture_processed", 0),
    })


@app.route("/api/enrollment/status")
def api_enrollment_status():
    return jsonify(enrollment_status)


@app.route("/api/attendance/start", methods=["POST"])
def api_attendance_start():
    global attendance_status
    if attendance_status["running"]:
        return jsonify({"error": "Attendance session already running"}), 400

    data = request.json
    class_name = data.get("class_name", "CPV391").strip()
    late_minutes = int(data.get("late_minutes", LATE_MINUTES))
    source = data.get("source", RTSP_URL)

    attendance_status = {
        "running": True,
        "session_id": None,
        "class_name": class_name,
        "start_time": datetime.now().isoformat(),
        "students_detected": 0,
        "fps": 0.0,
        "recognized": [],
        "message": "Starting...",
    }

    thread = threading.Thread(
        target=_attendance_worker,
        args=(source, class_name, late_minutes),
        daemon=True,
    )
    thread.start()
    return jsonify({"status": "started"})


@app.route("/api/attendance/stop", methods=["POST"])
def api_attendance_stop():
    global attendance_status
    attendance_status["running"] = False
    return jsonify({"status": "stopping"})


@app.route("/api/attendance/status")
def api_attendance_status():
    return jsonify(attendance_status)


@app.route("/api/sessions")
def api_sessions():
    conn = init_db()
    sessions = get_all_sessions(conn)
    conn.close()
    return jsonify(sessions)


@app.route("/api/sessions/<session_id>/records")
def api_session_records(session_id):
    conn = init_db()
    records = get_attendance_for_session(conn, session_id)
    conn.close()
    return jsonify(records)


@app.route("/api/sessions/<session_id>/export")
def api_export_csv(session_id):
    conn = init_db()
    csv_path = export_report(conn, session_id)
    conn.close()
    if csv_path and os.path.exists(csv_path):
        return send_file(csv_path, as_attachment=True)
    return jsonify({"error": "Export failed"}), 500


# ── MJPEG video feed ──────────────────────────────────────────
def _generate_mjpeg():
    while True:
        with frame_lock:
            frame = current_frame
        if frame is not None:
            _, buf = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 70])
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
        else:
            # Send a blank frame
            blank = np.zeros((480, 640, 3), dtype=np.uint8)
            cv2.putText(blank, "No camera feed", (160, 240),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (100, 100, 100), 2)
            _, buf = cv2.imencode(".jpg", blank)
            yield (
                b"--frame\r\n"
                b"Content-Type: image/jpeg\r\n\r\n" + buf.tobytes() + b"\r\n"
            )
        # Giảm cực đại sleep để mjpeg stream nhanh nhất có thể (chạy ~60 FPS)
        time.sleep(0.03)


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ── Workers ───────────────────────────────────────────────────
def _enrollment_worker(student_id, name, source, samples_needed):
    global enrollment_status, current_frame

    cap = None
    conn = None
    try:
        conn = init_db()
        upsert_student(conn, student_id, name)
        delete_encodings_for_student(conn, student_id)

        cap = connect_camera(source)
        samples = []
        pose_order = enrollment_status.get("pose_order", ENROLL_POSE_SEQUENCE)
        enrollment_status["message"] = f"Camera connected. Pose 1/{len(pose_order)}: {ENROLL_POSE_LABELS[pose_order[0]]}."

        def _estimate_pose(face_info):
            lmk = np.asarray(face_info.get("landmarks"), dtype=np.float32)
            if lmk.shape != (5, 2):
                return "unknown"

            right_eye, left_eye, nose, right_mouth, left_mouth = lmk
            eye_center = (right_eye + left_eye) * 0.5
            mouth_center = (right_mouth + left_mouth) * 0.5
            x1p, y1p, x2p, y2p = face_info["bbox"]
            fw = max(1.0, float(x2p - x1p))
            eyemouth = max(1.0, float(mouth_center[1] - eye_center[1]))

            yaw_ratio = float((nose[0] - eye_center[0]) / fw)
            pitch_ratio = float((nose[1] - eye_center[1]) / eyemouth)

            if pitch_ratio < 0.38:
                return "up"
            if pitch_ratio > 0.76:
                return "down"
            if yaw_ratio < -0.11:
                return "left"
            if yaw_ratio > 0.11:
                return "right"
            return "front"

        frame_idx = 0
        while len(samples) < samples_needed and enrollment_status["running"]:
            frame = read_frame(cap)
            if frame is None:
                time.sleep(0.01)
                continue
                
            frame_idx += 1
            display = get_display_frame(frame)
                
            rgb = preprocess_frame(frame)
            # Tắt tính năng băm nhỏ ảnh (SAHI) khi Enrollment vì sinh viên ngồi ngay sát màn hình
            # Cải thiện tốc độ nhận diện nhanh gấp 6 lần ở mục này
            faces = detect_faces(rgb, use_sahi=False)

            status_text = f"Samples: {len(samples)}/{samples_needed}"

            capture_needed = enrollment_status.get("capture_requested", 0) > enrollment_status.get("capture_processed", 0)

            if len(faces) == 0:
                status_text += " | No face detected"
            else:
                img_h, img_w = rgb.shape[:2]

                def _is_full_face(face_info):
                    x1f, y1f, x2f, y2f = face_info["bbox"]
                    fwf = x2f - x1f
                    fhf = y2f - y1f
                    if fwf <= 0 or fhf <= 0:
                        return False

                    # Reject partial faces touching frame borders.
                    margin = max(8, int(min(img_h, img_w) * 0.03))
                    if x1f <= margin or y1f <= margin or x2f >= (img_w - margin) or y2f >= (img_h - margin):
                        return False

                    lmk = np.asarray(face_info.get("landmarks"))
                    if lmk.shape != (5, 2):
                        return False

                    # Landmarks should stay safely inside the box.
                    if (np.min(lmk[:, 0]) < x1f + fwf * 0.08 or
                        np.max(lmk[:, 0]) > x2f - fwf * 0.08 or
                        np.min(lmk[:, 1]) < y1f + fhf * 0.08 or
                        np.max(lmk[:, 1]) > y2f - fhf * 0.08):
                        return False

                    return True

                valid_faces = [f for f in faces if _is_full_face(f)]
                if not valid_faces:
                    status_text += " | Need full face in frame"
                    if len(faces) > 1:
                        status_text += " (others ignored)"
                    if capture_needed and enrollment_status.get("capture_processed", 0) < enrollment_status.get("capture_requested", 0):
                        enrollment_status["capture_processed"] = enrollment_status.get("capture_processed", 0) + 1
                        status_text += " | Capture failed"
                    with frame_lock:
                        current_frame = display.copy()
                    enrollment_status["progress"] = len(samples)
                    enrollment_status["message"] = status_text
                    continue

                # Enrollment priority: pick the largest face (closest to camera), ignore others.
                face = max(
                    valid_faces,
                    key=lambda f: max(1, (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
                )
                if len(valid_faces) > 1:
                    status_text += f" | {len(valid_faces)} full faces, using largest"
                elif len(faces) > 1:
                    status_text += f" | {len(faces)} faces detected, only 1 full face accepted"

                x1, y1, x2, y2 = face["bbox"]
                face_crop = crop_face(rgb, (x1, y1, x2, y2))
                face_h, face_w = face_crop.shape[:2]
                detected_pose = _estimate_pose(face)

                pose_idx = enrollment_status.get("current_pose_index", 0)
                expected_pose = pose_order[min(pose_idx, len(pose_order) - 1)]
                expected_pose_label = ENROLL_POSE_LABELS.get(expected_pose, expected_pose)
                detected_pose_label = ENROLL_POSE_LABELS.get(detected_pose, detected_pose)

                scale_x = display.shape[1] / rgb.shape[1]
                scale_y = display.shape[0] / rgb.shape[0]
                d_x1, d_y1 = int(x1 * scale_x), int(y1 * scale_y)
                d_x2, d_y2 = int(x2 * scale_x), int(y2 * scale_y)
                cv2.rectangle(display, (d_x1, d_y1), (d_x2, d_y2), BBOX_COLOR, 2)

                if not capture_needed:
                    status_text += f" | Pose: {detected_pose_label} | Need: {expected_pose_label}"
                else:
                    # Keep pose as guidance, but do not hard-block capture by classifier.
                    if detected_pose != expected_pose:
                        status_text += f" | Pose hint: {detected_pose_label}, saving for {expected_pose_label}"

                    if face_h < MIN_FACE_SIZE or face_w < MIN_FACE_SIZE:
                        status_text += " | Face too small"
                    elif estimate_blur(face_crop) < BLUR_THRESHOLD:
                        status_text += " | Too blurry"
                    elif estimate_brightness(face_crop) < 40:
                        status_text += " | Too dark"
                    else:
                        embedding = extract_features(rgb, face)
                        if embedding is not None:
                            samples.append(embedding)
                            enrollment_status["captured_poses"].append(expected_pose)
                            next_idx = min(len(samples), len(pose_order) - 1)
                            enrollment_status["current_pose_index"] = next_idx
                            enrollment_status["current_pose"] = pose_order[next_idx]
                            enrollment_status["current_pose_label"] = ENROLL_POSE_LABELS[pose_order[next_idx]]
                            status_text += f" | Captured {expected_pose_label}"
                        else:
                            status_text += " | Cannot extract features"

                    enrollment_status["capture_processed"] = enrollment_status.get("capture_processed", 0) + 1

            # If user pressed capture but no valid face at this frame, consume one capture request.
            if capture_needed and enrollment_status.get("capture_processed", 0) < enrollment_status.get("capture_requested", 0):
                enrollment_status["capture_processed"] = enrollment_status.get("capture_processed", 0) + 1
                status_text += " | Capture failed - keep full face in frame"

            # Draw info on display
            cv2.putText(display, status_text, (10, 30),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            # Progress bar
            pct = len(samples) / samples_needed
            bar_w = int(display.shape[1] * 0.6)
            bar_y = display.shape[0] - 30
            cv2.rectangle(display, (10, bar_y), (10 + bar_w, bar_y + 20), (50, 50, 50), -1)
            cv2.rectangle(display, (10, bar_y), (10 + int(bar_w * pct), bar_y + 20), BBOX_COLOR, -1)

            with frame_lock:
                current_frame = display.copy()

            enrollment_status["progress"] = len(samples)
            enrollment_status["message"] = status_text
            # Bỏ sleep 0.05 ở đây để webcam quét liên tục, mượt mà

        if len(samples) >= samples_needed:
            filtered = remove_outliers(samples)
            if SAVE_ALL_SAMPLES:
                for emb in filtered:
                    insert_face_encoding(conn, student_id, emb, quality=1.0)
            else:
                mean_emb = np.mean(filtered, axis=0)
                insert_face_encoding(conn, student_id, mean_emb, quality=1.0)

            enrollment_status["success"] = True
            enrollment_status["message"] = f"{name} registered with 5 pose vectors."
        else:
            enrollment_status["message"] = "Enrollment cancelled."

    except Exception as e:
        import traceback
        traceback.print_exc()
        enrollment_status["message"] = f"Error: {str(e)}"
    finally:
        if cap is not None:
            release_camera(cap)
        if conn is not None:
            conn.close()

    enrollment_status["running"] = False
    enrollment_status["done"] = True

    # Clear frame after a delay
    time.sleep(2)
    with frame_lock:
        current_frame = None


def _attendance_worker(source, class_name, late_minutes):
    global attendance_status, current_frame

    cap = None
    conn = None
    try:
        conn = init_db()
        known_db = load_encodings_by_course(conn, class_name)

        if not known_db:
            attendance_status["message"] = f"No students enrolled in class {class_name}!"
            attendance_status["running"] = False
            conn.close()
            return

        student_names = {}
        for sid in known_db:
            info = get_student(conn, sid)
            student_names[sid] = info["name"] if info else sid

        start_time = datetime.now()
        session_id = create_session(conn, class_name, start_time, late_minutes)
        attendance_status["session_id"] = session_id
        attendance_status["start_time"] = start_time.isoformat()
        attendance_status["message"] = f"Session started: {class_name}"

        state = {}
        cap = connect_camera(source)
        frame_idx = 0
        fps_counter = 0
        fps_timer = time.time()
        no_face_inference_streak = 0
        track_map = {}
        track_identity = {}
        next_track_id = 1

        def _bbox_iou(a, b):
            ax1, ay1, ax2, ay2 = a
            bx1, by1, bx2, by2 = b
            ix1 = max(ax1, bx1)
            iy1 = max(ay1, by1)
            ix2 = min(ax2, bx2)
            iy2 = min(ay2, by2)
            iw = max(0, ix2 - ix1)
            ih = max(0, iy2 - iy1)
            inter = iw * ih
            if inter <= 0:
                return 0.0
            area_a = max(1, (ax2 - ax1) * (ay2 - ay1))
            area_b = max(1, (bx2 - bx1) * (by2 - by1))
            return inter / float(area_a + area_b - inter)

        def _smooth_bbox(old_box, new_box, alpha=BBOX_SMOOTH_ALPHA):
            return (
                int(alpha * new_box[0] + (1 - alpha) * old_box[0]),
                int(alpha * new_box[1] + (1 - alpha) * old_box[1]),
                int(alpha * new_box[2] + (1 - alpha) * old_box[2]),
                int(alpha * new_box[3] + (1 - alpha) * old_box[3]),
            )

        cached_drawings = []

        while attendance_status["running"]:
            recognized_list = []
            frame = read_frame(cap)
            if frame is None:
                continue

            frame_idx += 1
            display = get_display_frame(frame)
            faces = []
            do_inference = True
            
            if FRAME_SKIP > 1 and frame_idx % FRAME_SKIP != 0:
                do_inference = False
                # Trạng thái Frame Skip (Bỏ qua suy luận AI để giảm tải CPU)
                # Nhưng vẫn TỐN CÔNG vẽ lại Bounding Box cũ để stream không bị "giật giật"
                for bbox, color, label, label_y in cached_drawings:
                    cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(display, label, (bbox[0], label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICKNESS)
            else:
                # Trạng thái suy luận AI (Chạy YOLO + SFace)
                rgb = preprocess_frame(frame)
                if PIPELINE_MODE == "balanced":
                    # Balanced mode uses person tracker first; skip redundant full-frame face detection.
                    faces = [{"bbox": (0, 0, 1, 1)}]
                else:
                    use_sahi_now = no_face_inference_streak >= 2
                    faces = detect_faces(rgb, use_sahi=use_sahi_now)
                    scale_x = display.shape[1] / rgb.shape[1]
                    scale_y = display.shape[0] / rgb.shape[0]
                
                cached_drawings.clear()

            if do_inference and PIPELINE_MODE != "balanced" and len(faces) == 0:
                no_face_inference_streak += 1
                if no_face_inference_streak >= ATTENDANCE_NOFACE_GRACE_INFERENCES:
                    handle_absent_web(conn, session_id, state)
            elif do_inference:
                det_items = []
                now_t = datetime.now()

                if PIPELINE_MODE == "balanced":
                    # Balanced pipeline: person tracking first, face recognition per track.
                    people_tracks = track_people(frame)
                    if len(people_tracks) == 0:
                        no_face_inference_streak += 1
                    else:
                        no_face_inference_streak = 0

                    img_h, img_w = rgb.shape[:2]
                    for person in people_tracks:
                        tid = int(person["track_id"])
                        px1, py1, px2, py2 = person["bbox"]
                        px1 = max(0, px1)
                        py1 = max(0, py1)
                        px2 = min(img_w, px2)
                        py2 = min(img_h, py2)
                        if px2 - px1 < MIN_FACE_SIZE or py2 - py1 < MIN_FACE_SIZE:
                            continue

                        sid = None
                        conf = 0.0
                        dist = 99.0

                        need_recog = (tid not in track_identity) or (frame_idx % TRACK_RECOG_INTERVAL == 0)
                        if need_recog:
                            person_crop = rgb[py1:py2, px1:px2]
                            local_faces = detect_faces(person_crop, use_sahi=False)
                            if local_faces:
                                face_local = max(
                                    local_faces,
                                    key=lambda f: max(1, (f["bbox"][2] - f["bbox"][0]) * (f["bbox"][3] - f["bbox"][1]))
                                )
                                lx1, ly1, lx2, ly2 = face_local["bbox"]
                                gx1, gy1, gx2, gy2 = px1 + lx1, py1 + ly1, px1 + lx2, py1 + ly2
                                glmk = np.asarray(face_local["landmarks"], dtype=np.float32).copy()
                                glmk[:, 0] += px1
                                glmk[:, 1] += py1
                                face_global = {
                                    "bbox": (gx1, gy1, gx2, gy2),
                                    "landmarks": glmk,
                                    "det_score": float(face_local.get("det_score", 1.0)),
                                }
                                emb = extract_features(rgb, face_global)
                                if emb is not None:
                                    sid, dist, conf = match_embedding(emb, known_db)
                                    if sid is not None:
                                        track_identity[tid] = {
                                            "sid": sid,
                                            "conf": conf,
                                            "last_seen_frame": frame_idx,
                                        }

                        if sid is None and tid in track_identity:
                            prev = track_identity[tid]
                            if (frame_idx - prev["last_seen_frame"]) <= TRACK_KEEP_ID_MISSES:
                                sid = prev["sid"]
                                conf = prev["conf"]

                        if sid is not None:
                            sname = student_names.get(sid, sid)
                            label = f"#{tid} {sname} ({conf:.0%})"
                            color = BBOX_COLOR

                            if sid not in state:
                                delta_min = (now_t - start_time).total_seconds() / 60
                                status = "Present" if delta_min <= late_minutes else "Late"
                                state[sid] = {
                                    "first_seen": now_t, "last_seen": now_t,
                                    "status": status, "is_inside": True,
                                    "last_change_time": now_t, "last_confidence": conf,
                                }
                                insert_attendance_if_new(conn, session_id, sid, now_t, status, conf)
                                open_movement_interval(conn, session_id, sid, now_t)
                            else:
                                state[sid]["last_seen"] = now_t
                                state[sid]["last_confidence"] = conf
                                update_last_seen(conn, session_id, sid, now_t, conf)
                                if not state[sid]["is_inside"]:
                                    state[sid]["is_inside"] = True
                                    state[sid]["last_change_time"] = now_t
                                    open_movement_interval(conn, session_id, sid, now_t)

                            status_lbl = state[sid]["status"]
                            label += f" [{status_lbl}]"
                            recognized_list.append({
                                "student_id": sid,
                                "name": sname,
                                "status": status_lbl,
                                "confidence": round(conf * 100, 1),
                                "time": now_t.strftime("%H:%M:%S"),
                            })
                        else:
                            label = f"#{tid} Unknown"
                            color = UNKNOWN_BBOX_COLOR

                        det_items.append({
                            "bbox": (px1, py1, px2, py2),
                            "color": color,
                            "label": label,
                            "label_y": py1 - 10 if py1 - 10 > 10 else py2 + 20,
                        })

                else:
                    # Legacy pipeline fallback.
                    no_face_inference_streak = 0
                    for face in faces:
                        x1, y1, x2, y2 = face["bbox"]
                        embedding = extract_features(rgb, face)
                        if embedding is None:
                            continue

                        sid, dist, conf = match_embedding(embedding, known_db)

                        d_x1 = int(x1 * scale_x)
                        d_y1 = int(y1 * scale_y)
                        d_x2 = int(x2 * scale_x)
                        d_y2 = int(y2 * scale_y)

                        if sid is not None:
                            sname = student_names.get(sid, sid)
                            label = f"{sname} ({conf:.0%})"
                            color = BBOX_COLOR

                            if sid not in state:
                                delta_min = (now_t - start_time).total_seconds() / 60
                                status = "Present" if delta_min <= late_minutes else "Late"
                                state[sid] = {
                                    "first_seen": now_t, "last_seen": now_t,
                                    "status": status, "is_inside": True,
                                    "last_change_time": now_t, "last_confidence": conf,
                                }
                                insert_attendance_if_new(conn, session_id, sid, now_t, status, conf)
                                open_movement_interval(conn, session_id, sid, now_t)
                            else:
                                state[sid]["last_seen"] = now_t
                                state[sid]["last_confidence"] = conf
                                update_last_seen(conn, session_id, sid, now_t, conf)
                                if not state[sid]["is_inside"]:
                                    state[sid]["is_inside"] = True
                                    state[sid]["last_change_time"] = now_t
                                    open_movement_interval(conn, session_id, sid, now_t)

                            status_lbl = state[sid]["status"]
                            label += f" [{status_lbl}]"
                            recognized_list.append({
                                "student_id": sid,
                                "name": sname,
                                "status": status_lbl,
                                "confidence": round(conf * 100, 1),
                                "time": now_t.strftime("%H:%M:%S"),
                            })
                        else:
                            label = f"Unknown ({dist:.2f})"
                            color = UNKNOWN_BBOX_COLOR

                        det_items.append({
                            "bbox": (d_x1, d_y1, d_x2, d_y2),
                            "color": color,
                            "label": label,
                            "label_y": d_y1 - 10 if d_y1 - 10 > 10 else d_y2 + 20,
                        })

                # Associate detections to existing tracks to reduce jitter and keep visual continuity.
                unmatched_det_idx = set(range(len(det_items)))
                for tid in list(track_map.keys()):
                    tr = track_map[tid]
                    best_idx = None
                    best_iou = 0.0
                    for di in list(unmatched_det_idx):
                        iou = _bbox_iou(tr["bbox"], det_items[di]["bbox"])
                        if iou > best_iou:
                            best_iou = iou
                            best_idx = di

                    if best_idx is not None and best_iou >= TRACK_IOU_THRESHOLD:
                        det = det_items[best_idx]
                        tr["bbox"] = _smooth_bbox(tr["bbox"], det["bbox"])
                        tr["color"] = det["color"]
                        tr["label"] = det["label"]
                        tr["last_seen"] = frame_idx
                        tr["missed"] = 0
                        unmatched_det_idx.remove(best_idx)
                    else:
                        tr["missed"] += 1

                # Create tracks for remaining detections.
                for di in unmatched_det_idx:
                    det = det_items[di]
                    track_map[next_track_id] = {
                        "bbox": det["bbox"],
                        "color": det["color"],
                        "label": det["label"],
                        "last_seen": frame_idx,
                        "missed": 0,
                    }
                    next_track_id += 1

                # Prune stale tracks.
                for tid in list(track_map.keys()):
                    if track_map[tid]["missed"] > TRACK_MAX_MISSES:
                        del track_map[tid]

                handle_absent_web(conn, session_id, state)
                
                # Update frontend recognized list safely
                attendance_status["students_detected"] = len(state)

                cached_drawings.clear()
                for tid, tr in track_map.items():
                    bbox = tr["bbox"]
                    color = tr["color"]
                    label = tr["label"]
                    label_y = bbox[1] - 10 if bbox[1] - 10 > 10 else bbox[3] + 20
                    cached_drawings.append((bbox, color, label, label_y))

                # Draw smoothed track boxes.
                for bbox, color, label, label_y in cached_drawings:
                    cv2.rectangle(display, (bbox[0], bbox[1]), (bbox[2], bbox[3]), color, 2)
                    cv2.putText(display, label, (bbox[0], label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICKNESS)

            # FPS
            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                attendance_status["fps"] = round(fps_counter / elapsed, 1)
                fps_counter = 0
                fps_timer = time.time()

            # HUD
            hud = f"Class: {class_name} | FPS: {attendance_status['fps']:.1f} | Students: {len(state)}"
            cv2.putText(display, hud, (10, 25),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)

            with frame_lock:
                current_frame = display.copy()

            if recognized_list:
                attendance_status["recognized"] = recognized_list
            
        # Finalize
        end_time = datetime.now()
        finalize_session(conn, session_id, end_time)
        for sid in state:
            set_checkout_time(conn, session_id, sid, end_time)
            close_last_movement_interval(conn, session_id, sid, end_time)

        export_report(conn, session_id)

        attendance_status["message"] = f"Session ended. Duration: {end_time - start_time}"

    except Exception as e:
        import traceback
        traceback.print_exc()
        attendance_status["message"] = f"Error: {str(e)}"
    finally:
        if cap is not None:
            release_camera(cap)
        if conn is not None:
            conn.close()

    attendance_status["running"] = False
    time.sleep(2)
    with frame_lock:
        current_frame = None


def handle_absent_web(conn, session_id, state):
    now_t = datetime.now()
    for sid, info in state.items():
        if not info["is_inside"]:
            continue
        gap = (now_t - info["last_seen"]).total_seconds()
        if gap >= ABSENCE_THRESHOLD_SEC:
            info["is_inside"] = False
            info["last_change_time"] = now_t
            close_last_movement_interval(conn, session_id, sid, now_t)


# ── Run ───────────────────────────────────────────────────────
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000, debug=False, threaded=True)
