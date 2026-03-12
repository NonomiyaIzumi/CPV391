"""
Flask web UI for Smart Classroom Attendance System.
"""
import os
import threading
import time
from datetime import datetime

import cv2
import numpy as np
from flask import Flask, render_template, request, jsonify, Response, send_file

from config import (
    RTSP_URL, REQUIRED_SAMPLES, LATE_MINUTES,
    FRAME_SKIP, ABSENCE_THRESHOLD_SEC,
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
from detect import detect_faces, crop_face
from recognize import extract_features, match_embedding
from enrollment import estimate_blur, estimate_brightness, remove_outliers
from report import export_report

app = Flask(__name__)

# ── Global state ──────────────────────────────────────────────
camera_lock = threading.Lock()
current_frame = None        # latest display frame (BGR)
frame_lock = threading.Lock()

enrollment_status = {
    "running": False,
    "progress": 0,
    "total": 0,
    "message": "",
    "done": False,
    "success": False,
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
    samples = int(data.get("samples", REQUIRED_SAMPLES))
    source = data.get("source", RTSP_URL)

    if not student_id or not name:
        return jsonify({"error": "student_id and name are required"}), 400

    enrollment_status = {
        "running": True, "progress": 0, "total": samples,
        "message": "Starting...", "done": False, "success": False,
    }

    thread = threading.Thread(
        target=_enrollment_worker,
        args=(student_id, name, source, samples),
        daemon=True,
    )
    thread.start()
    return jsonify({"status": "started"})


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
        time.sleep(0.015)


@app.route("/video_feed")
def video_feed():
    return Response(
        _generate_mjpeg(),
        mimetype="multipart/x-mixed-replace; boundary=frame",
    )


# ── Workers ───────────────────────────────────────────────────
def _enrollment_worker(student_id, name, source, samples_needed):
    global enrollment_status, current_frame

    try:
        conn = init_db()
        upsert_student(conn, student_id, name)
        delete_encodings_for_student(conn, student_id)

        cap = connect_camera(source)
        samples = []
        enrollment_status["message"] = "Camera connected. Look at the camera."

        frame_idx = 0
        while len(samples) < samples_needed and enrollment_status["running"]:
            frame = read_frame(cap)
            if frame is None:
                continue
                
            frame_idx += 1
            display = get_display_frame(frame)
            
            # Chỉ chạy nhận diện 1 lần mỗi FRAME_SKIP frames để tránh quá tải CPU,
            # các frame còn lại chỉ cập nhật lên màn hình để web mượt mà.
            if FRAME_SKIP > 1 and frame_idx % FRAME_SKIP != 0:
                with frame_lock:
                    current_frame = display.copy()
                continue
                
            rgb = preprocess_frame(frame)
            faces = detect_faces(rgb)

            status_text = f"Samples: {len(samples)}/{samples_needed}"

            if len(faces) == 0:
                status_text += " | No face detected"
            elif len(faces) > 1:
                status_text += " | Multiple faces! Only 1 allowed"
            else:
                face = faces[0]
                x1, y1, x2, y2 = face["bbox"]
                face_crop = crop_face(rgb, (x1, y1, x2, y2))
                face_h, face_w = face_crop.shape[:2]

                scale_x = display.shape[1] / rgb.shape[1]
                scale_y = display.shape[0] / rgb.shape[0]
                d_x1, d_y1 = int(x1 * scale_x), int(y1 * scale_y)
                d_x2, d_y2 = int(x2 * scale_x), int(y2 * scale_y)
                cv2.rectangle(display, (d_x1, d_y1), (d_x2, d_y2), BBOX_COLOR, 2)

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
                        status_text += f" | ✓ Captured!"

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

        release_camera(cap)

        if len(samples) >= samples_needed:
            filtered = remove_outliers(samples)
            if SAVE_ALL_SAMPLES:
                for emb in filtered:
                    insert_face_encoding(conn, student_id, emb, quality=1.0)
            else:
                mean_emb = np.mean(filtered, axis=0)
                insert_face_encoding(conn, student_id, mean_emb, quality=1.0)

            enrollment_status["success"] = True
            enrollment_status["message"] = f"✓ {name} registered! ({len(filtered)} embeddings saved)"
        else:
            enrollment_status["message"] = "Enrollment cancelled."

        conn.close()

    except Exception as e:
        enrollment_status["message"] = f"Error: {str(e)}"

    enrollment_status["running"] = False
    enrollment_status["done"] = True

    # Clear frame after a delay
    time.sleep(2)
    with frame_lock:
        current_frame = None


def _attendance_worker(source, class_name, late_minutes):
    global attendance_status, current_frame

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

        while attendance_status["running"]:
            frame = read_frame(cap)
            if frame is None:
                time.sleep(0.01)
                continue

            frame_idx += 1
            if FRAME_SKIP > 1 and frame_idx % FRAME_SKIP != 0:
                display = get_display_frame(frame)
                with frame_lock:
                    current_frame = display.copy()
                continue

            rgb = preprocess_frame(frame)
            faces = detect_faces(rgb)
            display = get_display_frame(frame)

            scale_x = display.shape[1] / rgb.shape[1]
            scale_y = display.shape[0] / rgb.shape[0]

            recognized_list = []

            if len(faces) == 0:
                handle_absent_web(conn, session_id, state)
            else:
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

                    now_t = datetime.now()

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

                    cv2.rectangle(display, (d_x1, d_y1), (d_x2, d_y2), color, 2)
                    label_y = d_y1 - 10 if d_y1 - 10 > 10 else d_y2 + 20
                    cv2.putText(display, label, (d_x1, label_y),
                                cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICKNESS)

                handle_absent_web(conn, session_id, state)

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

            attendance_status["students_detected"] = len(state)
            if recognized_list:
                attendance_status["recognized"] = recognized_list

        # Finalize
        end_time = datetime.now()
        finalize_session(conn, session_id, end_time)
        for sid in state:
            set_checkout_time(conn, session_id, sid, end_time)
            close_last_movement_interval(conn, session_id, sid, end_time)

        release_camera(cap)
        export_report(conn, session_id)
        conn.close()

        attendance_status["message"] = f"Session ended. Duration: {end_time - start_time}"

    except Exception as e:
        attendance_status["message"] = f"Error: {str(e)}"

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
