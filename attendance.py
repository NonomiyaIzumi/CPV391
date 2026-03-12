"""
Real-time attendance workflow: multi-face detection + recognition,
Present/Late logic, movement tracking.
Uses InsightFace pretrained model (SCRFD + ArcFace) with GPU acceleration.
"""
import argparse
import sys
import time
from datetime import datetime

import cv2
import numpy as np

from config import (
    RTSP_URL,
    FRAME_SKIP,
    LATE_MINUTES,
    ABSENCE_THRESHOLD_SEC,
    SHOW_VIDEO,
    BBOX_COLOR,
    UNKNOWN_BBOX_COLOR,
    FONT_SCALE,
    FONT_THICKNESS,
)
from camera import connect_camera, read_frame, release_camera
from preprocess import preprocess_frame, get_display_frame
from detect import detect_and_encode
from recognize import match_embedding
from database import (
    init_db,
    load_all_encodings,
    create_session,
    finalize_session,
    insert_attendance_if_new,
    update_last_seen,
    set_checkout_time,
    open_movement_interval,
    close_last_movement_interval,
    get_student,
)
from report import export_report


def handle_absent(conn, session_id: str, state: dict):
    """Mark students as absent if not seen for ABSENCE_THRESHOLD_SEC."""
    now_t = datetime.now()
    for sid, info in state.items():
        if not info["is_inside"]:
            continue
        gap = (now_t - info["last_seen"]).total_seconds()
        if gap >= ABSENCE_THRESHOLD_SEC:
            info["is_inside"] = False
            info["last_change_time"] = now_t
            close_last_movement_interval(conn, session_id, sid, now_t)


def attendance(source, class_name: str, late_minutes: int = LATE_MINUTES):
    """
    Main attendance loop using InsightFace GPU.

    Press 'q' to end the class session.
    """
    conn = init_db()

    # Load known faces
    known_db = load_all_encodings(conn)
    if not known_db:
        print("[Attendance] ERROR: No enrolled students found! Run enrollment.py first.")
        conn.close()
        return None

    student_names = {}
    for sid in known_db:
        info = get_student(conn, sid)
        student_names[sid] = info["name"] if info else sid

    print(f"[Attendance] Loaded {len(known_db)} students from DB.")

    # Create session
    start_time = datetime.now()
    session_id = create_session(conn, class_name, start_time, late_minutes)
    print(f"[Attendance] Session started: {session_id}")
    print(f"[Attendance] Class: {class_name} | Late window: {late_minutes} min")
    print(f"[Attendance] Start time: {start_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print("[Attendance] Press 'q' to end class.\n")

    # State tracking
    state: dict[str, dict] = {}
    cap = connect_camera(source)
    frame_idx = 0
    fps_counter = 0
    fps_timer = time.time()
    current_fps = 0.0

    try:
        while True:
            frame = read_frame(cap)
            if frame is None:
                time.sleep(0.01)
                continue

            frame_idx += 1

            # Frame skip for performance
            if FRAME_SKIP > 1 and frame_idx % FRAME_SKIP != 0:
                if SHOW_VIDEO:
                    display = get_display_frame(frame)
                    cv2.imshow("Attendance", display)
                    if cv2.waitKey(1) & 0xFF == ord("q"):
                        break
                continue

            # Process frame — InsightFace does detect + encode in one GPU pass
            rgb = preprocess_frame(frame)
            faces = detect_and_encode(rgb)
            display = get_display_frame(frame)

            # Compute display scale
            scale_x = display.shape[1] / rgb.shape[1]
            scale_y = display.shape[0] / rgb.shape[0]

            if len(faces) == 0:
                handle_absent(conn, session_id, state)
            else:
                for face in faces:
                    x1, y1, x2, y2 = [int(v) for v in face.bbox]
                    embedding = face.embedding  # 512-D vector

                    if embedding is None:
                        continue

                    sid, dist, conf = match_embedding(embedding, known_db)

                    # Scale bbox for display
                    d_x1 = int(x1 * scale_x)
                    d_y1 = int(y1 * scale_y)
                    d_x2 = int(x2 * scale_x)
                    d_y2 = int(y2 * scale_y)

                    now_t = datetime.now()

                    if sid is not None:
                        name = student_names.get(sid, sid)
                        label = f"{name} ({conf:.0%})"
                        color = BBOX_COLOR

                        if sid not in state:
                            delta_min = (now_t - start_time).total_seconds() / 60
                            status = "Present" if delta_min <= late_minutes else "Late"

                            state[sid] = {
                                "first_seen": now_t,
                                "last_seen": now_t,
                                "status": status,
                                "is_inside": True,
                                "last_change_time": now_t,
                                "last_confidence": conf,
                            }

                            insert_attendance_if_new(conn, session_id, sid, now_t, status, conf)
                            open_movement_interval(conn, session_id, sid, now_t)
                            print(f"  [{status}] {name} ({sid}) - {now_t.strftime('%H:%M:%S')}")
                        else:
                            state[sid]["last_seen"] = now_t
                            state[sid]["last_confidence"] = conf
                            update_last_seen(conn, session_id, sid, now_t, conf)

                            if not state[sid]["is_inside"]:
                                state[sid]["is_inside"] = True
                                state[sid]["last_change_time"] = now_t
                                open_movement_interval(conn, session_id, sid, now_t)
                                print(f"  [RETURN] {name} ({sid}) - {now_t.strftime('%H:%M:%S')}")

                        # Status indicator
                        status_label = state[sid]["status"]
                        label += f" [{status_label}]"
                    else:
                        label = f"Unknown ({dist:.2f})"
                        color = UNKNOWN_BBOX_COLOR

                    # Draw bbox and label
                    cv2.rectangle(display, (d_x1, d_y1), (d_x2, d_y2), color, 2)
                    label_y = d_y1 - 10 if d_y1 - 10 > 10 else d_y2 + 20
                    cv2.putText(
                        display, label, (d_x1, label_y),
                        cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, color, FONT_THICKNESS,
                    )

                handle_absent(conn, session_id, state)

            # FPS counter
            fps_counter += 1
            elapsed = time.time() - fps_timer
            if elapsed >= 1.0:
                current_fps = fps_counter / elapsed
                fps_counter = 0
                fps_timer = time.time()

            # HUD
            hud_lines = [
                f"Class: {class_name} | FPS: {current_fps:.1f} | GPU Accelerated",
                f"Students detected: {len(state)} | Time: {datetime.now().strftime('%H:%M:%S')}",
            ]
            for i, line in enumerate(hud_lines):
                cv2.putText(
                    display, line, (10, 25 + i * 25),
                    cv2.FONT_HERSHEY_SIMPLEX, FONT_SCALE, (255, 255, 255), FONT_THICKNESS,
                )

            if SHOW_VIDEO:
                cv2.imshow("Attendance", display)
                key = cv2.waitKey(1) & 0xFF
                if key == ord("q"):
                    break

    except KeyboardInterrupt:
        print("\n[Attendance] Interrupted by user.")

    # Finalize
    end_time = datetime.now()
    finalize_session(conn, session_id, end_time)

    for sid in state:
        set_checkout_time(conn, session_id, sid, end_time)
        close_last_movement_interval(conn, session_id, sid, end_time)

    release_camera(cap)

    print(f"\n[Attendance] Session ended: {end_time.strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"[Attendance] Duration: {(end_time - start_time)}")
    print(f"[Attendance] Total students recorded: {len(state)}")

    # Export report
    csv_path = export_report(conn, session_id)
    print(f"[Attendance] Report exported: {csv_path}")

    conn.close()
    return session_id


def main():
    parser = argparse.ArgumentParser(description="Real-time attendance system")
    parser.add_argument("--source", default=RTSP_URL, help="RTSP URL or camera index")
    parser.add_argument("--class_name", default="CPV391", help="Class name")
    parser.add_argument("--late_minutes", type=int, default=LATE_MINUTES, help="Late window (minutes)")
    args = parser.parse_args()

    attendance(args.source, args.class_name, args.late_minutes)


if __name__ == "__main__":
    main()
