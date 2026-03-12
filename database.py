"""
SQLite database: schema, CRUD, embedding serialization.
"""
import sqlite3
import uuid
from datetime import datetime

import numpy as np

from config import DB_PATH

# ── Schema DDL ─────────────────────────────────────────────────
_DDL = """
CREATE TABLE IF NOT EXISTS students (
    student_id TEXT PRIMARY KEY,
    name       TEXT NOT NULL,
    created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS face_encodings (
    encoding_id INTEGER PRIMARY KEY AUTOINCREMENT,
    student_id  TEXT NOT NULL,
    encoding    BLOB NOT NULL,
    quality     REAL,
    created_at  DATETIME DEFAULT CURRENT_TIMESTAMP,
    FOREIGN KEY(student_id) REFERENCES students(student_id)
);

CREATE TABLE IF NOT EXISTS class_sessions (
    session_id   TEXT PRIMARY KEY,
    class_name   TEXT,
    start_time   DATETIME NOT NULL,
    end_time     DATETIME,
    late_minutes INTEGER DEFAULT 10
);

CREATE TABLE IF NOT EXISTS attendance_records (
    session_id    TEXT NOT NULL,
    student_id    TEXT NOT NULL,
    first_seen    DATETIME NOT NULL,
    last_seen     DATETIME NOT NULL,
    status        TEXT NOT NULL,
    checkout_time DATETIME,
    confidence    REAL,
    PRIMARY KEY(session_id, student_id),
    FOREIGN KEY(session_id) REFERENCES class_sessions(session_id),
    FOREIGN KEY(student_id) REFERENCES students(student_id)
);

CREATE TABLE IF NOT EXISTS movement_log (
    log_id     INTEGER PRIMARY KEY AUTOINCREMENT,
    session_id TEXT NOT NULL,
    student_id TEXT NOT NULL,
    time_in    DATETIME NOT NULL,
    time_out   DATETIME,
    FOREIGN KEY(session_id) REFERENCES class_sessions(session_id),
    FOREIGN KEY(student_id) REFERENCES students(student_id)
);

CREATE INDEX IF NOT EXISTS idx_face_enc_student
    ON face_encodings(student_id);
CREATE INDEX IF NOT EXISTS idx_att_session
    ON attendance_records(session_id);
CREATE INDEX IF NOT EXISTS idx_move_session_student
    ON movement_log(session_id, student_id);

CREATE TABLE IF NOT EXISTS student_courses (
    student_id  TEXT NOT NULL,
    course_name TEXT NOT NULL,
    PRIMARY KEY(student_id, course_name),
    FOREIGN KEY(student_id) REFERENCES students(student_id)
);

CREATE INDEX IF NOT EXISTS idx_student_courses_course
    ON student_courses(course_name);
"""


# ── Helpers ────────────────────────────────────────────────────
def serialize_embedding(vec: np.ndarray) -> bytes:
    return vec.astype(np.float32).tobytes()


def deserialize_embedding(blob: bytes) -> np.ndarray:
    return np.frombuffer(blob, dtype=np.float32)


# ── Connection ─────────────────────────────────────────────────
def init_db(db_path: str = DB_PATH) -> sqlite3.Connection:
    conn = sqlite3.connect(db_path)
    conn.row_factory = sqlite3.Row
    conn.executescript(_DDL)
    conn.commit()
    return conn


# ── Students ───────────────────────────────────────────────────
def upsert_student(conn: sqlite3.Connection, student_id: str, name: str):
    conn.execute(
        """INSERT INTO students(student_id, name)
           VALUES (?, ?)
           ON CONFLICT(student_id) DO UPDATE SET name = excluded.name""",
        (student_id, name),
    )
    conn.commit()


def get_student(conn: sqlite3.Connection, student_id: str):
    row = conn.execute(
        "SELECT * FROM students WHERE student_id = ?", (student_id,)
    ).fetchone()
    return dict(row) if row else None


def get_all_students(conn: sqlite3.Connection):
    rows = conn.execute("SELECT * FROM students ORDER BY student_id").fetchall()
    return [dict(r) for r in rows]


# ── Face Encodings ─────────────────────────────────────────────
def insert_face_encoding(
    conn: sqlite3.Connection,
    student_id: str,
    vec: np.ndarray,
    quality: float = 1.0,
):
    blob = serialize_embedding(vec)
    conn.execute(
        "INSERT INTO face_encodings(student_id, encoding, quality) VALUES (?, ?, ?)",
        (student_id, blob, quality),
    )
    conn.commit()


def delete_encodings_for_student(conn: sqlite3.Connection, student_id: str):
    conn.execute(
        "DELETE FROM face_encodings WHERE student_id = ?", (student_id,)
    )
    conn.commit()


def load_all_encodings(conn: sqlite3.Connection) -> dict:
    """Return {student_id: [np.array, ...]}."""
    rows = conn.execute("SELECT student_id, encoding FROM face_encodings").fetchall()
    known: dict[str, list[np.ndarray]] = {}
    for row in rows:
        vec = deserialize_embedding(row["encoding"])
    return known


def load_encodings_by_course(conn: sqlite3.Connection, course_name: str) -> dict:
    """Return {student_id: [np.array, ...]} for students enrolled in `course_name`."""
    rows = conn.execute(
        """SELECT f.student_id, f.encoding 
           FROM face_encodings f
           JOIN student_courses sc ON f.student_id = sc.student_id
           WHERE sc.course_name = ?""",
        (course_name,)
    ).fetchall()
    known: dict[str, list[np.ndarray]] = {}
    for row in rows:
        vec = deserialize_embedding(row["encoding"])
        known.setdefault(row["student_id"], []).append(vec)
    return known


# ── Courses ────────────────────────────────────────────────────
def add_student_to_course(conn: sqlite3.Connection, student_id: str, course_name: str):
    conn.execute(
        "INSERT OR IGNORE INTO student_courses(student_id, course_name) VALUES (?, ?)",
        (student_id, course_name),
    )
    conn.commit()


def remove_student_from_course(conn: sqlite3.Connection, student_id: str, course_name: str):
    conn.execute(
        "DELETE FROM student_courses WHERE student_id = ? AND course_name = ?",
        (student_id, course_name),
    )
    conn.commit()


def get_student_courses(conn: sqlite3.Connection, student_id: str):
    rows = conn.execute(
        "SELECT course_name FROM student_courses WHERE student_id = ?",
        (student_id,)
    ).fetchall()
    return [r["course_name"] for r in rows]


def get_students_by_course(conn: sqlite3.Connection, course_name: str):
    rows = conn.execute(
        """SELECT s.* FROM students s
           JOIN student_courses sc ON s.student_id = sc.student_id
           WHERE sc.course_name = ?""",
        (course_name,)
    ).fetchall()
    return [dict(r) for r in rows]


def delete_student(conn: sqlite3.Connection, student_id: str):
    conn.execute("DELETE FROM face_encodings WHERE student_id = ?", (student_id,))
    conn.execute("DELETE FROM student_courses WHERE student_id = ?", (student_id,))
    # We keep attendance_records and movement_log for reporting history
    conn.execute("DELETE FROM students WHERE student_id = ?", (student_id,))
    conn.commit()


# ── Class Sessions ─────────────────────────────────────────────
def create_session(
    conn: sqlite3.Connection,
    class_name: str,
    start_time: datetime,
    late_minutes: int = 10,
) -> str:
    sid = str(uuid.uuid4())
    conn.execute(
        """INSERT INTO class_sessions(session_id, class_name, start_time, late_minutes)
           VALUES (?, ?, ?, ?)""",
        (sid, class_name, start_time.isoformat(), late_minutes),
    )
    conn.commit()
    return sid


def finalize_session(
    conn: sqlite3.Connection, session_id: str, end_time: datetime
):
    conn.execute(
        "UPDATE class_sessions SET end_time = ? WHERE session_id = ?",
        (end_time.isoformat(), session_id),
    )
    conn.commit()


def get_session(conn: sqlite3.Connection, session_id: str):
    row = conn.execute(
        "SELECT * FROM class_sessions WHERE session_id = ?", (session_id,)
    ).fetchone()
    return dict(row) if row else None


def get_all_sessions(conn: sqlite3.Connection):
    rows = conn.execute(
        "SELECT * FROM class_sessions ORDER BY start_time DESC"
    ).fetchall()
    return [dict(r) for r in rows]


# ── Attendance Records ─────────────────────────────────────────
def insert_attendance_if_new(
    conn: sqlite3.Connection,
    session_id: str,
    student_id: str,
    first_seen: datetime,
    status: str,
    confidence: float,
):
    existing = conn.execute(
        """SELECT 1 FROM attendance_records
           WHERE session_id = ? AND student_id = ?""",
        (session_id, student_id),
    ).fetchone()
    if existing:
        return False
    conn.execute(
        """INSERT INTO attendance_records
           (session_id, student_id, first_seen, last_seen, status, confidence)
           VALUES (?, ?, ?, ?, ?, ?)""",
        (
            session_id,
            student_id,
            first_seen.isoformat(),
            first_seen.isoformat(),
            status,
            confidence,
        ),
    )
    conn.commit()
    return True


def update_last_seen(
    conn: sqlite3.Connection,
    session_id: str,
    student_id: str,
    last_seen: datetime,
    confidence: float,
):
    conn.execute(
        """UPDATE attendance_records
           SET last_seen = ?, confidence = ?
           WHERE session_id = ? AND student_id = ?""",
        (last_seen.isoformat(), confidence, session_id, student_id),
    )
    conn.commit()


def set_checkout_time(
    conn: sqlite3.Connection,
    session_id: str,
    student_id: str,
    checkout_time: datetime,
):
    conn.execute(
        """UPDATE attendance_records SET checkout_time = ?
           WHERE session_id = ? AND student_id = ?""",
        (checkout_time.isoformat(), session_id, student_id),
    )
    conn.commit()


def get_attendance_for_session(conn: sqlite3.Connection, session_id: str):
    rows = conn.execute(
        """SELECT a.*, s.name
           FROM attendance_records a
           JOIN students s ON a.student_id = s.student_id
           WHERE a.session_id = ?
           ORDER BY a.first_seen""",
        (session_id,),
    ).fetchall()
    return [dict(r) for r in rows]


# ── Movement Log ───────────────────────────────────────────────
def open_movement_interval(
    conn: sqlite3.Connection,
    session_id: str,
    student_id: str,
    time_in: datetime,
):
    conn.execute(
        """INSERT INTO movement_log(session_id, student_id, time_in)
           VALUES (?, ?, ?)""",
        (session_id, student_id, time_in.isoformat()),
    )
    conn.commit()


def close_last_movement_interval(
    conn: sqlite3.Connection,
    session_id: str,
    student_id: str,
    time_out: datetime,
):
    row = conn.execute(
        """SELECT log_id FROM movement_log
           WHERE session_id = ? AND student_id = ? AND time_out IS NULL
           ORDER BY log_id DESC LIMIT 1""",
        (session_id, student_id),
    ).fetchone()
    if row:
        conn.execute(
            "UPDATE movement_log SET time_out = ? WHERE log_id = ?",
            (time_out.isoformat(), row["log_id"]),
        )
        conn.commit()
