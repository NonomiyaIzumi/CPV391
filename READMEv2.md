# Smart Classroom Face Recognition Attendance System (assignment.md)

Tài liệu này mô tả đầy đủ bài toán, kiến trúc, workflow, hướng triển khai code, thiết kế database, pseudo-code chi tiết cho hệ thống điểm danh lớp học bằng nhận diện khuôn mặt.

---

## 1. Bài toán

### 1.1 Mục tiêu
Xây dựng hệ thống điểm danh tự động trong lớp học sử dụng camera IP. Hệ thống nhận luồng RTSP, xử lý video theo thời gian thực để phát hiện khuôn mặt, nhận diện danh tính sinh viên, ghi nhận điểm danh theo thời gian.

### 1.2 Phạm vi
- **Trong lớp học**: nhiều sinh viên cùng xuất hiện trong 1 khung hình
- **Thời gian thực**: xử lý liên tục trong suốt buổi học
- **Phân loại sinh viên theo môn**: sinh viên được đăng ký theo từng môn học (course) riêng biệt
- **2 giai đoạn**:
  - Enrollment: đăng ký khuôn mặt sinh viên và gán môn học
  - Attendance: điểm danh theo môn học, theo dõi hiện diện, xuất báo cáo

### 1.3 Đầu vào, đầu ra
**Đầu vào**
- RTSP URL của camera IP
- Thông tin sinh viên khi enrollment: `student_id`, `name`
- Tham số hệ thống: ngưỡng nhận diện, FPS xử lý, late window

**Đầu ra**
- Bảng điểm danh theo buổi học
- Log ra vào lớp theo thời gian (tùy chọn)
- Báo cáo xuất file CSV

### 1.4 Yêu cầu chức năng (Functional Requirements)
- Kết nối RTSP, đọc frame ổn định
- Tiền xử lý frame để tăng độ chính xác
- Phát hiện nhiều khuôn mặt trong 1 frame
- Trích xuất embedding khuôn mặt
- Quản lý Sinh viên - Môn học: Cho phép gán nhiều môn cho 1 sinh viên, xóa môn, xóa sinh viên
- So khớp embedding với dữ liệu đã đăng ký (chỉ duyệt các sinh viên được gán vào môn học hiện tại)
- Ghi điểm danh: `first_seen`, `last_seen`, `status`, `checkout_time`
- Quy tắc Present/Late theo cửa sổ 10 phút đầu
- Theo dõi biến mất xuất hiện lại (movement tracking)

### 1.5 Yêu cầu phi chức năng (Non-functional Requirements)
- Độ trễ thấp: xử lý gần thời gian thực (mục tiêu 5–15 FPS tùy máy)
- Chịu được ánh sáng thay đổi, góc quay lệch
- Dễ mở rộng số sinh viên
- Dữ liệu được lưu trữ bền vững, truy xuất nhanh

---

## 2. Hướng tiếp cận, mô hình sử dụng

### 2.1 Face Detection
Mục tiêu detector là tìm bounding box của từng khuôn mặt trong frame.

Khuyến nghị:
- **RetinaFace**: độ chính xác tốt trong cảnh đông người, nhiều góc
- **MTCNN**: dễ tích hợp, đủ tốt cho bài tập

Output: danh sách hộp mặt `(x1, y1, x2, y2)`.

### 2.2 Face Recognition
Cách làm mặc định dùng mô hình metric learning có sẵn.

- Thư viện: `face_recognition` (dlib)
- Mô hình: ResNet-based deep metric learning
- Output: embedding 128 chiều (vector float)

**Cách “train”**
- Không train mạng từ đầu
- “Train” ở đây là Enrollment: lấy nhiều ảnh của sinh viên, tạo embedding, lưu vào DB
- Khi Attendance: lấy embedding mới, so khớp với embeddings trong DB bằng khoảng cách Euclid hoặc cosine

### 2.3 Chiến lược so khớp (Matching)
- Tính khoảng cách từ embedding mới tới từng embedding đã lưu
- Chọn khoảng cách nhỏ nhất
- Nếu khoảng cách <= `TOLERANCE` thì xác nhận

Gợi ý tham số ban đầu:
- Với `face_recognition`, `tolerance` thường trong khoảng `0.45–0.60`
- Bắt đầu thử `0.50`, sau đó điều chỉnh theo dữ liệu thực tế

---

## 3. Kiến trúc hệ thống

### 3.1 Kiến trúc mức cao
```
Enrollment Module
        ↓
Face Embedding Database
        ↓
Real-time Attendance Module
        ↓
Attendance Database + Report Export
```

### 3.2 Module chức năng
- `camera.py`: kết nối RTSP, đọc frame
- `preprocess.py`: resize, normalize, tăng chất lượng ảnh
- `detect.py`: phát hiện khuôn mặt
- `recognize.py`: tạo embedding, match danh tính
- `database.py`: SQLite schema, CRUD
- `enrollment.py`: workflow đăng ký khuôn mặt
- `attendance.py`: workflow điểm danh thời gian thực
- `report.py`: xuất báo cáo

---

## 4. Workflow chi tiết

## 4.1 Enrollment Workflow

### 4.1.1 Mục tiêu
Tạo dữ liệu tham chiếu cho mỗi sinh viên. Dữ liệu tham chiếu có thể là:
- 1 embedding đại diện (mean vector)
- N embeddings mẫu (khuyến nghị) để tăng độ bền khi so khớp

### 4.1.2 Workflow
```
RTSP stream
  ↓
Frame capture
  ↓
Preprocess
  ↓
Detect face (phải đúng 1 mặt)
  ↓
Quality check (face size, blur, brightness)
  ↓
Encode embedding
  ↓
Collect N samples
  ↓
Remove outliers
  ↓
Store to DB
```

### 4.1.3 Quality check gợi ý
- `MIN_FACE_SIZE`: loại bỏ mặt quá nhỏ (ví dụ < 80 px)
- Blur: dùng variance of Laplacian, nếu < ngưỡng thì bỏ
- Brightness: nếu frame quá tối thì bỏ

### 4.1.4 Outlier removal gợi ý
- Tính embedding trung tâm `c = mean(embeddings)`
- Tính dist của từng embedding tới `c`
- Bỏ top 10–20% dist lớn nhất
- Tính mean lần cuối làm embedding đại diện

---

## 4.2 Attendance Workflow

### 4.2.1 Mục tiêu
- Nhận diện nhiều sinh viên trong lớp theo thời gian thực (chỉ điểm danh các sinh viên có đăng ký môn học đó)
- Ghi nhận `first_seen`, `status`, cập nhật `last_seen`
- Theo dõi ra vào lớp bằng `movement_log` (tùy chọn)

### 4.2.2 Workflow tổng quát
```
Teacher Start Class (nhập `class_name`)
  ↓
Create session in DB
  ↓
Load embeddings to RAM cache (chỉ load khuôn mặt của sinh viên thuộc `class_name`)
  ↓
Loop frames:
  preprocess
  detect faces
  encode
    match
    update attendance state + DB
    handle absent threshold
  ↓
Teacher End Class
  ↓
Finalize session, checkout_time
  ↓
Export report
```

### 4.2.3 Quy tắc Present, Late
- `late_window_minutes = 10` (có thể cấu hình)
- Nếu `first_seen - start_time <= late_window_minutes` → Present
- Nếu lớn hơn → Late

### 4.2.4 Theo dõi ra vào (movement tracking)
- Nếu `now - last_seen >= ABSENCE_THRESHOLD_SEC` và sinh viên đang được coi là “trong lớp” thì:
  - ghi `time_out`
  - đặt `is_inside = False`
- Nếu match lại sinh viên khi `is_inside = False` thì:
  - ghi `time_in`
  - đặt `is_inside = True`

---

## 5. Tiền xử lý ảnh (Pre-processing)

### 5.1 Mục tiêu
Tăng tính ổn định của detector, giảm ảnh hưởng ánh sáng, tăng FPS.

### 5.2 Pipeline gợi ý
- Resize frame theo chiều rộng `RESIZE_WIDTH` (ví dụ 960 hoặc 720)
- BGR → RGB (phù hợp thư viện)
- Histogram equalization hoặc CLAHE (tùy detector)
- Gaussian blur nhẹ (kernel 3x3) để giảm noise

Lưu ý: với detector deep learning, thường chỉ cần RGB + normalize đúng chuẩn model.

---

## 6. Thiết kế Database (SQLite)

Thiết kế tập trung vào 2 mục tiêu:
- Lưu nhiều embeddings mỗi sinh viên
- Quản lý đa môn học cho một sinh viên (N-N relationship)
- Quản lý điểm danh theo buổi học, dễ xuất báo cáo

### 6.1 Bảng và quan hệ
- `students` 1–N `face_encodings`
- `class_sessions` 1–N `attendance_records`
- `attendance_records` 1–N `movement_log` (tùy chọn)

### 6.2 Schema

#### 6.2.1 students
| Column     | Type | Notes |
|-----------|------|------|
| student_id | TEXT | PK |
| name       | TEXT | NOT NULL |
| created_at | DATETIME | default CURRENT_TIMESTAMP |

#### 6.2.2 face_encodings
| Column      | Type | Notes |
|------------|------|------|
| encoding_id | INTEGER | PK AUTOINCREMENT |
| student_id  | TEXT | FK → students |
| encoding    | BLOB | float32 array bytes |
| quality     | REAL | optional |
| created_at  | DATETIME | default CURRENT_TIMESTAMP |

#### 6.2.3 student_courses
| Column      | Type | Notes |
|------------|------|------|
| student_id  | TEXT | FK → students, PK |
| course_name | TEXT | PK |

#### 6.2.4 class_sessions
| Column        | Type | Notes |
|--------------|------|------|
| session_id    | TEXT | PK uuid |
| class_name    | TEXT | optional |
| start_time    | DATETIME | NOT NULL |
| end_time      | DATETIME | nullable |
| late_minutes  | INTEGER | default 10 |

#### 6.2.4 attendance_records
| Column        | Type | Notes |
|--------------|------|------|
| session_id    | TEXT | FK → class_sessions |
| student_id    | TEXT | FK → students |
| first_seen    | DATETIME | NOT NULL |
| last_seen     | DATETIME | NOT NULL |
| status        | TEXT | Present, Late |
| checkout_time | DATETIME | nullable |
| confidence    | REAL | optional |
Primary key: `(session_id, student_id)`

#### 6.2.5 movement_log (optional)
| Column     | Type | Notes |
|-----------|------|------|
| log_id     | INTEGER | PK AUTOINCREMENT |
| session_id | TEXT | FK |
| student_id | TEXT | FK |
| time_in    | DATETIME | NOT NULL |
| time_out   | DATETIME | nullable |

### 6.3 Index gợi ý
- `CREATE INDEX idx_face_enc_student ON face_encodings(student_id);`
- `CREATE INDEX idx_att_session ON attendance_records(session_id);`
- `CREATE INDEX idx_move_session_student ON movement_log(session_id, student_id);`

### 6.4 DDL tham khảo
```sql
CREATE TABLE IF NOT EXISTS students (
  student_id TEXT PRIMARY KEY,
  name TEXT NOT NULL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP
);

CREATE TABLE IF NOT EXISTS face_encodings (
  encoding_id INTEGER PRIMARY KEY AUTOINCREMENT,
  student_id TEXT NOT NULL,
  encoding BLOB NOT NULL,
  quality REAL,
  created_at DATETIME DEFAULT CURRENT_TIMESTAMP,
  FOREIGN KEY(student_id) REFERENCES students(student_id)
);

CREATE TABLE IF NOT EXISTS student_courses (
  student_id TEXT NOT NULL,
  course_name TEXT NOT NULL,
  PRIMARY KEY(student_id, course_name),
  FOREIGN KEY(student_id) REFERENCES students(student_id)
);

CREATE TABLE IF NOT EXISTS class_sessions (
  session_id TEXT PRIMARY KEY,
  class_name TEXT,
  start_time DATETIME NOT NULL,
  end_time DATETIME,
  late_minutes INTEGER DEFAULT 10
);

CREATE TABLE IF NOT EXISTS attendance_records (
  session_id TEXT NOT NULL,
  student_id TEXT NOT NULL,
  first_seen DATETIME NOT NULL,
  last_seen DATETIME NOT NULL,
  status TEXT NOT NULL,
  checkout_time DATETIME,
  confidence REAL,
  PRIMARY KEY(session_id, student_id),
  FOREIGN KEY(session_id) REFERENCES class_sessions(session_id),
  FOREIGN KEY(student_id) REFERENCES students(student_id)
);

CREATE TABLE IF NOT EXISTS movement_log (
  log_id INTEGER PRIMARY KEY AUTOINCREMENT,
  session_id TEXT NOT NULL,
  student_id TEXT NOT NULL,
  time_in DATETIME NOT NULL,
  time_out DATETIME,
  FOREIGN KEY(session_id) REFERENCES class_sessions(session_id),
  FOREIGN KEY(student_id) REFERENCES students(student_id)
);

CREATE INDEX IF NOT EXISTS idx_face_enc_student ON face_encodings(student_id);
CREATE INDEX IF NOT EXISTS idx_student_courses_course ON student_courses(course_name);
CREATE INDEX IF NOT EXISTS idx_att_session ON attendance_records(session_id);
CREATE INDEX IF NOT EXISTS idx_move_session_student ON movement_log(session_id, student_id);
```

---

## 7. Cách triển khai code (kế hoạch chi tiết)

### 7.1 Cài đặt thư viện
```bash
pip install opencv-python numpy face_recognition
pip install mtcnn
```

Nếu dùng RetinaFace hoặc InsightFace, cài theo tài liệu package tương ứng.

### 7.2 Cấu hình (config.py)
Các tham số nên gom vào 1 file:
- `RTSP_URL`
- `DB_PATH`
- `RESIZE_WIDTH`
- `FRAME_SKIP`
- `TOLERANCE`
- `LATE_MINUTES`
- `ABSENCE_THRESHOLD_SEC`
- `MIN_FACE_SIZE`
- `REQUIRED_SAMPLES` (enrollment)
- `SAVE_ALL_SAMPLES` (enrollment)

### 7.3 Cách chạy
Enrollment:
```bash
python enrollment.py --rtsp "<RTSP_URL>" --student_id "SE12345" --name "Nguyen Van A" --samples 20
```

Attendance:
```bash
python attendance.py --rtsp "<RTSP_URL>" --class_name "PRJ301" --late_minutes 10
```

Export report:
```bash
python report.py --session_id "<SESSION_ID>" --out "attendance.csv"
```

---

## 8. Pseudo-code full hệ thống (rất chi tiết)

### 8.1 Data structure dùng trong RAM
```
known_db: dict student_id -> list[embedding_vector]

attendance_state: dict student_id -> {
  first_seen: datetime,
  last_seen: datetime,
  status: "Present" | "Late",
  is_inside: bool,
  last_change_time: datetime,
  last_confidence: float
}
```

### 8.2 database.py
```
FUNCTION init_db(db_path):
  conn = sqlite_connect(db_path)
  execute DDL
  return conn

FUNCTION serialize_embedding(vec_float32):
  return vec_float32.tobytes()

FUNCTION deserialize_embedding(blob):
  return np.frombuffer(blob, dtype=float32)

FUNCTION upsert_student(conn, student_id, name):
  if exists(student_id):
    UPDATE students SET name=name WHERE student_id=student_id
  else:
    INSERT INTO students(student_id, name)

FUNCTION insert_face_encoding(conn, student_id, vec, quality):
  blob = serialize_embedding(vec)
  INSERT INTO face_encodings(student_id, encoding, quality)

FUNCTION load_all_encodings(conn):
  rows = SELECT student_id, encoding FROM face_encodings
  known = empty dict
  for row in rows:
    vec = deserialize_embedding(row.encoding)
    known[row.student_id].append(vec)
  return known

FUNCTION create_session(conn, class_name, start_time, late_minutes):
  sid = uuid4()
  INSERT INTO class_sessions(session_id, class_name, start_time, late_minutes)
  return sid

FUNCTION finalize_session(conn, session_id, end_time):
  UPDATE class_sessions SET end_time=end_time WHERE session_id=session_id

FUNCTION insert_attendance_if_new(conn, session_id, student_id, first_seen, status, confidence):
  if exists attendance_records(session_id, student_id):
    return
  INSERT INTO attendance_records(session_id, student_id, first_seen, last_seen=first_seen, status, confidence)

FUNCTION update_last_seen(conn, session_id, student_id, last_seen, confidence):
  UPDATE attendance_records SET last_seen=last_seen, confidence=confidence
  WHERE session_id=session_id AND student_id=student_id

FUNCTION set_checkout_time(conn, session_id, student_id, checkout_time):
  UPDATE attendance_records SET checkout_time=checkout_time
  WHERE session_id=session_id AND student_id=student_id

FUNCTION open_movement_interval(conn, session_id, student_id, time_in):
  INSERT INTO movement_log(session_id, student_id, time_in, time_out=NULL)

FUNCTION close_last_movement_interval(conn, session_id, student_id, time_out):
  row = SELECT log_id FROM movement_log
        WHERE session_id=session_id AND student_id=student_id AND time_out IS NULL
        ORDER BY log_id DESC LIMIT 1
  if row exists:
    UPDATE movement_log SET time_out=time_out WHERE log_id=row.log_id
```

### 8.3 camera.py
```
FUNCTION connect_rtsp(url):
  cap = cv2.VideoCapture(url)
  if not cap.isOpened():
    raise "Cannot open RTSP stream"
  return cap

FUNCTION read_frame(cap):
  ok, frame = cap.read()
  if not ok:
    return None
  return frame

FUNCTION release_camera(cap):
  cap.release()
```

### 8.4 preprocess.py
```
FUNCTION resize_keep_ratio(frame, width):
  compute new height by ratio
  return cv2.resize(frame, (width, new_height))

FUNCTION preprocess_frame(frame):
  f = resize_keep_ratio(frame, RESIZE_WIDTH)
  rgb = cv2.cvtColor(f, BGR2RGB)

  if USE_HIST_EQ:
    rgb = hist_equalization_on_luma(rgb)   # optional

  if USE_BLUR:
    rgb = gaussian_blur(rgb)

  return rgb
```

### 8.5 detect.py
```
FUNCTION detect_faces(rgb):
  bboxes = DETECTOR.infer(rgb)  # list of (x1, y1, x2, y2)
  bboxes = filter bboxes by min face size
  bboxes = clamp to image boundary
  return bboxes
```

### 8.6 recognize.py
```
FUNCTION encode_faces(rgb, bboxes):
  locations = convert bbox to (top,right,bottom,left)
  embeddings = face_recognition.face_encodings(rgb, locations)
  return embeddings

FUNCTION euclidean(a, b):
  return sqrt(sum((a-b)^2))

FUNCTION match_embedding(query_vec, known_db):
  best_student = None
  best_dist = +INF

  for student_id in known_db:
    for ref_vec in known_db[student_id]:
      dist = euclidean(query_vec, ref_vec)
      if dist < best_dist:
        best_dist = dist
        best_student = student_id

  if best_dist <= TOLERANCE:
    confidence = dist_to_confidence(best_dist)  # optional mapping
    return best_student, best_dist, confidence

  return None, best_dist, 0.0
```

### 8.7 enrollment.py
```
FUNCTION estimate_blur(face_crop):
  gray = to_grayscale(face_crop)
  return variance(Laplacian(gray))

FUNCTION enrollment(student_id, name, rtsp_url):
  conn = init_db(DB_PATH)
  upsert_student(conn, student_id, name)

  cap = connect_rtsp(rtsp_url)

  samples = []
  while len(samples) < REQUIRED_SAMPLES:
    frame = read_frame(cap)
    if frame is None:
      continue

    rgb = preprocess_frame(frame)
    bboxes = detect_faces(rgb)

    if len(bboxes) != 1:
      continue

    box = bboxes[0]
    face_crop = crop(rgb, box)

    if face_crop size < MIN_FACE_SIZE:
      continue

    if estimate_blur(face_crop) < BLUR_THRESHOLD:
      continue

    emb = encode_faces(rgb, [box])[0]
    samples.append(emb)

  filtered = remove_outliers(samples)
  rep = mean(filtered)

  if SAVE_ALL_SAMPLES:
    for e in filtered:
      insert_face_encoding(conn, student_id, e, quality=1.0)
  else:
    insert_face_encoding(conn, student_id, rep, quality=1.0)

  release_camera(cap)
  close conn
```

### 8.8 attendance.py
```
FUNCTION handle_absent(conn, session_id, state):
  now_t = now()
  for student_id in state:
    gap = seconds(now_t - state[student_id].last_seen)
    if state[student_id].is_inside == True and gap >= ABSENCE_THRESHOLD_SEC:
      state[student_id].is_inside = False
      state[student_id].last_change_time = now_t
      close_last_movement_interval(conn, session_id, student_id, time_out=now_t)

FUNCTION attendance(rtsp_url, class_name, late_minutes):
  conn = init_db(DB_PATH)

  start_time = now()
  session_id = create_session(conn, class_name, start_time, late_minutes)

  known_db = load_all_encodings(conn)
  state = empty dict

  cap = connect_rtsp(rtsp_url)
  frame_idx = 0

  while not END_SIGNAL:
    frame = read_frame(cap)
    if frame is None:
      continue

    frame_idx += 1
    if FRAME_SKIP > 1 and frame_idx % FRAME_SKIP != 0:
      continue

    rgb = preprocess_frame(frame)
    bboxes = detect_faces(rgb)

    if len(bboxes) == 0:
      handle_absent(conn, session_id, state)
      continue

    embeddings = encode_faces(rgb, bboxes)

    for emb in embeddings:
      sid, dist, conf = match_embedding(emb, known_db)
      if sid is None:
        continue

      t = now()

      if sid not in state:
        delta_min = minutes(t - start_time)
        status = "Present" if delta_min <= late_minutes else "Late"

        state[sid] = {
          first_seen=t,
          last_seen=t,
          status=status,
          is_inside=True,
          last_change_time=t,
          last_confidence=conf
        }

        insert_attendance_if_new(conn, session_id, sid, t, status, conf)
        open_movement_interval(conn, session_id, sid, time_in=t)

      else:
        state[sid].last_seen = t
        state[sid].last_confidence = conf
        update_last_seen(conn, session_id, sid, t, conf)

        if state[sid].is_inside == False:
          state[sid].is_inside = True
          state[sid].last_change_time = t
          open_movement_interval(conn, session_id, sid, time_in=t)

    handle_absent(conn, session_id, state)

  end_time = now()
  finalize_session(conn, session_id, end_time)

  for sid in state:
    set_checkout_time(conn, session_id, sid, end_time)
    close_last_movement_interval(conn, session_id, sid, time_out=end_time)

  release_camera(cap)
  close conn
  export_report(session_id)
  return session_id
```

---

## 9. Tối ưu hiệu năng, độ chính xác

### 9.1 Tối ưu hiệu năng
- Resize frame xuống 720p hoặc 960px width
- Frame skip 2–3 tùy CPU
- Cache embeddings vào RAM
- Detector dùng GPU nếu có

### 9.2 Tăng độ chính xác
- Enrollment nhiều mẫu, nhiều góc
- Lưu nhiều embeddings mỗi sinh viên
- Tune tolerance theo thực nghiệm
- Lọc khuôn mặt nhỏ, mờ

---

## 10. Kiểm thử, đánh giá

### 10.1 Test case quan trọng
- 1 người, nhiều người
- Ánh sáng mạnh, ánh sáng yếu
- Che khẩu trang một phần
- Đứng xa camera
- Sinh viên ra khỏi lớp 2 phút rồi quay lại

### 10.2 Chỉ số đánh giá
- True Positive Rate: nhận đúng sinh viên
- False Positive Rate: nhận nhầm sinh viên
- FPS: số frame xử lý mỗi giây
- Latency: độ trễ từ frame đến kết quả

---

## 11. Kết luận
Tài liệu mô tả đầy đủ hệ thống:
- RTSP streaming
- Frame capture, preprocess
- Multi-face detection
- Embedding-based recognition
- Attendance logging, late detection
- Movement tracking
- Export report