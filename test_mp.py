import time
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

model_path = 'models/face_detection_full_range.tflite'

BaseOptions = mp.tasks.BaseOptions
FaceDetector = mp.tasks.vision.FaceDetector
FaceDetectorOptions = mp.tasks.vision.FaceDetectorOptions
VisionRunningMode = mp.tasks.vision.RunningMode

options = FaceDetectorOptions(
    base_options=BaseOptions(model_asset_path=model_path),
    running_mode=VisionRunningMode.IMAGE)

with FaceDetector.create_from_options(options) as detector:
    img = np.zeros((800, 800, 3), dtype=np.uint8)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=img)
    
    # Warmup
    for _ in range(5):
        detector.detect(mp_image)
        
    t0 = time.time()
    n = 30
    for _ in range(n):
        detector.detect(mp_image)
    t1 = time.time()
    
    print(f"Time per frame: {(t1-t0)/n*1000:.2f} ms")
    print(f"FPS: {n/(t1-t0):.2f}")
