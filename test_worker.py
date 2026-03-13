import sys
import traceback
from app import _attendance_worker

if __name__ == "__main__":
    print("Testing attendance worker...")
    try:
        # Pass a mock video or "0" for webcam. If NO webcam exists, it throws earlier.
        # But let's assume "0" opens.
        _attendance_worker("0", "CPV391", 10)
    except Exception as e:
        traceback.print_exc()
