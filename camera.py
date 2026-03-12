"""
Camera module: connect to RTSP stream or local webcam, read frames.
"""
import cv2


def connect_camera(source):
    """
    Open a video source.

    Parameters
    ----------
    source : str or int
        RTSP URL (e.g. "rtsp://...") or device index (0 for webcam).
        String "0" is auto-converted to int 0.
    """
    if isinstance(source, str) and source.isdigit():
        source = int(source)

    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        raise RuntimeError(f"Cannot open video source: {source}")

    print(f"[Camera] Connected to source: {source}")
    print(
        f"[Camera] Resolution: "
        f"{int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))}x"
        f"{int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))} @ "
        f"{cap.get(cv2.CAP_PROP_FPS):.1f} FPS"
    )
    return cap


def read_frame(cap: cv2.VideoCapture):
    """Read one frame. Returns None on failure."""
    ok, frame = cap.read()
    if not ok or frame is None:
        return None
    return frame


def release_camera(cap: cv2.VideoCapture):
    """Release camera resources."""
    if cap is not None:
        cap.release()
    cv2.destroyAllWindows()
