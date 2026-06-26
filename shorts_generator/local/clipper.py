"""Local clipping: ffmpeg subclip + OpenCV face-aware or shot-aware vertical crop.

Two stages per highlight:
  1. Cut the source video to [start, end] with ffmpeg (re-encoded, audio kept).
  2. Reframe the cut to the target aspect ratio.

Crop modes:
  * face (default) — Haar cascade face tracking with smoothing.
  * shot           — Detect shot boundaries, then for each shot find the
                     maximum action area and lock the crop there until the
                     next shot change. Prevents mid-shot drift.
"""
import os
import subprocess
from typing import Dict, List, Optional, Tuple

from ..config import LOCAL_OUTPUT_DIR


def _ratio(aspect_ratio: str) -> float:
    """Parse '9:16' → 9/16, '1:1' → 1.0."""
    try:
        w, h = aspect_ratio.split(":")
        return float(w) / float(h)
    except (ValueError, ZeroDivisionError):
        return 9.0 / 16.0


def _cut_subclip(source_path: str, start: float, end: float, out_path: str) -> str:
    """ffmpeg -ss start -to end → re-encoded mp4 with audio."""
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", source_path,
        "-ss", f"{start:.3f}",
        "-to", f"{end:.3f}",
        "-c:v", "libx264", "-preset", "fast", "-crf", "20",
        "-c:a", "aac", "-b:a", "128k",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    return out_path


def _reframe_vertical(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Crop the cut clip to the target aspect ratio, tracking faces if possible."""
    try:
        import cv2  # type: ignore
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0

    # Compute the largest crop that fits inside the frame at the target ratio.
    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))

    last_center: Optional[Tuple[int, int]] = None
    smoothing = 0.15  # how aggressively to chase a new face position
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40))
        if len(faces) > 0:
            # Pick the largest face — usually the speaker.
            x, y, w, h = max(faces, key=lambda f: f[2] * f[3])
            cx = x + w // 2
            cy = y + h // 2
            if last_center is None:
                last_center = (cx, cy)
            else:
                lx, ly = last_center
                last_center = (
                    int(lx + (cx - lx) * smoothing),
                    int(ly + (cy - ly) * smoothing),
                )
        if last_center is None:
            last_center = (src_w // 2, src_h // 2)

        cx, cy = last_center
        x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
        y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        writer.write(cropped)

    cap.release()
    writer.release()

    # Mux audio from the cut clip back onto the silent reframed video.
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(silent_path)
    return out_path


def _detect_shot_boundaries(
    cap,
    threshold: float = 30.0,
    min_shot_frames: int = 15,
) -> List[int]:
    """Return frame indices where shot changes occur.

    Uses frame-difference histogram comparison. A shot boundary is declared
    when the mean absolute difference between consecutive grayscale frames
    exceeds `threshold` and at least `min_shot_frames` have passed since the
    last boundary.
    """
    import cv2

    prev_gray = None
    boundaries: List[int] = []
    frame_idx = 0
    last_boundary = -min_shot_frames

    while True:
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            diff = cv2.absdiff(gray, prev_gray)
            mean_diff = float(diff.mean())
            if mean_diff > threshold and (frame_idx - last_boundary) >= min_shot_frames:
                boundaries.append(frame_idx)
                last_boundary = frame_idx
        prev_gray = gray
        frame_idx += 1

    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
    return boundaries


def _find_action_center(
    cap,
    start_frame: int,
    end_frame: int,
    crop_w: int,
    crop_h: int,
) -> Tuple[int, int]:
    """For a shot range, compute the most active region and return its center.

    Motion is estimated via optical-flow magnitude averaged over the shot.
    The crop window is positioned so its center sits on the highest-motion area.
    Falls back to frame center if motion is negligible.
    """
    import cv2
    import numpy as np

    cap.set(cv2.CAP_PROP_POS_FRAMES, start_frame)
    prev_gray = None
    motion_map = None
    frame_count = 0

    for _ in range(start_frame, end_frame):
        ret, frame = cap.read()
        if not ret:
            break
        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        if prev_gray is not None:
            flow = cv2.calcOpticalFlowFarneback(
                prev_gray, gray, None,
                pyr_scale=0.5, levels=3, winsize=15,
                iterations=3, poly_n=5, poly_sigma=1.2, flags=0,
            )
            mag = np.sqrt(flow[..., 0] ** 2 + flow[..., 1] ** 2)
            if motion_map is None:
                motion_map = mag
            else:
                motion_map += mag
            frame_count += 1
        prev_gray = gray

    if motion_map is None or frame_count == 0:
        src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        return src_w // 2, src_h // 2

    # Average motion and find the best crop position
    avg_motion = motion_map / frame_count
    # Downsample motion map to a grid for speed
    h, w = avg_motion.shape
    grid_h = max(1, h // crop_h)
    grid_w = max(1, w // crop_w)
    pooled = cv2.resize(avg_motion, (grid_w, grid_h), interpolation=cv2.INTER_AREA)

    # Find the grid cell with max motion
    max_y, max_x = np.unravel_index(np.argmax(pooled), pooled.shape)
    cx = int((max_x + 0.5) * crop_w)
    cy = int((max_y + 0.5) * crop_h)

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    cx = max(crop_w // 2, min(src_w - crop_w // 2, cx))
    cy = max(crop_h // 2, min(src_h - crop_h // 2, cy))
    return cx, cy


def _reframe_shot_aware(in_path: str, out_path: str, aspect_ratio: str) -> str:
    """Crop the cut clip to the target aspect ratio, locking per-shot centers.

    Steps:
      1. Detect shot boundaries via frame differencing.
      2. For each shot, compute the maximum action area via optical flow.
      3. Lock the crop center for that shot; no smoothing between shots.
    """
    try:
        import cv2  # type: ignore
        import numpy as np
    except ImportError as e:
        raise RuntimeError(
            "opencv-python is required for --mode local. Install it with:\n"
            "    pip install -r requirements-local.txt"
        ) from e

    target_ratio = _ratio(aspect_ratio)
    cap = cv2.VideoCapture(in_path)
    if not cap.isOpened():
        raise RuntimeError(f"could not open {in_path}")

    src_w = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    src_h = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS) or 30.0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    if target_ratio < src_w / src_h:
        crop_h = src_h
        crop_w = int(crop_h * target_ratio)
    else:
        crop_w = src_w
        crop_h = int(crop_w / target_ratio)
    crop_w = max(2, crop_w - (crop_w % 2))
    crop_h = max(2, crop_h - (crop_h % 2))

    # Detect shot boundaries
    boundaries = _detect_shot_boundaries(cap)
    shot_ranges = []
    prev = 0
    for b in boundaries:
        shot_ranges.append((prev, b))
        prev = b
    shot_ranges.append((prev, total_frames))

    # Pre-compute action center for each shot
    shot_centers: List[Tuple[int, int]] = []
    for start_f, end_f in shot_ranges:
        cx, cy = _find_action_center(cap, start_f, end_f, crop_w, crop_h)
        shot_centers.append((cx, cy))

    # Render with locked per-shot centers
    silent_path = out_path + ".silent.mp4"
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(silent_path, fourcc, fps, (crop_w, crop_h))

    current_shot = 0
    frame_idx = 0
    cap.set(cv2.CAP_PROP_POS_FRAMES, 0)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        # Advance shot if we've crossed a boundary
        while current_shot < len(shot_ranges) - 1 and frame_idx >= shot_ranges[current_shot][1]:
            current_shot += 1

        cx, cy = shot_centers[current_shot]
        x0 = max(0, min(src_w - crop_w, cx - crop_w // 2))
        y0 = max(0, min(src_h - crop_h, cy - crop_h // 2))
        cropped = frame[y0:y0 + crop_h, x0:x0 + crop_w]
        writer.write(cropped)
        frame_idx += 1

    cap.release()
    writer.release()

    # Mux audio back
    cmd = [
        "ffmpeg", "-y", "-loglevel", "error",
        "-i", silent_path,
        "-i", in_path,
        "-c:v", "copy",
        "-c:a", "aac", "-b:a", "128k",
        "-map", "0:v:0", "-map", "1:a:0?",
        "-shortest",
        out_path,
    ]
    subprocess.run(cmd, check=True)
    os.remove(silent_path)
    return out_path


def crop_clip_local(
    source_path: str,
    start_time: float,
    end_time: float,
    aspect_ratio: str,
    out_path: str,
    crop_mode: str = "face",
) -> str:
    """Cut + reframe one highlight, returning the local mp4 path.

    Args:
        crop_mode: "face" (default) or "shot" (shot-aware action centering).
    """
    cut_path = out_path + ".cut.mp4"
    try:
        _cut_subclip(source_path, start_time, end_time, cut_path)
        if crop_mode == "shot":
            _reframe_shot_aware(cut_path, out_path, aspect_ratio)
        else:
            _reframe_vertical(cut_path, out_path, aspect_ratio)
    finally:
        if os.path.exists(cut_path):
            os.remove(cut_path)
    return out_path


def crop_highlights_local(
    source_path: str,
    highlights: List[Dict],
    aspect_ratio: str = "9:16",
    out_dir: Optional[str] = None,
    crop_mode: str = "face",
) -> List[Dict]:
    out_dir = out_dir or LOCAL_OUTPUT_DIR
    os.makedirs(out_dir, exist_ok=True)
    results: List[Dict] = []
    for i, h in enumerate(highlights, 1):
        out_path = os.path.join(out_dir, f"short_{i:02d}.mp4")
        print(f"[clip/local] {i}/{len(highlights)}: {h.get('title', '(untitled)')} [{crop_mode} mode]", flush=True)
        try:
            crop_clip_local(
                source_path,
                float(h["start_time"]),
                float(h["end_time"]),
                aspect_ratio,
                out_path,
                crop_mode=crop_mode,
            )
            results.append({**h, "clip_url": out_path})
        except Exception as e:
            print(f"[clip/local] {i} failed: {e}", flush=True)
            results.append({**h, "clip_url": None, "error": str(e)})
    return results
