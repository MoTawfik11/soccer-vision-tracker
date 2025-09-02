import cv2
import logging
from typing import List, Optional
import numpy as np

# -----------------------------
# Logging configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)


def read_video(video_path: str) -> List[np.ndarray]:
    """
    Reads a video from the given path and returns a list of frames.

    Args:
        video_path (str): Path to the video file.

    Returns:
        List[np.ndarray]: List of frames read from the video.
    """
    frames: List[np.ndarray] = []
    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            logging.error(f"Failed to open video: {video_path}")
            return frames

        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)

        cap.release()
        logging.info(f"Read {len(frames)} frames from video: {video_path}")

    except Exception as e:
        logging.error(f"Error reading video {video_path}: {e}")

    return frames


def save_video(output_video_frames: List[np.ndarray], output_video_path: str, fps: Optional[int] = 24):
    """
    Saves a list of frames to a video file.

    Args:
        output_video_frames (List[np.ndarray]): List of frames to save.
        output_video_path (str): Path to save the output video.
        fps (int, optional): Frames per second. Defaults to 24.
    """
    if not output_video_frames:
        logging.warning("No frames to save.")
        return

    try:
        height, width = output_video_frames[0].shape[:2]
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter(output_video_path, fourcc, fps, (width, height))

        for frame in output_video_frames:
            out.write(frame)

        out.release()
        logging.info(f"Video saved successfully at {fps} FPS: {output_video_path}")

    except Exception as e:
        logging.error(f"Error saving video to {output_video_path}: {e}")
