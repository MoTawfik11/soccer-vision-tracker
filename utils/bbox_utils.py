import logging
from typing import Tuple

# -----------------------------
# Logging configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)


def get_center_of_bbox(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Returns the center point (x, y) of a bounding box.

    Args:
        bbox (Tuple[int, int, int, int]): Bounding box in format (x1, y1, x2, y2).

    Returns:
        Tuple[int, int]: Center coordinates (x_center, y_center).
    """
    try:
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int((y1 + y2) / 2)
    except Exception as e:
        logging.warning(f"get_center_of_bbox failed for bbox {bbox}: {e}")
        return 0, 0


def get_bbox_width(bbox: Tuple[int, int, int, int]) -> int:
    """
    Returns the width of a bounding box.

    Args:
        bbox (Tuple[int, int, int, int]): Bounding box in format (x1, y1, x2, y2).

    Returns:
        int: Width of the bounding box.
    """
    try:
        return bbox[2] - bbox[0]
    except Exception as e:
        logging.warning(f"get_bbox_width failed for bbox {bbox}: {e}")
        return 0


def measure_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
    """
    Returns the Euclidean distance between two points.

    Args:
        p1 (Tuple[int, int]): First point (x, y).
        p2 (Tuple[int, int]): Second point (x, y).

    Returns:
        float: Distance between points.
    """
    try:
        return ((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2) ** 0.5
    except Exception as e:
        logging.warning(f"measure_distance failed between points {p1} and {p2}: {e}")
        return float('inf')


def measure_xy_distance(p1: Tuple[int, int], p2: Tuple[int, int]) -> Tuple[int, int]:
    """
    Returns the horizontal (dx) and vertical (dy) distance between two points.

    Args:
        p1 (Tuple[int, int]): First point (x, y).
        p2 (Tuple[int, int]): Second point (x, y).

    Returns:
        Tuple[int, int]: Differences (dx, dy) where dx = x1 - x2, dy = y1 - y2.
    """
    try:
        return p1[0] - p2[0], p1[1] - p2[1]
    except Exception as e:
        logging.warning(f"measure_xy_distance failed between points {p1} and {p2}: {e}")
        return 0, 0


def get_foot_position(bbox: Tuple[int, int, int, int]) -> Tuple[int, int]:
    """
    Returns the approximate foot position of a player in a bounding box.
    Assumes feet are at the bottom center of the box.

    Args:
        bbox (Tuple[int, int, int, int]): Bounding box in format (x1, y1, x2, y2).

    Returns:
        Tuple[int, int]: Coordinates of the foot position (x_center, y_bottom).
    """
    try:
        x1, y1, x2, y2 = bbox
        return int((x1 + x2) / 2), int(y2)
    except Exception as e:
        logging.warning(f"get_foot_position failed for bbox {bbox}: {e}")
        return 0, 0
