import numpy as np
import cv2
import logging
from config import COURT_WIDTH, COURT_LENGTH

# -----------------------------
# Logging configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)

class ViewTransformer:
    """
    Transforms image coordinates to real-world court coordinates using perspective transformation.
    Provides methods to transform single points and to update tracks with transformed positions.
    """

    def __init__(self):
        self.court_width = COURT_WIDTH
        self.court_length = COURT_LENGTH

        # Source pixel coordinates from the image
        self.pixel_vertices = np.array([
            [110, 1035],
            [265, 275],
            [910, 260],
            [1640, 915]
        ], dtype=np.float32)

        # Target coordinates (real-world court)
        self.target_vertices = np.array([
            [0, self.court_width],
            [0, 0],
            [self.court_length, 0],
            [self.court_length, self.court_width]
        ], dtype=np.float32)

        # Compute the perspective transform matrix
        self.perspective_transformer = cv2.getPerspectiveTransform(
            self.pixel_vertices,
            self.target_vertices
        )

    def transform_point(self, point):
        """
        Transforms a single point from image space to court space.
        Returns None if invalid or outside court.
        """
        if point is None or len(point) != 2:
            logging.warning(f"Invalid point: {point}")
            return None
        try:
            p = (int(point[0]), int(point[1]))
            if cv2.pointPolygonTest(self.pixel_vertices, p, False) < 0:
                return None
            reshaped_point = np.array(point, dtype=np.float32).reshape(-1, 1, 2)
            transformed = cv2.perspectiveTransform(reshaped_point, self.perspective_transformer)
            return transformed.reshape(-1, 2)
        except Exception as e:
            logging.warning(f"Failed to transform point {point}: {e}")
            return None

    def add_transformed_position_to_tracks(self, tracks):
        """
        Adds court-space positions to tracks safely.
        Skips any missing or invalid positions.
        """
        if not isinstance(tracks, dict):
            logging.warning("Tracks is not a dictionary.")
            return

        for object_name, object_tracks in tracks.items():
            if not isinstance(object_tracks, list):
                logging.warning(f"Tracks for {object_name} is not a list. Skipping.")
                continue

            for frame_num, track in enumerate(object_tracks):
                if not isinstance(track, dict):
                    logging.debug(f"Frame {frame_num} for {object_name} is not a dict. Skipping.")
                    continue

                for track_id, track_info in track.items():
                    try:
                        position = track_info.get('position')
                        if position is None or len(position) != 2:
                            logging.debug(f"Missing or invalid 'position' for {object_name}, frame {frame_num}, track {track_id}")
                            tracks[object_name][frame_num][track_id]['position_transformed'] = None
                            continue

                        transformed_position = self.transform_point(np.array(position))
                        if transformed_position is not None:
                            transformed_position = transformed_position.squeeze().tolist()
                        tracks[object_name][frame_num][track_id]['position_transformed'] = transformed_position
                    except Exception as e:
                        logging.warning(f"Error transforming position for {object_name}, frame {frame_num}, track {track_id}: {e}")
                        tracks[object_name][frame_num][track_id]['position_transformed'] = None
