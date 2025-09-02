import logging
from typing import Dict, Any, Union
from sklearn.cluster import KMeans
import numpy as np
from config import CROP_RATIO

# -----------------------------
# Logging configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)


class TeamAssigner:
    """
    Assigns players to teams based on their uniform colors.
    Uses KMeans clustering to detect dominant player colors and assign
    each player to one of two teams. Caches results for performance.
    """

    def __init__(self):
        self.team_colors: Dict[int, np.ndarray] = {}
        self.player_team_dict: Dict[int, int] = {}
        self.kmeans: Union[KMeans, None] = None

    def _get_clustering_model(self, image: np.ndarray) -> Union[KMeans, None]:
        """
        Returns a KMeans model (2 clusters) for the input image region.
        """
        if image is None or image.size == 0:
            logging.warning("Empty image passed to clustering model")
            return None

        image_2d = image.reshape(-1, 3)
        if len(image_2d) < 2:
            logging.warning("Image too small for clustering")
            return None

        try:
            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
            kmeans.fit(image_2d)
            return kmeans
        except Exception as e:
            logging.warning(f"KMeans clustering failed: {e}")
            return None

    def _get_player_color(self, frame: np.ndarray, bbox: Union[list, tuple]) -> np.ndarray:
        """
        Extracts the dominant color of a player from the top portion of their bounding box.
        """
        try:
            if frame is None or frame.size == 0 or not bbox or len(bbox) != 4:
                logging.warning("Invalid frame or bbox for player color")
                return np.array([0, 0, 0], dtype=float)

            x1, y1, x2, y2 = map(int, bbox)
            image = frame[y1:y2, x1:x2]
            if image.size == 0:
                logging.warning("Empty player bbox image, using fallback color")
                return np.array([0, 0, 0], dtype=float)

            top_half_image = image[: max(1, int(image.shape[0] / CROP_RATIO)), :]
            kmeans = self._get_clustering_model(top_half_image)
            if kmeans is None:
                logging.warning("Clustering returned None, using mean color")
                return image.mean(axis=(0, 1))

            labels = kmeans.labels_.reshape(top_half_image.shape[0], top_half_image.shape[1])
            # Use corner pixels to detect background cluster
            corner_clusters = [labels[0, 0], labels[0, -1], labels[-1, 0], labels[-1, -1]]
            non_player_cluster = max(set(corner_clusters), key=corner_clusters.count)
            player_cluster = 1 - non_player_cluster

            return kmeans.cluster_centers_[player_cluster]
        except Exception as e:
            logging.warning(f"Failed to get player color: {e}")
            return np.array([0, 0, 0], dtype=float)

    def assign_team_color(self, frame: np.ndarray, player_detections: Dict[int, Dict[str, Any]]) -> None:
        """
        Assigns team colors based on detected players in the first frame.
        """
        try:
            if frame is None or frame.size == 0 or not player_detections:
                logging.error("Invalid frame or empty player detections")
                return

            player_colors = []
            for _, player_detection in player_detections.items():
                bbox = player_detection.get("bbox")
                if bbox is None or len(bbox) != 4:
                    logging.warning("Skipping invalid bbox for team assignment")
                    continue
                player_colors.append(self._get_player_color(frame, bbox))

            if len(player_colors) < 2:
                logging.error("Not enough valid players detected to assign team colors")
                return

            kmeans = KMeans(n_clusters=2, init="k-means++", n_init=10)
            kmeans.fit(player_colors)
            self.kmeans = kmeans
            self.team_colors[1] = kmeans.cluster_centers_[0]
            self.team_colors[2] = kmeans.cluster_centers_[1]
            logging.info(f"Team colors assigned: {self.team_colors}")
        except Exception as e:
            logging.error(f"Team color assignment failed: {e}")

    def get_player_team(self, frame: np.ndarray, player_bbox: Union[list, tuple], player_id: int) -> int:
        """
        Returns the team ID (1 or 2) for a player based on their color.
        Caches results to improve performance.
        """
        try:
            if player_id in self.player_team_dict:
                return self.player_team_dict[player_id]

            if frame is None or frame.size == 0 or player_bbox is None or len(player_bbox) != 4:
                logging.warning(f"Invalid input for player {player_id}, returning -1")
                return -1

            player_color = self._get_player_color(frame, player_bbox)
            if self.kmeans is None:
                logging.warning("KMeans model not initialized, returning -1")
                return -1

            team_id = self.kmeans.predict(player_color.reshape(1, -1))[0] + 1

            # Special override example (optional, can remove if unnecessary)
            if player_id == 91:
                team_id = 1

            self.player_team_dict[player_id] = team_id
            return team_id
        except Exception as e:
            logging.warning(f"Failed to get team for player {player_id}: {e}")
            return -1
