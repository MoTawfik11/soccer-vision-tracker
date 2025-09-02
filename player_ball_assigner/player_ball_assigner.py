import logging
from typing import Dict, Union
from utils.bbox_utils import get_center_of_bbox, measure_distance
from config import MAX_PLAYER_BALL_DISTANCE

# -----------------------------
# Logging configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)


class PlayerBallAssigner:
    """
    Assigns the ball to the closest player based on distance to player feet.

    Uses a maximum distance threshold to determine if the ball is close enough
    to any player.
    """

    def __init__(self):
        self.max_player_ball_distance: float = MAX_PLAYER_BALL_DISTANCE

    def assign_ball_to_player(
        self, 
        players: Dict[int, Dict[str, Union[list, tuple]]], 
        ball_bbox: Union[list, tuple]
    ) -> int:
        """
        Assigns the ball to the closest player within max distance.

        For each player, calculates the distance from the ball to both left and right feet
        (bottom corners of the bounding box). Assigns the ball to the closest player
        within the defined maximum distance. Returns -1 if no player is close enough.

        Args:
            players (dict): Dictionary of players where each key is player_id and value is a dict containing 'bbox'.
            ball_bbox (list or tuple): Bounding box of the ball [x1, y1, x2, y2].

        Returns:
            int: player_id of the player closest to the ball, or -1 if no player is close enough.
        """
        if not players or ball_bbox is None or len(ball_bbox) != 4:
            logging.warning("Invalid players dictionary or ball_bbox")
            return -1

        try:
            ball_position = get_center_of_bbox(ball_bbox)
        except Exception as e:
            logging.warning(f"Failed to get ball position: {e}")
            return -1

        minimum_distance = float('inf')
        assigned_player = -1

        for player_id, player in players.items():
            try:
                player_bbox = player.get('bbox')
                if player_bbox is None or len(player_bbox) != 4:
                    logging.warning(f"Invalid bbox for player {player_id}, skipping")
                    continue

                # Measure distance to left and right feet (bottom corners of bbox)
                distance_left = measure_distance((player_bbox[0], player_bbox[3]), ball_position)
                distance_right = measure_distance((player_bbox[2], player_bbox[3]), ball_position)
                distance = min(distance_left, distance_right)

                # Assign the ball if within max distance and closer than previous
                if distance < self.max_player_ball_distance and distance < minimum_distance:
                    minimum_distance = distance
                    assigned_player = player_id

            except Exception as e:
                logging.warning(f"Failed to process player {player_id}: {e}")
                continue

        if assigned_player == -1:
            logging.info("No player is close enough to the ball in this frame")

        return assigned_player
