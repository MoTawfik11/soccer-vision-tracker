import warnings
import logging
import numpy as np
from typing import List, Tuple, Optional

from utils.video_utils import read_video, save_video
from trackers.tracker import Tracker
from team_assigner.team_assigner import TeamAssigner
from player_ball_assigner.player_ball_assigner import PlayerBallAssigner
from view_transformer.view_transformer import ViewTransformer
from config import VIDEO_FPS, VIDEO_CODEC

# -----------------------------
# Ignore future warnings
# -----------------------------
warnings.filterwarnings("ignore", category=FutureWarning)

# -----------------------------
# Logging configuration
# -----------------------------
logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(asctime)s - %(message)s'
)

# -----------------------------
# Video Loading
# -----------------------------
def load_video(video_path: str) -> List[np.ndarray]:
    try:
        frames = read_video(video_path)
        if not frames:
            logging.error(f"No frames loaded from {video_path}. Exiting pipeline.")
            return []
        logging.info(f"Loaded {len(frames)} frames from {video_path}.")
        return frames
    except Exception as e:
        logging.error(f"Failed to read video {video_path}: {e}")
        return []

# -----------------------------
# Tracker Initialization
# -----------------------------
def initialize_tracker(model_path: str, frames: List[np.ndarray], stub_path: str = None) -> Tuple[Optional[Tracker], dict]:
    if not frames:
        return None, {}
    try:
        tracker = Tracker(model_path)
        tracks = tracker.get_object_tracks(
            frames,
            read_from_stub=True if stub_path else False,
            stub_path=stub_path
        )
        if not tracks or 'players' not in tracks or 'ball' not in tracks:
            logging.warning("Tracks are incomplete or missing. Continuing pipeline with caution.")
        tracker.add_position_to_tracks(tracks)
        logging.info("Tracker initialized and object tracks prepared.")
        return tracker, tracks
    except Exception as e:
        logging.error(f"Failed to initialize tracker: {e}")
        return None, {}

# -----------------------------
# View Transformation
# -----------------------------
def apply_view_transformation(tracks: dict) -> None:
    if not tracks:
        logging.warning("No tracks available for view transformation.")
        return
    try:
        transformer = ViewTransformer()
        transformer.add_transformed_position_to_tracks(tracks)
        logging.info("Applied view transformation to tracks.")
    except Exception as e:
        logging.warning(f"Failed to apply view transformation: {e}")

# -----------------------------
# Ball Position Interpolation
# -----------------------------
def interpolate_ball_positions(tracker: Tracker, tracks: dict) -> None:
    if not tracker or 'ball' not in tracks:
        logging.warning("Cannot interpolate ball positions: tracker or ball tracks missing.")
        return
    try:
        tracks["ball"] = tracker.interpolate_ball_positions(tracks["ball"])
        logging.info("Interpolated missing ball positions.")
    except Exception as e:
        logging.warning(f"Failed to interpolate ball positions: {e}")

# -----------------------------
# Assign Player Teams
# -----------------------------
def assign_player_teams(video_frames: List[np.ndarray], tracks: dict) -> Optional[TeamAssigner]:
    if not tracks or 'players' not in tracks or not video_frames:
        logging.warning("Cannot assign teams: tracks or video frames missing.")
        return None
    try:
        team_assigner = TeamAssigner()
        team_assigner.assign_team_color(video_frames[0], tracks['players'][0])
        num_frames = len(video_frames)
        for frame_num, player_track in enumerate(tracks['players']):
            if frame_num >= num_frames:
                break
            for player_id, track in player_track.items():
                team = team_assigner.get_player_team(
                    video_frames[frame_num], track.get('bbox', None), player_id
                )
                track['team'] = team
                track['team_color'] = team_assigner.team_colors.get(team, (0, 0, 0))
        logging.info("Player teams assigned.")
        return team_assigner
    except Exception as e:
        logging.warning(f"Failed to assign player teams: {e}")
        return None

# -----------------------------
# Assign Ball Possession
# -----------------------------
def assign_ball_possession(tracks: dict) -> np.ndarray:
    if not tracks or 'players' not in tracks or 'ball' not in tracks:
        logging.warning("Cannot assign ball possession: tracks missing.")
        return np.array([])
    try:
        player_assigner = PlayerBallAssigner()
        team_ball_control: List[int] = []
        last_team: int = -1
        num_frames = len(tracks['players'])

        for frame_num, player_track in enumerate(tracks['players']):
            if frame_num >= num_frames or frame_num >= len(tracks['ball']):
                break

            ball_frame = tracks['ball'][frame_num]
            if not isinstance(ball_frame, dict) or 1 not in ball_frame or \
               not ball_frame[1].get('bbox') or np.any(np.isnan(ball_frame[1]['bbox'])):
                logging.debug(f"Ball missing or invalid at frame {frame_num}. Using last team: {last_team}")
                team_ball_control.append(last_team)
                continue

            ball_bbox = ball_frame[1]['bbox']
            assigned_player = player_assigner.assign_ball_to_player(player_track, ball_bbox)

            if assigned_player != -1:
                player_track[assigned_player]['has_ball'] = True
                team_ball_control.append(player_track[assigned_player]['team'])
                last_team = player_track[assigned_player]['team']
            else:
                team_ball_control.append(team_ball_control[-1] if team_ball_control else -1)

        team_ball_control_array = np.array(team_ball_control)
        logging.info("Ball possession assigned per frame.")
        return team_ball_control_array
    except Exception as e:
        logging.warning(f"Failed to assign ball possession: {e}")
        return np.array([])
    
# -----------------------------
# Draw and Save Video
# -----------------------------
def draw_and_save(
    tracker: Tracker,
    video_frames: List[np.ndarray],
    tracks: dict,
    team_ball_control: np.ndarray,
    output_path: str
) -> None:
    if not tracker or not video_frames or not tracks or team_ball_control.size == 0:
        logging.warning("Cannot draw or save video: missing data.")
        return
    try:
        output_frames = tracker.draw_annotations(video_frames, tracks, team_ball_control)
        logging.info("Annotations drawn on video frames.")
        save_video(output_frames, output_path, fps=VIDEO_FPS)
        logging.info(f"Output video saved to '{output_path}'.")
    except Exception as e:
        logging.warning(f"Failed to draw/save video: {e}")

# -----------------------------
# Main Pipeline
# -----------------------------
def main():
    try:
        video_frames = load_video('input_videos/CV_Task.mp4')
        if not video_frames:
            return

        tracker, tracks = initialize_tracker(
            "models/train/weights/best.pt",
            video_frames,
            stub_path='stubs/track_stubs.pkl'
        )
        if not tracker or not tracks:
            return

        apply_view_transformation(tracks)
        interpolate_ball_positions(tracker, tracks)
        assign_player_teams(video_frames, tracks)
        team_ball_control = assign_ball_possession(tracks)
        draw_and_save(tracker, video_frames, tracks, team_ball_control, 'output_videos/output_video.avi')
    except Exception as e:
        logging.error(f"Pipeline failed: {e}")

if __name__ == '__main__':
    main()
