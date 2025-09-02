from ultralytics import YOLO
import supervision as sv
import pickle
import os
import numpy as np
import pandas as pd
import cv2
import logging
from utils import get_center_of_bbox, get_bbox_width, get_foot_position
from config import VIDEO_FPS, VIDEO_CODEC

# Configure logging format for info and warnings
logging.basicConfig(level=logging.INFO, format='[%(levelname)s] %(asctime)s - %(message)s')

class Tracker:
    def __init__(self, model_path):
        """
        Initialize the Tracker object.
        - Loads YOLO model from given path.
        - Initializes ByteTrack tracker from supervision library.
        """
        self.model = YOLO(model_path)
        self.tracker = sv.ByteTrack()
        logging.info(f"YOLO model loaded from {model_path}")

    def add_position_to_tracks(self, tracks):
        """
        Compute and add a 'position' key to each track:
        - For ball: use bbox center
        - For players/referees: use foot position
        """
        for obj_name, object_tracks in tracks.items():
            for frame_num, track in enumerate(object_tracks):
                for track_id, track_info in track.items():
                    bbox = track_info.get('bbox')
                    if bbox is None:
                        continue
                    position = get_center_of_bbox(bbox) if obj_name == 'ball' else get_foot_position(bbox)
                    tracks[obj_name][frame_num][track_id]['position'] = position
        logging.info("Positions added to tracks")

    def interpolate_ball_positions(self, ball_positions):
        """
        Interpolates missing ball positions using pandas DataFrame:
        - Linear interpolation
        - Backfill missing data
        """
        try:
            ball_positions_list = [x.get(1, {}).get('bbox', [0,0,0,0]) for x in ball_positions]
            df_ball = pd.DataFrame(ball_positions_list, columns=['x1','y1','x2','y2'])
            df_ball.interpolate(inplace=True)
            df_ball.fillna(method='bfill', inplace=True)
            interpolated = [{1: {"bbox": x}} for x in df_ball.to_numpy().tolist()]
            logging.info("Ball positions interpolated")
            return interpolated
        except Exception as e:
            logging.warning(f"Failed to interpolate ball positions: {e}")
            return ball_positions

    def detect_frames(self, frames):
        """
        Detect objects in frames using YOLO model in batches:
        - Uses small batch size for memory efficiency
        - Logs failures per batch without stopping processing
        """
        batch_size = 20
        detections = []
        for i in range(0, len(frames), batch_size):
            try:
                batch_det = self.model.predict(frames[i:i+batch_size], conf=0.1)
                detections += batch_det
            except Exception as e:
                logging.warning(f"Detection failed for frames {i}-{i+batch_size}: {e}")
        logging.info(f"Detected objects in {len(frames)} frames")
        return detections

    def get_object_tracks(self, frames, read_from_stub=False, stub_path=None):
        """
        Main function to generate tracks for players, referees, and ball.
        - Optionally reads from a saved stub (pickle file) to avoid re-detection.
        - Converts GoalKeeper class to player for tracking purposes.
        - Tracks players/referees using ByteTrack.
        - Ball is tracked separately using YOLO detections.
        """
        if read_from_stub and stub_path and os.path.exists(stub_path):
            with open(stub_path,'rb') as f:
                tracks = pickle.load(f)
            logging.info(f"Loaded tracks from stub: {stub_path}")
            return tracks

        detections = self.detect_frames(frames)
        tracks = {"players": [], "referees": [], "ball": []}

        for frame_num, detection in enumerate(detections):
            try:
                cls_names = detection.names
                cls_names_inv = {v:k for k,v in cls_names.items()}
                detection_supervision = sv.Detections.from_ultralytics(detection)

                # Convert GoalKeeper to player
                for i, class_id in enumerate(detection_supervision.class_id):
                    if cls_names[class_id] == "goalkeeper":
                        detection_supervision.class_id[i] = cls_names_inv["player"]

                detection_with_tracks = self.tracker.update_with_detections(detection_supervision)

                # Initialize empty dictionaries for this frame
                tracks["players"].append({})
                tracks["referees"].append({})
                tracks["ball"].append({})

                # Process tracked players/referees
                for det in detection_with_tracks:
                    bbox = det[0].tolist()
                    cls_id = det[3]
                    track_id = det[4]
                    if cls_id == cls_names_inv.get('player'):
                        tracks["players"][frame_num][track_id] = {"bbox": bbox}
                    elif cls_id == cls_names_inv.get('referee'):
                        tracks["referees"][frame_num][track_id] = {"bbox": bbox}

                # Process ball detections separately
                for det in detection_supervision:
                    bbox = det[0].tolist()
                    cls_id = det[3]
                    if cls_id == cls_names_inv.get('ball'):
                        tracks["ball"][frame_num][1] = {"bbox": bbox}

            except Exception as e:
                logging.warning(f"Failed to process frame {frame_num}: {e}")

        # Save tracks to stub if requested
        if stub_path:
            try:
                with open(stub_path,'wb') as f:
                    pickle.dump(tracks, f)
                logging.info(f"Tracks saved to stub: {stub_path}")
            except Exception as e:
                logging.warning(f"Failed to save stub: {e}")

        return tracks

    def draw_ellipse(self, frame, bbox, color, track_id=None):
        """
        Draw an ellipse for player/referee and optional ID rectangle.
        - Ellipse represents the feet area.
        - Rectangle with track_id displays player number.
        """
        try:
            y2 = int(bbox[3])
            x_center, _ = get_center_of_bbox(bbox)
            width = get_bbox_width(bbox)
            cv2.ellipse(frame, (x_center, y2), (int(width), int(0.35*width)), 0, -45, 235, color, 2)
            
            if track_id is not None:
                rect_w, rect_h = 40, 20
                x1, x2 = x_center - rect_w//2, x_center + rect_w//2
                y1, y2_rect = (y2 - rect_h//2)+15, (y2 + rect_h//2)+15
                cv2.rectangle(frame, (x1, y1), (x2, y2_rect), color, cv2.FILLED)
                x1_text = x1 + 12 - (10 if track_id>99 else 0)
                cv2.putText(frame, f"{track_id}", (x1_text, y1+15), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0,0,0), 2)
        except Exception as e:
            logging.warning(f"Failed to draw ellipse: {e}")
        return frame

    def draw_triangle(self, frame, bbox, color):
        """
        Draw a triangle to indicate the ball or player possession.
        - Positioned above the object (bbox top)
        - Always drawn if bbox exists, ignoring NaNs
        """
        try:
            x, y_center = get_center_of_bbox(bbox)
            y = int(bbox[1])
            pts = np.array([[x,y], [x-10, y-20], [x+10, y-20]])
            cv2.drawContours(frame, [pts], 0, color, cv2.FILLED)
            cv2.drawContours(frame, [pts], 0, (0,0,0), 2)
        except Exception as e:
            logging.warning(f"Failed to draw triangle: {e}")
        return frame

    def draw_team_ball_control(self, frame, frame_num, team_ball_control):
        """
        Draw a semi-transparent rectangle showing the ball control percentages.
        """
        try:
            overlay = frame.copy()
            cv2.rectangle(overlay, (1350, 850), (1900,970), (255,255,255), -1)
            cv2.addWeighted(overlay, 0.4, frame, 0.6, 0, frame)
            team_ball = team_ball_control[:frame_num+1]
            team1_frames = np.sum(team_ball==1)
            team2_frames = np.sum(team_ball==2)
            total = team1_frames + team2_frames if (team1_frames + team2_frames)>0 else 1
            cv2.putText(frame, f"Team 1 Ball Control: {team1_frames/total*100:.2f}%", (1400,900),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
            cv2.putText(frame, f"Team 2 Ball Control: {team2_frames/total*100:.2f}%", (1400,950),
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0,0,0), 3)
        except Exception as e:
            logging.warning(f"Failed to draw team ball control: {e}")
        return frame

    def draw_annotations(self, video_frames, tracks, team_ball_control):
        """
        Draw all annotations for each frame:

        - Players with ellipse + ID rectangle
        - Players with ball have a triangle on top
        - Referees with ellipse
        - Ball represented as a triangle
        - Team ball control overlay
        - Robust: ignores missing bboxes or NaNs, logs warnings but continues
        """

        output_frames = []

        for frame_num, frame in enumerate(video_frames):
            try:
                frame = frame.copy()
                player_dict = tracks.get("players", [])[frame_num]
                ball_dict = tracks.get("ball", [])[frame_num]
                referee_dict = tracks.get("referees", [])[frame_num]

                # --- Draw Players ---
                for track_id, player in player_dict.items():
                    bbox = player.get("bbox", [0,0,0,0])
                    color = player.get("team_color", (0,0,255))
                    
                    # Draw ellipse + ID
                    frame = self.draw_ellipse(frame, bbox, color, track_id)

                    # Draw triangle if player has the ball
                    if player.get('has_ball', False):
                        frame = self.draw_triangle(frame, bbox, (0,0,255))

                # --- Draw Referees ---
                for _, referee in referee_dict.items():
                    bbox = referee.get("bbox", [0,0,0,0])
                    frame = self.draw_ellipse(frame, bbox, (0,255,255))

                # --- Draw Ball ---
                for _, ball in ball_dict.items():
                    bbox = ball.get("bbox", [0,0,0,0])
                    frame = self.draw_triangle(frame, bbox, (0,255,0))

                # --- Draw Team Ball Control Overlay ---
                frame = self.draw_team_ball_control(frame, frame_num, team_ball_control)

                logging.info(f"Annotations drawn for frame {frame_num}")

            except Exception as e:
                logging.warning(f"Failed to draw annotations for frame {frame_num}: {e}")

            output_frames.append(frame)

        logging.info("All annotations drawn on video frames.")
        return output_frames
