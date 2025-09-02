# config.py
# -----------------------------
# Video settings
VIDEO_FPS = 24
VIDEO_CODEC = 'XVID'

# YOLO settings
BATCH_SIZE = 32
CONF_THRESHOLD = 0.3  # Confidence threshold for detection

# Player/Team settings
CROP_RATIO = 1.7  # Top-half cropping ratio for player color extraction
MAX_PLAYER_BALL_DISTANCE = 70  # Maximum distance to assign ball to player

# Ellipse and rectangle drawing settings
ELLIPSE_RATIO = 0.35  # For ellipse axes calculation
RECT_WIDTH = 40
RECT_HEIGHT = 20

# Court settings for perspective transform
COURT_WIDTH = 68
COURT_LENGTH = 23.32

PIXEL_VERTICES = [
    [110, 1035],
    [265, 275],
    [910, 260],
    [1640, 915]
]

TARGET_VERTICES = [
    [0, COURT_WIDTH],
    [0, 0],
    [COURT_LENGTH, 0],
    [COURT_LENGTH, COURT_WIDTH]
]
