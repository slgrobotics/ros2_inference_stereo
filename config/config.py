# config.py

class Camera:    # parameters related to camera
    # =====================================================
    # Camera configuration parameters.
    #
    # Defines resolution, frame rate, and device identifiers for left and right
    # CSI cameras. These settings are used across capture, calibration, and
    # streaming components to ensure consistent image dimensions and timing.
    #
    # Note: to use full FOV use RAW_*=1640x1232  (or full IMX219 res 3280x2464)
    #       run ../tests/print_sensor_modes.py and look for "crop_limits: (0, 0, 3280, 2464)"
    # =====================================================

    RAW_WIDTH = 1640  # what IMX219 camera mode is used during capture
    RAW_HEIGHT = 1232
    SCALE_BY = 2   # 1 = no scaling, 2 = half...
    WIDTH = RAW_WIDTH // SCALE_BY     # what is passed after capture to be processed
    HEIGHT = RAW_HEIGHT // SCALE_BY
    FPS = 30
    LEFT = 0
    RIGHT = 1
    CAMERA_STEREO_BASE = 0.060  # meters, Waveshare stereo camera for Jetson Nano

class Stereo:    # parameters related to stereo algorithms
    # =====================================================
    # Stereo vision and depth estimation parameters.
    #
    # Contains checkerboard geometry used for calibration as well as filtering
    # thresholds for disparity and 3D point extraction. These values directly
    # affect calibration accuracy and depth computation quality.
    #
    # Note:
    # CHESSBOARD_SIZE and SQUARE_SIZE must match your physical calibration board.
    # =====================================================

    CHESSBOARD_SIZE = (8, 6)   # inner corners (across, down)
    #SQUARE_SIZE = 0.02821     # meters, small board
    SQUARE_SIZE = 0.06290      # meters, large board
    # Depth filtering (reasonable defaults):
    MIN_VALID_DISP = 1.0
    MAX_RANGE_M = 5.0

class Calib:     # parameters used during calibration
    # =====================================================
    # Dataset capture and calibration workflow parameters.
    #
    # Defines storage locations, capture timing, and dataset management options
    # used during stereo image acquisition and calibration. Includes settings for
    # automatic cleanup of invalid pairs and output file naming.
    #
    # Ensures consistent dataset structure and reproducible calibration runs.
    # =====================================================

    PAIR_DIR = f"stereo_pairs_{Camera.WIDTH}x{Camera.HEIGHT}"
    GRID_SIZE = 10
    NUM_PAIRS = 50
    INTERVAL_SEC = 2.0
    FLUSH_FRAMES = 4
    FLASH_SEC = 0.5
    IMAGE_EXT = "*.png"
    IMAGE_EXTENSIONS = ("*.png", "*.jpg", "*.jpeg", "*.bmp")
    DELETE_BAD_AUTOMATICALLY = True   # set True to auto-delete pairs where either side fails
    CALIBRATION_FILE = f"calib_{Camera.WIDTH}x{Camera.HEIGHT}.npz"  # produced during calibration

