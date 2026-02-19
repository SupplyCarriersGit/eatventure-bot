###############################
###    WINDOW & UI SETTINGS   ###
###############################

# WINDOW_TITLE: The exact title of the scrcpy window (visible at the top of the window)
WINDOW_TITLE = "EatventureAuto"

# Window dimensions used for capturing and relative positioning
WINDOW_WIDTH = 300 * 1.2
WINDOW_HEIGHT = 650 * 1.2

# Debug and Visualization Settings
DEBUG = True
DEBUG_VISION = False  # Enables masked view for tuning pixel density
ShowForbiddenArea = False  # Enables a visual overlay showing forbidden zones in red


###############################
###  DIRECTORY & FILE PATHS ###
###############################

TEMPLATES_DIR = "templates"
ASSETS_DIR = "Assets"
LOGS_DIR = "logs"


###############################
###   DETECTION THRESHOLDS  ###
###############################

# General template matching confidence (0.0 - 1.0)
MATCH_THRESHOLD = 0.98

# Specific thresholds for different game assets
RED_ICON_THRESHOLD = 0.94
NEW_LEVEL_RED_ICON_THRESHOLD = 0.95
STATS_RED_ICON_THRESHOLD = 0.97
UPGRADE_STATION_THRESHOLD = 0.92
BOX_THRESHOLD = 0.97
UNLOCK_THRESHOLD = 0.95
NEW_LEVEL_THRESHOLD = 0.98

# Detection gate settings
RED_ICON_MIN_MATCHES = 1
NEW_LEVEL_RED_ICON_MIN_MATCHES = 1
RED_ICON_PIXEL_THRESHOLD = 50  # Min red pixels in ROI to trigger
RED_ICON_DILATE_KERNEL = 3     # Size of dilation kernel to 'inflate' red pixels

# Red Color HSV bounds (wider range for better detection)
RED_HSV_LOWER1 = (0, 100, 100)
RED_HSV_UPPER1 = (15, 255, 255)
RED_HSV_LOWER2 = (165, 100, 100)
RED_HSV_UPPER2 = (180, 255, 255)

# Color verification for Red Icons
RED_ICON_COLOR_CHECK = True
RED_ICON_COLOR_MIN_RATIO = 1.15
RED_ICON_COLOR_MIN_MEAN = 35
RED_ICON_COLOR_SAMPLE_SIZE = 24

# Position refinement and verification
RED_ICON_VERIFY_PADDING = 24
RED_ICON_VERIFY_TOLERANCE = 12
RED_ICON_REFINE_RADIUS = 18
RED_ICON_REFINE_THRESHOLD_DROP = 0.02

# Upgrade station specific detection
UPGRADE_STATION_COLOR_CHECK = False
UPGRADE_STATION_REFINE_RADIUS = 28
UPGRADE_STATION_CLICK_REFINE_RADIUS = 18


###############################
###  MOUSE & INTERACTION    ###
###############################

# Base delays for human-like movement and interaction
CLICK_DELAY = 0.036
MOUSE_MOVE_DELAY = 0.004
MOUSE_DOWN_UP_DELAY = 0.006
DOUBLE_CLICK_DELAY = 0.02

# Mouse movement retry and correction logic
MOUSE_MOVE_RETRIES = 3
MOUSE_MOVE_RETRY_DELAY = 0.002
MOUSE_TARGET_SETTLE_DELAY = 0.001
MOUSE_TARGET_TIMEOUT = 0.04
MOUSE_TARGET_CHECK_INTERVAL = 0.001
MOUSE_TARGET_HOVER_DELAY = 0.001
MOUSE_STABILIZE_DURATION = 0.004
MOUSE_TARGET_RETRIES = 2
MOUSE_TARGET_CORRECTION_DELAY = 0.001

# Stability delays before clicking
MOUSE_PRE_CLICK_STABILIZE_BASE = 0.0015
MOUSE_PRE_CLICK_STABILIZE_MAX = 0.012
MOUSE_PRE_CLICK_STABILIZE_DISTANCE_FACTOR = 0.00003

# Click retry logic for robustness
MOUSE_CLICK_RETRY_COUNT = 2
MOUSE_CLICK_RETRY_SETTLE_DELAY = 0.0015


###############################
###    SCROLLING BEHAVIOR   ###
###############################

# Start position for search scrolls (relative to window)
SCROLL_START_POS = (180, 390)

# Distance in pixels for a single "standard" scroll step
SCROLL_PIXEL_STEP = 120
SCROLL_DISTANCE_RATIO = 1  # Default multiplier for non-incremental scrolls

# Continuous Incremental Oscillating Search settings
MAX_SCROLL_CYCLES = 15  # Maximum widening steps before resetting to cycle 1
SCROLL_INCREMENT_STEP = 2  # Arithmetic progression increment
SCROLL_INTERVAL_PAUSE = 0.5  # Time to wait between individual scroll steps
CYCLE_PAUSE_DURATION = 0.5  # Time to wait after a full oscillation sequence

# Visual smoothness and stability
SCROLL_DURATION = 0.6  # How long each drag action takes
SCROLL_STEP_COUNT = 60  # Intermediate steps for smooth cursor movement
SCROLL_MIN_INTERVAL = 0.005  # Throttle between consecutive drag steps
SCROLL_SETTLE_DELAY = 0.15  # Wait time after a scroll for UI to stop moving
SCROLL_SEGMENTS = 1
SCROLL_MIN_SEGMENT_DISTANCE = 20
SCROLL_SEGMENT_SETTLE_DELAY = 0.05  # Short pause before scanning during a multi-segment scroll
SCROLL_ASSET_SCAN_INTERVAL = 0.005  # Frequency of background asset scans during scroll

# Scroll detection settings
SCROLL_RED_ICON_MIN_DISTANCE = 56
SCROLL_RED_ICON_THRESHOLD_DROP = 0.01
SCROLL_RED_ICON_MIN_MATCHES = 1
SCROLL_ASSET_FULL_SCAN_EVERY = 1

# Interrupt assets to scan for DURING an active scroll
SCROLL_INTERRUPT_ASSET_THRESHOLD = 0.93
SCROLL_INTERRUPT_ASSET_TEMPLATES = (
    "upgradeStation", "unlock", "box1", "box2", "box3", "box4", "box5",
    "RedIcon", "RedIcon2", "RedIcon3", "RedIcon4", "RedIcon5", "RedIcon6",
    "RedIcon7", "RedIcon8", "RedIcon9", "RedIcon10", "RedIcon11", "RedIcon12",
    "RedIcon13", "RedIcon14", "RedIcon15", "RedIconNoBG"
)


###############################
###    BOT LOGIC & TIMING   ###
###############################

# Main loop execution speed
MAIN_LOOP_DELAY = 0.008

# Minimum time to wait between state handler executions
STATE_DELAY = 0.01
STATE_MIN_INTERVAL_DEFAULT = 0.005
STATE_MIN_INTERVALS = {
    "FIND_RED_ICONS": 0.005,
    "OPEN_BOXES": 0.005,
    "SCROLL": 0.005,
}

# Red Icon and detection offsets
RED_ICON_OFFSET_X = 10
RED_ICON_OFFSET_Y = 10

# Fixed click positions for specific UI elements
NEW_LEVEL_POS = (171, 434)
LEVEL_TRANSITION_POS = (174, 520)
IDLE_CLICK_POS = (2, 390)
STATS_UPGRADE_POS = (270, 304)
STATS_UPGRADE_BUTTON_POS = (310, 698)
NEW_LEVEL_BUTTON_POS = (30, 692)

# Timing for interaction sequences
UPGRADE_HOLD_DURATION = 5  # How long to hold the upgrade button
UPGRADE_CLICK_INTERVAL = 0.008
UPGRADE_SEARCH_INTERVAL = 0.02
UPGRADE_CHECK_INTERVAL = 0.045
STATS_UPGRADE_CLICK_DURATION = 2
STATS_UPGRADE_CLICK_DELAY = 0.006
STATS_ICON_PADDING = 20

# UI render and settle delays
IDLE_CLICK_SETTLE_DELAY = 0.006
IDLE_CLICK_COOLDOWN = 0.06

# Red Icon and detection logic constants
RED_ICON_MIN_DISTANCE = 80
RED_ICON_MERGE_PROXIMITY = 10
RED_ICON_MERGE_BUCKET_SIZE = 10

# Upgrade station interaction settings
UPGRADE_STATION_SEARCH_MAX_ATTEMPTS = 5
UPGRADE_STATION_RELAXED_THRESHOLD_DROP = 0.05
UPGRADE_STATION_RELAXED_ATTEMPT_TRIGGER = 2

# Level transition and completion settings
LEVEL_TRANSITION_MAX_ATTEMPTS = 5
LEVEL_COMPLETION_RECENCY_WINDOW = 5.0
NEW_LEVEL_FAIL_COOLDOWN = 15.0

NEW_LEVEL_BUTTON_DELAY = 0.5
NEW_LEVEL_FOLLOWUP_DELAY = 0.3
TRANSITION_POST_CLICK_DELAY = 0.5
TRANSITION_RETRY_DELAY = 0.1
UNLOCK_POST_CLICK_DELAY = 0.8
WAIT_UNLOCK_RETRY_DELAY = 0.02
PRE_UNLOCK_DELAY = 0.0
UNLOCK_BACKOFF_THRESHOLD = 5
UNLOCK_MAX_RETRY_DELAY = 0.5

# Performance caching
CAPTURE_CACHE_TTL = 0.008
NEW_LEVEL_RED_ICON_CACHE_TTL = 0.01
RED_ICON_STABILITY_CACHE_TTL = 0.3
RED_ICON_STABILITY_RADIUS = 14
RED_ICON_STABILITY_MIN_HITS = 3
RED_ICON_STABILITY_MAX_HISTORY = 10

# Scan regions for Red Icons
NEW_LEVEL_RED_ICON_X_MIN = 40
NEW_LEVEL_RED_ICON_X_MAX = 60
NEW_LEVEL_RED_ICON_Y_MIN = 665
NEW_LEVEL_RED_ICON_Y_MAX = 680

UPGRADE_RED_ICON_X_MIN = 280
UPGRADE_RED_ICON_X_MAX = 310
UPGRADE_RED_ICON_Y_MIN = 665
UPGRADE_RED_ICON_Y_MAX = 680

# Background monitoring frequency
NEW_LEVEL_INTERRUPT_INTERVAL = 0.02
NEW_LEVEL_MONITOR_INTERVAL = 0.01
NEW_LEVEL_OVERRIDE_COOLDOWN = 0.25


###############################
### ADAPTIVE TUNER SETTINGS ###
###############################

ADAPTIVE_TUNER_ENABLED = True
ADAPTIVE_TUNER_ALPHA = 0.2  # EMA smoothing factor

# Success rate thresholds for triggering delay adjustments
ADAPTIVE_TUNER_CLICK_LOW_THRESHOLD = 0.85
ADAPTIVE_TUNER_CLICK_HIGH_THRESHOLD = 0.97
ADAPTIVE_TUNER_SEARCH_LOW_THRESHOLD = 0.70
ADAPTIVE_TUNER_SEARCH_HIGH_THRESHOLD = 0.90

# Step values for delay adjustments
ADAPTIVE_TUNER_CLICK_DELAY_STEP = 0.01
ADAPTIVE_TUNER_MOVE_DELAY_STEP = 0.001
ADAPTIVE_TUNER_CLICK_DECREMENT = 0.005
ADAPTIVE_TUNER_MOVE_DECREMENT = 0.001
ADAPTIVE_TUNER_SEARCH_INTERVAL_STEP = 0.01
ADAPTIVE_TUNER_UPGRADE_INTERVAL_STEP = 0.001
ADAPTIVE_TUNER_SEARCH_DECREMENT = 0.005
ADAPTIVE_TUNER_UPGRADE_DECREMENT = 0.001

# Range limits for adaptive delays
ADAPTIVE_TUNER_MIN_CLICK_DELAY = 0.035
ADAPTIVE_TUNER_MAX_CLICK_DELAY = 0.11
ADAPTIVE_TUNER_MIN_MOVE_DELAY = 0.003
ADAPTIVE_TUNER_MAX_MOVE_DELAY = 0.012
ADAPTIVE_TUNER_MIN_UPGRADE_INTERVAL = 0.006
ADAPTIVE_TUNER_MAX_UPGRADE_INTERVAL = 0.012
ADAPTIVE_TUNER_MIN_SEARCH_INTERVAL = 0.015
ADAPTIVE_TUNER_MAX_SEARCH_INTERVAL = 0.05


###############################
###  AI VISION & LEARNING   ###
###############################

AI_VISION_ENABLED = True
AI_VISION_ALPHA = 0.2
AI_VISION_ALPHA_MAX = 0.45
AI_VISION_CONFIDENCE_BOOST = 0.3
AI_VISION_CONFIDENCE_THRESHOLD = 0.8  # Confidence above this level triggers boost

# Box detection specific AI settings
AI_BOX_THRESHOLD_MIN = 0.85
AI_BOX_THRESHOLD_MAX = 0.995
AI_BOX_MISS_WINDOW = 3
AI_BOX_MISS_STEP = 0.005

# Threshold limits for AI-driven detection
AI_RED_ICON_THRESHOLD_MIN = 0.92
AI_RED_ICON_THRESHOLD_MAX = 0.985
AI_RED_ICON_MARGIN = 0.01
AI_RED_ICON_MISS_WINDOW = 2
AI_RED_ICON_MISS_STEP = 0.006

AI_NEW_LEVEL_THRESHOLD_MIN = 0.965
AI_NEW_LEVEL_THRESHOLD_MAX = 0.995
AI_NEW_LEVEL_MISS_WINDOW = 2
AI_NEW_LEVEL_MISS_STEP = 0.004

AI_NEW_LEVEL_RED_ICON_THRESHOLD_MIN = 0.92
AI_NEW_LEVEL_RED_ICON_THRESHOLD_MAX = 0.99
AI_NEW_LEVEL_RED_ICON_MISS_WINDOW = 2
AI_NEW_LEVEL_RED_ICON_MISS_STEP = 0.005

AI_UPGRADE_STATION_THRESHOLD_MIN = 0.9
AI_UPGRADE_STATION_THRESHOLD_MAX = 0.99
AI_UPGRADE_STATION_MISS_WINDOW = 2
AI_UPGRADE_STATION_MISS_STEP = 0.005

AI_STATS_UPGRADE_THRESHOLD_MIN = 0.9
AI_STATS_UPGRADE_THRESHOLD_MAX = 0.99
AI_STATS_UPGRADE_MISS_WINDOW = 2
AI_STATS_UPGRADE_MISS_STEP = 0.005

# Persistence files
AI_VISION_STATE_FILE = f"{LOGS_DIR}/vision_state.json"
AI_VISION_SAVE_INTERVAL = 1.0

# Historical Learning
AI_LEARNING_ENABLED = True
AI_LEARNING_STATE_FILE = f"{LOGS_DIR}/learning_state.json"
AI_LEARNING_SAVE_INTERVAL = 1.5
AI_LEARNING_RECORDS_LIMIT = 120
AI_LEARNING_THREAD_JOIN_TIMEOUT = 1.0

# Learning range limits
AI_LEARNING_MIN_CLICK_DELAY = 0.035
AI_LEARNING_MAX_CLICK_DELAY = 0.12
AI_LEARNING_MIN_MOVE_DELAY = 0.002
AI_LEARNING_MAX_MOVE_DELAY = 0.012
AI_LEARNING_MIN_UPGRADE_INTERVAL = 0.006
AI_LEARNING_MAX_UPGRADE_INTERVAL = 0.013
AI_LEARNING_MIN_SEARCH_INTERVAL = 0.012
AI_LEARNING_MAX_SEARCH_INTERVAL = 0.05


###############################
###  TELEGRAM NOTIFICATIONS ###
###############################

TELEGRAM_ENABLED = False
TELEGRAM_BOT_TOKEN = ""
TELEGRAM_CHAT_ID = ""


###############################
###     FORBIDDEN ZONES     ###
###############################

# Zones prevent the bot from clicking on critical UI elements
# Each zone is defined by name and bounding box (min/max X and Y)
FORBIDDEN_ZONES = [
    {
        "name": "General bottom bar",
        "x_min": 60, "x_max": 280, "y_min": 668, "y_max": 1000
    },
    {
        "name": "Zone 1: Right side menu area",
        "x_min": 290, "x_max": 350, "y_min": 93, "y_max": 270
    },
    {
        "name": "Zone 2: Left side top menu area",
        "x_min": 0, "x_max": 60, "y_min": 50, "y_max": 280
    },
    {
        "name": "Zone 3: Left side bottom menu area",
        "x_min": 0, "x_max": 60, "y_min": 590, "y_max": 667
    },
    {
        "name": "Zone 4: Top center notification area",
        "x_min": 145, "x_max": 200, "y_min": 65, "y_max": 110
    },
    {
        "name": "Zone 5: Bottom navigation bar",
        "x_min": 55, "x_max": 285, "y_min": 660, "y_max": 725
    },
    {
        "name": "Zone 6: Top bar area",
        "x_min": 0, "x_max": 360, "y_min": 0, "y_max": 70
    }
]

# Coordinate limits for searching Red Icons
MAX_SEARCH_Y = 660
EXTENDED_SEARCH_Y = 710
