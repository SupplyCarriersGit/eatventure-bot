# =============================================================================
# WINDOW & SYSTEM CONFIGURATION
# =============================================================================
# WINDOW_TITLE: The exact title of the scrcpy window (visible at the top of the window).
WINDOW_TITLE = "EatventureAuto"
# WINDOW_WIDTH/HEIGHT: Target dimensions for the game window scaling.
WINDOW_WIDTH = 300 * 1.2
WINDOW_HEIGHT = 650 * 1.2

# =============================================================================
# GAME ELEMENT DETECTION AREA
# =============================================================================
# MAX_SEARCH_Y: Maximum Y coordinate to scan for most game elements (to avoid UI bars).
MAX_SEARCH_Y = 660
# EXTENDED_SEARCH_Y: Larger scan area for specific initial checks.
EXTENDED_SEARCH_Y = 710

# =============================================================================
# DIRECTORY PATHS
# =============================================================================
TEMPLATES_DIR = "templates"    # Folder containing image templates for matching
ASSETS_DIR = "Assets"          # Folder containing game assets/icons
LOGS_DIR = "logs"              # Folder where log files are stored
SCREENSHOTS_DIR = "screenshots" # Folder for debug screenshots

# =============================================================================
# DEBUG & VISUALIZATION
# =============================================================================
DEBUG = True                   # Enable verbose logging and debug features
SAVE_SCREENSHOTS = False       # Save every capture to disk (high disk usage!)
# ShowForbiddenArea: Displays a red overlay on forbidden click zones for debugging.
ShowForbiddenArea = False

# =============================================================================
# TELEGRAM NOTIFICATIONS
# =============================================================================
TELEGRAM_ENABLED = True      # Enable/Disable all Telegram messages
TELEGRAM_BOT_TOKEN = "8244889019:AAFFqf1dn4d3LbHf3tenOXEBaoruj3FWkR0"        # Your bot's API token from @BotFather
TELEGRAM_CHAT_ID = "770506304"          # Your unique chat ID to receive notifications

# =============================================================================
# DETECTION THRESHOLDS
# =============================================================================
# General matching sensitivity (0.0 to 1.0). Higher = stricter.
MATCH_THRESHOLD = 0.975
# Specific threshold for detecting various game elements.
RED_ICON_THRESHOLD = 0.92
NEW_LEVEL_RED_ICON_THRESHOLD = 0.95
STATS_RED_ICON_THRESHOLD = 0.94
UPGRADE_STATION_THRESHOLD = 0.9
UPGRADE_STATION_COLOR_CHECK = False   # Double-check upgrade station matches for color consistency
BOX_THRESHOLD = 0.97
BOX_MIN_MATCHES = 2                   # Minimum template matches required to confirm a box
UNLOCK_THRESHOLD = 0.9
NEW_LEVEL_THRESHOLD = 0.98

# Refinement radii for pinpointing assets more accurately.
UPGRADE_STATION_REFINE_RADIUS = 28
UPGRADE_STATION_CLICK_REFINE_RADIUS = 18

# Minimum number of template matches required to confirm detection.
RED_ICON_MIN_MATCHES = 1
NEW_LEVEL_RED_ICON_MIN_MATCHES = 1

# =============================================================================
# RED ICON COLOR VALIDATION
# =============================================================================
RED_ICON_COLOR_CHECK = True      # Double-check detected icons for actual red color
RED_ICON_COLOR_MIN_RATIO = 1.1   # Relaxed for better detection
RED_ICON_COLOR_MIN_MEAN = 35     # Minimum brightness of the red channel
RED_ICON_COLOR_SAMPLE_SIZE = 12  # Number of pixels to sample for color check
RED_ICON_VERIFY_PADDING = 24     # Area around icon to scan for color confirmation
RED_ICON_VERIFY_TOLERANCE = 12   # Distance from center to check for color
RED_ICON_REFINE_RADIUS = 18      # Local search area to pinpoint icon center
RED_ICON_REFINE_THRESHOLD_DROP = 0.02 # Allowed sensitivity drop during refinement
RED_ICON_PRIORITY_TEMPLATE_LIMIT = 8 # Max templates to use for fast-scanning

# =============================================================================
# GAME ELEMENT POSITIONS (RELATIVE TO WINDOW)
# =============================================================================
UPGRADE_POS = (320, 726)              # Main upgrade button position
NEW_LEVEL_POS = (40, 700)             # Location of the new level/map button
REDICON_CHECK_POS = (60, 704)         # Area to check for the map's red notification
LEVEL_TRANSITION_POS = (175, 513)     # Center of the screen during transitions
NEW_LEVEL_CONFIRM_POS = (175, 513)    # Position of the confirm button after clicking new level
IDLE_CLICK_POS = (2, 390)             # Position for "keeping the game active" clicks
INIT_CLICK_POS = (180, 472)           # Initial click to focus/start
STATS_UPGRADE_POS = (270, 304)        # Position of upgrade in the stats menu
STATS_UPGRADE_BUTTON_POS = (310, 698) # The actual "BUY" button in stats menu
NEW_LEVEL_BUTTON_POS = (30, 692)      # Button to confirm moving to a new city

# =============================================================================
# RED ICON DETECTION ZONES
# =============================================================================
# Detection zones for specific red icons (e.g., map notifications).
NEW_LEVEL_RED_ICON_X_MIN = 40
NEW_LEVEL_RED_ICON_X_MAX = 60
NEW_LEVEL_RED_ICON_Y_MIN = 665
NEW_LEVEL_RED_ICON_Y_MAX = 680

UPGRADE_RED_ICON_X_MIN = 280
UPGRADE_RED_ICON_X_MAX = 310
UPGRADE_RED_ICON_Y_MIN = 665
UPGRADE_RED_ICON_Y_MAX = 680

# Offsets applied when clicking a detected red icon to hit the center accurately.
RED_ICON_OFFSET_X = 10
RED_ICON_OFFSET_Y = 10

# =============================================================================
# BOT BEHAVIOR & TIMING
# =============================================================================
SEARCH_INTERVAL = 0.25                # How often to scan the screen (seconds)
RED_ICON_CYCLE_COUNT = 3              # How many times to try clicking an icon
RED_ICON_MAX_PER_SCAN = 2             # Max icons to process in one go
STATS_UPGRADE_CLICK_DURATION = 3      # How long to hold the stats upgrade button
STATS_UPGRADE_CLICK_DELAY = 0.015     # Wait time after stats click
STATS_ICON_PADDING = 20               # Extra search area around stats icons
CAPTURE_CACHE_TTL = 0.015             # Synced with monitor interval (prev: 0.03)
NEW_LEVEL_RED_ICON_CACHE_TTL = 0.015  # Synced with monitor interval (prev: 0.03)
RED_ICON_STABILITY_CACHE_TTL = 0.3    # Increased window for stability
RED_ICON_STABILITY_RADIUS = 14        # Max distance an icon can move and be "stable"
RED_ICON_STABILITY_MIN_HITS = 1       # Trust detection immediately
RED_ICON_STABILITY_MAX_HISTORY = 10   # Max history points for stability tracking
UPGRADE_HOLD_DURATION = 5             # Optimized for faster restaurant rotation
UPGRADE_CLICK_INTERVAL = 0.01         # Rapid-fire click interval during upgrades
UPGRADE_SEARCH_INTERVAL = 0.015       # Scan speed while looking for stations
UPGRADE_CHECK_INTERVAL = 0.05         # Frequency of checking station status
IDLE_CLICK_SETTLE_DELAY = 0.012       # Wait after an idle click
IDLE_CLICK_COOLDOWN = 0.05            # Min time between idle clicks
NEW_LEVEL_BUTTON_DELAY = 0.05         # Wait after clicking map button
NEW_LEVEL_FOLLOWUP_DELAY = 0.05       # Wait before confirming transition
TRANSITION_POST_CLICK_DELAY = 0.25    # Wait after clicking "Fly to New City"
TRANSITION_RETRY_DELAY = 0.05         # Wait before retrying failed transition
UNLOCK_POST_CLICK_DELAY = 0.12        # Wait after clicking "Unlock"
WAIT_UNLOCK_RETRY_DELAY = 0.1         # Frequency of checking for unlock button

# =============================================================================
# MOUSE CONTROL SETTINGS
# =============================================================================
CLICK_DELAY = 0.045                   # Balanced delay for game recognition
MOUSE_MOVE_DELAY = 0.008              # Aligned with system timer resolution
MOUSE_DOWN_UP_DELAY = 0.012           # Ensuring the game registers the press
DOUBLE_CLICK_DELAY = 0.04             # Stable double-click timing
MIN_CLICK_INTERVAL = 0.015            # Safety margin between any two clicks
MOUSE_POSITION_TOLERANCE = 2          # Allowed error in pixels for cursor position
MOUSE_MOVE_RETRIES = 3                # Times to retry moving cursor if it fails
MOUSE_MOVE_RETRY_DELAY = 0.005        # Wait between movement retries
MOUSE_TARGET_SETTLE_DELAY = 0.005     # Time for cursor to "stop" visually
MOUSE_TARGET_TIMEOUT = 0.08           # Max time to wait for cursor to reach target
MOUSE_TARGET_CHECK_INTERVAL = 0.005   # Frequency of checking cursor position
MOUSE_TARGET_HOVER_DELAY = 0.005      # Wait after reaching target before action
MOUSE_STABILIZE_DURATION = 0.008      # Time cursor must stay still to be "stable"
MOUSE_TARGET_RETRIES = 2              # Extra attempts to correct position
MOUSE_TARGET_CORRECTION_DELAY = 0.005 # Wait between position corrections
MOUSE_PRE_CLICK_STABILIZE_BASE = 0.01 # Slightly increased for stability (prev: 0.008)
MOUSE_PRE_CLICK_STABILIZE_MAX = 0.02  # Slightly increased for stability (prev: 0.015)
MOUSE_PRE_CLICK_STABILIZE_DISTANCE_FACTOR = 0.00003 # Delay based on travel distance
MOUSE_CLICK_RETRY_COUNT = 2           # Attempts to click if first fails
MOUSE_CLICK_RETRY_SETTLE_DELAY = 0.01 # Wait between click retries

# =============================================================================
# SCROLL SETTINGS (GENERAL)
# =============================================================================
SCROLL_DURATION = 0.15                 # Faster, more responsive drag
NO_ICON_SCROLL_DURATION = 0.15         # Consistent searching speed
SCROLL_STEP_COUNT = 15                # Reduced granularity for speed
SCROLL_MIN_INTERVAL = 0.02            # Increased to prevent input flood (prev: 0.01)
SCROLL_SETTLE_DELAY = 0.07             # Increased for rendering safety (prev: 0.05)
SCROLL_SEGMENTS = 1                   # Single segment for maximum speed
SCROLL_MIN_SEGMENT_DISTANCE = 10      # Smaller segments
SCROLL_SEGMENT_SETTLE_DELAY = 0.02     # Tightened for faster scanning
SCROLL_ASSET_SCAN_INTERVAL = 0.02     # Aligned with Min Interval (prev: 0.015)
SCROLL_DISTANCE_RATIO = 1.0           # Percentage of the full scroll distance to use
SCROLL_ASSET_FULL_SCAN_EVERY = 1      # Scan every segment for max accuracy
SCROLL_UP_CYCLES = 15                  # Increased number of up-scrolls
MAX_SCROLL_CYCLES = 25                 # Increased max scrolls before forcing a direction reset
NEW_LEVEL_INTERRUPT_INTERVAL = 0.015  # Interval to check for interrupts during sleep (seconds)
NEW_LEVEL_MONITOR_INTERVAL = 0.015    # How often background thread scans for new level (seconds)
NEW_LEVEL_OVERRIDE_COOLDOWN = 0.2     # Minimum time between priority new-level clicks (seconds)

# =============================================================================
# SCROLL DIRECTIONAL POSITIONS
# =============================================================================
# Positions define where the drag starts and ends for each direction.
SCROLL_UP_START_POS = (180, 500)
SCROLL_UP_END_POS = (180, 420)
SCROLL_DOWN_START_POS = (180, 420)
SCROLL_DOWN_END_POS = (180, 500)

# Timings for directional scrolling
SCROLL_UP_DURATION = 0.15
SCROLL_DOWN_DURATION = 0.15
SCROLL_UP_STEP_COUNT = 15
SCROLL_DOWN_STEP_COUNT = 15

# =============================================================================
# SCROLL INTERRUPT & DETECTION
# =============================================================================
SCROLL_RED_ICON_MIN_DISTANCE = 20     # Min distance between icons during scroll
SCROLL_RED_ICON_THRESHOLD_DROP = 0.04 # Sensitivity boost while scrolling
SCROLL_RED_ICON_MIN_MATCHES = 1       # Matches needed to stop scroll for an icon
SCROLL_INTERRUPT_ASSET_THRESHOLD = 0.93 # Sensitivity for stopping scroll for assets
# Assets that, if seen, will immediately stop the scrolling to interact.
SCROLL_INTERRUPT_ASSET_TEMPLATES = (
    "upgradeStation", "unlock", "box1", "box2", "box3", "box4", "box5",
    "RedIcon", "RedIcon2", "RedIcon3", "RedIcon4", "RedIcon5", "RedIcon6",
    "RedIcon7", "RedIcon8", "RedIcon9", "RedIcon10", "RedIcon11", "RedIcon12",
    "RedIcon13", "RedIcon14", "RedIcon15", "RedIconNoBG",
)

# =============================================================================
# NO-ICON & FORBIDDEN SCROLLING
# =============================================================================
NO_ICON_SCROLL_UP_COUNT = 20           # Increased search cycles
NO_ICON_SCROLL_DOWN_COUNT = 15         # Increased search cycles
FORBIDDEN_ICON_MAX_SCROLLS = 3        # Max tries to scroll an icon out of a blocked zone
FORBIDDEN_ICON_SCROLL_COOLDOWN = 0.015# Wait between rescue attempts

# =============================================================================
# STATE MACHINE SETTINGS
# =============================================================================
STATE_DELAY = 0.04                    # Tightened for faster completion
MAIN_LOOP_DELAY = 0.01                # Wait time at the end of each full cycle
STATE_MIN_INTERVAL_DEFAULT = 0.02     # Min time a state must run
# Enforced minimum runtimes for specific states to prevent rapid switching.
STATE_MIN_INTERVALS = {
    "FIND_RED_ICONS": 0.05,
    "OPEN_BOXES": 0.05,
    "SCROLL": 0.1,
}
NORMAL_SCROLL_UP_COUNT = 12            # Up-scrolls in a normal work cycle
NORMAL_SCROLL_DOWN_COUNT = 10          # Down-scrolls in a normal work cycle

# =============================================================================
# ADAPTIVE PERFORMANCE TUNER
# =============================================================================
ADAPTIVE_TUNER_ENABLED = True         # Auto-adjust timings based on success rates
ADAPTIVE_TUNER_ALPHA = 0.2            # Smoothing factor for adjustments (EMA)
ADAPTIVE_TUNER_MIN_CLICK_DELAY = 0.035
ADAPTIVE_TUNER_MAX_CLICK_DELAY = 0.11
ADAPTIVE_TUNER_MIN_MOVE_DELAY = 0.008
ADAPTIVE_TUNER_MAX_MOVE_DELAY = 0.012
ADAPTIVE_TUNER_MIN_UPGRADE_INTERVAL = 0.006
ADAPTIVE_TUNER_MAX_UPGRADE_INTERVAL = 0.012
ADAPTIVE_TUNER_MIN_SEARCH_INTERVAL = 0.015
ADAPTIVE_TUNER_MAX_SEARCH_INTERVAL = 0.05

# =============================================================================
# AI VISION OPTIMIZER
# =============================================================================
AI_VISION_ENABLED = True              # Auto-adjust thresholds based on confidence
AI_VISION_ALPHA = 0.35                # How fast the AI learns from success (increased for faster adaptation)
AI_VISION_ALPHA_MAX = 0.6             # Maximum learning speed (increased for faster adaptation)
AI_VISION_CONFIDENCE_BOOST = 0.3      # Weight given to high-confidence matches
AI_RED_ICON_THRESHOLD_MIN = 0.88      # Raised to prevent false positives (prev: 0.85)
AI_RED_ICON_THRESHOLD_MAX = 0.985     # Highest the red icon sensitivity can go
AI_RED_ICON_MARGIN = 0.01             # Safety buffer for thresholds
AI_RED_ICON_MISS_WINDOW = 2           # Misses before lowering sensitivity
AI_RED_ICON_MISS_STEP = 0.006         # How much to lower sensitivity on miss
AI_NEW_LEVEL_THRESHOLD_MIN = 0.97     # Raised for accuracy (prev: 0.965)
AI_NEW_LEVEL_THRESHOLD_MAX = 0.995
AI_NEW_LEVEL_MISS_WINDOW = 2
AI_NEW_LEVEL_MISS_STEP = 0.004
AI_NEW_LEVEL_RED_ICON_THRESHOLD_MIN = 0.93 # Raised for accuracy (prev: 0.92)
AI_NEW_LEVEL_RED_ICON_THRESHOLD_MAX = 0.99
AI_NEW_LEVEL_RED_ICON_MISS_WINDOW = 2
AI_NEW_LEVEL_RED_ICON_MISS_STEP = 0.005
AI_UPGRADE_STATION_THRESHOLD_MIN = 0.92 # Raised for accuracy (prev: 0.9)
AI_UPGRADE_STATION_THRESHOLD_MAX = 0.99
AI_UPGRADE_STATION_MISS_WINDOW = 2
AI_UPGRADE_STATION_MISS_STEP = 0.005
AI_STATS_UPGRADE_THRESHOLD_MIN = 0.92 # Raised for accuracy (prev: 0.9)
AI_STATS_UPGRADE_THRESHOLD_MAX = 0.99
AI_STATS_UPGRADE_MISS_WINDOW = 2
AI_STATS_UPGRADE_MISS_STEP = 0.005
AI_VISION_STATE_FILE = f"{LOGS_DIR}/vision_state.json" # Where AI saves its progress
AI_VISION_SAVE_INTERVAL = 1.0         # How often to save AI state (seconds)

# =============================================================================
# HISTORICAL LEARNER
# =============================================================================
AI_LEARNING_ENABLED = True            # Track long-term performance trends
AI_LEARNING_STATE_FILE = f"{LOGS_DIR}/learning_state.json"
AI_LEARNING_SAVE_INTERVAL = 1.5
AI_LEARNING_PAIR_WINDOW = 2
AI_LEARNING_BATCH_WINDOW = 7
AI_LEARNING_THREAD_INTERVAL = 0.05
AI_LEARNING_AUTOWRITE_CONFIG = False   # Automatically update this file with best settings
AI_LEARNING_MIN_CLICK_DELAY = 0.035
AI_LEARNING_MAX_CLICK_DELAY = 0.12
AI_LEARNING_MIN_MOVE_DELAY = 0.002
AI_LEARNING_MAX_MOVE_DELAY = 0.012
AI_LEARNING_MIN_UPGRADE_INTERVAL = 0.006
AI_LEARNING_MAX_UPGRADE_INTERVAL = 0.013
AI_LEARNING_MIN_SEARCH_INTERVAL = 0.012
AI_LEARNING_MAX_SEARCH_INTERVAL = 0.05
AI_LEARNING_EMA_ALPHA = 0.18
AI_LEARNING_PROFILE_BLEND_TOP_K = 3
AI_LEARNING_MIN_IMPROVEMENT_RATIO = 0.03
AI_LEARNING_APPLY_COOLDOWN = 1.2

# =============================================================================
# FORBIDDEN ZONES (UI PROTECTION)
# =============================================================================
# These zones prevent the bot from clicking on critical UI elements (menus, etc.)
# Defined by: X_MIN, X_MAX, Y_MIN, Y_MAX coordinates

# Bottom navigation bar area
FORBIDDEN_CLICK_X_MIN = 60
FORBIDDEN_CLICK_X_MAX = 280
FORBIDDEN_CLICK_Y_MIN = 668

# Zone 1: Right side menu area
FORBIDDEN_ZONE_1_X_MIN = 290
FORBIDDEN_ZONE_1_X_MAX = 350
FORBIDDEN_ZONE_1_Y_MIN = 93
FORBIDDEN_ZONE_1_Y_MAX = 270

# Zone 2: Left side top menu area
FORBIDDEN_ZONE_2_X_MIN = 0
FORBIDDEN_ZONE_2_X_MAX = 60
FORBIDDEN_ZONE_2_Y_MIN = 50
FORBIDDEN_ZONE_2_Y_MAX = 280

# Zone 3: Left side bottom menu area
FORBIDDEN_ZONE_3_X_MIN = 0
FORBIDDEN_ZONE_3_X_MAX = 60
FORBIDDEN_ZONE_3_Y_MIN = 590
FORBIDDEN_ZONE_3_Y_MAX = 667

# Zone 4: Top center notification area
FORBIDDEN_ZONE_4_X_MIN = 145
FORBIDDEN_ZONE_4_X_MAX = 200
FORBIDDEN_ZONE_4_Y_MIN = 65
FORBIDDEN_ZONE_4_Y_MAX = 110

# Zone 5: Bottom navigation bar (Extended)
FORBIDDEN_ZONE_5_X_MIN = 55
FORBIDDEN_ZONE_5_X_MAX = 285
FORBIDDEN_ZONE_5_Y_MIN = 660
FORBIDDEN_ZONE_5_Y_MAX = 725

# Zone 6: Top status bar area
FORBIDDEN_ZONE_6_X_MIN = 0
FORBIDDEN_ZONE_6_X_MAX = 360
FORBIDDEN_ZONE_6_Y_MIN = 0
FORBIDDEN_ZONE_6_Y_MAX = 70
