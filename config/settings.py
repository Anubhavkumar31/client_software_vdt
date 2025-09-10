"""
Application settings and configurations
"""

import os
import sys

# ================================
# UI CONSTRAINTS & SIZING
# ================================

class UIConstraints:
    """UI layout constraints and limits"""
    
    # Splitter limits (pixels)
    MIN_TOP_HEIGHT = 220      # Top pane (charts) minimum height
    MIN_BOTTOM_HEIGHT = 250   # Bottom pane (tables/proximity) minimum height
    MAX_TOP_HEIGHT = None     # Set to specific value if needed (e.g., 900)
    MAX_BOTTOM_HEIGHT = None  # Set to specific value if needed (e.g., 900)
    
    # Margins and spacing
    RIGHT_MARGIN_PX = 300     # Right margin for scrolling
    TABLE_RIGHT_MARGIN = 50   # Small margin for table scrollbar
    
    # Widget sizing
    COMBO_BOX_MAX_VISIBLE = 12
    DEFAULT_SECTION_SIZE = 380  # Table column default width
    HEADER_HEIGHT_OFFSET = 20   # Space for pipe selection row

class ScrollSettings:
    """Scroll-related settings"""
    
    SCROLL_SCALE = 3          # Scroll sensitivity (5-10 for gentler)
    SINGLE_STEP_SIZE = 2      # Scroll bar single step
    PAGE_STEP_SIZE = 100      # Scroll bar page step
    SINGLE_STEP_INCREMENT = 10 # Scroll bar single step increment
    
    # Scroll policies
    HORIZONTAL_POLICY = "ScrollBarAsNeeded"  # or "ScrollBarAlwaysOff"
    VERTICAL_POLICY = "ScrollBarAlwaysOn"
    
    # Scroll modes
    SCROLL_MODE = "ScrollPerPixel"

# ================================
# CHART & VIEW SETTINGS  
# ================================

class ChartSettings:
    """Chart and view configuration"""
    
    # Web view fixed sizes
    MAIN_WEB_VIEW_WIDTH = 2500
    MAIN_WEB_VIEW_HEIGHT = 650
    
    SECONDARY_WEB_VIEW_WIDTH = 2500
    SECONDARY_WEB_VIEW_HEIGHT = 600
    
    # Chart minimum dimensions
    CHART_MIN_WIDTH = 2200
    CHART_MIN_HEIGHT = 1400
    
    PROXIMITY_CHART_MIN_WIDTH = 2000
    PROXIMITY_CHART_MIN_HEIGHT = 900
    
    # Scrollbar theme
    HANDLE_RADIUS = 10
    BUTTON_WIDTH_HEIGHT = 22    # Arrow circle size
    SCROLLBAR_HEIGHT = 14       # Unified height for all top bars
    SCROLLBAR_WIDTH = 16

class ViewSettings:
    """View and display settings"""
    
    # Loading dialog
    LOADING_DIALOG_WIDTH = 400
    LOADING_DIALOG_HEIGHT = 200
    
    # No defects container
    NO_DEFECTS_CONTAINER_MAX_WIDTH = 500
    NO_DEFECTS_CONTAINER_MAX_HEIGHT = 200
    NO_DEFECTS_CONTAINER_MIN_WIDTH = 400
    NO_DEFECTS_CONTAINER_MIN_HEIGHT = 150
    
    # Create project card
    CREATE_PROJECT_CARD_WIDTH = 420
    SELECT_PIPE_CARD_WIDTH = 500

# ================================
# FILE & PATH SETTINGS
# ================================

class FileSettings:
    """File and path configurations"""
    
    # Directory structure
    PICKLE_DATA_DIR = "pickle_data"
    PIPES_DATA_DIR = "pipes_data" 
    PIPETALLY_MAIN_DIR = "pipetally_main"
    REPORT_DIR = "report"
    
    # Pipe directory patterns
    PIPE_DIR_PATTERNS = [
        "pipe_{pipe_idx}",
        "pipe-{pipe_idx}", 
        "Pipe_{pipe_idx}"
    ]
    
    # File extensions
    PICKLE_EXTENSIONS = [".pkl"]
    EXCEL_EXTENSIONS = [".xlsx", ".xls"]
    CSV_EXTENSIONS = [".csv"]
    
    # Report files
    FINAL_REPORT_FILE = "FR.pdf"
    PRELIMINARY_REPORT_FILE = "PR.pdf"
    
    # Manual
    MANUAL_PATH = os.path.join("manual", "user_manual.pdf")
    
    # Icons and UI files
    UI_DIR = "ui"
    ICONS_DIR = os.path.join(UI_DIR, "icons")
    
    LANDING_UI = os.path.join(UI_DIR, "landing.ui")
    MAIN_WINDOW_UI = os.path.join(UI_DIR, "main_window.ui")
    GRAPHS_UI = os.path.join(UI_DIR, "graphs_ui.py")
    
    # Icon files
    ARROW_DOWN_SVG = os.path.join(ICONS_DIR, "arrow_down.svg")
    ARROW_UP_SVG = os.path.join(ICONS_DIR, "arrow_up.svg")
    ARROW_LEFT_SVG = os.path.join(ICONS_DIR, "arrow_left.svg")
    ARROW_RIGHT_SVG = os.path.join(ICONS_DIR, "arrow_right.svg")
    VDT_ANIMATION_GIF = os.path.join(ICONS_DIR, "VDT_ani.gif")
    VDT_WATERMARK_HTML = os.path.join(ICONS_DIR, "VDT_watermark.html")
    
    # Backend files
    BACKEND_DIR = "backend"
    BACKEND_FILES_DIR = os.path.join(BACKEND_DIR, "files")
    
    # Temporary file patterns
    TEMP_TALLY_PREFIX = "pipe_tally_"
    
    # Google Earth paths by platform
    GOOGLE_EARTH_PATHS = {
        "win32": r"C:\Program Files\Google\Google Earth Pro\client\googleearth.exe",
        "darwin": "/Applications/Google Earth Pro.app/Contents/MacOS/Google Earth Pro", 
        "linux": "/usr/bin/google-earth-pro"
    }

# ================================
# TIMING & PERFORMANCE SETTINGS
# ================================

class TimingSettings:
    """Timing and performance configurations"""
    
    # Splash screen
    SPLASH_DISPLAY_TIME = 1200  # milliseconds
    
    # Loading delays and timeouts
    THREAD_TIMEOUT = 30000      # 30 seconds
    UI_UPDATE_INTERVAL = 100    # milliseconds
    TABLE_FILL_DELAY = 0        # milliseconds between chunks
    
    # Timer intervals
    STATUS_TIMER_INTERVAL = 100  # milliseconds
    SCROLLBAR_NUDGE_DELAY = 100  # milliseconds
    SCROLLBAR_ARM_DELAY = 120    # milliseconds
    SCROLLBAR_SETUP_DELAY = 500  # milliseconds
    
    # Batch processing
    DEFAULT_CHUNK_SIZE = 300    # rows per batch
    MIN_CHUNK_SIZE = 50         # minimum chunk size
    MAX_CHUNK_SIZE = 500        # maximum chunk size
    
    # Event processing
    EVENT_PROCESSING_TIME = 50   # milliseconds

# ================================
# DEVELOPMENT & DEBUG SETTINGS
# ================================

class DebugSettings:
    """Debug and development settings"""
    
    DEBUG_MODE = False
    VERBOSE_LOGGING = False
    SHOW_PERFORMANCE_METRICS = False
    
    # Console output colors (if implementing colored logging)
    LOG_COLORS = {
        'DEBUG': '\033[36m',    # Cyan
        'INFO': '\033[32m',     # Green  
        'WARNING': '\033[33m',  # Yellow
        'ERROR': '\033[31m',    # Red
        'CRITICAL': '\033[35m', # Magenta
        'RESET': '\033[0m'      # Reset
    }

# ================================
# EXPORTED SETTINGS OBJECTS
# ================================

# Create instances for easy import
UI_CONSTRAINTS = UIConstraints()
SCROLL_SETTINGS = ScrollSettings()
CHART_SETTINGS = ChartSettings()
VIEW_SETTINGS = ViewSettings()
FILE_SETTINGS = FileSettings()
TIMING_SETTINGS = TimingSettings()
DEBUG_SETTINGS = DebugSettings()

# Platform detection
CURRENT_PLATFORM = sys.platform
IS_FROZEN = getattr(sys, 'frozen', False)

def get_google_earth_path():
    """Get Google Earth Pro path for current platform"""
    return FILE_SETTINGS.GOOGLE_EARTH_PATHS.get(CURRENT_PLATFORM, "")

def is_windows():
    """Check if running on Windows"""
    return CURRENT_PLATFORM == "win32"

def is_mac():
    """Check if running on macOS"""
    return CURRENT_PLATFORM == "darwin"

def is_linux():
    """Check if running on Linux"""
    return CURRENT_PLATFORM.startswith("linux")
