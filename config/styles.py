# config/styles.py
"""
Application styles and themes
"""

# ================================
# SCROLLBAR STYLES
# ================================
SCROLLBAR_STYLE = """
QScrollBar:vertical {
    background: #2b2b2b;
    width: 14px;
}
QScrollBar::handle:vertical {
    background: #555;
    min-height: 20px;
}
QScrollBar::handle:vertical:hover {
    background: #777;
}
QScrollBar:horizontal {
    background: #2b2b2b;
    height: 14px;
}
QScrollBar::handle:horizontal {
    background: #555;
    min-width: 20px;
}
QScrollBar::handle:horizontal:hover {
    background: #777;
}
"""

# ================================
# MENU BAR STYLES
# ================================
MENU_BAR_STYLE = """
QMenuBar {
    background-color: #000000;
    color: white;
}
QMenuBar::item {
    background: transparent;
    padding: 4px 12px;
}
QMenuBar::item:selected {
    background: #333333;
    color: white;
}

/* Dropdown menus stay white */
QMenu {
    background-color: #ffffff;
    color: black;
    border: 1px solid #cccccc;
}
QMenu::item:selected {
    background: #c0c0c0;
    color: #000000;
}
"""

# ================================
# BUTTON STYLES
# ================================
class ButtonStyles:
    """Button style definitions"""

    LOAD_BUTTON = """
        QPushButton {
            background-color: #3498db;
            color: white;
            border: 1px solid #2980b9;
            border-radius: 6px;
            padding: 4px 12px;
            font-weight: 500;
        }
        QPushButton:hover {
            background-color: #2980b9;
        }
        QPushButton:pressed {
            background-color: #1f5f8a;
        }
        QPushButton:disabled {
            background-color: #a6a6a6;
            color: #f0f0f0;
            border: 2px solid #6e6e6e;
        }
    """

    DIGSHEET_BUTTON = """
        QPushButton {
            background: white;
            border: 1px solid #3498db;
            color: #3498db;
            border-radius: 6px;
            padding: 4px 12px;
            font-weight: 500;
        }
        QPushButton:hover {
            background: #ecf6fd;
        }
        QPushButton:pressed {
            background: #d0e9fa;
        }
        QPushButton:disabled {
            color: #a0a0a0;
            background: #f5f5f5;
            border: 2px solid #6e6e6e;
        }
    """

BUTTON_STYLES = ButtonStyles()

# ================================
# COMBOBOX STYLES
# ================================
def get_combobox_style(arrow_path: str) -> str:
    """Get combobox style with arrow path"""
    return f"""
        QComboBox {{
            padding: 4px 8px;
            border: 2px solid #000000;
            border-radius: 6px;
            background: white;
        }}
        QComboBox::drop-down {{
            subcontrol-origin: padding;
            subcontrol-position: top right;
            width: 24px;
            border-left: 2px solid #000000;
        }}
        QComboBox::down-arrow {{
            image: url({arrow_path});
            width: 12px;
            height: 12px;
        }}
        QComboBox QAbstractItemView {{
            border: 2px solid #000000;
            selection-background-color: #3498db;
            selection-color: white;
        }}
    """

# ================================
# DIALOG STYLES
# ================================
class DialogStyles:
    """Dialog style definitions"""

    MODERN_LOADING = """
        QDialog {
            background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
                stop:0 #f0f0f0, stop:1 #e0e0e0);
            border: 2px solid #3498db;
            border-radius: 10px;
        }
        QLabel {
            color: #2c3e50;
            font-family: 'Segoe UI', Arial;
        }
        QProgressBar {
            border: 2px solid #bdc3c7;
            border-radius: 8px;
            background-color: #ecf0f1;
            text-align: center;
            font-weight: bold;
            color: #2c3e50;
        }
        QProgressBar::chunk {
            background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                stop:0 #3498db, stop:1 #2980b9);
            border-radius: 6px;
        }
    """

DIALOG_STYLES = DialogStyles()

# ================================
# TABLE STYLES
# ================================
class TableStyles:
    """Table style definitions"""

    HEADER_STYLE = """
        QHeaderView::section {
            font-weight: bold;
            background-color: #f0f0f0;
            border: 1px solid #d0d0d0;
            padding: 5px;
            text-align: center;
        }
    """

    VERTICAL_HEADER_STYLE = """
        QHeaderView::section {
            font-weight: bold;
            background-color: #f0f0f0;
            border: 1px solid #d0d0d0;
            padding: 5px;
            text-align: center;
            min-width: 40px;
        }
    """

TABLE_STYLES = TableStyles()

# ================================
# OVERLAY STYLES
# ================================
class OverlayStyles:
    """Overlay and message style definitions"""

    NO_DEFECTS_LABEL = """
        QLabel {
            font-size: 16pt;
            color: #666666;
            font-weight: bold;
            background-color: #f8f8f8;
            border: 2px dashed #cccccc;
            border-radius: 10px;
            padding: 20px;
            margin: 10px;
        }
    """

    SELECT_PIPE_CONTAINER = "background-color: rgba(255, 255, 255, 180);"
    SELECT_PIPE_CARD = """
        QFrame {
            background-color: #ffffff;
            border-radius: 16px;
            border: 1px solid #d0d0d0;
            padding: 30px;
        }
    """
    SELECT_PIPE_TITLE = "font-size: 22pt; font-weight: 600; color: #2c3e50;"
    SELECT_PIPE_SUBTITLE = "font-size: 12pt; color: #555; margin-top: 10px;"
    SELECT_PIPE_HINT = "font-size: 10pt; color: #888; margin-top: 15px;"

    CREATE_PROJECT_CONTAINER = "background-color: rgba(245, 247, 250, 200);"
    CREATE_PROJECT_CARD = """
        QFrame {
            background-color: #ffffff;
            border-radius: 14px;
            border: 1px solid #e0e0e0;
            padding: 30px 20px;
        }
    """
    CREATE_PROJECT_TITLE = "font-size: 20pt; font-weight: 600; color: #2c3e50;"
    CREATE_PROJECT_SUBTITLE = "font-size: 12pt; color: #555;"

    LOADING_DIALOG_TITLE  = "font-size: 16px; font-weight: bold; margin-bottom: 10px;"
    LOADING_DIALOG_STATUS = "font-size: 12px; color: #7f8c8d;"
    LOADING_DIALOG_TIME   = "font-size: 10px; color: #95a5a6;"

OVERLAY_STYLES = OverlayStyles()

# ================================
# SPLITTER STYLES
# ================================
SPLITTER_STYLE = """
    QSplitter::handle#MidBarHandle { background: #16181c; }
    #MidBarFrame { background: #16181c; }
    QTabBar::tab {
        color: #d8d8d8;
        padding: 6px 14px;
        margin: 0px;
        border: 0;
        background: transparent;
    }
    QTabBar::tab:selected {
        color: white;
        font-weight: 600;
    }
"""

# ================================
# MAIN WINDOW STYLE
# ================================
MAIN_WINDOW_STYLE = """
    QMainWindow {
        background-color: #FFFFFF;
        color: #000000;
    }
"""

# ================================
# CUSTOM SCROLLBAR THEME BUILDER
# ================================
def build_scrollbar_theme(handle_radius: int, btn_wh: int, bar_h: int, bar_w: int,
                          left_path: str, right_path: str, up_path: str, down_path: str) -> tuple[str, str]:
    """
    Build horizontal and vertical QSS for the custom top bars and vertical bars.
    Returns (horizontal_style, vertical_style).
    """
    h_style = f"""
    QScrollBar#proxTopBar:horizontal,
    QScrollBar#mainTopBar:horizontal,
    QScrollBar#tableTopBar:horizontal {{
        height:{bar_h}px;
        background: transparent;
        margin: 0 {btn_wh + 3}px 0 {btn_wh + 3}px;
        padding: 0;
        border: 0;
    }}

    /* handle (thumb) */
    QScrollBar#proxTopBar::handle:horizontal,
    QScrollBar#mainTopBar::handle:horizontal,
    QScrollBar#tableTopBar::handle:horizontal {{
        min-width: 40px;
        border-radius:{handle_radius}px;
        border:1px solid rgba(0,0,0,0.18);
        background:#d9d9d9;
    }}
    QScrollBar#proxTopBar::handle:horizontal:hover,
    QScrollBar#mainTopBar::handle:horizontal:hover,
    QScrollBar#tableTopBar::handle:horizontal:hover {{
        background:#bfbfbf; border-color:rgba(0,0,0,0.28);
    }}
    QScrollBar#proxTopBar::handle:horizontal:pressed,
    QScrollBar#mainTopBar::handle:horizontal:pressed,
    QScrollBar#tableTopBar::handle:horizontal:pressed {{
        background:#9a9a9a; border-color:rgba(0,0,0,0.38);
    }}

    /* arrows */
    QScrollBar#proxTopBar::sub-line:horizontal,
    QScrollBar#mainTopBar::sub-line:horizontal,
    QScrollBar#tableTopBar::sub-line:horizontal {{
        width:{btn_wh}px; height:{btn_wh}px;
        subcontrol-origin: margin;
        subcontrol-position: left;
        border: none;
        border-radius:{btn_wh//2}px;
        background:#e9e9e9;
        image: url("{left_path}");
    }}
    QScrollBar#proxTopBar::add-line:horizontal,
    QScrollBar#mainTopBar::add-line:horizontal,
    QScrollBar#tableTopBar::add-line:horizontal {{
        width:{btn_wh}px; height:{btn_wh}px;
        subcontrol-origin: margin;
        subcontrol-position: right;
        border: none;
        border-radius:{btn_wh//2}px;
        background:#e9e9e9;
        image: url("{right_path}");
    }}

    /* hover states */
    QScrollBar#proxTopBar::sub-line:horizontal:hover,
    QScrollBar#mainTopBar::sub-line:horizontal:hover,
    QScrollBar#tableTopBar::sub-line:horizontal:hover,
    QScrollBar#proxTopBar::add-line:horizontal:hover,
    QScrollBar#mainTopBar::add-line:horizontal:hover,
    QScrollBar#tableTopBar::add-line:horizontal:hover {{
        background:#d6d6d6;
    }}
    QScrollBar#proxTopBar::sub-line:horizontal:pressed,
    QScrollBar#mainTopBar::sub-line:horizontal:pressed,
    QScrollBar#tableTopBar::sub-line:horizontal:pressed,
    QScrollBar#proxTopBar::add-line:horizontal:pressed,
    QScrollBar#mainTopBar::add-line:horizontal:pressed,
    QScrollBar#tableTopBar::add-line:horizontal:pressed {{
        background:#c2c2c2;
    }}

    /* pages transparent */
    QScrollBar#proxTopBar::add-page:horizontal,
    QScrollBar#proxTopBar::sub-page:horizontal,
    QScrollBar#mainTopBar::add-page:horizontal,
    QScrollBar#mainTopBar::sub-page:horizontal,
    QScrollBar#tableTopBar::add-page:horizontal,
    QScrollBar#tableTopBar::sub-page:horizontal {{
        background: transparent;
    }}
    """

    v_style = f"""
    QScrollBar:vertical {{
        width:{bar_w}px;
        margin:{btn_wh + 8}px 0;
        background: transparent;
    }}
    QScrollBar::handle:vertical {{
        min-height:40px;
        border-radius:{handle_radius}px;
        border:1px solid rgba(0,0,0,0.18);
        background:#d9d9d9;
    }}
    QScrollBar::handle:vertical:hover  {{ background:#bfbfbf; border-color:rgba(0,0,0,0.28); }}
    QScrollBar::handle:vertical:pressed{{ background:#9a9a9a; border-color:rgba(0,0,0,0.38); }}

    QScrollBar::sub-line:vertical {{
        height:{btn_wh}px; width:{btn_wh}px;
        subcontrol-origin: margin;
        subcontrol-position: top;
        border:none; border-radius:{btn_wh//2}px;
        background:#e9e9e9;
        image: url("{up_path}");
    }}
    QScrollBar::add-line:vertical {{
        height:{btn_wh}px; width:{btn_wh}px;
        subcontrol-origin: margin;
        subcontrol-position: bottom;
        border:none; border-radius:{btn_wh//2}px;
        background:#e9e9e9;
        image: url("{down_path}");
    }}
    QScrollBar::sub-line:vertical:hover,
    QScrollBar::add-line:vertical:hover {{ background:#d6d6d6; }}
    QScrollBar::sub-line:vertical:pressed,
    QScrollBar::add-line:vertical:pressed {{ background:#c2c2c2; }}

    QScrollBar::add-page:vertical,
    QScrollBar::sub-page:vertical {{ background: transparent; }}
    """

    return h_style, v_style

# ================================
# ERROR DIALOG STYLE
# ================================
ERROR_DIALOG_TEXT_STYLE = """
    font-size: 10pt;
    font-family: Consolas;
    color: #aa0000;
"""

# ================================
# WEB VIEW FORCE SCROLLBAR CSS
# ================================
WEB_VIEW_SCROLLBAR_CSS = """
::-webkit-scrollbar {
    width: 16px !important;
    height: 16px !important;
    display: block !important;
}
::-webkit-scrollbar-track {
    background: #f0f0f0 !important;
}
::-webkit-scrollbar-thumb {
    background: #888 !important;
    border-radius: 4px !important;
}
html, body {
    overflow: scroll !important;
}
"""

# ================================
# CHART WRAPPER HTML STYLES
# ================================
CHART_WRAPPER_CSS = """
* {
    scrollbar-width: auto !important;
    -webkit-appearance: auto !important;
}
html, body {
    height: 100%;
    margin: 0;
    overflow: hidden;
}
.wrap {
    height: 100vh;
    width: 100vw;
    overflow: scroll !important;
    overflow-x: scroll !important;
    overflow-y: scroll !important;
    scrollbar-width: auto !important;
    -ms-overflow-style: scrollbar !important;
}
.wrap::-webkit-scrollbar {
    width: 18px !important;
    height: 18px !important;
    background: #f5f5f5 !important;
    display: block !important;
}
.wrap::-webkit-scrollbar-track {
    background: #e0e0e0 !important;
    border: 1px solid #ccc !important;
}
.wrap::-webkit-scrollbar-thumb {
    background: #666 !important;
    border: 2px solid #999 !important;
    border-radius: 2px !important;
}
.wrap::-webkit-scrollbar-thumb:hover {
    background: #333 !important;
}
.wrap::-webkit-scrollbar-corner {
    background: #e0e0e0 !important;
}
iframe {
    border: 0;
    width: {width}px !important;
    height: {height}px !important;
    min-width: {width}px !important;
    min-height: {height}px !important;
    display: block;
}
"""
