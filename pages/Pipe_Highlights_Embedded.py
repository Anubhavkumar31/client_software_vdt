from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QLineEdit, 
    QFrame, QGridLayout, QSizePolicy, QScrollArea, QSplitter
)
from PyQt6.QtCore import Qt, pyqtSignal, QPropertyAnimation, QEasingCurve
from PyQt6.QtGui import QFont, QPalette, QColor, QLinearGradient, QBrush
from matplotlib.backends.backend_qtagg import FigureCanvasQTAgg as FigureCanvas
from matplotlib.figure import Figure
import pandas as pd
import numpy as np
from PyQt6.QtWidgets import QGraphicsDropShadowEffect
from PyQt6.QtGui import QColor
import os

class PipeHighlightEmbedded(QWidget):
    def __init__(self, parent=None, pipe_tally_df=None, project_root=None):
        super().__init__(parent)
        self.pipe_tally_df = pipe_tally_df if pipe_tally_df is not None else pd.DataFrame()
        self.project_root = project_root 
        print(f"🔍 Initializing Embedded Pipeline Highlights...")
        if not self.pipe_tally_df.empty:
            print(f"✅ Using loaded pipe tally data ({len(self.pipe_tally_df)} rows)")
        else:
            print("⚠️ No pipe tally data provided")
            
        # Initialize constants
        self._initialize_constants()
        
        # Calculate statistics
        self._calculate_statistics()
        
        # Setup UI
        self._setup_ui()
        

    def _initialize_constants(self):
        """Load pipeline constants from constants.xlsx or fallback to defaults"""
        constants_file = os.path.join(self.project_root, "constants.xlsx")
        print("🔍 Looking for constants at:", constants_file)
        if os.path.exists(constants_file):
            try:
                df = pd.read_excel(constants_file)
                row = df.iloc[0].to_dict()   # take first row as dictionary

                # Dynamically set attributes
                for key, value in row.items():
                    setattr(self, key, value)

                print(f"✅ Loaded constants from {constants_file}")
                return
            except Exception as e:
                print(f"⚠️ Failed to load constants.xlsx: {e}")

        # Fallback defaults if Excel missing or invalid
        print("⚠️ Using fallback default constants")
        self.CONTRACTOR = 'ZZZ'
        self.IP_TYPE = 'MFL'
        self.MEDIUM = 'Oil'
        self.TYPE_PIPE = 'ZZZ'
        self.GRADE_PIPE = 'ZZZ'
        self.DIA = 340
        self.WT = 7.1
        self.DP = 3.67
        self.OP = 0
        self.MAOP = 11
        self.DF = 0.72
        self.UTS = 413.686
        self.SMYS = 2493.8

        
    def _calculate_statistics(self):
        """Calculate statistics from the loaded DataFrame"""
        print("🔍 Calculating statistics...")
        
        if self.pipe_tally_df.empty:
            print("⚠️ DataFrame is empty, using default values")
            self.TOT_ANAL = 0
            self.INT_ANAL = 0
            self.EXT_ANAL = 0
            self.ERF_95 = 0
            self.ERF_95_1 = 0
            self.ERF_1 = 0
            self.DEP_25 = 0
            self.DEP_25_50 = 0
            self.DEP_50_80 = 0
            self.DEP_80_100 = 0
            return

        try:
            # Total anomalies
            self.TOT_ANAL = len(self.pipe_tally_df)
            
            # Internal/External anomalies
            type_col = None
            for col in ['Type', 'Feature Type', 'Anomaly Type']:
                if col in self.pipe_tally_df.columns:
                    type_col = col
                    break
            
            if type_col:
                type_series = self.pipe_tally_df[type_col].astype(str).str.lower()
                self.INT_ANAL = len(self.pipe_tally_df[type_series.str.contains('internal', na=False)])
                self.EXT_ANAL = len(self.pipe_tally_df[type_series.str.contains('external', na=False)])
            else:
                self.INT_ANAL = 0
                self.EXT_ANAL = 0

            # ERF statistics
            erf_col = None
            for col in ['ERF (ASME B31G)', 'ERF', 'Engineering Risk Factor']:
                if col in self.pipe_tally_df.columns:
                    erf_col = col
                    break
            
            if erf_col:
                erf_data = pd.to_numeric(self.pipe_tally_df[erf_col], errors='coerce').dropna()
                if len(erf_data) > 0:
                    self.ERF_95 = len(erf_data[erf_data < 0.95])
                    self.ERF_95_1 = len(erf_data[(erf_data >= 0.95) & (erf_data < 1)])
                    self.ERF_1 = len(erf_data[erf_data >= 1])
                else:
                    self.ERF_95 = self.ERF_95_1 = self.ERF_1 = 0
            else:
                self.ERF_95 = self.ERF_95_1 = self.ERF_1 = 0

            # Depth statistics
            depth_col = None
            for col in ['Depth %', 'Depth % ', 'Depth Percentage']:
                if col in self.pipe_tally_df.columns:
                    depth_col = col
                    break
            
            if depth_col:
                depth_data = pd.to_numeric(self.pipe_tally_df[depth_col], errors='coerce').dropna()
                if len(depth_data) > 0:
                    self.DEP_25 = len(depth_data[depth_data < 25])
                    self.DEP_25_50 = len(depth_data[(depth_data >= 25) & (depth_data < 50)])
                    self.DEP_50_80 = len(depth_data[(depth_data >= 50) & (depth_data < 80)])
                    self.DEP_80_100 = len(depth_data[(depth_data >= 80) & (depth_data <= 100)])
                else:
                    self.DEP_25 = self.DEP_25_50 = self.DEP_50_80 = self.DEP_80_100 = 0
            else:
                self.DEP_25 = self.DEP_25_50 = self.DEP_50_80 = self.DEP_80_100 = 0

            print("✅ Statistics calculation completed successfully")

        except Exception as e:
            print(f"❌ Error calculating statistics: {e}")
            # Set default values on error
            self.TOT_ANAL = len(self.pipe_tally_df) if not self.pipe_tally_df.empty else 0
            self.INT_ANAL = self.EXT_ANAL = 0
            self.ERF_95 = self.ERF_95_1 = self.ERF_1 = 0
            self.DEP_25 = self.DEP_25_50 = self.DEP_50_80 = self.DEP_80_100 = 0

    def _setup_ui(self):
        """Setup the main UI layout with PREMIUM DESIGN"""
        # ✅ Set size policy for main widget
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # ✨ PREMIUM: Set main widget background with gradient
        self.setStyleSheet("""
            QWidget {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:1,
                    stop:0 #f8fafc, stop:0.5 #f1f5f9, stop:1 #e2e8f0);
                font-family: 'Segoe UI', 'Arial', sans-serif;
            }
        """)
        
        # Main layout for the widget
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)
        
        # ✅ Create scroll area with PREMIUM styling
        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        scroll_area.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll_area.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # ✨ PREMIUM: Custom scrollbar styling
        scroll_area.setStyleSheet("""
            QScrollArea {
                border: none;
                background: transparent;
            }
            QScrollBar:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #e2e8f0, stop:1 #cbd5e1);
                width: 12px;
                border-radius: 6px;
            }
            QScrollBar::handle:vertical {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #64748b, stop:1 #475569);
                border-radius: 6px;
                min-height: 30px;
            }
            QScrollBar::handle:vertical:hover {
                background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
                    stop:0 #475569, stop:1 #334155);
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
        """)
        
        # ✅ Create the scrollable content widget
        content_widget = QWidget()
        content_widget.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        content_widget.setMinimumHeight(1600)
        
        content_layout = QVBoxLayout(content_widget)
        content_layout.setContentsMargins(25, 25, 25, 25)
        content_layout.setSpacing(25)
        
        # # ✨ PREMIUM: Enhanced title with gradient and shadow effect
        # title = QLabel("🔧 Pipeline Highlights Dashboard")
        # title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        # title_font = QFont("Segoe UI", 24, QFont.Weight.Bold)
        # title.setFont(title_font)
        # title.setStyleSheet("""
        #     QLabel {
        #         color: qlineargradient(x1:0, y1:0, x2:1, y2:0,
        #             stop:0 #1e40af, stop:0.5 #3b82f6, stop:1 #60a5fa);
        #         padding: 20px;
        #         margin: 15px;
        #         background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
        #             stop:0 rgba(255,255,255,0.9), stop:1 rgba(248,250,252,0.9));
        #         border-radius: 16px;
        #         border: 2px solid rgba(59, 130, 246, 0.2);
        #     }
        # """)
        # content_layout.addWidget(title)
        self._create_hero_header(content_layout, title="Pipeline Highlights Dashboard",
                         subtitle="Comprehensive Overview of Pipeline")

        
        # ✅ General Info Section with PREMIUM design
        self._create_general_info_section(content_layout)
        
        # ✅ Statistics and Charts Section with PREMIUM design
        self._create_statistics_section(content_layout)
        
        # ✅ Add stretch
        content_layout.addStretch(1)
        
        # Set the content widget to scroll area
        scroll_area.setWidget(content_widget)
        
        # Add scroll area to main layout
        main_layout.addWidget(scroll_area)
        
        print("✅ PREMIUM UI setup completed successfully!")

    # def _create_general_info_section(self, parent_layout):
    #     """Create the PREMIUM general information section"""
    #     # ✨ PREMIUM: Glass-morphism effect frame
    #     info_frame = QFrame()
    #     info_frame.setFrameStyle(QFrame.Shape.NoFrame)
    #     info_frame.setStyleSheet("""
    #         QFrame {
    #             background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
    #                 stop:0 rgba(255,255,255,0.95), 
    #                 stop:0.5 rgba(248,250,252,0.9), 
    #                 stop:1 rgba(241,245,249,0.95));
    #             border: 1px solid rgba(148, 163, 184, 0.3);
    #             border-radius: 20px;
    #             padding: 25px;
    #             margin: 15px;
    #         }
    #     """)
        
    #     info_layout = QVBoxLayout(info_frame)
    #     info_layout.setSpacing(20)
        
    #     # ✨ PREMIUM: Enhanced section title with icon
    #     info_title = QLabel("📊 General Pipeline Information")
    #     info_title_font = QFont("Segoe UI", 18, QFont.Weight.Bold)
    #     info_title.setFont(info_title_font)
    #     info_title.setStyleSheet("""
    #         QLabel {
    #             color: #1e293b;
    #             padding: 12px 20px;
    #             background: qlineargradient(x1:0, y1:0, x2:1, y2:0,
    #                 stop:0 rgba(59, 130, 246, 0.1), 
    #                 stop:1 rgba(147, 197, 253, 0.1));
    #             border-left: 5px solid #3b82f6;
    #             border-radius: 12px;
    #             margin-bottom: 10px;
    #         }
    #     """)
    #     info_layout.addWidget(info_title)
        
    #     # ✨ PREMIUM: Enhanced grid with better styling
    #     info_grid = QGridLayout()
    #     info_grid.setSpacing(18)
    #     info_grid.setColumnStretch(1, 1)
    #     info_grid.setColumnStretch(3, 1)
    #     info_grid.setColumnStretch(5, 1)
        
    #     # Info data organized in 3 columns
    #     info_items = [
    #         # Column 1
    #         [("🏢 Contractor:", self.CONTRACTOR),
    #          ("🔍 IP Type:", self.IP_TYPE),
    #          ("🛢️ Medium:", self.MEDIUM),
    #          ("⚙️ Type of Pipe:", self.TYPE_PIPE),
    #          ("📋 Grade of Pipe:", self.GRADE_PIPE)],
    #         # Column 2
    #         [("📏 Diameter (mm):", str(self.DIA)),
    #          ("📐 Wall Thickness (mm):", str(self.WT)),
    #          ("🔺 Design Pressure (MPa):", str(self.DP)),
    #          ("⚡ Operating Pressure (MPa):", str(self.OP)),
    #          ("", "")],
    #         # Column 3
    #         [("🔋 MAOP (MPa):", str(self.MAOP)),
    #          ("⚖️ Design Factor:", str(self.DF)),
    #          ("💪 UTS (MPa):", str(self.UTS)),
    #          ("🔩 SMYS (MPa):", str(self.SMYS)),
    #          ("", "")]
    #     ]
        
    #     # ✨ PREMIUM: Enhanced grid items with better styling
    #     for col_idx, column_items in enumerate(info_items):
    #         for row_idx, (label_text, value_text) in enumerate(column_items):
    #             if label_text:
    #                 # ✨ PREMIUM: Enhanced label styling
    #                 label = QLabel(label_text)
    #                 label.setStyleSheet("""
    #                     QLabel {
    #                         font-weight: 600; 
    #                         color: #374151; 
    #                         font-size: 13pt;
    #                         padding: 8px 12px;
    #                         background: rgba(249, 250, 251, 0.8);
    #                         border-radius: 8px;
    #                         border-left: 3px solid #6366f1;
    #                     }
    #                 """)
    #                 label.setAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
    #                 info_grid.addWidget(label, row_idx, col_idx * 2)
                    
    #                 # ✨ PREMIUM: Enhanced value input styling
    #                 value_edit = QLineEdit(value_text)
    #                 value_edit.setReadOnly(True)
    #                 value_edit.setFixedWidth(130)
    #                 value_edit.setFixedHeight(42)
    #                 value_edit.setStyleSheet("""
    #                     QLineEdit {
    #                         background: qlineargradient(x1:0, y1:0, x2:0, y2:1,
    #                             stop:0 #ffffff, stop:1 #f8fafc);
    #                         border: 2px solid #e2e8f0;
    #                         padding: 10px 15px;
    #                         border-radius: 10px;
    #                         font-size: 12pt;
    #                         font-weight: 600;
    #                         color: #1f2937;
    #                         selection-background-color: #3b82f6;
    #                     }
    #                     QLineEdit:focus {
    #                         border: 2px solid #3b82f6;
    #                         background: #ffffff;
    #                     }
    #                     QLineEdit:hover {
    #                         border: 2px solid #60a5fa;
    #                         background: #ffffff;
    #                     }
    #                 """)
    #                 info_grid.addWidget(value_edit, row_idx, col_idx * 2 + 1)
        
    #     info_layout.addLayout(info_grid)
    #     parent_layout.addWidget(info_frame)

    def _create_hero_header(self, parent_layout, title: str, subtitle: str | None = None):
        ACCENT_A = "#2563eb"   # blue-600
        ACCENT_B = "#60a5fa"   # sky-400
        BORDER   = "rgba(148,163,184,0.35)"  # slate-400 @ 35%
        TITLE    = "#0f172a"   # slate-900
        SUBTITLE = "#64748b"   # slate-500
        CHIP_BG  = "#f8fafc"   # slate-50
        CHIP_BR  = "#e2e8f0"   # slate-200
        CHIP_TX  = "#334155"   # slate-700

        header = QFrame()
        header.setObjectName("heroHeader")
        header.setStyleSheet(f"""
            QFrame#heroHeader {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #ffffff, stop:1 #f9fafb);
                border: 1px solid {BORDER};
                border-radius: 18px;
            }}
            QFrame#accentBar {{
                background: qlineargradient(x1:0,y1:0,x2:0,y2:1, stop:0 {ACCENT_A}, stop:1 {ACCENT_B});
                border-radius: 6px;
            }}
            QLabel#heroTitle {{
                color: {TITLE};
                font-family: 'Segoe UI','Arial';
                font-size: 22pt;                /* clean, not cartoonish */
                font-weight: 800;               /* bold but classy */
                letter-spacing: 0.2px;
            }}
            QLabel#heroSubtitle {{
                color: {SUBTITLE};
                font-size: 11.5pt;
                font-weight: 500;
            }}
            QLabel#chip {{
                background: {CHIP_BG};
                border: 1px solid {CHIP_BR};
                color: {CHIP_TX};
                padding: 6px 10px;
                border-radius: 9999px;          /* pill */
                font-size: 10.5pt;
                font-weight: 600;
            }}
        """)
        # soft drop shadow for the white card
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(30)             # how “spread out” the shadow looks
        shadow.setOffset(0, 6)               # x=0, y=6px → shadow below the card
        shadow.setColor(QColor(15, 23, 42, 30))  # RGBA (dark blue-grey with low alpha)
        header.setGraphicsEffect(shadow)


        # soft drop shadow for the white card
        shadow = QGraphicsDropShadowEffect(self)
        shadow.setBlurRadius(24)
        shadow.setOffset(0, 6)
        shadow.setColor(QColor(15, 23, 42, 40))  # slate-900 @ ~16% alpha
        header.setGraphicsEffect(shadow)

        # layout structure
        h = QHBoxLayout(header)
        h.setContentsMargins(18, 14, 18, 14)
        h.setSpacing(16)

        accent = QFrame()
        accent.setObjectName("accentBar")
        accent.setFixedWidth(10)
        h.addWidget(accent)

        # text column (title + subtitle)
        text_col = QVBoxLayout()
        text_col.setSpacing(2)
        title_lbl = QLabel(title)
        title_lbl.setObjectName("heroTitle")
        title_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_lbl = QLabel(subtitle or "")
        subtitle_lbl.setObjectName("heroSubtitle")
        subtitle_lbl.setAlignment(Qt.AlignmentFlag.AlignCenter)
        subtitle_lbl.setVisible(bool(subtitle))

        text_col.addWidget(title_lbl)
        text_col.addWidget(subtitle_lbl)
        h.addLayout(text_col, 1)

        # right-side meta chips (date/version/etc.)
        meta = QHBoxLayout()
        meta.setSpacing(8)

        # Example chips: “Report” and current date; customize or remove as needed
        chip1 = QLabel("Report")
        chip1.setObjectName("chip")

        from datetime import datetime
        # chip2 = QLabel(datetime.now().strftime("%b %d, %Y"))
        # chip2.setObjectName("chip")

        meta.addWidget(chip1)
        # meta.addWidget(chip2)
        meta.addStretch(1)

        h.addLayout(meta, 0)

        parent_layout.addWidget(header)


    def _create_general_info_section(self, parent_layout):
        """General info section with white card + colored label chips + soft value boxes"""

        # ---- Theme knobs (easy to tweak later) ----
        ACCENT_BORDER = "#c7d2fe"   # indigo-200
        ACCENT_GRAD_A = "#eef2ff"   # indigo-50
        ACCENT_GRAD_B = "#e0e7ff"   # indigo-100
        VALUE_BG      = "#f8fafc"   # slate-50
        VALUE_BORDER  = "#e2e8f0"   # slate-200
        VALUE_BORDER_FOCUS = "#60a5fa"  # sky-400
        TEXT_PRIMARY  = "#1f2937"   # slate-800
        TITLE_BAR_BG1 = "rgba(59, 130, 246, 0.08)"  # blue-500 @ 8%
        TITLE_BAR_BG2 = "rgba(147, 197, 253, 0.08)" # blue-300 @ 8%

        # ---- Card (white) ----
        info_frame = QFrame()
        info_frame.setObjectName("infoFrame")
        info_frame.setFrameStyle(QFrame.Shape.NoFrame)
        info_frame.setStyleSheet(f"""
            QFrame#infoFrame {{
                background: #ffffff;                      /* ← pure white card */
                border: 1px solid rgba(148,163,184,0.35);
                border-radius: 18px;
                padding: 24px;
                margin: 12px 8px;
            }}
            /* Label chips */
            QLabel#infoLabelChip {{
                color: {TEXT_PRIMARY};
                font-size: 13pt;
                font-weight: 600;
                padding: 8px 12px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                            stop:0 {ACCENT_GRAD_A}, stop:1 {ACCENT_GRAD_B});
                border: 2px solid {ACCENT_BORDER};
                border-radius: 12px;
            }}
            /* Value boxes */
            QLineEdit#infoValueBox {{
                background: {VALUE_BG};
                border: 2px solid {VALUE_BORDER};
                border-radius: 12px;
                padding: 10px 14px;
                font-size: 12pt;
                font-weight: 600;
                color: {TEXT_PRIMARY};
                selection-background-color: {VALUE_BORDER_FOCUS};
            }}
            QLineEdit#infoValueBox:focus {{
                background: #ffffff;
                border: 2px solid {VALUE_BORDER_FOCUS};
            }}
            QLineEdit#infoValueBox:hover {{
                border: 2px solid #94a3b8;              /* slate-400 */
                background: #ffffff;
            }}
        """)

        info_layout = QVBoxLayout(info_frame)
        info_layout.setSpacing(18)
        info_layout.setContentsMargins(10, 6, 10, 10)

        # ---- Section title (subtle ribbon) ----
        info_title = QLabel("📊 General Pipeline Information")
        info_title.setObjectName("infoTitle")
        info_title.setStyleSheet(f"""
            QLabel#infoTitle {{
                color: #0f172a;                          /* slate-900 */
                font-family: 'Segoe UI', 'Arial';
                font-size: 18pt;
                font-weight: 700;
                padding: 12px 18px;
                background: qlineargradient(x1:0,y1:0,x2:1,y2:0,
                            stop:0 {TITLE_BAR_BG1}, stop:1 {TITLE_BAR_BG2});
                border-left: 6px solid #3b82f6;          /* blue-500 */
                border-radius: 12px;
            }}
        """)
        info_layout.addWidget(info_title)

        # ---- Grid layout (labels + values) ----
        grid = QGridLayout()
        grid.setHorizontalSpacing(18)
        grid.setVerticalSpacing(14)
        # label columns expand, value columns stay compact
        grid.setColumnStretch(0, 2)
        grid.setColumnStretch(1, 3)
        grid.setColumnStretch(2, 2)
        grid.setColumnStretch(3, 3)
        grid.setColumnStretch(4, 2)
        grid.setColumnStretch(5, 3)

        # Data (3 columns x 5 rows)
        info_items = [
            # col 0-1
            [("📋 Contractor:", self.CONTRACTOR),
            ("📏 Diameter (mm):", str(self.DIA)),
            ("🧪 MAOP (MPa):", str(self.MAOP))],

            # col 2-3
            [("🔎 IP Type:", self.IP_TYPE),
            ("📐 Wall Thickness (mm):", str(self.WT)),
            ("⚖️ Design Factor:", str(self.DF))],

            # col 4-5
            [("🛢️ Medium:", self.MEDIUM),
            ("🔺 Design Pressure (MPa):", str(self.DP)),
            ("💪 UTS (MPa):", str(self.UTS))],

            # next row
            [("⚙️ Type of Pipe:", self.TYPE_PIPE),
            ("⚡ Operating Pressure (MPa):", str(self.OP)),
            ("🔩 SMYS (MPa):", str(self.SMYS))],

            [("🧾 Grade of Pipe:", self.GRADE_PIPE),
            ("", ""), ("", "")]
        ]

        def add_field(r, c, label_text, value_text):
            if not label_text:
                return
            lbl = QLabel(label_text)
            lbl.setObjectName("infoLabelChip")
            lbl.setAlignment(Qt.AlignmentFlag.AlignVCenter | Qt.AlignmentFlag.AlignRight)

            val = QLineEdit(value_text)
            val.setObjectName("infoValueBox")
            val.setReadOnly(True)
            val.setMinimumWidth(160)
            val.setMinimumHeight(40)

            grid.addWidget(lbl, r, c)
            grid.addWidget(val, r, c + 1)

        # place fields: 3 pairs per row → columns (0,1), (2,3), (4,5)
        for row_idx, row in enumerate(info_items):
            # row is list of up to 3 (label, value) tuples
            for col_block, (label_text, value_text) in enumerate(row):
                add_field(row_idx, col_block * 2, label_text, value_text)

        info_layout.addLayout(grid)
        parent_layout.addWidget(info_frame)




    def _create_statistics_section(self, parent_layout):
        """Simple working statistics and charts section"""
        
        splitter = QSplitter(Qt.Orientation.Horizontal)
        
        # Simple stats frame
        stats_frame = QFrame()
        stats_frame.setFixedWidth(350)
        stats_frame.setStyleSheet("""
            QFrame {
                background: #f5f5f5;
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        
        stats_layout = QVBoxLayout(stats_frame)
        stats_layout.setSpacing(15)
        
        # Simple title
        stats_title = QLabel("📈 Pipeline Statistics")
        stats_title.setStyleSheet("""
            QLabel {
                font-size: 16pt;
                font-weight: bold;
                color: #333;
                padding: 10px;
                background: white;
                border-radius: 8px;
            }
        """)
        stats_layout.addWidget(stats_title)
        
        # Add groups
        self._create_stats_group(stats_layout, "🔢 Anomaly Counts", [
            ("Total Anomalies", self.TOT_ANAL, "#3b82f6"),
            ("Internal Anomalies", self.INT_ANAL, "#ef4444"),
            ("External Anomalies", self.EXT_ANAL, "#06b6d4"),
        ])
        
        self._create_stats_group(stats_layout, "⚡ ERF Analysis", [
            ("0.95 > ", self.ERF_95, "#10b981"),
            ("0.95 ≤ ERF < 1", self.ERF_95_1, "#f59e0b"),
            ("ERF ≥ 1", self.ERF_1, "#ef4444"),
        ])
        
        # self._create_stats_group(stats_layout, "📊 Depth Distribution", [
        #     ("Depth < 25%", self.DEP_25, "#8b5cf6"),
        #     ("25% ≤ Depth < 50%", self.DEP_25_50, "#10b981"),
        #     ("50% ≤ Depth < 80%", self.DEP_50_80, "#f59e0b"),
        #     ("80% ≤ Depth ≤ 100%", self.DEP_80_100, "#ef4444"),
        # ])
        
        stats_layout.addStretch()
        
        # Charts frame (keep your existing charts code)
        charts_frame = QFrame()
        charts_frame.setStyleSheet("""
            QFrame {
                background: white;
                border: 1px solid #ccc;
                border-radius: 10px;
                padding: 15px;
            }
        """)
        
        charts_layout = QVBoxLayout(charts_frame)
        
        charts_title = QLabel("📊 Data Visualization")
        charts_title.setStyleSheet("""
            QLabel {
                font-size: 16pt;
                font-weight: bold;
                color: #333;
                padding: 10px;
                text-align: center;
            }
        """)
        charts_title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        charts_layout.addWidget(charts_title)
        
        # Your existing matplotlib code
        self.figure = Figure(figsize=(8, 12), dpi=90, facecolor='white')
        self.canvas = FigureCanvas(self.figure)
        self.canvas.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.canvas.setMinimumSize(500, 650)
        
        charts_layout.addWidget(self.canvas, 1)
        self._create_charts()
        
        # Add to splitter
        splitter.addWidget(stats_frame)
        splitter.addWidget(charts_frame)
        splitter.setSizes([350, 950])
        
        parent_layout.addWidget(splitter, 1)



    # def _create_stats_group(self, parent_layout, group_title, stats_data):
    #     """Simple working statistics group"""
        
    #     # Simple group frame
    #     group_frame = QFrame()
    #     group_frame.setStyleSheet("""
    #         QFrame {
    #             background: white;
    #             border: 1px solid #ddd;
    #             border-radius: 8px;
    #             padding: 15px;
    #             margin: 5px;
    #         }
    #     """)
        
    #     group_layout = QVBoxLayout(group_frame)
    #     group_layout.setSpacing(8)
        
    #     # Simple header
    #     header = QLabel(group_title)
    #     header.setStyleSheet("""
    #         QLabel {
    #             font-size: 14pt;
    #             font-weight: bold;
    #             color: #333;
    #             padding-bottom: 10px;
    #             border-bottom: 1px solid #eee;
    #         }
    #     """)
    #     group_layout.addWidget(header)
        
    #     # Simple stats entries
    #     for label_text, value, color in stats_data:
    #         # Container for each stat
    #         stat_widget = QWidget()
    #         stat_layout = QHBoxLayout(stat_widget)
    #         stat_layout.setContentsMargins(5, 5, 5, 5)
            
    #         # Simple label
    #         label = QLabel(label_text + ":")
    #         label.setStyleSheet("font-size: 12pt; color: #555;")
            
    #         # Simple value
    #         value_label = QLabel(str(value))
    #         value_label.setFixedWidth(60)
    #         value_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    #         value_label.setStyleSheet(f"""
    #             QLabel {{
    #                 background: {color};
    #                 color: white;
    #                 font-weight: bold;
    #                 padding: 5px;
    #                 border-radius: 4px;
    #                 font-size: 12pt;
    #             }}
    #         """)
            
    #         stat_layout.addWidget(label)
    #         stat_layout.addStretch()
    #         stat_layout.addWidget(value_label)
            
    #         group_layout.addWidget(stat_widget)
        
    #     parent_layout.addWidget(group_frame)



    # def _create_stats_group(self, parent_layout, group_title, stats_data):
    #     group_frame = QFrame()
    #     group_frame.setStyleSheet("""
    #         QFrame {
    #             background: rgba(255, 255, 255, 0.86);
    #             border: 1px solid rgba(203, 213, 225, 0.6);
    #             border-radius: 12px;
    #             padding: 16px;
    #             margin: 8px 0;
    #         }
    #     """)
    #     outer = QVBoxLayout(group_frame)
    #     outer.setSpacing(12)
    #     outer.setContentsMargins(16, 12, 16, 12)

    #     header = QLabel(group_title)
    #     header.setTextFormat(Qt.TextFormat.PlainText)
    #     header.setStyleSheet("""
    #         QLabel {
    #             color: #111827;
    #             font-size: 15pt;
    #             font-weight: 700;
    #             padding: 8px 0 12px 0;
    #             border-bottom: 2px solid #e5e7eb;
    #         }
    #     """)
    #     outer.addWidget(header)

    #     grid = QGridLayout()
    #     grid.setContentsMargins(0, 0, 0, 0)
    #     grid.setHorizontalSpacing(12)
    #     grid.setVerticalSpacing(10)
    #     outer.addLayout(grid)

    #     for r, (label_text, value, color) in enumerate(stats_data):
    #         lbl = QLabel(label_text)  # no ":" so it wraps clean on tiny widths if ever needed
    #         lbl.setTextFormat(Qt.TextFormat.PlainText)  # ← critical for "<" texts
    #         lbl.setWordWrap(False)
    #         lbl.setMinimumHeight(36)
    #         lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    #         lbl.setStyleSheet("""
    #             QLabel {
    #                 color: #1f2937;                   /* darker for contrast */
    #                 font-size: 12.5pt;
    #                 font-weight: 600;
    #                 padding: 7px 10px;
    #                 background: rgba(249, 250, 251, 0.95);
    #                 border-left: 3px solid #6366f1;
    #                 border-radius: 8px;
    #             }
    #         """)

    #         val = QLabel(str(value))
    #         val.setAlignment(Qt.AlignmentFlag.AlignCenter)
    #         val.setMinimumHeight(36)
    #         val.setFixedWidth(120)
    #         val.setStyleSheet(f"""
    #             QLabel {{
    #                 background: #ffffff;
    #                 border: 2px solid {color};
    #                 border-radius: 10px;
    #                 font-size: 14pt;
    #                 font-weight: 700;
    #                 color: {color};
    #                 padding: 6px;
    #             }}
    #         """)

    #         grid.addWidget(lbl, r, 0)
    #         grid.addWidget(val, r, 1, alignment=Qt.AlignmentFlag.AlignRight)

    #     parent_layout.addWidget(group_frame)

    def _create_stats_group(self, parent_layout, group_title, stats_data):
        group_frame = QFrame()
        group_frame.setStyleSheet("""
            QFrame {
                background: rgba(255, 255, 255, 0.86);
                border: 1px solid rgba(203, 213, 225, 0.6);
                border-radius: 12px;
                padding: 16px;
                margin: 8px 0;
            }
        """)
        outer = QVBoxLayout(group_frame)
        outer.setSpacing(12)
        outer.setContentsMargins(16, 12, 16, 12)

        header = QLabel(group_title)
        header.setTextFormat(Qt.TextFormat.PlainText)
        header.setStyleSheet("""
            QLabel {
                color: #111827;
                font-size: 15pt;
                font-weight: 700;
                padding: 8px 0 12px 0;
                border-bottom: 2px solid #e5e7eb;
            }
        """)
        outer.addWidget(header)

        grid = QGridLayout()
        grid.setContentsMargins(0, 0, 0, 0)
        grid.setHorizontalSpacing(12)
        grid.setVerticalSpacing(10)
        grid.setColumnStretch(0, 4)   # label column grows
        grid.setColumnStretch(1, 2)   # value column stays compact
        outer.addLayout(grid)

        for r, (label_text, value, color) in enumerate(stats_data):
            lbl = QLabel(label_text)
            lbl.setTextFormat(Qt.TextFormat.PlainText)
            lbl.setWordWrap(True)  # allow two lines when narrow
            lbl.setMinimumHeight(36)
            lbl.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
            lbl.setStyleSheet("""
                QLabel {
                    color: #1f2937;
                    font-size: 12.5pt;
                    font-weight: 600;
                    padding: 7px 10px;
                    background: rgba(249, 250, 251, 0.95);
                    border-left: 3px solid #6366f1;
                    border-radius: 8px;
                }
            """)
            lbl.setToolTip(label_text)

            val = QLabel(str(value))
            val.setAlignment(Qt.AlignmentFlag.AlignCenter)
            val.setMinimumHeight(36)
            val.setFixedWidth(120)
            val.setStyleSheet(f"""
                QLabel {{
                    background: #ffffff;
                    border: 2px solid {color};
                    border-radius: 10px;
                    font-size: 14pt;
                    font-weight: 700;
                    color: {color};
                    padding: 6px;
                }}
            """)

            grid.addWidget(lbl, r, 0)
            grid.addWidget(val, r, 1, alignment=Qt.AlignmentFlag.AlignRight)

        parent_layout.addWidget(group_frame)



        

    def _create_charts(self):
        """Create PREMIUM styled pie charts"""
        try:
            print("🔍 Creating PREMIUM pie charts...")
            
            # ✨ PREMIUM: Clear and setup figure with premium styling
            self.figure.clear()
            self.figure.patch.set_facecolor('#fefefe')
            
            # ✨ Enhanced spacing and margins
            self.figure.subplots_adjust(
                top=0.96,
                bottom=0.04,
                left=0.08,
                right=0.72,
                hspace=0.75
            )
            
            axs = self.figure.subplots(2, 1)
            
            def autopct_func(pct):
                return f'{pct:.1f}%' if pct >= 1.5 else ''

            # ✨ PREMIUM: Enhanced color schemes
            premium_colors_1 = ['#ef4444', '#06b6d4']  # Red, Cyan
            premium_colors_2 = ['#10b981', '#f59e0b', '#ef4444']  # Green, Amber, Red  
            premium_colors_3 = ['#8b5cf6', '#10b981', '#f59e0b', '#dc2626']  # Purple, Green, Amber, Red

            # ✨ Chart 1: Internal/External Anomalies
            labels_1 = ['Internal ML', 'External ML']
            sizes_1 = [self.INT_ANAL, self.EXT_ANAL]
            
            if sum(sizes_1) > 0:
                wedges, texts, autotexts = axs[0].pie(
                    sizes_1, 
                    colors=premium_colors_1, 
                    autopct=autopct_func, 
                    startangle=90,
                    textprops={'fontsize': 12, 'fontweight': 'bold', 'color': 'white'},
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'antialiased': True}
                )
                for autotext in autotexts:
                    autotext.set_fontsize(13)
                    autotext.set_fontweight('bold')
                    
                legend1 = axs[0].legend(labels_1, loc='center left', bbox_to_anchor=(1.05, 0.5), 
                                       fontsize=12, frameon=True, fancybox=True, shadow=True)
                legend1.get_frame().set_facecolor('white')
                legend1.get_frame().set_alpha(0.9)
            else:
                axs[0].text(0.5, 0.5, '📊 No Data Available', ha='center', va='center', 
                           fontsize=16, color='#6b7280', transform=axs[0].transAxes)
            
            axs[0].set_title('🔴 Internal vs External Anomalies', fontsize=16, fontweight='bold', 
                           pad=35, color='#1f2937')

            # ✨ Chart 2: ERF Distribution
            labels_2 = ['ERF < 0.95', '0.95 ≤ ERF < 1', 'ERF ≥ 1']
            sizes_2 = [self.ERF_95, self.ERF_95_1, self.ERF_1]
            
            if sum(sizes_2) > 0:
                wedges, texts, autotexts = axs[1].pie(
                    sizes_2, 
                    colors=premium_colors_2, 
                    autopct=autopct_func, 
                    startangle=90,
                    textprops={'fontsize': 12, 'fontweight': 'bold', 'color': 'white'},
                    wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'antialiased': True}
                )
                for autotext in autotexts:
                    autotext.set_fontsize(13)
                    autotext.set_fontweight('bold')
                    
                legend2 = axs[1].legend(labels_2, loc='center left', bbox_to_anchor=(1.05, 0.5), 
                                       fontsize=12, frameon=True, fancybox=True, shadow=True)
                legend2.get_frame().set_facecolor('white')
                legend2.get_frame().set_alpha(0.9)
            else:
                axs[1].text(0.5, 0.5, '⚡ No Data Available', ha='center', va='center', 
                           fontsize=16, color='#6b7280', transform=axs[1].transAxes)
            
            axs[1].set_title('⚡ Engineering Risk Factor (ERF) Analysis', fontsize=16, fontweight='bold', 
                           pad=35, color='#1f2937')

            # # ✨ Chart 3: Depth Distribution
            # labels_3 = ['Shallow\n(< 25%)', 'Moderate\n(25% - 50%)', 'Deep\n(50% - 80%)', 'Critical\n(≥ 80%)']
            # sizes_3 = [self.DEP_25, self.DEP_25_50, self.DEP_50_80, self.DEP_80_100]
            
            # if sum(sizes_3) > 0:
            #     wedges, texts, autotexts = axs[2].pie(
            #         sizes_3, 
            #         colors=premium_colors_3, 
            #         autopct=autopct_func, 
            #         startangle=90,
            #         textprops={'fontsize': 12, 'fontweight': 'bold', 'color': 'white'},
            #         wedgeprops={'edgecolor': 'white', 'linewidth': 2, 'antialiased': True}
            #     )
            #     for autotext in autotexts:
            #         autotext.set_fontsize(13)
            #         autotext.set_fontweight('bold')
                    
            #     legend3 = axs[2].legend(labels_3, loc='center left', bbox_to_anchor=(1.05, 0.5), 
            #                            fontsize=12, frameon=True, fancybox=True, shadow=True)
            #     legend3.get_frame().set_facecolor('white')
            #     legend3.get_frame().set_alpha(0.9)
            # else:
            #     axs[2].text(0.5, 0.5, '📊 No Data Available', ha='center', va='center', 
            #                fontsize=16, color='#6b7280', transform=axs[2].transAxes)
            
            # axs[2].set_title('📊 Corrosion Depth Distribution Analysis', fontsize=16, fontweight='bold', 
            #                pad=35, color='#1f2937')

            # ✨ PREMIUM: Ensure all axes are equal and clean
            for ax in axs:
                ax.axis('equal')
                ax.set_facecolor('#fefefe')

            # ✨ Force canvas update
            self.canvas.draw()
            
            print("✅ PREMIUM pie charts created successfully")

        except Exception as e:
            print(f"❌ Error creating pie charts: {e}")
            import traceback
            traceback.print_exc()
            
            # Create error message in chart area
            self.figure.clear()
            ax = self.figure.add_subplot(111)
            ax.text(0.5, 0.5, f'❌ Error creating charts:\n{str(e)}', 
                    ha='center', va='center', fontsize=14, color='#ef4444',
                    transform=ax.transAxes, bbox=dict(boxstyle="round,pad=0.3", facecolor='#fee2e2'))
            ax.set_xlim(0, 1)
            ax.set_ylim(0, 1)
            ax.axis('off')
            self.canvas.draw()
