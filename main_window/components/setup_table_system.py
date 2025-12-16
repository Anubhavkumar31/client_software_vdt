from PIL.ImageQt import QPixmap
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QAbstractItemView, QWidget, QVBoxLayout, QFrame, QLabel, QSizePolicy


def setup_table_scroll(table):
    from PyQt6.QtWidgets import QHeaderView, QAbstractItemView, QAbstractScrollArea
    from PyQt6.QtCore import Qt

    # Show scrollbars when needed (or keep AlwaysOn if you prefer)
    table.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
    table.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOn)

    # per-pixel scrolling for smooth behavior
    table.setHorizontalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)
    table.setVerticalScrollMode(QAbstractItemView.ScrollMode.ScrollPerPixel)

    # don't let the view auto-adjust its size to contents (prevents hiding scrollbars)
    table.setSizeAdjustPolicy(QAbstractScrollArea.SizeAdjustPolicy.AdjustIgnored)

    # Configure horizontal header: interactive sizing and a large default width so total width > viewport
    header = table.horizontalHeader()
    header.setStretchLastSection(False)
    header.setSectionResizeMode(QHeaderView.ResizeMode.Interactive)

    # <- Increase default section size to force horizontal overflow.
    # Set this to a higher value if you have many columns (try 220 - 320).
    header.setDefaultSectionSize(380)

    # Configure vertical header (row height)
    vheader = table.verticalHeader()
    vheader.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
    vheader.setDefaultSectionSize(40)

    # Set slower scroll speed
    table.verticalScrollBar().setSingleStep(15)


def setup_table_system(self):
    """
    -------------------------------------------------------------
    DEFECT TABLE SYSTEM
    -------------------------------------------------------------
    Configures the defect table behavior:
      • Row selection mode (single row)
      • Scroll settings
      • Connect signals:
            selection → update digsheet button
      • Handles table styling and helper labels:
            "No defects found", "Select a pipe", etc.

    Ensures table reacts instantly to user interaction and
    controls the enable/disable logic for dependent buttons.
    -------------------------------------------------------------
    """
    tw = self.ui.tableWidgetDefect
    tw.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
    tw.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)

    setup_table_scroll(tw)

    try: tw.itemSelectionChanged.disconnect()
    except: pass
    tw.itemSelectionChanged.connect(self.update_digsheet_button_state)

    try: tw.cellClicked.disconnect()
    except: pass
    tw.cellClicked.connect(lambda *_: self.update_digsheet_button_state())

    self._setup_no_defects_label()
    self._setup_select_pipe_label()
    _setup_create_project_label(self)
    _show_create_project_message(self)
    self._setup_table_styling()


def _setup_create_project_label(self):
    """Create a centered overlay for 'Create Project' message"""
    central = self.centralWidget()
    self._create_proj_container = QWidget(central)
    self._create_proj_container.setGeometry(central.rect())
    self._create_proj_container.setStyleSheet("""
        background-color: rgba(245, 247, 250, 200);
    """)

    layout = QVBoxLayout(self._create_proj_container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # Main card
    card = QFrame()
    card.setFixedWidth(420)
    card.setStyleSheet("""
        QFrame {
            background-color: #ffffff;
            border-radius: 14px;
            border: 1px solid #e0e0e0;
            padding: 30px 20px;
        }
    """)
    card_layout = QVBoxLayout(card)
    card_layout.setSpacing(20)
    card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # Proper icon (no cropping)
    icon_label = QLabel()
    pixmap = QPixmap("icons/folder.png")  # ✅ use your own folder.png here
    if not pixmap.isNull():
        pixmap = pixmap.scaled(64, 64, Qt.AspectRatioMode.KeepAspectRatio,
                               Qt.TransformationMode.SmoothTransformation)
        icon_label.setPixmap(pixmap)
    else:
        icon_label.setText("📁")  # fallback emoji
        icon_label.setStyleSheet("font-size: 48px;")

    icon_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    card_layout.addWidget(icon_label)

    # Title
    title = QLabel("Create the Project")
    title.setStyleSheet("""
        font-size: 20pt;
        font-weight: 600;
        color: #2c3e50;
    """)
    title.setAlignment(Qt.AlignmentFlag.AlignCenter)
    card_layout.addWidget(title)

    # Divider
    divider = QFrame()
    divider.setFrameShape(QFrame.Shape.HLine)
    divider.setStyleSheet("color: #e0e0e0; margin: 8px 0;")
    card_layout.addWidget(divider)

    # Subtitle (fixed clipping issue)
    subtitle = QLabel("Go to <b>File → Create Project</b> in the menu bar")
    subtitle.setWordWrap(True)
    subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
    subtitle.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
    subtitle.setStyleSheet("""
        font-size: 12pt;
        color: #555;
    """)
    card_layout.addWidget(subtitle)

    layout.addWidget(card)
    self._create_proj_container.hide()


def _show_create_project_message(self):
    """Show 'Create the Project in File' message, hide table + scrollbars."""
    try:
        if hasattr(self, '_create_proj_container') and self._create_proj_container:
            self._create_proj_container.show()

        if hasattr(self.ui, 'tableWidgetDefect'):
            self.ui.tableWidgetDefect.hide()

        if hasattr(self, '_no_defects_container') and self._no_defects_container:
            self._no_defects_container.hide()

        if hasattr(self, 'table_scrollbar') and self.table_scrollbar:
            self.table_scrollbar.hide()   # 👈 also hide table top bar

        print("📋 Displaying 'Create the Project in File' message")
    except Exception as e:
        print(f"Error showing create project message: {e}")