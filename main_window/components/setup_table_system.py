import traceback

from PIL.ImageQt import QPixmap
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QAbstractItemView, QWidget, QVBoxLayout, QFrame, QLabel, QSizePolicy

from main_section_view.workers.digsheet_abs_worker import _abs_col_index_silent
from main_section_view.workers.table_data_worker import _setup_table_styling
from main_section_view.utils import update_digsheet_button_state


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
    # tw.itemSelectionChanged.connect(lambda : update_digsheet_button_state(self))
    tw.itemSelectionChanged.connect(lambda: on_defect_selection_changed(self))
    try: tw.cellClicked.disconnect()
    except: pass
    # tw.cellClicked.connect(lambda *_: update_digsheet_button_state(self))
    # tw.cellClicked.connect(lambda *_: on_defect_selection_changed(self))
    _setup_no_defects_label(self)
    _setup_select_pipe_label(self)
    _setup_create_project_label(self)
    _show_create_project_message(self)
    _setup_table_styling(self)

def on_defect_selection_changed(self):
    # Existing logic (DON’T remove this)
    update_digsheet_button_state(self)

    # Your debug print
    debug_print_selected_defect(self)

def debug_print_selected_defect(self):
    tw = self.ui.tableWidgetDefect

    sel_model = tw.selectionModel()
    rows = [idx.row() for idx in sel_model.selectedRows()] or [i.row() for i in tw.selectedIndexes()]
    rows = list(dict.fromkeys(rows))

    if len(rows) != 1:
        return

    row = rows[0]

    # ---- Absolute Distance ----
    abs_col = _abs_col_index_silent(self)
    abs_val = ""
    if abs_col is not None:
        item = tw.item(row, abs_col)
        if item:
            abs_val = item.text().strip()

    # ---- s_no (Defect No) ----
    s_no_val = ""
    for c in range(tw.columnCount()):
        hdr = tw.horizontalHeaderItem(c)
        name = hdr.text().strip().lower() if hdr else ""
        if name == "defect_id":
            item = tw.item(row, c)
            if item:
                s_no_val = item.text().strip()
            break
    try:
        print(f"ROW CLICKED → Defect_id: {s_no_val} | ABS: {abs_val}")
    except Exception as e:
        traceback.print_exc()

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

def _setup_select_pipe_label(self):
    """Create a polished overlay asking user to select a pipe"""
    central = self.centralWidget()
    self._select_pipe_container = QWidget(central)
    self._select_pipe_container.setGeometry(central.rect())
    self._select_pipe_container.setStyleSheet("""
        background-color: rgba(255, 255, 255, 180);  /* frosted background */
    """)

    layout = QVBoxLayout(self._select_pipe_container)
    layout.setContentsMargins(0, 0, 0, 0)
    layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # --- Inner card widget ---
    card = QFrame()
    card.setFixedWidth(500)
    card.setStyleSheet("""
        QFrame {
            background-color: #ffffff;
            border-radius: 16px;
            border: 1px solid #d0d0d0;
            padding: 30px;
        }
    """)
    card_layout = QVBoxLayout(card)
    card_layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

    # Icon
    icon_label = QLabel("📂")
    icon_label.setStyleSheet("font-size: 42px;")
    card_layout.addWidget(icon_label, alignment=Qt.AlignmentFlag.AlignCenter)

    # Title
    title = QLabel("No Pipe Selected")
    title.setStyleSheet("""
        font-size: 22pt;
        font-weight: 600;
        color: #2c3e50;
    """)
    card_layout.addWidget(title, alignment=Qt.AlignmentFlag.AlignCenter)

    # Subtitle
    subtitle = QLabel("Please choose a pipe number from the list above to continue.")
    subtitle.setWordWrap(True)
    subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
    subtitle.setStyleSheet("""
        font-size: 12pt;
        color: #555;
        margin-top: 10px;
    """)
    card_layout.addWidget(subtitle)

    # Hint / efficiency tip
    hint = QLabel("💡 You can also type a pipe number directly in the box.")
    hint.setWordWrap(True)
    hint.setAlignment(Qt.AlignmentFlag.AlignCenter)
    hint.setStyleSheet("""
        font-size: 10pt;
        color: #888;
        margin-top: 15px;
    """)
    card_layout.addWidget(hint)

    layout.addWidget(card)
    self._select_pipe_container.hide()

def _setup_no_defects_label(self):
    """Create and setup the 'No Defects Found' label with absolute positioning"""
    # Create a container widget to control sizing
    self._no_defects_container = QWidget()
    self._no_defects_container.setMaximumSize(500, 200)
    self._no_defects_container.setMinimumSize(400, 150)

    # Set size policy to prevent expansion
    self._no_defects_container.setSizePolicy(
        QSizePolicy.Policy.Fixed,
        QSizePolicy.Policy.Fixed
    )

    # Create the layout for the container
    container_layout = QVBoxLayout(self._no_defects_container)
    container_layout.setContentsMargins(0, 0, 0, 0)

    # Create the actual label
    self._no_defects_label = QLabel("No Defects Found in this Pipe")
    self._no_defects_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
    self._no_defects_label.setStyleSheet("""
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
    """)

    container_layout.addWidget(self._no_defects_label)
    self._no_defects_container.hide()

    # Add to parent WITHOUT layout management
    table_parent = self.ui.tableWidgetDefect.parentWidget()
    if table_parent:
        self._no_defects_container.setParent(table_parent)
        # Position at specific coordinates (x=100, y=50)
        self._no_defects_container.move(500, 50)  # ← TWEAK THESE VALUES
