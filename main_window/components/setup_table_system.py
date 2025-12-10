from PyQt6.QtWidgets import QAbstractItemView


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
    self._setup_create_project_label()
    self._show_create_project_message()
    self._setup_table_styling()
