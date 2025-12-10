from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton


def create_hide_show_table(self):
    """
    create hide/show table button and style it and place it after tabswitcher dropdown
    """
    self.btnToggleTable = QPushButton("Show Table", self)
    self._table_hidden = True  # table is hidden initially
    self.btnToggleTable.setEnabled(False)

    self.btnToggleTable.setToolTip("Toggle table visibility (Heatmap only)")
    self.btnToggleTable.setCursor(Qt.CursorShape.PointingHandCursor)

    self.btnToggleTable.setStyleSheet("""
                QPushButton {
                    background-color: #FFFFFF;
                    color: #000000;
                    border: 1.5px solid #000000;
                    border-radius: 6px;
                    padding: 4px 12px;
                    font-weight: 500;
                    min-width: 80px;
                }
                QPushButton:hover {
                    background-color: #d6d3ce;
                }
                QPushButton:pressed {
                    background-color: #111111;
                    color: white;
                }
                QPushButton:disabled {
                    background-color: #a6a6a6;
                    color: #f0f0f0;
                    border: 2px solid #6e6e6e;
                }
            """)
    parent = self.ui.comboBoxPipe.parentWidget()
    row = parent.layout()

    if row:
        # --- Remove if already present (safe reload)
        try:
            row.removeWidget(self.btnToggleTable)
        except Exception:
            pass

        self.btnToggleTable.setParent(parent)

        # Insert right after tabSwitcherDropdown
        dropdown_pos = row.indexOf(self.tabSwitcherDropdown)
        row.insertWidget(dropdown_pos + 1, self.btnToggleTable)

    else:
        self.btnToggleTable.setParent(parent)

    # Initialize flag - default is shown (False = not hidden)
    self._table_hidden = True
    self.btnToggleTable.setText("Show Table")

    return self.btnToggleTable

