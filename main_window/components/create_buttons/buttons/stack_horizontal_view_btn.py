from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QPushButton


def create_stack_H_btn(self):
    """
    create stack and horizontal view button and style it and place it after show/hide table button
    """
    self.btnToggleHmLayout = QPushButton("Stack", self)
    self.btnToggleHmLayout.setEnabled(False)

    self.btnToggleHmLayout.setToolTip("Toggle dual-heatmap layout (side-by-side / stacked)")
    self.btnToggleHmLayout.setCursor(Qt.CursorShape.PointingHandCursor)

    self.btnToggleHmLayout.setStyleSheet("""
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
        # --- Remove if already present
        try:
            row.removeWidget(self.btnToggleHmLayout)
        except Exception:
            pass

        self.btnToggleHmLayout.setParent(parent)

        # Insert after Show Table button
        # (Show Table is at dropdown_pos + 1 → so Stack goes at +2)
        dropdown_pos = row.indexOf(self.tabSwitcherDropdown)
        row.insertWidget(dropdown_pos + 2, self.btnToggleHmLayout)

    else:
        self.btnToggleHmLayout.setParent(parent)

    return self.btnToggleHmLayout
