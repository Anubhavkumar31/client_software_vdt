import os

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QComboBox


def create_tabSwitcher_dropdown(self):
    """
    This block creates the tab-switcher dropdown you added
     (the combobox used to jump between Heatmap / 3D / LineChart / Telemetry / etc.).
    """
    self.tabSwitcherDropdown = QComboBox(self)
    self.tabSwitcherDropdown.setToolTip("Switch between chart tabs")
    self.tabSwitcherDropdown.setCursor(Qt.CursorShape.PointingHandCursor)
    self.tabSwitcherDropdown.setMinimumWidth(120)
    self.tabSwitcherDropdown.setMaximumWidth(150)

    # Style the dropdown to match your other buttons
    arrow_path = os.path.join(os.path.dirname(__file__), "ui", "icons", "arrow_down.svg").replace("\\", "/")

    self.tabSwitcherDropdown.setStyleSheet(f"""
              QComboBox {{
                  background-color: #FFFFFF;
                  color: #000000;
                  border: 1.5px solid #000000;
                  border-radius: 6px;
                  padding: 4px 12px;
                  font-weight: 500;
              }}
              QComboBox:hover {{
                  background-color: #d6d3ce;
              }}
              QComboBox:pressed {{
                  background-color: #111111;
                  color: white;
              }}
              QComboBox:disabled {{
                  background-color: #a6a6a6;     /* same as Load button */
                  color: #f0f0f0;                /* same as Load button */
                  border: 2px solid #6e6e6e;     /* same as Load button */
              }}
              QComboBox::drop-down {{
                  subcontrol-origin: padding;
                  subcontrol-position: top right;
                  width: 20px;
                  border-left: 1.5px solid #000000;
              }}
              QComboBox::down-arrow {{
                  image: url({arrow_path});
                  width: 12px;
                  height: 12px;
              }}
          """)
    self.tabSwitcherDropdown.setEnabled(False)

    # Populate dropdown with tab names from tabWidgetM
    for i in range(self.ui.tabWidgetM.count()):
        tab_text = self.ui.tabWidgetM.tabText(i)
        self.tabSwitcherDropdown.addItem(tab_text)

    self.tabSwitcherDropdown.setCurrentIndex(0)

    # Add the dropdown right after the filter button (pos + 3)
    _parent = self.ui.comboBoxPipe.parentWidget()
    if _parent and _parent.layout():
        pos = _parent.layout().indexOf(self.btnOpenFilterDlg)
        _parent.layout().insertWidget(pos + 1, self.tabSwitcherDropdown)
    else:
        self.tabSwitcherDropdown.setParent(_parent)

    return self.tabSwitcherDropdown
