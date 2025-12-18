"""
this is where main intital ui was created using the mainwindow.ui from ui folder in main_window.
it creates instances of menubar and other ui elements
"""
import os

from PyQt6 import uic, QtWidgets
from PyQt6.QtCore import Qt

# base_dir = os.path.dirname(__file__)
# # ui_path_main = os.path.join(base_dir, "ui", "main_window.ui")
# # ui_path_main = os.path.join(
# #     base_dir,
# #     "..",      # go OUT of main_window folder
# #     "ui",
# #     "main_window.ui"
# # )
# ui_path_main = os.path.join(
#     os.path.dirname(os.path.dirname(__file__)),
#     "ui",
#     "main_window.ui"
# )
#
# ui_path_main = os.path.abspath(ui_path_main)
BASE_DIR = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
# Explanation:
# dirname(__file__) -> setup_ui/
# dirname(dirname(__file__)) -> components/
# dirname(dirname(dirname(__file__))) -> main_window/

ui_path_main = os.path.join(BASE_DIR, "ui", "main_window.ui")

Form, Window = uic.loadUiType(ui_path_main)

Form, Window = uic.loadUiType(ui_path_main)


def setup_ui(self):
    self.ui = Form()

    self.ui.setupUi(self)
    # Hide unwanted menu actions
    if hasattr(self.ui, "action_Pipe_Locator"):
        self.ui.action_Pipe_Locator.setVisible(False)

    if hasattr(self.ui, "action_ERF"):
        self.ui.action_ERF.setVisible(True)

    if hasattr(self.ui, "action_Pipe_Sch"):
        self.ui.action_Pipe_Sch.setVisible(True)
        self.ui.action_Pipe_Sch.setEnabled(False)

    for tb in self.findChildren(QtWidgets.QToolBar):
        if self.toolBarArea(tb) == Qt.ToolBarArea.LeftToolBarArea:
            self.removeToolBar(tb)
            tb.setParent(None)
            tb.deleteLater()

    menubar_design(self)


def menubar_design(self):
    self.menuBar().setStyleSheet("""
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
    """)