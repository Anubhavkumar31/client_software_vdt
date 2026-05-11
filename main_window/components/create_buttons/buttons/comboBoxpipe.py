import os
import sys

from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QComboBox
from pathlib import Path
def resource_path(relative_path: str) -> str:
    """
    Get absolute path to resource, works for dev and PyInstaller EXE.
    """
    if getattr(sys, 'frozen', False):
        # Running in PyInstaller bundle
        base_path = Path(sys._MEIPASS)
    else:
        # Running in normal Python
        base_path = Path(__file__).resolve().parents[4]
        # print("base path: " , base_path)

    return str(base_path / relative_path)


def comboBoxPipe_setup(self):
    self.ui.comboBoxPipe.setEditable(True)

    arrow_path = Path(resource_path("ui/icons/arrow_down.svg")).as_posix()

    self.ui.comboBoxPipe.setStyleSheet(f"""
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
           """)
    self.ui.comboBoxPipe.clear()
    self.ui.comboBoxPipe.addItem("-Pipe-")
    self.ui.comboBoxPipe.setMaxVisibleItems(7)
    self.ui.comboBoxPipe.completer().setCompletionMode(
        QtWidgets.QCompleter.CompletionMode.PopupCompletion
    )
    self.ui.comboBoxPipe.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)