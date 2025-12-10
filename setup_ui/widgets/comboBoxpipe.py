import os

from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QComboBox


def comboBoxPipe_setup(self):
    self.ui.comboBoxPipe.setEditable(True)

    arrow_path = os.path.join(os.path.dirname(__file__), "ui", "icons", "arrow_down.svg").replace("\\", "/")

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
    self.ui.comboBoxPipe.setMaxVisibleItems(12)
    self.ui.comboBoxPipe.completer().setCompletionMode(
        QtWidgets.QCompleter.CompletionMode.PopupCompletion
    )
    self.ui.comboBoxPipe.setInsertPolicy(QComboBox.InsertPolicy.NoInsert)