from PyQt6.QtWidgets import QPushButton


def create_digsheet_btn(self):
    """
    This block does three things:

    1️⃣ Create the Digsheet button
    2️⃣ Style it
    3️⃣ Dynamically position it next to comboBoxPipe
    """

    self.btnDigsheetAbs = QPushButton("Digsheet")
    self.btnDigsheetAbs.setToolTip("Select an Absolute Distance cell in the defect table (on Heatmap/3D) to enable.")
    self.btnDigsheetAbs.setEnabled(False)
    self.btnDigsheetAbs.setStyleSheet("""
               QPushButton {
                   background: white;
                   border: 1px solid #3498db;
                   color: #3498db;
                   border-radius: 6px;
                   padding: 4px 12px;
                   font-weight: 500;
               }
               QPushButton:hover {
                   background: #ecf6fd;
               }
               QPushButton:pressed {
                   background: #d0e9fa;
               }
               QPushButton:disabled {
                   color: #a0a0a0;
                   background: #f5f5f5;
                   border: 2px solid #6e6e6e;
               }
           """)
    try:
        _parent = self.ui.comboBoxPipe.parentWidget()
        _lay = _parent.layout()
        if _lay is not None:
            pos = _lay.indexOf(self.ui.comboBoxPipe)
            if pos != -1:
                _lay.insertWidget(pos + 1, self.btnDigsheetAbs)
            else:
                _lay.addWidget(self.btnDigsheetAbs)
        else:
            self.btnDigsheetAbs.setParent(_parent)
    except Exception:
        self.statusBar().addPermanentWidget(self.btnDigsheetAbs)

    return self.btnDigsheetAbs