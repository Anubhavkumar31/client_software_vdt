from PyQt6.QtWidgets import QPushButton


def create_Load_btn(self):
    """
    This block does three things:

    1️⃣ Create the Load button
    2️⃣ Style it
    3️⃣ Dynamically position it next to comboBoxPipe adn between digsheet
    """
    self.btnLoadPipe = QPushButton("Load")
    self.btnLoadPipe.setEnabled(False)
    self.btnLoadPipe.setStyleSheet("""
               QPushButton {
                   background-color: #3498db;
                   color: white;
                   border: 1px solid #2980b9;
                   border-radius: 6px;
                   padding: 4px 12px;
                   font-weight: 500;
               }
               QPushButton:hover {
                   background-color: #2980b9;
               }
               QPushButton:pressed {
                   background-color: #1f5f8a;
               }
               QPushButton:disabled {
               background-color: #a6a6a6;   
               color: #f0f0f0;              
               border: 2px solid #6e6e6e;   
           }
           """)
    _parent = self.ui.comboBoxPipe.parentWidget()
    _lay = _parent.layout()
    if _lay is not None:
        pos = _lay.indexOf(self.ui.comboBoxPipe)
        if pos != -1:
            _lay.insertWidget(pos + 1, self.btnLoadPipe)
        else:
            _lay.addWidget(self.btnLoadPipe)
    else:
        self.btnLoadPipe.setParent(_parent)

    return self.btnLoadPipe