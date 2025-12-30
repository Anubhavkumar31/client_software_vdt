import pandas as pd
from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QRadioButton, QGridLayout, QSpinBox, QLineEdit, QLabel, QPushButton, \
    QMessageBox, QProgressDialog, QApplication


class SeverityDialog(QDialog):
    def __init__(self,parent=None):
        super().__init__(parent)
        self.setWindowTitle("Severity Calculation")
        self.setFixedWidth(340)

        v = QVBoxLayout(self)

        self.rb_all = QRadioButton("Full Pipe Tally")
        self.rb_sel = QRadioButton("Only Selected Rows")
        self.rb_all.setChecked(True)

        v.addWidget(self.rb_all)
        v.addWidget(self.rb_sel)

        grid = QGridLayout()

        self.start = QSpinBox()
        self.end = QSpinBox()
        self.start.setMaximum(999999)
        self.end.setMaximum(999999)

        self.D_input = QLineEdit()
        self.D_input.setPlaceholderText("Outside Diameter (mm)")

        grid.addWidget(QLabel("From Row:"),0,0)
        grid.addWidget(self.start,0,1)
        grid.addWidget(QLabel("To Row:"),1,0)
        grid.addWidget(self.end,1,1)
        grid.addWidget(QLabel("Pipe OD (mm):"),2,0)
        grid.addWidget(self.D_input,2,1)

        v.addLayout(grid)

        btn = QPushButton("Run Severity")
        btn.clicked.connect(self.accept)
        v.addWidget(btn)


def launch_severity(parent):
    dlg = SeverityDialog(parent)
    if not dlg.exec():
        return

    try:
        D = float(dlg.D_input.text())
    except:
        QMessageBox.warning(parent,"Missing Diameter","Enter Pipe Outside Diameter")
        return

    if dlg.rb_all.isChecked():
        start = 0
        end = len(parent.df)-1
    else:
        start = dlg.start.value()-1
        end   = dlg.end.value()-1

    total = end - start + 1

    progress = QProgressDialog("Calculating Severity...", None, 0, 100, parent)
    progress.setWindowTitle("Severity Engine")
    progress.setWindowModality(Qt.WindowModality.ApplicationModal)
    progress.show()

    df = parent.df.copy()

    # -------- PHASE 1 : Row calculation (0–80%)
    for count, i in enumerate(range(start, end+1), 1):
        row = df.loc[i]
        df.loc[i,"Severity"] = calc_severity_safe(
            row.get("ERF (ASME B31G)",0),
            row.get("Depth (mm)",0),
            row.get("WT (mm)",0),
            row.get("Length (mm)",0),
            D
        )
        progress.setValue(int((count/total)*80))
        QApplication.processEvents()

    # -------- PHASE 2 : Excel write (80–90%)
    progress.setLabelText("Saving to Excel...")
    progress.setValue(85)
    QApplication.processEvents()
    df["Severity"] = pd.to_numeric(df["Severity"], errors="coerce")
    df.to_excel(parent.EXCEL_PATH, index=False)

    # -------- PHASE 3 : Reload table (90–100%)
    progress.setLabelText("Reloading Pipe Tally...")
    progress.setValue(92)
    QApplication.processEvents()
    parent.reload_from_excel()
    progress.setValue(100)
    QApplication.processEvents()

    progress.close()
    QMessageBox.information(parent,"Done","Severity Calculated ✔")







def calc_severity_safe(erf, depth, t, L, D):

    if t <= 0 or D <= 0:
        return ""

    if erf > 1:
        return 1

    if depth >= 0.8 * t:
        return 2

    if 0.95 <= erf < 1:
        return 3

    if 0.2 * t <= depth < 0.8 * t:
        return 4

    if L >= 0.2 * D and depth >= 0.1 * t:
        return 5

    return 0