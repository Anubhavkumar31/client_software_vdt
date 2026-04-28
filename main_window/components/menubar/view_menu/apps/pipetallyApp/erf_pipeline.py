
import sys, math, pandas as pd
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtCore import QObject, QThread, pyqtSignal

from PyQt6.QtWidgets import QProgressDialog
from PyQt6.QtCore import Qt


# FILE = r"C:\Users\admin\Downloads\Pipe_Tally_8inch (1).xlsx"

# ================= FORMULAS =================

def asme_b31g(D,t,L,d,SMYS,MAOP):
    flow = 1.1 * SMYS
    M = math.sqrt(1 + 0.8 * (L*L)/(D*t))
    k = (1 - (2/3)*(d/t)) / (1 - ((2/3)*(d/t))/M)
    Pf = (2 * flow * t) / D * k
    Psafe = Pf / 1.39
    return MAOP/Psafe, Psafe

def mod_b31g(D,t,L,d,SMYS,MAOP):
    sigma_f = 1.1 * SMYS
    M = math.sqrt(1 + 0.6275*(L/math.sqrt(D*t))**2)
    rsf = (1 - 0.85*(d/t)) / (1 - (0.85*(d/t))/M)
    Pf = (2*sigma_f*t/D)*rsf
    Psafe = Pf/1.39
    return MAOP/Psafe, Psafe

def dnv_f101(D,t,L,d,SMTS,P_op):
    F = 0.90*0.67
    Q = math.sqrt(1+0.31*(L/(D*t))**2)
    Pf = (2*SMTS*t/(D-t))*((1-d/t)/(1-d/(t*Q)))
    Psafe = F*Pf
    return P_op/Psafe, Psafe

def shell_92(D,t,L,d,SMYS,MAOP):
    sigma_f = 1.15 * SMYS
    M = math.sqrt(1+0.31*(L/math.sqrt(D*t))**2)
    rsf = (1-0.9*(d/t))/(1-(0.9*(d/t))/M)
    Pf = (2*sigma_f*t/D)*rsf
    Psafe = Pf/1.5
    return MAOP/Psafe, Psafe

# ================= UTILS =================

def fixcol(df, name):
    for c in df.columns:
        if c.strip().lower() == name.strip().lower():
            return c
    return name

# ================= WORKER =================

class ERFWorker(QObject):
    finished = pyqtSignal(str)
    progress = pyqtSignal(int)   # now emits percent

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def run(self, EXCEL_PATH):
        try:
            df = pd.read_excel(EXCEL_PATH)
            total = len(df)

            D = self.settings["D"]
            SMYS = self.settings["SMYS"]
            SMTS = self.settings["SMTS"]
            MAOP = self.settings["MAOP"]
            P_op = self.settings["P_op"]
            std = self.settings["std"]
            write = self.settings["write"]

            # -------- Phase 1 (0–80%)
            for count, (i, row) in enumerate(df.iterrows(), 1):

                L = row["Length (mm)"]
                d = row["Depth (mm)"]
                t = row["WT (mm)"]

                if std["asme"]:
                    erf, ps = asme_b31g(D, t, L, d, SMYS, MAOP)
                    if write["erf"]:   df.loc[i, fixcol(df, "ERF (ASME B31G)")] = erf
                    if write["psafe"]: df.loc[i, fixcol(df, "Psafe (ASME B31G) Barg")] = ps

                if std["mod"]:
                    erf, ps = mod_b31g(D, t, L, d, SMYS, MAOP)
                    if write["erf"]:   df.loc[i, fixcol(df, "ERF (MOD B31G)")] = erf
                    if write["psafe"]: df.loc[i, fixcol(df, "Psafe (MOD B31G)")] = ps

                if std["dnv"]:
                    erf, ps = dnv_f101(D, t, L, d, SMTS, P_op)
                    if write["erf"]:   df.loc[i, fixcol(df, "ERF (DNV-RP-F101 )")] = erf
                    if write["psafe"]: df.loc[i, fixcol(df, "Psafe (DNV-RP-F101 )")] = ps

                if std["shell"]:
                    erf, ps = shell_92(D, t, L, d, SMYS, MAOP)
                    if write["erf"]:   df.loc[i, fixcol(df, "ERF (SHELL 92 )")] = erf
                    if write["psafe"]: df.loc[i, fixcol(df, "Psafe (SHELL 92)")] = ps

                self.progress.emit(int((count / total) * 80))

            # -------- Phase 2 (80–90%)
            self.progress.emit(85)
            df.to_excel(EXCEL_PATH, index=False)

            # -------- Phase 3 (90–100%)
            self.progress.emit(95)
            self.progress.emit(100)

            self.finished.emit("success")

        except Exception as e:
            self.finished.emit(str(e))


# ================= DIALOG =================

class ERFBatchDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Batch ERF Assessment")
        self.setFixedWidth(420)
        root = QVBoxLayout(self)

        self.cb_asme = QCheckBox("ASME B31G")
        self.cb_mod = QCheckBox("Modified B31G")
        self.cb_dnv = QCheckBox("DNV RP F101")
        self.cb_shell = QCheckBox("SHELL 92")
        for c in (self.cb_asme,self.cb_mod,self.cb_dnv,self.cb_shell):
            c.setChecked(True)

        box = QGroupBox("Assessment Standards")
        l = QVBoxLayout(box)
        for c in (self.cb_asme,self.cb_mod,self.cb_dnv,self.cb_shell):
            l.addWidget(c)
        root.addWidget(box)

        self.cb_write_erf = QCheckBox("Write ERF")
        self.cb_write_psafe = QCheckBox("Write Psafe")
        self.cb_write_erf.setChecked(True)
        self.cb_write_psafe.setChecked(True)
        root.addWidget(self.cb_write_erf)
        root.addWidget(self.cb_write_psafe)

        grid = QGridLayout()
        self.D=QLineEdit(); self.SMYS=QLineEdit(); self.SMTS=QLineEdit()
        self.MAOP=QLineEdit(); self.P_op=QLineEdit()
        for e in (self.D,self.SMYS,self.SMTS,self.MAOP,self.P_op):
            e.setValidator(QDoubleValidator())

        labels = ["Outside Diameter (mm)","SMYS (MPa)","SMTS (MPa)","MAOP (MPa)","P-op (MPa)"]
        for i,(lbl,w) in enumerate(zip(labels,[self.D,self.SMYS,self.SMTS,self.MAOP,self.P_op])):
            grid.addWidget(QLabel(lbl), i,0)
            grid.addWidget(w,i,1)
        root.addLayout(grid)

        btn = QPushButton("Run Assessment")
        btn.clicked.connect(self.accept)
        root.addWidget(btn)

# ================= LAUNCHER =================


def launch_erf_batch(parent, EXCEL_PATH):
    dlg = ERFBatchDialog(parent)
    if not dlg.exec():
        return

    settings = {
        "D": float(dlg.D.text()),
        "SMYS": float(dlg.SMYS.text()),
        "SMTS": float(dlg.SMTS.text()),
        "MAOP": float(dlg.MAOP.text()),
        "P_op": float(dlg.P_op.text()),
        "std": {
            "asme": dlg.cb_asme.isChecked(),
            "mod": dlg.cb_mod.isChecked(),
            "dnv": dlg.cb_dnv.isChecked(),
            "shell": dlg.cb_shell.isChecked()
        },
        "write": {
            "erf": dlg.cb_write_erf.isChecked(),
            "psafe": dlg.cb_write_psafe.isChecked()
        }
    }

    parent.erf_progress = QProgressDialog("Calculating ERF...", None, 0, 100, parent)
    parent.erf_progress.setWindowTitle("ERF Batch")
    parent.erf_progress.setWindowModality(Qt.WindowModality.ApplicationModal)
    parent.erf_progress.show()

    parent.erf_thread = QThread(parent)
    parent.erf_worker = ERFWorker(settings)
    parent.erf_worker.moveToThread(parent.erf_thread)

    parent.erf_worker.progress.connect(
        lambda p: parent.erf_progress.setValue(p))

    parent.erf_worker.finished.connect(parent.erf_thread.quit)
    parent.erf_worker.finished.connect(parent.erf_worker.deleteLater)
    parent.erf_thread.finished.connect(parent.erf_thread.deleteLater)

    parent.erf_worker.finished.connect(lambda msg: (
        parent.erf_progress.setLabelText("Reloading Pipe Tally..."),
        parent.erf_progress.setValue(92),
        QApplication.processEvents(),
        parent.reload_from_excel(),
        parent.erf_progress.setValue(100),
        parent.erf_progress.close(),
        QMessageBox.information(parent, "Done", "ERF Updated ✔")
    ))

    parent.erf_thread.started.connect(lambda: parent.erf_worker.run(EXCEL_PATH))
    parent.erf_thread.start()

