import sys, math, pandas as pd
from PyQt6.QtWidgets import *
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtCore import QThread, pyqtSignal

FILE = r"C:\Users\admin\Downloads\Pipe_Tally_8inch (1).xlsx"

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

class ERFWorker(QThread):
    done = pyqtSignal(str)

    def __init__(self, settings):
        super().__init__()
        self.settings = settings

    def run(self):
        print("\n🔹 Worker thread started")
        try:
            df = pd.read_excel(FILE)
            print("✔ Pipe tally loaded")

            D = self.settings["D"]
            SMYS = self.settings["SMYS"]
            SMTS = self.settings["SMTS"]
            MAOP = self.settings["MAOP"]
            P_op = self.settings["P_op"]
            std = self.settings["std"]
            write = self.settings["write"]

            for i,row in df.iterrows():
                L = row["Length (mm)"]
                d = row["Depth (mm)"]
                t = row["WT (mm)"]

                print(f"\n========= ROW {i+1} =========")
                print(f"L={L}  d={d}  t={t}")

                if std["asme"]:
                    erf, ps = asme_b31g(D,t,L,d,SMYS,MAOP)
                    print(f"[ASME] ERF={erf:.5f} Psafe={ps:.5f}")
                    if write["erf"]:
                        df.loc[i, fixcol(df,"ERF (ASME B31G)")] = erf
                    if write["psafe"]:
                        df.loc[i, fixcol(df,"Psafe (ASME B31G) Barg")] = ps

                if std["mod"]:
                    erf, ps = mod_b31g(D,t,L,d,SMYS,MAOP)
                    print(f"[MOD ] ERF={erf:.5f} Psafe={ps:.5f}")
                    if write["erf"]:
                        df.loc[i, fixcol(df,"ERF (MOD B31G)")] = erf
                    if write["psafe"]:
                        df.loc[i, fixcol(df,"Psafe (MOD B31G)")] = ps

                if std["dnv"]:
                    erf, ps = dnv_f101(D,t,L,d,SMTS,P_op)
                    print(f"[DNV ] ERF={erf:.5f} Psafe={ps:.5f}")
                    if write["erf"]:
                        df.loc[i, fixcol(df,"ERF (DNV-RP-F101 )")] = erf
                    if write["psafe"]:
                        df.loc[i, fixcol(df,"Psafe (DNV-RP-F101 )")] = ps

                if std["shell"]:
                    erf, ps = shell_92(D,t,L,d,SMYS,MAOP)
                    print(f"[SHEL] ERF={erf:.5f} Psafe={ps:.5f}")
                    if write["erf"]:
                        df.loc[i, fixcol(df,"ERF (SHELL 92 )")] = erf
                    if write["psafe"]:
                        df.loc[i, fixcol(df,"Psafe (SHELL 92)")] = ps

            df.to_excel(FILE,index=False)
            print("💾 Excel updated successfully")
            self.done.emit("success")

        except Exception as e:
            print("❌ ERROR:", e)
            self.done.emit(str(e))

# ================= DIALOG =================

class ERFBatchDialog(QDialog):
    def __init__(self):
        super().__init__()
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

# ================= MAIN =================
def launch_erf_batch(parent):
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

    print("\n🚀 Starting batch ERF...")

    # 👇 THIS LINE SAVES YOUR LIFE
    parent.erf_worker = ERFWorker(settings)

    parent.erf_worker.done.connect(
        lambda msg:
            QMessageBox.information(parent,"Done","Pipe tally updated ✔") if msg=="success"
            else QMessageBox.critical(parent,"Error",msg)
    )

    parent.erf_worker.start()




# ================= RUN =================
if __name__ == "__main__":
    app = QApplication(sys.argv)

    dlg = ERFBatchDialog()
    if dlg.exec():

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

        print("\n🚀 Starting batch ERF...")
        worker = ERFWorker(settings)
        worker.done.connect(lambda msg:
            QMessageBox.information(None,"Done","Pipe tally updated safely ✔") if msg=="success"
            else QMessageBox.critical(None,"Error",msg)
        )
        worker.start()

    sys.exit(app.exec())


