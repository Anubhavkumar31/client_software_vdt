import sys, traceback
from pathlib import Path
from PyQt6.QtWidgets import QMessageBox

def global_error_handler(exctype, value, tb):
    # Format error
    formatted = "".join(traceback.format_exception(exctype, value, tb))

    # 🔥 Print to Pycharm terminal
    print("\n" + "=" * 80)
    print("🔥 ERROR OCCURRED:")
    print(formatted)
    print("=" * 80)

    # 🔥 Show in Message Box
    msg = QMessageBox()
    msg.setWindowTitle("Application Error")
    msg.setIcon(QMessageBox.Icon.Critical)
    msg.setText("An error occurred!")
    msg.setDetailedText(formatted)   # <-- opens on clicking "Show Details"
    msg.exec()

# Install hook
sys.excepthook = global_error_handler

import tempfile, uuid, runpy
import os

# from Data_Gen.DataGenApp import ScriptRunnerApp
from main_window.main_window import MyMainWindow

os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = "--disable-logging --log-level=3 --disable-features=AccessibilityAriaVirtualContent"


# main.py
import sys
import os
try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView, QWebEnginePage
except ImportError:
    # Some builds moved QWebEnginePage to a separate submodule
    from PyQt6.QtWebEngineCore import QWebEnginePage
    from PyQt6.QtWebEngineWidgets import QWebEngineView

import matplotlib
# matplotlib.use("Qt5Agg")
# matplotlib.use("QtAgg")


# PyQt6 Core
from PyQt6 import uic, QtWidgets
from PyQt6.QtCore import ( Qt, QTimer )
# PyQt6 GUI
from PyQt6.QtGui import QMovie

# PyQt6 Widgets
from PyQt6.QtWidgets import (
    QApplication, QLabel, QMessageBox
)


# Project imports (leave as-is)
from reportlab.pdfgen import canvas  # noqa
from pages.XYZ import XYZ  # noqa
from pages.metrics import Metric_Dialog  # noqa
from pages.errorBox import Error_Dialog  # noqa
from backend.heatmap import HeatmapWindow as hm, pre_process, pre_process2  # noqa



def resource_path(relative_path):
    """
    Resolve paths relative to the project root (client_software_vdt),
    independent of current working directory.
    """
    if getattr(sys, "frozen", False):
        base_path = Path(sys._MEIPASS)
    else:
        # main.py lives in client_software_vdt → this IS the project root
        base_path = Path(__file__).resolve().parent

    return str(base_path / relative_path)

def _dump_tally_to_temp(df):
    import pickle
    p = os.path.join(tempfile.gettempdir(), f"pipe_tally_{uuid.uuid4().hex}.pkl")
    with open(p, "wb") as f: pickle.dump(df, f)
    return p


base_dir = os.path.dirname(__file__)
ui_path = os.path.join(base_dir, "ui", "landing.ui")
SplashScreen, SplashWindow = uic.loadUiType(ui_path)
ui_path_main = os.path.join(base_dir, "ui", "main_window.ui")
Form, Window = uic.loadUiType(ui_path_main)



class SplashScreenWidget(QtWidgets.QWidget, SplashScreen):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)


class MainApp(QApplication):
    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        self.splash = None
        self.main_window = None
        from PyQt6.QtGui import QIcon
        self.setWindowIcon(QIcon(resource_path(r"D:\Aamna\client_software\client_software_vdt\ui\icons\vdt-logo.png")))

    def show_splash_screen(self):
        self.splash = SplashScreenWidget()
        # Tool flag prevents taskbar entry for splash screen
        self.splash.setWindowFlags(
            Qt.WindowType.FramelessWindowHint |
            Qt.WindowType.Tool |
            Qt.WindowType.WindowStaysOnTopHint
        )
        label = self.splash.findChild(QLabel, 'label')
        if label:
            gif_path = os.path.join(os.path.dirname(__file__), "ui", "icons", "VDT_ani.gif")
            self.movie = QMovie(gif_path)
            label.setMovie(self.movie)
            self.movie.start()
        self.splash.show()
        QApplication.processEvents()

    def close_splash_screen(self):
        if self.splash:
            self.splash.close()
            self.splash = None

    def show_main_window(self):
        self.main_window = MyMainWindow()
        # Set application icon for taskbar
        self.main_window.setWindowIcon(self.windowIcon())
        self.main_window.show()
        QApplication.processEvents()

    def start(self):
        self.show_splash_screen()
        QTimer.singleShot(1200, self.initialize_app)

    def initialize_app(self):
        self.show_main_window()
        # Close splash after main window is fully shown
        QTimer.singleShot(50, self.close_splash_screen)



#sdfsf
if __name__ == "__main__":
    # Handle special modes in the frozen EXE so it doesn't relaunch the main UI
    if "--run-digsheet-abs" in sys.argv:
        i = sys.argv.index("--run-digsheet-abs")
        tally_pkl = sys.argv[i+1]
        abs_val = sys.argv[i+2]
        project_root = sys.argv[i+3] if len(sys.argv) > i+3 else None
        print("MAIN FILE:", Path(__file__).resolve())
        print("BASE DIR :", Path(__file__).resolve().parent)
        print("DIG EXISTS:", (Path(__file__).resolve().parent / "dig" / "digsheet_abs.py").exists())
        dig_py_abs = resource_path("dig/digsheet_abs.py")


        # 👇 Pass along all arguments, including project_root
        sys.argv = [dig_py_abs, tally_pkl, abs_val]
        if project_root:
            sys.argv.append(project_root)



        runpy.run_path(dig_py_abs, run_name="__main__")
        sys.exit(0)



    if "--run-digsheet" in sys.argv:
        i = sys.argv.index("--run-digsheet")
        tally_pkl = sys.argv[i+1]
        project_root = sys.argv[i+2] if len(sys.argv) > i+2 else None

        dig_py = resource_path(os.path.join("dig", "dig_sheet.py"))
        sys.argv = [dig_py, tally_pkl]
        if project_root:
            sys.argv.append(project_root)

        runpy.run_path(dig_py, run_name="__main__")
        sys.exit(0)


    app = MainApp(sys.argv)
    app.start()
    sys.exit(app.exec())
