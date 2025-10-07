# run_with_bootstrap.py
import sys
from PyQt6.QtWidgets import QApplication
import main_copy2 as appmod
from bootstrap_ui_patch import init_after_show

if __name__ == "__main__":
    app = appmod.MainApp(sys.argv)
    app.start()
    from PyQt6.QtCore import QTimer
    def go():
        if app.main_window:
            init_after_show(app, app.main_window)
    QTimer.singleShot(0, go)
    sys.exit(app.exec())
