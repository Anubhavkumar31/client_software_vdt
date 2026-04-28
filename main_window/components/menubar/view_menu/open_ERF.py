# from pages.erf1 import ERF1App as ERF



# def open_ERF(self):
#     import threading
#
#     # Inner function - no self parameter
#     def run_erf():
#         erf_app = ERF(self.project_root)
#         erf_app.run()
#
#     # Start ERF calculator in a background thread
#     threading.Thread(target=run_erf, daemon=True).start()
from menubar.view_menu.apps.erfapp import ERFWindow

def open_ERF(self):
    if not hasattr(self, "erf_window") or self.erf_window is None:
        self.erf_window = ERFWindow(self.project_root)
    self.erf_window.show()
    self.erf_window.raise_()
    self.erf_window.activateWindow()
