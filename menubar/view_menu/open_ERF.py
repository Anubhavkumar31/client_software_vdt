from pages.erf1 import ERF1App as ERF

def open_ERF(self):
    import threading

    # Inner function - no self parameter
    def run_erf():
        erf_app = ERF(self.project_root)
        erf_app.run()

    # Start ERF calculator in a background thread
    threading.Thread(target=run_erf, daemon=True).start()