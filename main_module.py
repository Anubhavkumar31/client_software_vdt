# PIE_DV_NEW/main.py
import sys
from app.app import MainApp

if __name__ == "__main__":
    # Create and start the Qt application
    app = MainApp(sys.argv)
    app.start()
    sys.exit(app.exec())
