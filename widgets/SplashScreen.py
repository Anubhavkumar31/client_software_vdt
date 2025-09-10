from PyQt6 import QtWidgets
from config.paths import SplashScreen


class SplashScreenWidget(QtWidgets.QWidget, SplashScreen):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setupUi(self)
