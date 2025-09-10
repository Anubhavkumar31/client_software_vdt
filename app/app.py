# app/app.py
import sys
from PyQt6.QtWidgets import QApplication
from app.main_window.window import MyMainWindow
from widgets.SplashScreen import SplashScreenWidget
from PyQt6.QtCore import Qt, QTimer
from PyQt6.QtGui import QMovie
from PyQt6.QtWidgets import QLabel
import os
from config.paths import resource_path

class MainApp(QApplication):
    def __init__(self, sys_argv):
        super().__init__(sys_argv)
        self.splash = None
        self.main_window = None

    def show_splash_screen(self):
        self.splash = SplashScreenWidget()
        self.splash.setWindowFlags(Qt.WindowType.FramelessWindowHint)
        label = self.splash.findChild(QLabel, 'label')
        if label:
            gif_path = resource_path(os.path.join( "ui", "icons", "VDT_ani.gif"))
            self.movie = QMovie(gif_path)
            label.setMovie(self.movie)
            self.movie.start()
        self.splash.show()

    def close_splash_screen(self):
        if self.splash:
            self.splash.close()

    def show_main_window(self):
        self.main_window = MyMainWindow()
        self.main_window.show()

    def start(self):
        self.show_splash_screen()
        self.timer = QTimer(self)
        self.timer.setSingleShot(True)
        self.timer.timeout.connect(self.initialize_app)
        self.timer.start(1200)

    def initialize_app(self):
        self.close_splash_screen()
        self.show_main_window()
