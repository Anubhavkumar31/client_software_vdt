from pathlib import Path
import sys
import os

def get_app_dir():
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(os.getcwd())
