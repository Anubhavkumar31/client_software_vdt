# utils.py
import os
import sys

def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)










# pyinstaller --noconfirm --clean --onefile main.py ^
# --add-data "ui;ui" ^
# --add-data "dig;dig" ^
# --add-data "pages;pages" ^
# --add-data "backend;backend" ^
# --add-data "pipeline_schema;pipeline_schema" ^
# --add-data "final_report;final_report" ^
# --add-data "preliminary_report;preliminary_report" ^
# --add-data "pipetally;pipetally" ^
# --add-data "manual;manual" ^
# --hidden-import=PyQt6.QtWebEngineWidgets ^
# --hidden-import=matplotlib.backends.backend_qt5agg ^
# --add-data "pipe_tally.pkl;."

