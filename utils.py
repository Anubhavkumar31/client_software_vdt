# # utils.py
import os
import sys

def resource_path(relative_path):
    if getattr(sys, 'frozen', False):
        return os.path.join(sys._MEIPASS, relative_path)
    return os.path.join(os.path.abspath("."), relative_path)










# # # pyinstaller --noconfirm --clean --onefile main.py ^
# # # --add-data "ui;ui" ^
# # # --add-data "dig;dig" ^
# # # --add-data "pages;pages" ^
# # # --add-data "backend;backend" ^
# # # --add-data "pipeline_schema;pipeline_schema" ^
# # # --add-data "final_report;final_report" ^
# # # --add-data "preliminary_report;preliminary_report" ^
# # # --add-data "pipetally;pipetally" ^
# # # --add-data "manual;manual" ^
# # # --hidden-import=PyQt6.QtWebEngineWidgets ^
# # # --hidden-import=matplotlib.backends.backend_qt5agg ^
# # # --add-data "pipe_tally.pkl;."




# import PyInstaller.__main__
# import os

# # Folders in your project that must be bundled
# FOLDERS = [
#     "ui", "pages", "backend", "Data_Gen",
#     "dig", "pipetally",
#     "final_report", "preliminary_report", "manual"
# ]

# # File extensions you want to make sure are included
# STATIC_EXTS = (".html", ".csv", ".xlsx", ".xls", ".pdf", ".ui", ".gif", ".svg", ".png", ".ico")

# add_data = []

# for folder in FOLDERS:
#     if os.path.exists(folder):
#         # Always include the folder itself
#         add_data.append(f"--add-data={folder};{folder}")
#         # Walk through it to check for static files
#         for root, dirs, files in os.walk(folder):
#             for file in files:
#                 if file.lower().endswith(STATIC_EXTS):
#                     src = os.path.join(root, file)
#                     # preserve subfolder structure relative to project root
#                     rel = os.path.relpath(src, ".")
#                     add_data.append(f"--add-data={rel};{os.path.dirname(rel)}")

# PyInstaller.__main__.run([
#     "test1.py",
#     "--onefile",
#     "--noconsole",
#     "--clean",
#     *add_data
# ])
