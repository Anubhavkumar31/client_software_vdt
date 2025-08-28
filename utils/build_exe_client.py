# import os
# import PyInstaller.__main__

# # Switch to project root (parent of utils/)
# ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
# os.chdir(ROOT)

# PyInstaller.__main__.run([
#     "--noconfirm",
#     "--onefile",
#     "--windowed",
#     "--exclude-module", "PyQt5",
#     "--add-data", "ui;ui/",
#     "--add-data", "pages;pages/",
#     "--add-data", "backend;backend/",
#     "--add-data", "Data_Gen;Data_Gen/",
#     "--add-data", "dig;dig/",
#     "--add-data", "manual;manual/",
#     "--add-data", "pipetally;pipetally/",
#     "--add-data", "final_report;final_report/",
#     "--add-data", "preliminary_report;preliminary_report/",
#     "--hidden-import", "PyQt6.QtWebEngineWidgets",
#     "--hidden-import", "PyQt6.QtWebEngineCore",
#     "--hidden-import", "PyQt6.QtPrintSupport",
#     "--collect-all", "numpy",
#     "--collect-all", "pandas",
#     "--name", "test_client",   # 👈 Name of final exe
#     "test.py"
# ])

import os
import shutil
import PyInstaller.__main__

# Switch to project root (parent of utils/)
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

# --- Clean previous build outputs ---
for folder in ["build", "dist"]:
    folder_path = os.path.join(ROOT, folder)
    if os.path.exists(folder_path):
        print(f"Removing old {folder_path} ...")
        shutil.rmtree(folder_path)

spec_file = os.path.join(ROOT, "test_client.spec")
if os.path.exists(spec_file):
    print(f"Removing old {spec_file} ...")
    os.remove(spec_file)

# --- Build EXE with PyInstaller ---
PyInstaller.__main__.run([
    "--noconfirm",
    "--onefile",           
    "--windowed",          
    "--exclude-module", "PyQt5",
    "--collect-all", "numpy",
    "--collect-all", "pandas",
    "--collect-all", "kaleido",    
    "--hidden-import", "kaleido",  
    "--hidden-import", "PyQt6.QtWebEngineWidgets",
    "--hidden-import", "PyQt6.QtWebEngineCore",
    "--hidden-import", "PyQt6.QtPrintSupport",
    "--add-data", "ui;ui/",
    "--add-data", "pages;pages/",
    "--add-data", "backend;backend/",
    "--add-data", "Data_Gen;Data_Gen/",
    "--add-data", "dig;dig/",
    "--add-data", "manual;manual/",
    "--add-data", "pipetally;pipetally/",
    "--add-data", "final_report;final_report/",
    "--add-data", "preliminary_report;preliminary_report/",
    "--name", "test_client",  
    "test5.py"
])
