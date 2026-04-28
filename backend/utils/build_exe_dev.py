import os
import shutil
import PyInstaller.__main__

# -----------------------------
# Setup paths
# -----------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

print(f"📁 DEV BUILD from: {ROOT}")

DIST_PATH = os.path.join(ROOT, "dist_dev")
BUILD_PATH = os.path.join(ROOT, "build_dev")
EXE_NAME = "main_client_dev"

# -----------------------------
# Clean old build
# -----------------------------
for folder in [BUILD_PATH, DIST_PATH]:
    if os.path.exists(folder):
        print(f"🧹 Removing {folder}")
        shutil.rmtree(folder)

# -----------------------------
# PyInstaller DEV config
# -----------------------------
PyInstaller.__main__.run([
    "--noconfirm",
    "--onedir",              # ✅ DEV mode (important)
    "--console",             # ✅ show logs
    "--clean",

    "--distpath", DIST_PATH,
    "--workpath", BUILD_PATH,

    # ❌ remove unnecessary PyQt bloat
    "--exclude-module", "PyQt5",
    "--exclude-module", "PyQt6.QtQuick",
    "--exclude-module", "PyQt6.QtQml",
    "--exclude-module", "PyQt6.QtWebEngine",

    # Core packages
    "--collect-all", "numpy",
    "--collect-all", "pandas",
    "--collect-all", "PIL",
    "--collect-all", "img2pdf",
    "--collect-all", "pywin32",

    # Hidden imports
    "--hidden-import", "tkinter",
    "--hidden-import", "PIL.ImageGrab",
    "--hidden-import", "win32print",
    "--hidden-import", "win32api",

    # 🔥 IMPORTANT (your digsheet fix)
    f"--add-data={os.path.join(ROOT, 'dig')};dig",

    # Other folders
    f"--add-data={os.path.join(ROOT, 'backend')};backend",
    f"--add-data={os.path.join(ROOT, 'ui')};ui",
    f"--add-data={os.path.join(ROOT, 'pages')};pages",
    f"--add-data={os.path.join(ROOT, 'pipeline_schema')};pipeline_schema",

    "--name", EXE_NAME,
    "main_latest1.py"
])

print("\n✅ DEV BUILD READY:")
print(f"👉 {DIST_PATH}\\{EXE_NAME}\\")