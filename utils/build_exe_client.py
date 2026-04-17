import os
import shutil

import warnings
warnings.filterwarnings("ignore")
import PyInstaller.__main__

# -----------------------------
# Setup paths
# -----------------------------
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.chdir(ROOT)

print(f"📁 Building from project root: {ROOT}")

DIST_PATH = os.path.join(ROOT, "dist")
BUILD_PATH = os.path.join(ROOT, "build")
SPEC_FILE = os.path.join(ROOT, "main_client_software.spec")
EXE_NAME = "main_client_software"
EXE_PATH = os.path.join(DIST_PATH, EXE_NAME, f"{EXE_NAME}.exe")

# -----------------------------
# Clean old build files
# -----------------------------
for folder in [BUILD_PATH, DIST_PATH]:
    if os.path.exists(folder):
        print(f"🧹 Removing old {folder} ...")
        shutil.rmtree(folder)

if os.path.exists(SPEC_FILE):
    print(f"🧹 Removing old spec file {SPEC_FILE} ...")
    os.remove(SPEC_FILE)

if os.path.exists(EXE_PATH):
    print(f"🧹 Removing old EXE {EXE_PATH} ...")
    os.remove(EXE_PATH)

# -----------------------------
# Build configuration
# -----------------------------
# Note: Using one-folder build (no --onefile) for instant startup speed.
# Uncomment "--onefile" below if you need a single EXE (will be slower on launch).

PyInstaller.__main__.run([
    "--noconfirm",
    "--windowed",
    "--clean",                # Cleans PyInstaller cache for fresh build
     "--onefile",             # Uncomment if single-file EXE is desired
    "--distpath", DIST_PATH,
    "--workpath", BUILD_PATH,
    #'--add-data', 'dig;dig',
    # '--hidden-import=pipeline_schema.pipeline_schema',
    # '--add-data=pipeline_schema;pipeline_schema',


    # Exclude unnecessary test packages
    "--exclude-module", "pandas.tests",
    "--exclude-module", "sklearn.tests",

    # Collect full package data
    "--collect-all", "numpy",
    "--collect-all", "pandas",
    "--collect-all", "kaleido",
    "--collect-all", "pywin32",
    "--collect-all", "img2pdf",
    "--collect-all", "PIL",
    "--collect-all", "scipy",

    # Hidden imports for GUI, imaging, printing, and PDF generation
    "--hidden-import", "tkinter",
    "--hidden-import", "tkinter.filedialog",
    "--hidden-import", "tkinter.messagebox",
    "--hidden-import", "PIL.ImageGrab",
    "--hidden-import", "win32print",
    "--hidden-import", "win32api",
    "--hidden-import", "win32con",
    "--hidden-import", "pywintypes",
    "--hidden-import", "pythoncom",
    "--hidden-import", "img2pdf",
    "--hidden-import", "scipy",
    "--hidden-import", "scipy.signal",
    "--hidden-import", "scipy.linalg",
    "--hidden-import", "scipy._lib",
    "--hidden-import", "scipy._lib.array_api_compat",
    "--hidden-import", "scipy._lib.array_api_compat.numpy",
    "--hidden-import", "scipy._lib.array_api_compat.numpy.fft",

    # Optional: specify UPX directory if installed to compress binaries
    # "--upx-dir", "C:\\path\\to\\upx",

    # Data folders to bundle
    f"--add-data={os.path.join(ROOT, 'backend')};backend/",
    f"--add-data={os.path.join(ROOT, 'dig')};dig/",
    f"--add-data={os.path.join(ROOT, 'final_report')};final_report/",
    f"--add-data={os.path.join(ROOT, 'main_section_view')};main_section_view/",
    f"--add-data={os.path.join(ROOT, 'main_window')};main_window/",
    f"--add-data={os.path.join(ROOT, 'manual')};manual/",
    f"--add-data={os.path.join(ROOT, 'menubar')};menubar/",
    f"--add-data={os.path.join(ROOT, 'pages')};pages/",
    f"--add-data={os.path.join(ROOT, 'pipeline_schema')};pipeline_schema/",
    f"--add-data={os.path.join(ROOT, 'pipetally')};pipetally/",
    f"--add-data={os.path.join(ROOT, 'preliminary_report')};preliminary_report/",
    f"--add-data={os.path.join(ROOT, 'ui')};ui/",








    "--name", EXE_NAME,
    "main_latest1.py"
])

print("\n✅ Build complete! Check:")
print(f"   → {os.path.join(DIST_PATH, EXE_NAME)}")
print("   (Opens instantly since it's a folder-based build)")
