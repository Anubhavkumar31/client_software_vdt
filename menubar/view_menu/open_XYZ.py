import os

from PyQt6.QtWidgets import QMessageBox, QInputDialog
import subprocess
from pathlib import Path

def open_XYZ(self):
    if not self.project_is_open:
        if self._ui_ready:
            self._project_required_popup()
        return
    try:
        # First check if a project is open
        if not self.project_is_open or not self.project_root:
            QMessageBox.warning(
                self,
                "No Project Open",
                "Please open a project first to load KML files from the project folder."
            )
            return

        # Search for KML files in the project folder
        kml_files = []
        project_path = Path(self.project_root)

        # Search for KML files in project root and subdirectories
        kml_patterns = ["*.kml", "*.KML"]
        for pattern in kml_patterns:
            kml_files.extend(project_path.glob(pattern))
            kml_files.extend(project_path.glob(f"**/{pattern}"))  # Search subdirectories too

        # Remove duplicates and convert to strings
        kml_files = list(set(str(f) for f in kml_files))

        if not kml_files:
            QMessageBox.information(
                self,
                "No KML Files Found",
                f"No KML files were found in the project folder:\n{self.project_root}\n\n"
                "Please ensure your KML files are placed in the project directory."
            )
            return

        # If multiple KML files found, let user choose
        kml_path = None
        if len(kml_files) == 1:
            kml_path = kml_files[0]
        else:
            # Show selection dialog for multiple KML files
            file_names = [os.path.basename(f) for f in kml_files]
            selected_file, ok = QInputDialog.getItem(
                self,
                "Select KML File",
                f"Found {len(kml_files)} KML files. Please select one to open:",
                file_names,
                0,
                False
            )
            if ok and selected_file:
                # Find the full path for the selected file
                kml_path = next((f for f in kml_files if os.path.basename(f) == selected_file), None)

        if not kml_path:
            return

        # Determine Google Earth Pro path based on platform
        if sys.platform == "win32":
            earth_path = r"C:\Program Files\Google\Google Earth Pro\client\googleearth.exe"
        elif sys.platform == "darwin":
            earth_path = "/Applications/Google Earth Pro.app/Contents/MacOS/Google Earth Pro"
        else:
            earth_path = "/usr/bin/google-earth-pro"

        # Check if Google Earth Pro is installed
        if not os.path.exists(earth_path):
            # Show installation message
            reply = QMessageBox.question(
                self,
                "Google Earth Pro Not Found",
                "Google Earth Pro is not installed on your system.\n\n"
                "Would you like to download and install it?\n\n"
                "Click 'Yes' to open the download page, or 'No' to cancel.",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.Yes
            )

            if reply == QMessageBox.StandardButton.Yes:
                # Open download page in default browser
                import webbrowser
                webbrowser.open("https://www.google.com/earth/versions/#earth-pro")
            return

        # Launch Google Earth Pro with the selected KML file
        try:
            subprocess.Popen([earth_path, kml_path])
            # QMessageBox.information(
            #     self,
            #     "Success",
            #     f"Google Earth Pro has been launched with:\n{os.path.basename(kml_path)}"
            # )
        except Exception as launch_error:
            QMessageBox.critical(
                self,
                "Launch Error",
                f"Failed to launch Google Earth Pro with the KML file:\n{str(launch_error)}"
            )

    except Exception as e:
        QMessageBox.critical(
            self,
            "Error",
            f"An unexpected error occurred while searching for KML files:\n{str(e)}"
        )