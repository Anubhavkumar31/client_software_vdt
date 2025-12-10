import os

from PyQt6.QtWidgets import QMessageBox, QInputDialog

from pathlib import Path

def open_pipe_tally(self):
    # Check if a project is open
    if not self.project_is_open or not self.project_root:
        QMessageBox.warning(
            self,
            "No Project Open",
            "Please create/open a project first to access the pipe tally file.\n\n"
            "Steps:\n"
            "1. Go to File → Create Project\n"
            "2. Select a project folder\n"
            "3. Then try accessing Pipe Tally again"
        )
        return

    if not hasattr(self, 'pipe_tally') or self.pipe_tally is None:
        QMessageBox.warning(
            self,
            "No Pipe Tally Loaded",
            "No pipe tally data is currently loaded from this project."
        )
        return

    # Search for pipe tally files ONLY in the project root directory (not subdirectories)
    pipe_tally_files = []
    project_path = Path(self.project_root)

    # Define pattern to match pipe tally related files (case-insensitive)
    # Matches: pipetally, pipe_tally, tally_pipe, pipe-tally, etc.
    import re
    tally_pattern = re.compile(r'.*(pipe.*tally|tally.*pipe|pipetally|pipe_tally|pipe-tally).*\.(xlsx?|csv)$',
                               re.IGNORECASE)

    # Search ONLY in project root (not subdirectories)
    # Search ONLY in pipetally_main subfolder
    pipetally_main_path = project_path / "pipetally_main"
    if not pipetally_main_path.is_dir():
        QMessageBox.warning(
            self,
            "Pipetally Directory Not Found",
            f"Could not find 'pipetally_main' folder in the project directory:\n{self.project_root}\n\n"
            "Please ensure the pipetally_main folder exists in your project."
        )
        return

    try:
        for file_path in pipetally_main_path.iterdir():  # Only direct children of pipetally_main
            if file_path.is_file() and tally_pattern.match(file_path.name):
                pipe_tally_files.append(str(file_path))

    except Exception as e:
        QMessageBox.critical(
            self,
            "Error",
            f"Error searching for pipe tally files:\n{e}"
        )
        return

    if not pipe_tally_files:
        QMessageBox.warning(
            self,
            "Pipe Tally File Not Found",
            f"Could not find any pipe tally files in the project root directory:\n{self.project_root}\n\n"
            "Looking for files containing: 'pipetally', 'pipe_tally', 'tally_pipe', etc.\n"
            "Note: Only searching in the root folder, not inside pipe subdirectories.\n\n"
            "The pipe tally data is loaded in memory, but the source file could not be located."
        )
        return

    # If multiple files found, let user choose
    pipe_tally_file = None
    if len(pipe_tally_files) == 1:
        pipe_tally_file = pipe_tally_files[0]
    else:
        # Show selection dialog for multiple pipe tally files
        file_names = [os.path.basename(f) for f in pipe_tally_files]
        selected_file, ok = QInputDialog.getItem(
            self,
            "Select Pipe Tally File",
            f"Found {len(pipe_tally_files)} pipe tally files in the root directory. Please select one to open:",
            file_names,
            0,
            False
        )
        if ok and selected_file:
            # Find the full path for the selected file
            pipe_tally_file = next((f for f in pipe_tally_files if os.path.basename(f) == selected_file), None)

    # Open the selected file
    if pipe_tally_file:
        try:
            os.startfile(pipe_tally_file)
        except Exception as e:
            self.open_Error(f"Failed to open pipe tally file:\n{e}")
    else:
        QMessageBox.information(self, "No Selection", "No file was selected.")