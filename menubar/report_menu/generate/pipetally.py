import os

from PyQt6.QtWidgets import QMessageBox, QInputDialog

from pathlib import Path

def open_pipe_tally(self):
    import os
    from pathlib import Path
    from PyQt6.QtWidgets import QMessageBox

    # 1. Check project
    if not self.project_is_open or not self.project_root:
        QMessageBox.warning(
            self,
            "No Project Open",
            "Please open or create a project first."
        )
        return

    project_path = Path(self.project_root)
    pipetally_path = project_path / "pipetally_main"

    # 2. Check folder exists
    if not pipetally_path.exists():
        QMessageBox.warning(
            self,
            "Missing Folder",
            f"'pipetally_main' folder not found in:\n{self.project_root}"
        )
        return

    # 3. Find files starting with "pipetally_main"
    try:
        matching_files = [
            f for f in pipetally_path.iterdir()
            if f.is_file() and f.name.lower().startswith("pipetally_main")
        ]
    except Exception as e:
        QMessageBox.critical(self, "Error", f"Error reading folder:\n{e}")
        return

    # 4. No file found
    if len(matching_files) == 0:
        QMessageBox.warning(
            self,
            "No File Found",
            "No file starting with 'pipetally_main' found in the pipetally_main folder."
        )
        return

    # 5. More than one file found
    if len(matching_files) > 1:
        QMessageBox.critical(
            self,
            "Multiple Files Found",
            "There can only be ONE file starting with 'pipetally_main' inside the pipetally_main folder.\n"
            "Please keep only one such file."
        )
        return

    # 6. Confirmation before opening
    file_to_open = matching_files[0]
    file_name = file_to_open.name

    reply = QMessageBox.question(
        self,
        "Open Pipe Tally",
        f"This will open '{file_name}' in Microsoft Excel.\n\nDo you want to continue?",
        QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
        QMessageBox.StandardButton.No
    )

    if reply == QMessageBox.StandardButton.Yes:
        try:
            os.startfile(str(file_to_open))
        except Exception as e:
            QMessageBox.critical(
                self,
                "Error",
                f"Failed to open file:\n{e}"
            )
    else:
        QMessageBox.information(self, "Cancelled", "Operation cancelled.")