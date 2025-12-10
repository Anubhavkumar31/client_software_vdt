import os

from PyQt6.QtWidgets import QMessageBox


def open_Preliminary_Report(self):
    # Check if a project is open
    if not self.project_is_open or not self.project_root:
        QMessageBox.warning(
            self,
            "No Project Open",
            "Please create/open a project first to access the Preliminary Report.\n\n"
            "Steps:\n"
            "1. Go to File → Create Project\n"
            "2. Select a project folder\n"
            "3. Then try accessing Preliminary Report again"
        )
        return

    # Look for PR.pdf in the report folder within project root
    report_dir = os.path.join(self.project_root, "report")
    prelim_report_path = os.path.join(report_dir, "PR.pdf")

    if not os.path.exists(prelim_report_path):
        QMessageBox.warning(
            self,
            "Preliminary Report Not Found",
            f"Could not find 'PR.pdf' in the report directory:\n{report_dir}\n\n"
            "Please ensure the report folder exists in your project and contains PR.pdf"
        )
        return

    # Open the preliminary report
    try:
        os.startfile(prelim_report_path)
    except Exception as e:
        self.open_Error(f"Failed to open Preliminary Report:\n{e}")