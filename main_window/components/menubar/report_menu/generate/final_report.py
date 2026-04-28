import os

from PyQt6.QtWidgets import QMessageBox


def open_Final_Report(self):
    # Check if a project is open
    if not self.project_is_open or not self.project_root:
        QMessageBox.warning(
            self,
            "No Project Open",
            "Please create/open a project first to access the Final Report."
        )
        return

    # Look for Final_Report.pdf in the report folder within project root
    report_dir = os.path.join(self.project_root, "report")
    final_report_path = os.path.join(report_dir, "FR.pdf")

    if not os.path.exists(final_report_path):
        QMessageBox.warning(
            self,
            "Final Report Not Found",
            f"Could not find 'Final_Report.pdf' in the report directory:\n{report_dir}"
        )
        return

    try:
        os.startfile(final_report_path)
    except Exception as e:
        self.open_Error(f"Failed to open Final Report:\n{e}")