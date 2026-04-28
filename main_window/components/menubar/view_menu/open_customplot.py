from PyQt6.QtWidgets import QMessageBox

from menubar.view_menu.apps.customPlotApp import ExcelDualAxisZoomChart


def open_customplot(self):
    if not self.project_is_open:
        QMessageBox.information(
            self,
            "Project Required",
            "Please create or open a project first."
        )
        return
    if not hasattr(self, "_custom_plot_window"):
        self._custom_plot_window = ExcelDualAxisZoomChart(self)

    self._custom_plot_window.show()
    self._custom_plot_window.raise_()
    self._custom_plot_window.activateWindow()