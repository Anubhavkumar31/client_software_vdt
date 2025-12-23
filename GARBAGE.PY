from test_main import ExcelDualAxisZoomChart


def open_customplot(self):
    if not hasattr(self, "_custom_plot_window"):
        self._custom_plot_window = ExcelDualAxisZoomChart(self)

    self._custom_plot_window.show()
    self._custom_plot_window.raise_()
    self._custom_plot_window.activateWindow()

# open_customplot(self)