import os

os.environ["QTWEBENGINE_DISABLE_SANDBOX"] = "1"
os.environ["QTWEBENGINE_CHROMIUM_FLAGS"] = (
    "--disable-gpu "
    "--disable-software-rasterizer "
    "--disable-gpu-compositing "
    "--log-level=3"
)



import sys
import json
import pandas as pd

from PyQt6.QtCore import Qt, QCoreApplication

QCoreApplication.setAttribute(
    Qt.ApplicationAttribute.AA_ShareOpenGLContexts
)

from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout,
    QPushButton, QFileDialog,
    QListWidget, QLabel, QMessageBox,
    QComboBox, QFrame
)

from PyQt6.QtWebEngineWidgets import QWebEngineView



from PyQt6.QtCore import QUrl


# ===================== RESIZABLE WEB VIEW =====================

class ResizableWebView(QWebEngineView):
    def resizeEvent(self, event):
        super().resizeEvent(event)
        self.page().runJavaScript(
            "if (window.chart) { chart.resize(); }"
        )


# ===================== ECHARTS HTML (THEME AWARE) =====================

HTML = """<!DOCTYPE html>
<html>
<head>
<meta charset="UTF-8">
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
html, body {
    margin: 0;
    width: 100%;
    height: 100%;
    overflow: hidden;
}
#chart { width: 100%; height: 100%; }
</style>
</head>
<body>
<div id="chart"></div>

<script>
const chart = echarts.init(document.getElementById("chart"));
window.chart = chart;

let resizeTimer;
window.addEventListener("resize", () => {
    clearTimeout(resizeTimer);
    resizeTimer = setTimeout(() => chart.resize(), 100);
});

function renderChart(payload) {

    const isDark = payload.theme === "dark";

    const bg = isDark ? "#0b1220" : "#ffffff";
    const text = isDark ? "#e5e7eb" : "#0f172a";
    const gridLine = isDark ? "#1f2937" : "#e5e7eb";

    const yAxes = payload.y_axes.map(a => ({
        type: "value",
        name: a.display_name,
        position: a.position,
        nameLocation: "middle",
        nameGap: 55,
        nameTextStyle: {
            color: a.color,
            fontWeight: "bold"
        },
        axisLine: { lineStyle: { color: a.color } },
        axisLabel: { color: text },
        splitLine: { lineStyle: { color: gridLine } }
    }));

    chart.setOption({
        backgroundColor: bg,

        tooltip: { trigger: "axis" },
        legend: { textStyle: { color: text } },

        grid: { left: 80, right: 120, top: 80, bottom: 140 },

        toolbox: {
            right: 30,
            top: 10,
            iconStyle: { borderColor: text },
            feature: {
                restore: {},
                saveAsImage: {}
            }
        },

        dataZoom: [
            { type: "inside", xAxisIndex: 0 },
            { type: "slider", xAxisIndex: 0, bottom: 20, height: 22 }
        ],

        xAxis: {
            type: "category",
            data: payload.x,
            name: payload.x_name,
            nameLocation: "middle",
            nameGap: 80,
            nameTextStyle: {
                color: text,
                fontWeight: "bold"
            },
            axisLabel: { color: text, rotate: 35 },
            axisLine: { lineStyle: { color: text } }
        },

        yAxis: yAxes,
        series: payload.series
    }, true);

    chart.resize();
}

window.renderChart = renderChart;

function applyThemeOnly(theme) {
    const isDark = theme === "dark";

    const bg = isDark ? "#0b1220" : "#ffffff";
    const text = isDark ? "#e5e7eb" : "#0f172a";
    const gridLine = isDark ? "#1f2937" : "#e5e7eb";

    chart.setOption({
        backgroundColor: bg,
        xAxis: {
            axisLabel: { color: text },
            axisLine: { lineStyle: { color: text } }
        },
        yAxis: {
            axisLabel: { color: text },
            splitLine: { lineStyle: { color: gridLine } }
        },
        legend: { textStyle: { color: text } }
    }, false);

    chart.resize();
}

window.applyThemeOnly = applyThemeOnly;

</script>
</body>
</html>
"""


# ===================== MAIN APP =====================

class ExcelDualAxisZoomChart(QMainWindow):
    def __init__(self):
        super().__init__()
        self.setWindowTitle("Excel → Analytics Chart")
        self.resize(1400, 780)

        self.df = None
        self.theme = "light"

        # ---- Widgets ----
        self.chart_type = QComboBox()
        self.chart_type.addItems(["Line", "Scatter", "Bar"])

        self.axis_count = QComboBox()
        self.axis_count.addItems(["1 Y-Axis", "2 Y-Axes"])

        self.x_list = QListWidget()
        self.y_list = QListWidget()
        self.x_list.setFixedHeight(120)
        self.y_list.setFixedHeight(160)

        self.x_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)
        self.y_list.setSelectionMode(QListWidget.SelectionMode.SingleSelection)

        self.load_btn = QPushButton("Load Excel")
        self.plot_btn = QPushButton("Plot Chart")
        self.reset_btn = QPushButton("Reset")
        self.theme_btn = QPushButton("🌙 Dark Mode")

        self.plot_btn.setObjectName("primary")
        self.plot_btn.setEnabled(False)

        self.web = ResizableWebView()
        self.web.setHtml(HTML, QUrl("about:blank"))

        # ---- Sidebar Layout ----
        sidebar = QVBoxLayout()
        sidebar.setSpacing(14)

        sidebar.addWidget(self._section_label("CHART"))
        sidebar.addWidget(QLabel("Chart Type"))
        sidebar.addWidget(self.chart_type)

        sidebar.addSpacing(10)
        sidebar.addWidget(self._divider())

        sidebar.addWidget(self._section_label("AXES"))
        sidebar.addWidget(QLabel("Y Axes Count"))
        sidebar.addWidget(self.axis_count)
        sidebar.addWidget(QLabel("X Axis"))
        sidebar.addWidget(self.x_list)
        sidebar.addWidget(QLabel("Y Axis"))
        sidebar.addWidget(self.y_list)

        sidebar.addSpacing(10)
        sidebar.addWidget(self._divider())

        sidebar.addWidget(self._section_label("ACTIONS"))
        sidebar.addWidget(self.load_btn)
        sidebar.addWidget(self.plot_btn)
        sidebar.addWidget(self.reset_btn)
        sidebar.addWidget(self.theme_btn)
        sidebar.addStretch()

        self.left = QWidget()
        self.left.setLayout(sidebar)
        self.left.setFixedWidth(300)

        self.apply_theme()

        main = QHBoxLayout()
        main.addWidget(self.left)
        main.addWidget(self.web, 1)

        container = QWidget()
        container.setLayout(main)
        self.setCentralWidget(container)

        # ---- Signals ----
        self.load_btn.clicked.connect(self.load_excel)
        self.plot_btn.clicked.connect(self.plot_chart)
        self.reset_btn.clicked.connect(self.reset_selection)
        self.axis_count.currentIndexChanged.connect(self.on_axis_count_changed)
        self.y_list.itemSelectionChanged.connect(self.enforce_y_limit)
        self.theme_btn.clicked.connect(self.toggle_theme)

    # ===================== THEME =====================

    def toggle_theme(self):
        self.theme = "dark" if self.theme == "light" else "light"
        self.theme_btn.setText("☀ Light Mode" if self.theme == "dark" else "🌙 Dark Mode")
        self.apply_theme()

        # 🔹 If chart already plotted, re-render fully
        if self.df is not None:
            self.plot_chart()
        else:
            # 🔹 Apply theme immediately even without data
            self.web.page().runJavaScript(
                f"applyThemeOnly('{self.theme}')"
            )

    def apply_theme(self):
        self.left.setStyleSheet(self._sidebar_style())

    def _sidebar_style(self):
        if self.theme == "light":
            return """
            QWidget {
                background:#f8fafc;
                color:#0f172a;
            }

            QLabel#section {
                color:#64748b;
                font-weight:600;
                letter-spacing:1px;
            }

            QComboBox, QListWidget {
                background:#ffffff;
                border:1px solid #e5e7eb;
                padding:6px;
            }

            /* 🔹 LIGHT MODE SELECTION */
            QListWidget::item:selected {
                background:#dbeafe;   /* light blue */
                color:#0f172a;        /* black text */
            }

            QPushButton {
                background:#ffffff;
                border:1px solid #e5e7eb;
                padding:8px;
            }

            QPushButton#primary {
                background:#2563eb;
                color:white;
                font-weight:600;
            }
            """
        else:
            return """
            QWidget {
                background:#0f172a;
                color:#e5e7eb;
            }

            QLabel#section {
                color:#94a3b8;
                font-weight:600;
                letter-spacing:1px;
            }

            QComboBox, QListWidget {
                background:#020617;
                border:1px solid #1f2937;
                padding:6px;
            }

            /* 🔹 DARK MODE SELECTION (LIGHT BLUE + WHITE TEXT) */
            QListWidget::item:selected {
                background:#2563eb;   /* light blue accent */
                color:#ffffff;        /* white text */
            }

            QPushButton {
                background:#020617;
                border:1px solid #1f2937;
                padding:8px;
            }

            QPushButton#primary {
                background:#38bdf8;
                color:#020617;
                font-weight:600;
            }
            """

    # ===================== UI HELPERS =====================

    def _section_label(self, text):
        lbl = QLabel(text)
        lbl.setObjectName("section")
        return lbl

    def _divider(self):
        line = QFrame()
        line.setFrameShape(QFrame.Shape.HLine)
        return line

    # ===================== LOGIC =====================

    def load_excel(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Excel", "", "Excel Files (*.xlsx *.xls)")
        if not path:
            return

        self.df = pd.read_excel(path)
        self.x_list.clear()
        self.y_list.clear()

        for col in self.df.columns:
            self.x_list.addItem(col)
            if pd.api.types.is_numeric_dtype(self.df[col]):
                self.y_list.addItem(col)

        self.plot_btn.setEnabled(True)

    def reset_selection(self):
        self.x_list.clearSelection()
        self.y_list.clearSelection()
        self.axis_count.setCurrentIndex(0)

    def on_axis_count_changed(self, idx):
        self.y_list.clearSelection()
        self.y_list.setSelectionMode(
            QListWidget.SelectionMode.SingleSelection
            if idx == 0 else QListWidget.SelectionMode.MultiSelection
        )

    def enforce_y_limit(self):
        if self.axis_count.currentIndex() == 1:
            sel = self.y_list.selectedItems()
            if len(sel) > 2:
                sel[-1].setSelected(False)

    def plot_chart(self):
        if self.df is None:
            return

        x = self.x_list.selectedItems()
        y = self.y_list.selectedItems()
        if not x:
            QMessageBox.warning(self, "Error", "Select X axis")
            return

        x_col = x[0].text()
        y_cols = [i.text() for i in y]

        if self.axis_count.currentIndex() == 0 and len(y_cols) != 1:
            QMessageBox.warning(self, "Error", "Select 1 Y axis")
            return
        if self.axis_count.currentIndex() == 1 and len(y_cols) != 2:
            QMessageBox.warning(self, "Error", "Select 2 Y axes")
            return

        colors = ["#22c55e", "#facc15"]
        series, y_axes = [], []

        for i, col in enumerate(y_cols):
            y_axes.append({
                "display_name": f"Y Axis {i+1}: {col}",
                "position": "left" if i == 0 else "right",
                "color": colors[i]
            })
            series.append({
                "name": col,
                "type": self.chart_type.currentText().lower(),
                "data": self.df[col].tolist(),
                "yAxisIndex": i
            })

        payload = {
            "theme": self.theme,
            "x": self.df[x_col].astype(str).tolist(),
            "x_name": f"X Axis: {x_col}",
            "series": series,
            "y_axes": y_axes
        }

        self.web.page().runJavaScript(
            f"renderChart({json.dumps(payload)})"
        )


# ===================== RUN =====================

if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ExcelDualAxisZoomChart()
    win.show()
    sys.exit(app.exec())
