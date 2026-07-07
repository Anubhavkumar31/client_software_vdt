import sys
import json
import math
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget,
    QVBoxLayout, QHBoxLayout, QGroupBox,
    QLineEdit, QPushButton, QRadioButton,
    QMessageBox, QLabel, QGridLayout,
    QScrollArea  # Add this import
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtGui import QDoubleValidator
from PyQt6.QtCore import Qt, pyqtSignal

# ================= ECHARTS HTML =================
ECHART_HTML = """
<!DOCTYPE html>
<html>
<head>
<meta charset="utf-8">
<script src="https://cdn.jsdelivr.net/npm/echarts@5/dist/echarts.min.js"></script>
<style>
html, body { width:100%; height:100%; margin:0; }
#chart { width:100%; height:100%; }
</style>
</head>
<body>
<div id="chart"></div>

<script>
let chart = null;

function initChart(theme) {
    if (chart) chart.dispose();
    chart = echarts.init(document.getElementById("chart"), theme);
}

function renderChart(data) {
    if (!chart || !data) return;

    chart.setOption({
        tooltip: { trigger: "axis" },
        legend: { data: ["Depth% vs Axial Length", "Actual Defect"] },
        xAxis: { type: "value", name: "Axial Length (mm)" },
        yAxis: { type: "value", name: "Depth (%)", max: 100 },
        dataZoom: [
            { type: "inside", xAxisIndex: 0 },
            { type: "slider", xAxisIndex: 0, bottom: 20, height: 22 }
        ],
        series: [
            { name: "Depth% vs Axial Length", type: "line", smooth: true, showSymbol: false, data: data.profile },
            { name: "Actual Defect", type: "scatter", symbolSize: 12, data: [[data.L, data.depth_pct]] }
        ]
    }, true);
}

window.initChart = initChart;
window.renderChart = renderChart;
window.onresize = () => chart && chart.resize();
</script>
</body>
</html>
"""


# ================= MAIN WINDOW =================
class ERFWindow(QMainWindow):
    def __init__(self, project_root=None):
        super().__init__()
        self.setWindowTitle("ERF Calculator")
        self.resize(450, 700)  # Slightly smaller initial size
        self.theme = "dark"
        self.last_chart_payload = None

        # Create central widget and main layout
        central = QWidget()
        self.setCentralWidget(central)
        main_layout = QVBoxLayout(central)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # ================= SCROLL AREA =================
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)
        scroll.setVerticalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAsNeeded)

        # Container widget for scroll area
        scroll_content = QWidget()
        scroll_layout = QVBoxLayout(scroll_content)
        scroll_layout.setSpacing(14)
        scroll_layout.setContentsMargins(10, 10, 10, 10)

        # ================= STANDARD =================
        std_layout = QHBoxLayout()
        self.rb_asme = QRadioButton("ASME B31G")
        self.rb_mod = QRadioButton("Mod B31G")
        self.rb_dnv = QRadioButton("DNV-RP-F101")
        self.rb_shell = QRadioButton("SHELL 92")
        self.rb_asme.setChecked(True)

        for rb in (self.rb_asme, self.rb_mod, self.rb_dnv, self.rb_shell):
            rb.toggled.connect(self.check_fields)
            std_layout.addWidget(rb)

        scroll_layout.addWidget(self.make_section("Assessment Standard", std_layout))

        # ================= INPUT PARAMETERS (3×3 GRID) =================
        v = QDoubleValidator()

        self.od_D = QLineEdit();
        self.od_D.setValidator(v)
        self.thickness_T = QLineEdit();
        self.thickness_T.setValidator(v)
        self.smys = QLineEdit();
        self.smys.setValidator(v)

        self.maop = QLineEdit();
        self.maop.setValidator(v)
        self.length_L = QLineEdit();
        self.length_L.setValidator(v)
        self.depth_d = QLineEdit();
        self.depth_d.setValidator(v)

        self.smts = QLineEdit();
        self.smts.setValidator(v)
        self.p_op = QLineEdit();
        self.p_op.setValidator(v)

        # Connect text changed signals
        for field in (self.od_D, self.thickness_T, self.smys, self.maop,
                      self.length_L, self.depth_d, self.smts, self.p_op):
            field.textChanged.connect(self.check_fields)

        grid = QGridLayout()
        grid.setHorizontalSpacing(20)
        grid.setVerticalSpacing(12)

        def cell(label, widget):
            box = QVBoxLayout()
            lbl = QLabel(label)
            lbl.setStyleSheet("font-size:11px;")
            box.addWidget(lbl)
            box.addWidget(widget)
            return box

        grid.addLayout(cell("Outside Diameter (mm)", self.od_D), 0, 0)
        grid.addLayout(cell("Wall Thickness (mm)", self.thickness_T), 0, 1)
        grid.addLayout(cell("SMYS (MPa)", self.smys), 0, 2)

        grid.addLayout(cell("MAOP (MPa)", self.maop), 2, 0)
        grid.addLayout(cell("Axial Length (mm)", self.length_L), 2, 1)
        grid.addLayout(cell("Depth (mm)", self.depth_d), 2, 2)

        grid.addLayout(cell("SMTS (MPa)", self.smts), 3, 0)
        grid.addLayout(cell("P-op (MPa)", self.p_op), 3, 1)

        scroll_layout.addWidget(self.make_section("Pipeline Parameters", grid))

        # ================= RESULTS =================
        res_grid = QGridLayout()
        self.erf_out = QLineEdit();
        self.erf_out.setReadOnly(True)
        self.safe_p_out = QLineEdit();
        self.safe_p_out.setReadOnly(True)

        res_grid.addLayout(cell("ERF", self.erf_out), 0, 0)
        res_grid.addLayout(cell("Psafe (kg/cm² )", self.safe_p_out), 0, 1)

        scroll_layout.addWidget(self.make_section("Results", res_grid))

        # ================= ACTIONS =================
        actions = QHBoxLayout()
        self.calc_btn = QPushButton("Calculate")
        self.calc_btn.setEnabled(False)  # Start disabled
        self.reset_btn = QPushButton("Reset")
        self.theme_btn = QPushButton("☀ Light Mode")

        self.calc_btn.clicked.connect(self.calculate_erf)
        self.reset_btn.clicked.connect(self.reset_fields)
        self.theme_btn.clicked.connect(self.toggle_theme)

        actions.addWidget(self.calc_btn)
        actions.addWidget(self.reset_btn)
        actions.addStretch()
        actions.addWidget(self.theme_btn)

        scroll_layout.addLayout(actions)

        # ================= CHART =================
        self.web = QWebEngineView()
        self.web.setMinimumHeight(400)  # Slightly larger minimum height
        self.web.setHtml(ECHART_HTML)
        self.web.loadFinished.connect(self.on_web_ready)

        scroll_layout.addWidget(self.web, 1)

        # Add scroll area to main layout
        scroll.setWidget(scroll_content)
        main_layout.addWidget(scroll)

        self.apply_theme()
        self.check_fields()  # Initial check

    # ================= SECTION BUILDER =================
    def make_section(self, title, content_layout):
        container = QWidget()
        v = QVBoxLayout(container)
        v.setContentsMargins(0, 0, 0, 0)
        v.setSpacing(6)

        header = QLabel(title)
        header.setObjectName("sectionHeader")

        body = QGroupBox()
        body.setLayout(content_layout)

        v.addWidget(header)
        v.addWidget(body)

        return container

    # ================= FIELD VALIDATION =================
    def check_fields(self):
        """Check if all required fields for the selected standard are filled"""
        required_fields = self.get_required_fields()

        # Check if all required fields have non-empty text
        all_filled = all(field.text().strip() for field in required_fields)

        # Also check if the values are valid numbers (optional)
        if all_filled:
            try:
                for field in required_fields:
                    float(field.text())
                self.calc_btn.setEnabled(True)
            except ValueError:
                self.calc_btn.setEnabled(False)
        else:
            self.calc_btn.setEnabled(False)

    def get_required_fields(self):
        """Return list of required fields for the selected standard"""
        if self.rb_asme.isChecked() or self.rb_mod.isChecked():
            # ASME B31G and Mod B31G require: D, T, SMYS, MAOP, L, d
            return [self.od_D, self.thickness_T, self.smys, self.maop,
                    self.length_L, self.depth_d]
        elif self.rb_dnv.isChecked():
            # DNV-RP-F101 requires: D, T, SMTS, P_op, L, d
            return [self.od_D, self.thickness_T, self.smts, self.p_op,
                    self.length_L, self.depth_d]
        elif self.rb_shell.isChecked():
            # SHELL 92 requires: D, T, SMYS, MAOP, L, d
            return [self.od_D, self.thickness_T, self.smys, self.maop,
                    self.length_L, self.depth_d]
        return []

    # ================= WEB READY =================
    def on_web_ready(self):
        self.web.page().runJavaScript(f"initChart('{self.theme}')")

    # ================= THEME =================
    def toggle_theme(self):
        self.theme = "light" if self.theme == "dark" else "dark"
        self.theme_btn.setText("☀ Light Mode" if self.theme == "dark" else "🌙 Dark Mode")
        self.apply_theme()
        self.web.page().runJavaScript(f"initChart('{self.theme}')")

        if self.last_chart_payload:
            self.web.page().runJavaScript(
                f"renderChart({json.dumps(self.last_chart_payload)})"
            )

    def apply_theme(self):
        if self.theme == "dark":
            self.setStyleSheet("""
            QWidget { background:#0f172a; color:#e5e7eb; }
            QLabel#sectionHeader { color:#94a3b8; font-weight:600; padding:6px 4px; }
            QGroupBox { border:1px solid #1f2937; border-radius:6px; padding:10px; }
            QLineEdit { background:#020617; border:1px solid #1f2937; padding:8px; }
            QPushButton { padding:8px 14px; }
            QPushButton:disabled { background:#1f2937; color:#4b5563; }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                border: none;
                background: #1f2937;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #374151;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
            """)
        else:
            self.setStyleSheet("""
            QWidget { background:#f8fafc; color:#0f172a; }
            QLabel#sectionHeader { color:#475569; font-weight:600; padding:6px 4px; }
            QGroupBox { border:1px solid #cbd5f5; border-radius:6px; padding:10px; }
            QLineEdit { background:#ffffff; border:1px solid #cbd5f5; padding:8px; }
            QPushButton { padding:8px 14px; }
            QPushButton:disabled { background:#e2e8f0; color:#94a3b8; }
            QScrollArea { border: none; background: transparent; }
            QScrollBar:vertical {
                border: none;
                background: #e2e8f0;
                width: 12px;
                margin: 0px;
            }
            QScrollBar::handle:vertical {
                background: #94a3b8;
                min-height: 20px;
                border-radius: 6px;
            }
            QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {
                border: none;
                background: none;
            }
            """)

    # ================= ERF DISPATCH =================
    def calculate_erf(self):
        if self.rb_asme.isChecked():
            self.calculate_asme_b31g()
        elif self.rb_mod.isChecked():
            self.calculate_mod_b31g()
        elif self.rb_dnv.isChecked():
            self.calculate_dnv_rp_f101_erf()
        else:
            self.calculate_shell_92()

    # ================= STANDARD METHODS =================
    def calculate_asme_b31g(self):
        print("asme b31g")
        self._common_erf()

    def calculate_mod_b31g(self):
        print("mod b31g selected and its erf getting calculated")
        try:
            # -------- inputs --------
            D = float(self.od_D.text())
            t = float(self.thickness_T.text())
            SMYS = float(self.smys.text())
            MAOP = float(self.maop.text())
            L = float(self.length_L.text())
            d = float(self.depth_d.text())

            # -------- Modified B31G math --------
            # 1) Flow stress
            sigma_f = 1.1 * SMYS

            # 2) Folias factor (Modified B31G)
            M = math.sqrt(1 + 0.6275 * (L / math.sqrt(D * t)) ** 2)

            # 3) Remaining strength factor
            rsf = (1 - 0.85 * (d / t)) / (1 - (0.85 * (d / t)) / M)

            # 4) Failure pressure
            Pf = (2 * sigma_f * t / D) * rsf

            # 5) Safe operating pressure
            Psafe = (Pf / 1.39) * 10.1972

            # 6) ERF
            ERF = MAOP / Psafe

            # -------- UI outputs --------
            self.erf_out.setText(f"{ERF:.3f}")
            self.safe_p_out.setText(f"{Psafe:.2f}")

            self._render_chart(L, d, t)

        except Exception:
            QMessageBox.critical(self, "Error", "Please enter valid numeric values")

    def calculate_dnv_rp_f101_erf(self, offshore=True):
        print("dnv rp f101")

        # -------- inputs --------
        D = float(self.od_D.text())
        t = float(self.thickness_T.text())
        SMTS = float(self.smts.text())
        P_op = float(self.p_op.text())
        L = float(self.length_L.text())
        d = float(self.depth_d.text())

        F1 = 0.90
        F2 = 0.67 if offshore else 0.72
        F = F1 * F2

        Q = math.sqrt(1 + 0.31 * (L / (D * t)) ** 2)

        P_fail = (2 * SMTS * t / (D - t)) * ((1 - d / t) / (1 - d / (t * Q)))

        Psafe = (F * P_fail) * 10.1972

        ERF = P_op / Psafe

        # -------- UI outputs --------
        self.erf_out.setText(f"{ERF:.3f}")
        self.safe_p_out.setText(f"{Psafe:.2f}")

        self._render_chart(L, d, t)
        return ERF

    def calculate_shell_92(self):
        print("shell 92")
        try:
            D = float(self.od_D.text())
            t = float(self.thickness_T.text())
            SMYS = float(self.smys.text())
            MAOP = float(self.maop.text())
            L = float(self.length_L.text())
            d = float(self.depth_d.text())

            # Shell-92
            sigma_f = 1.15 * SMYS
            M = math.sqrt(1 + 0.31 * (L / math.sqrt(D * t)) ** 2)
            rsf = (1 - 0.9 * (d / t)) / (1 - (0.9 * (d / t)) / M)

            Pf = (2 * sigma_f * t / D) * rsf
            Psafe = (Pf / 1.5) * 10.1972
            ERF = MAOP / Psafe

            self.erf_out.setText(f"{ERF:.3f}")
            self.safe_p_out.setText(f"{Psafe:.2f}")

            # chart hook
            x = list(range(0, int(max(500, L * 1.3)), 10))
            profile = [[i, 100 / (1 + i / 150)] for i in x]

            payload = {
                "profile": profile,
                "L": L,
                "depth_pct": (d / t) * 100
            }

            self.last_chart_payload = payload
            self.web.page().runJavaScript(
                f"renderChart({json.dumps(payload)})"
            )

        except Exception:
            QMessageBox.critical(self, "Error", "Please enter valid numeric values")

    # ================= COMMON ERF =================
    def _render_chart(self, L, d, t):
        x = list(range(0, int(max(500, L * 1.3)), 10))
        profile = [[i, 100 / (1 + i / 150)] for i in x]

        payload = {
            "profile": profile,
            "L": L,
            "depth_pct": (d / t) * 100
        }

        self.last_chart_payload = payload
        self.web.page().runJavaScript(
            f"renderChart({json.dumps(payload)})"
        )

    def _common_erf(self):
        try:
            L = float(self.length_L.text())
            d = float(self.depth_d.text())
            D = float(self.od_D.text())
            T = float(self.thickness_T.text())
            SMYS = float(self.smys.text())
            MAOP = float(self.maop.text())

            flow_stress = 1.1 * SMYS
            z_factor = (L * L) / (D * T)
            M = math.sqrt(1 + 0.8 * z_factor)
            y = 1 - (2 / 3) * (d / T)
            z = 1 - ((2 / 3) * (d / T)) / M
            k = y / z

            Estimated_failure_stress_level_SF = flow_stress * k
            estimate_failure_pressure = (2 * Estimated_failure_stress_level_SF * T) / D
            safe_operating_pressure = (estimate_failure_pressure / 1.39) * 10.1972
            ERF = MAOP / safe_operating_pressure

            self.erf_out.setText(f"{ERF:.3f}")
            self.safe_p_out.setText(f"{safe_operating_pressure:.2f}")

            x = list(range(0, int(max(500, L * 1.3)), 10))
            profile = [[i, 100 / (1 + i / 150)] for i in x]

            payload = {
                "profile": profile,
                "L": L,
                "depth_pct": (d / T) * 100
            }

            self.last_chart_payload = payload
            self.web.page().runJavaScript(
                f"renderChart({json.dumps(payload)})"
            )

        except Exception:
            QMessageBox.critical(self, "Error", "Please enter valid numeric values")

    def reset_fields(self):
        for f in (
                self.od_D, self.thickness_T, self.smys, self.maop,
                self.length_L, self.depth_d, self.smts, self.p_op
        ):
            f.clear()
        self.erf_out.clear()
        self.safe_p_out.clear()
        self.last_chart_payload = None
        self.calc_btn.setEnabled(False)  # Disable after reset


# ================= RUN =================
if __name__ == "__main__":
    app = QApplication(sys.argv)
    win = ERFWindow()
    win.show()
    sys.exit(app.exec())