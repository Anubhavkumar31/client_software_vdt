# bootstrap_ui_patch.py
from PyQt6.QtCore import QTimer
from PyQt6.QtWidgets import QPushButton

BOOTSTRAPISH_QT = """
QMainWindow { background:#f8f9fa; color:#212529; }
QStatusBar { background:#ffffff; border-top:1px solid #dee2e6; color:#495057; }

/* Headers / tables */
QHeaderView::section {
  background:#f8f9fa; color:#212529; font-weight:600;
  padding:6px 8px; border:1px solid #dee2e6;
}
QTableView, QTableWidget {
  gridline-color:#e9ecef; alternate-background-color:#fafafa;
  selection-background-color:#0d6efd; selection-color:#ffffff;
}
QTableView::item:selected, QTableWidget::item:selected { background:#0d6efd; color:#fff; }

/* Scrollbars */
QScrollBar:vertical, QScrollBar:horizontal { background:#f1f3f5; border:none; }
QScrollBar::handle:vertical, QScrollBar::handle:horizontal {
  background:#cfd4da; border-radius:6px; min-height:24px; min-width:24px;
}
QScrollBar::handle:vertical:hover, QScrollBar::handle:horizontal:hover { background:#adb5bd; }

/* Buttons */
QPushButton[bs="primary"] {
  background:#0d6efd; color:#fff; border:1px solid #0b5ed7;
  border-radius:8px; padding:6px 12px; font-weight:600;
}
QPushButton[bs="primary"]:hover { background:#0b5ed7; }
QPushButton[bs="outline"] {
  background:#fff; color:#0d6efd; border:1.5px solid #0d6efd;
  border-radius:8px; padding:6px 12px; font-weight:600;
}
QPushButton[bs="outline"]:hover { background:#e7f1ff; }
/* generic tables (top model view also benefits) */
QTableView, QTableWidget {
  gridline-color:#e9ecef;
  alternate-background-color:#fafafa;
  selection-background-color:#0d6efd;
  selection-color:#ffffff;
}
QHeaderView::section {
  background:#f8f9fa; color:#212529; font-weight:600;
  padding:6px 8px; border:1px solid #dee2e6;
}

"""

EMBEDDED_MINI_BOOTSTRAP = """
html, body { background:#ffffff !important; margin:0; padding:0;
  font-family:-apple-system,BlinkMacSystemFont,"Segoe UI",Roboto,Arial,sans-serif; }
"""

def _inject_css_into_qweb(qwebview):
    def _do_inject(_ok):
        js = """
        (function(){
          try {
            var style = document.createElement('style');
            style.type = 'text/css';
            style.innerHTML = `%s`;
            document.head.appendChild(style);
          } catch(e) {}
        })();
        """ % EMBEDDED_MINI_BOOTSTRAP.replace("`", "\\`").replace("\\", "\\\\")
        qwebview.page().runJavaScript(js)
    qwebview.loadFinished.connect(_do_inject)

def apply_bootstrap_like_theme(app, window):
    app.setStyleSheet(BOOTSTRAPISH_QT)

    # Tag a couple of buttons for styling (if present)
    for name in ("btnLoadPipe", "btnDigsheetAbs", "btnOpenFilterDlg"):
        btn = getattr(window, name, None)
        if isinstance(btn, QPushButton):
            btn.setProperty("bs", "primary" if name == "btnLoadPipe" else "outline")
            btn.style().unpolish(btn); btn.style().polish(btn)

    # Only inject a tiny CSS into the main web view (don’t reparent!)
    qweb = getattr(window, "web_view", None)
    if qweb:
        _inject_css_into_qweb(qweb)

def init_after_show(app, window):
    # call this once after the main window is created & shown
    QTimer.singleShot(0, lambda: apply_bootstrap_like_theme(app, window))
