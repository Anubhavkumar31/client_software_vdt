import os

from PyQt6.QtCore import QTimer, Qt, QUrl, pyqtSignal
from PyQt6.QtWebEngineCore import QWebEnginePage
from PyQt6.QtWebEngineWidgets import QWebEngineView

from main_section_view.utils import update_digsheet_button_state
from ui.graphs_ui import GraphApp
from PyQt6.QtGui import QAction
from PyQt6.QtWidgets import QMessageBox, QDialog


class ConsoleRelayPage(QWebEnginePage):
    """Catches JS console messages to ferry Plotly relayout/hover to Python."""
    relayout_json = pyqtSignal(dict)    # emits on plotly_relayout
    hover_json    = pyqtSignal(dict)    # (optional) emits on plotly_hover

    def javaScriptConsoleMessage(self, level, msg, line, source):
        if msg.startswith("RANGE:"):
            import json
            try:
                payload = json.loads(msg[6:])
                self.relayout_json.emit(payload)
            except Exception:
                pass
        elif msg.startswith("HOVER:"):
            import json
            try:
                payload = json.loads(msg[6:])
                self.hover_json.emit(payload)
            except Exception:
                pass
        # still let base handle logging
        return super().javaScriptConsoleMessage(level, msg, line, source)

class SyncPlotlyView(QWebEngineView):
    """
    A webview that, after the Plotly HTML loads, injects small JS hooks that:
      - listen for plotly_relayout and emit to Python
      - expose a JS function to apply ranges from Python
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self._page = ConsoleRelayPage(self)
        self.setPage(self._page)
        self._installed = False
        self._busy = False
        self.loadFinished.connect(self._install_hooks_if_needed)

    @property
    def relay(self) -> ConsoleRelayPage:
        return self._page

    def _install_hooks_if_needed(self, ok: bool):
        if not ok or self._installed:
            return

        js = r"""
        (function(){
          if (window.__pie_hooks_installed) return;
          window.__pie_hooks_installed = true;

          function getGraph(){
            let g = document.querySelector('.js-plotly-plot');
            if (!g) g = document.querySelector('div[data-plotly]');
            if (!g) {
              const cand = Array.from(document.querySelectorAll('div'));
              g = cand.find(d => d && d._fullLayout);
            }
            return g;
          }

          function emitRange(){
            const g = getGraph();
            if (!g || !window.Plotly) return;
            const x = g.layout?.xaxis?.range;
            const y = g.layout?.yaxis?.range;
            if (x && y) {
              try {
                console.log('RANGE:' + JSON.stringify({'xaxis.range':x, 'yaxis.range':y}));
              } catch(e){}
            }
          }

          function install(){
            const g = getGraph();
            if (!g || !window.Plotly) { setTimeout(install, 200); return; }

            // Catch all interactions that change zoom/pan
            g.on('plotly_relayout', emitRange);
            g.on('plotly_doubleclick', emitRange);
            g.on('plotly_afterplot', emitRange);
            g.on('plotly_redraw', emitRange);
            g.on('plotly_autosize', emitRange);
            g.on('plotly_restyle', emitRange);

            //  Support mouse wheel zoom
            g.addEventListener('wheel', () => setTimeout(emitRange, 200));

            // 🔹 Support laptop touchpad pinch / scroll gestures
            g.addEventListener('gesturechange', () => setTimeout(emitRange, 200));
            g.addEventListener('touchmove', () => setTimeout(emitRange, 200));

            // 🔹 Function called from Python to apply the other heatmap's range
            window.__pie_applyRelayout = function(payload){
              try {
                const g2 = getGraph();
                if (g2 && window.Plotly) Plotly.relayout(g2, payload);
              } catch(err){}
            };
          }

          install();
        })();
        """
        self.page().runJavaScript(js)
        self._installed = True


    def apply_relayout(self, payload: dict):
        """Apply ranges from the other view (with a feedback guard)."""
        if self._busy:
            return
        self._busy = True
        self.page().runJavaScript(
            f"window.__pie_applyRelayout({payload!r});",
            lambda _=None: self._clear_busy()
        )

    def _clear_busy(self):
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(0, lambda: setattr(self, "_busy", False))

#used in setup_menu_actions.py inside main_window.components
def _on_middle_tab_changed(self, index: int):
    # print("inside middole tab change ")
    if self._reverting_tab:
        return

    if not self.project_is_open:
        if self._ui_ready:
            self._project_required_popup()
        self._reverting_tab = True
        try:
            self.ui.tabWidgetM.setCurrentIndex(self._last_allowed_tab_index)
        finally:
            self._reverting_tab = False
        return

    self._last_allowed_tab_index = index

    # Get current tab name
    tab_text = self.ui.tabWidgetM.tabText(index).strip()
    # Fix: Switch the upper frame content correctly
    if hasattr(self, "top_stack"):
        if tab_text.lower() == "heatmap":
            # show the dual-heatmaps page
            self.top_stack.setCurrentWidget(self.dual_heatmaps_page)
        else:
            # show the single-chart page (for LineChart, 3D Graph, etc.)
            self.top_stack.setCurrentWidget(self.single_chart_page)

    # Always show table for LineChart and 3D Graph tabs
    if tab_text in {"LineChart", "Line Chart", "Line Plot", "3D Graph", "3D"}:
        self.bottom_stack.show()
        # Disable the toggle button for non-Heatmap tabs
        if hasattr(self, 'btnToggleTable'):
            self.btnToggleTable.setEnabled(False)
        if hasattr(self, "btnToggleHmLayout"):
            self.btnToggleHmLayout.setEnabled(False)
    # For Heatmap, respect the toggle flag
    elif tab_text == "Heatmap":
        if getattr(self, '_table_hidden', False):
            self.bottom_stack.hide()
        else:
            self.bottom_stack.show()
        # Enable the toggle button for Heatmap tab
        if hasattr(self, 'btnToggleTable'):
            self.btnToggleTable.setEnabled(True)
        if hasattr(self, "btnToggleHmLayout"):
            self.btnToggleHmLayout.setEnabled(True)
        QTimer.singleShot(100, lambda : _reset_splitter_ratio(self, 0.45))

    tab_switcher2(self)
    update_digsheet_button_state(self)

def syncdropdownwithtabs(self, index: int):
    """Sync dropdown when tab changes from other sources"""
    try:
        if hasattr(self, 'tabSwitcherDropdown'):
            self.tabSwitcherDropdown.blockSignals(True)
            self.tabSwitcherDropdown.setCurrentIndex(index)
            self.tabSwitcherDropdown.blockSignals(False)
    except Exception as e:
        print(f"Error syncing dropdown: {e}")



#helper functions for _on_middle_tab_changed
def _reset_splitter_ratio(self, top_ratio: float = 0.6):
    """Force consistent top/bottom height ratio for the stack layout."""
    if not hasattr(self, "splitter"):
        return

    def apply_ratio():
        sizes = self.splitter.sizes()
        total = sum(sizes) if sizes else self.splitter.height()
        if total > 0:
            top = int(total * top_ratio)
            bottom = total - top
            self.splitter.setSizes([top, bottom])
            # optional debug
            print(f"[DEBUG] Splitter resized: top={top}, bottom={bottom}, total={total}")

    # 🔹 Delay the resize slightly so the layout stabilizes first
    QTimer.singleShot(120, apply_ratio)



# ---------- guarded connections for heatmap/line/3D / action_graphs----------
def _connect_guarded_graph_controls(self):
    a = self.ui
    # QActions from menu/toolbar
    action_map = [
        ("actionHeatmap", "Heatmap"),
        ("action_LineChart", "LineChart"),
        ("action_3D_Graph", "3D"),
    ]
    if hasattr(self.ui, "action_graphs"):
        self.ui.action_graphs.triggered.connect(lambda : open_graphs_window(self))

    for aname, tab in action_map:
        act = getattr(a, aname, None)
        if isinstance(act, QAction):
            try: act.triggered.disconnect()
            except Exception: pass
            act.triggered.connect(lambda _=False, t=tab: lambda t: _guarded_open_tab(self, t))

    # Buttons / toolbuttons
    widget_map = [
        ("btnHeatmap", "Heatmap"),
        ("toolButtonHeatmap", "Heatmap"),
        ("btnLinechart", "LineChart"),
        ("toolButtonLine", "LineChart"),
        ("btn3D", "3D"),
        ("toolButton3D", "3D"),
    ]
    for wname, tab in widget_map:
        w = getattr(a, wname, None)
        if w is not None and hasattr(w, "clicked"):
            try: w.clicked.disconnect()
            except Exception: pass
            w.clicked.connect(lambda _=False, t=tab: lambda t: _guarded_open_tab(self, t))


def _guarded_open_tab(self, tab_name: str):
    if not self.project_is_open:
        if self._ui_ready:
            self._project_required_popup()
        return
    wanted = {
        "Heatmap": {"Heatmap"},
        "LineChart": {"LineChart", "Line Chart", "Line Plot"},
        "3D": {"3D Graph", "3D"},
    }.get(tab_name, {tab_name})

    tw = self.ui.tabWidgetM
    for i in range(tw.count()):
        if tw.tabText(i) in wanted:
            tw.setCurrentIndex(i)
            tab_switcher2(self)
            return
    QMessageBox.information(self, "Tab not found", f"Could not locate tab: {tab_name}")


def open_graphs_window(self):
    if self.pipe_tally is None:
        QMessageBox.warning(self, "No Pipe Tally", "Please create or load a project first.")
        return

    if self._central_graphs is None:
        self._central_graphs = GraphApp(self.pipe_tally,self.project_root)
    self.setCentralWidget(self._central_graphs)



#filter column button helper functions


def apply_column_filter(self):
    """Hide/show columns based on self._selected_columns + locked columns."""
    locked = set(getattr(self, "BACKEND_LOCKED_COLS", set()))

    # If we have no selection yet, treat as 'show all'
    if not self._selected_columns:
        self._selected_columns = set(self._current_headers_for_filter()) | locked

    names_to_keep = set(self._selected_columns) | locked

    # Prefer bottom QTableWidgetDefect if it has columns
    if hasattr(self.ui, "tableWidgetDefect") and self.ui.tableWidgetDefect.columnCount() > 0:
        header_map = {
            c: (self.ui.tableWidgetDefect.horizontalHeaderItem(c).text()
                if self.ui.tableWidgetDefect.horizontalHeaderItem(c) else f"Col {c}")
            for c in range(self.ui.tableWidgetDefect.columnCount())
        }
        for c, name in header_map.items():
            hide = (name not in names_to_keep) and (name not in locked)
            self.ui.tableWidgetDefect.setColumnHidden(c, hide)
        QTimer.singleShot(0, self._refresh_table_scrollbars)
        return

    # Fallback to the top QTableView
    if hasattr(self.ui, "tableView") and self.ui.tableView.model() is not None:
        model = self.ui.tableView.model()
        header_names = [str(model.headerData(c, Qt.Orientation.Horizontal)) for c in range(model.columnCount())]
        for c, name in enumerate(header_names):
            hide = (name not in names_to_keep) and (name not in locked)
            self.ui.tableView.setColumnHidden(c, hide)


def _apply_heatmap_layout(self, mode: str = None):
    """Apply horizontal (side-by-side) or vertical (stacked) layout for dual heatmaps"""
    # Use provided mode or fall back to current mode
    if mode is None:
        mode = getattr(self, '_hm_layout_mode', 'horizontal')

    # Safety checks
    if not hasattr(self, 'top_hsplit'):
        print("Warning: top_hsplit not found, skipping layout change")
        return

    self._hm_layout_mode = mode

    # Change splitter orientation
    if mode == "horizontal":
        self.top_hsplit.setOrientation(Qt.Orientation.Horizontal)
        if hasattr(self, 'btnToggleHmLayout'):
            self.btnToggleHmLayout.setText("stack" if mode == "horizontal" else "side-by-side")
        # Apply 50-50 split
        total = self.top_hsplit.width()
        left = int(total * 0.38)
        right = total - left
        self.top_hsplit.setSizes([left, right])
    else:  # vertical
        self.top_hsplit.setOrientation(Qt.Orientation.Vertical)
        if hasattr(self, 'btnToggleHmLayout'):
            self.btnToggleHmLayout.setText("Side-by-side")
        # Apply 50-50 split
        total = self.top_hsplit.height()
        top = (total // 2) - 95
        bottom = total - top
        self.top_hsplit.setSizes([top, bottom])

    print(f"Heatmap layout changed to: {mode}")


#tab_switcher used in on middle tab changed and its helper func
def tab_switcher2(self, *_):
    if not self.project_is_open:
        self._show_watermark()
        return
    try:
        tab = self.ui.tabWidgetM.tabText(self.ui.tabWidgetM.currentIndex())

        if tab == "Heatmap":
            # Only proceed if UI is fully initialized
            if not hasattr(self, 'top_stack'):
                print("Warning: top_stack not yet initialized, skipping heatmap view")
                return

            # Set dual mode layout
            _set_top_mode(self, "dual")

            # Load both heatmaps into the splitter
            if self.hhmap and hasattr(self, 'web_view_left'):
                _load_scrollable_chart(self, self.web_view_left, self.hhmap, min_w=2200, min_h=1400)
            else:
                if hasattr(self, 'web_view_left'):
                    self.web_view_left.setUrl(QUrl())

            if self.phmap and hasattr(self, 'web_view_right'):
                _load_scrollable_chart(self, self.web_view_right, self.phmap, min_w=2200, min_h=1400)
            else:
                if hasattr(self, 'web_view_right'):
                    self.web_view_right.setUrl(QUrl())

            # Apply the current layout mode
            _apply_heatmap_layout(self, self._hm_layout_mode)
            # --- 🔄 Synchronize zoom/pan between both heatmaps ---
            try:
                if hasattr(self, "web_view_left") and hasattr(self, "web_view_right"):
                    self.web_view_left.relay.relayout_json.connect(
                        lambda payload: self._sync_heatmap_range(self.web_view_right, payload)
                    )
                    self.web_view_right.relay.relayout_json.connect(
                        lambda payload: self._sync_heatmap_range(self.web_view_left, payload)
                    )
                    print("✅ Heatmap synchronization connections established")
            except Exception as sync_err:
                print(f"⚠️ Heatmap sync setup failed: {sync_err}")

            left_pixel_offset = 120     # your desired vertical pixel scroll offset for left heatmap
            right_pixel_offset = 120     # desired offset for right heatmap

            QTimer.singleShot(100, lambda: self.left_scroll.verticalScrollBar().setValue(left_pixel_offset))
            QTimer.singleShot(100, lambda: self.right_scroll.verticalScrollBar().setValue(right_pixel_offset))


            self.bottom_stack.setCurrentIndex(0)
            QTimer.singleShot(100, lambda : _arm_main_topbar(self))



        elif tab in ("LineChart", "Line Chart", "Line Plot"):
            if self.lplot:
                _load_scrollable_chart(self, self.web_view, self.lplot, min_w=2200, min_h=1400)
            else:
                self.web_view.setUrl(QUrl())
            if self.prox_linechart and os.path.exists(self.prox_linechart):
                self.bottom_stack.setCurrentIndex(2)
                _load_scrollable_chart(self, self.web_view2, self.prox_linechart, min_w=2000, min_h=900)
                QTimer.singleShot(0, lambda : _arm_topbar(self))
                QTimer.singleShot(120, lambda : _arm_topbar(self))  # small safety nudge
                QTimer.singleShot(500, lambda: _setup_web_view_scrollbars(self, self.web_view2))
            else:
                self.bottom_stack.setCurrentIndex(0)
                self.web_view2.setUrl(QUrl())
            # Setup scrollbar for line chart main view
            QTimer.singleShot(100, lambda : _arm_main_topbar(self))

        elif tab in ("3D Graph", "3D"):
            if self.pipe3d:
                try:
                    _load_scrollable_chart(self, self.web_view, self.pipe3d, min_w=2200, min_h=1400)
                except AttributeError:
                    self.web_view.setUrl(QUrl.fromLocalFile(self.pipe3d))
            else:
                self.web_view.setUrl(QUrl())
            self.bottom_stack.setCurrentIndex(0)
            self.web_view2.setUrl(QUrl())
            # Setup scrollbar for 3D graph
            QTimer.singleShot(100, lambda : _arm_main_topbar(self))

        update_digsheet_button_state(self)
    except Exception as e:
        self.open_Error(e)








def _load_scrollable_chart(self, view: QWebEngineView, html_path: str, min_w: int = 2200, min_h: int = 1400):
    if not html_path or not os.path.exists(html_path):
        view.setUrl(QUrl())
        return
    effective_min_w = max(0, min_w - self._right_margin_px)

    safe = html_path.replace('\\', '/')
    wrapper = f"""<!doctype html>
<html>
<head>
<meta charset="utf-8">
<style>
* {{
    scrollbar-width: auto !important;
    -webkit-appearance: auto !important;
}}
html, body {{ 
    height: 100%; 
    margin: 0; 
    overflow: hidden;
}}
.wrap {{ 
    height: 100vh; 
    width: 100vw; 
    overflow: scroll !important;
    overflow-x: scroll !important;
    overflow-y: scroll !important;
    scrollbar-width: auto !important;
    -ms-overflow-style: scrollbar !important;
}}
.wrap::-webkit-scrollbar {{
    width: 18px !important;
    height: 18px !important;
    background: #f5f5f5 !important;
    display: block !important;
}}
.wrap::-webkit-scrollbar-track {{
    background: #e0e0e0 !important;
    border: 1px solid #ccc !important;
}}
.wrap::-webkit-scrollbar-thumb {{
    background: #666 !important;
    border: 2px solid #999 !important;
    border-radius: 2px !important;
}}
.wrap::-webkit-scrollbar-thumb:hover {{
    background: #333 !important;
}}
.wrap::-webkit-scrollbar-corner {{
    background: #e0e0e0 !important;
}}
iframe {{ 
    border: 0; 
    width: {effective_min_w}px !important; 
    height: {min_h}px !important;
    min-width: {effective_min_w}px !important;
    min-height: {min_h}px !important;
    display: block;
}}
</style>
</head>
<body>
<div class="wrap" id="scrollContainer">
<iframe sandbox="allow-scripts allow-same-origin allow-forms" src="file:///{safe}"></iframe>
</div>
<script>
// Force scrollbars to be visible
document.addEventListener('DOMContentLoaded', function() {{
const container = document.getElementById('scrollContainer');

// Force a reflow to ensure scrollbars appear
container.style.overflow = 'hidden';
setTimeout(() => {{
    container.style.overflow = 'scroll';
    container.style.overflowX = 'scroll';
    container.style.overflowY = 'scroll';
}}, 10);

// Trigger scroll to force scrollbar appearance
container.scrollLeft = 1;
container.scrollTop = 1;
setTimeout(() => {{
    container.scrollLeft = 0;
    container.scrollTop = 0;
}}, 100);
}});
</script>
</body>
</html>"""
    base = QUrl.fromLocalFile(os.path.dirname(html_path) + os.sep)
    view.setHtml(wrapper, base)



def _arm_topbar(self, virtual_max: int = 2000):
    """Re-sync the top scrollbar with the inner QScrollArea hbar and enable mapping."""
    try:
        inner = self.web_scroll_area.horizontalScrollBar()
        imin, imax = inner.minimum(), inner.maximum()
        rng = max(1, imax - imin)
        # map inner -> top
        top_val = int(round(((inner.value() - imin) / rng) * virtual_max))
        self._hscroll_ready = True
        self.top_scrollbar.blockSignals(True)
        self.top_scrollbar.setRange(0, virtual_max)
        self.top_scrollbar.setPageStep(100)
        self.top_scrollbar.setSingleStep(10)
        self.top_scrollbar.setValue(top_val)
        self.top_scrollbar.blockSignals(False)
    except Exception:
        # don't crash UI if something is missing during early init
        self._hscroll_ready = True


def _arm_main_topbar(self, virtual_max: int = 2000):
    """Re-sync the main top scrollbar with the inner QScrollArea hbar and enable mapping."""
    try:
        inner = self.main_web_scroll_area.horizontalScrollBar()
        imin, imax = inner.minimum(), inner.maximum()
        rng = max(1, imax - imin)
        # map inner -> top
        top_val = int(round(((inner.value() - imin) / rng) * virtual_max))
        self._hscroll_ready_main = True
        self.main_top_scrollbar.blockSignals(True)
        self.main_top_scrollbar.setRange(0, virtual_max)
        self.main_top_scrollbar.setPageStep(100)
        self.main_top_scrollbar.setSingleStep(10)
        self.main_top_scrollbar.setValue(top_val)
        self.main_top_scrollbar.blockSignals(False)
    except Exception:
        # don't crash UI if something is missing during early init
        self._hscroll_ready_main = True


def _set_top_mode(self, mode: str):
    """mode: 'dual' for heatmaps, 'single' for line/3D"""
    mode = mode.lower()
    if mode == "dual":
        # show the dual heatmaps page on top
        self.top_stack.setCurrentWidget(self.dual_heatmaps_page)
        self.main_top_scrollbar.hide()
    else:
        # show the single chart page on top
        self.top_stack.setCurrentWidget(self.single_chart_page)
        self.main_top_scrollbar.show()

    # optional: blank out views that aren't visible so you never see stale content
    if self.top_stack.currentWidget() is self.single_chart_page:
        # blank dual views
        try:
            self.web_view_left.setHtml("<html></html>")
            self.web_view_right.setHtml("<html></html>")
        except Exception:
            pass
    else:
        # blank single view
        try:
            self.web_view.setHtml("<html></html>")
        except Exception:
            pass


def _sync_heatmap_range(self, target_view, payload):
    """Synchronize zoom/pan between both heatmaps."""
    if not isinstance(target_view, SyncPlotlyView):
        return

    clean_payload = {}
    if "xaxis.range" in payload:
        clean_payload["xaxis.range"] = payload["xaxis.range"]
    if "yaxis.range" in payload:
        clean_payload["yaxis.range"] = payload["yaxis.range"]

    # Apply to the other view
    target_view.apply_relayout(clean_payload)


def _setup_web_view_scrollbars(self, web_view):
    """Force scrollbars to be visible on QWebEngineView"""
    try:
        # Enable scrollbars at the widget level
        web_view.page().settings().setAttribute(
            web_view.page().settings().WebAttribute.ShowScrollBars, True
        )

        # Inject CSS to force scrollbar visibility
        css = """
        ::-webkit-scrollbar { 
            width: 16px !important; 
            height: 16px !important; 
            display: block !important; 
        }
        ::-webkit-scrollbar-track { 
            background: #f0f0f0 !important; 
        }
        ::-webkit-scrollbar-thumb { 
            background: #888 !important; 
            border-radius: 4px !important; 
        }
        html, body { 
            overflow: scroll !important; 
        }
        """

        web_view.page().runJavaScript(f"""
        var style = document.createElement('style');
        style.textContent = `{css}`;
        document.head.appendChild(style);
        """)
    except Exception as e:
        print(f"Error setting up scrollbars: {e}")