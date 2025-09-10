# 🍰 PIE_DV_NEW — Modularity & Architecture (Single-file Main Window)

This document explains how the app is structured today and how to extend it safely. It’s written for developers who will maintain or add features.

(THIS IS ONLY FOR WHEN YOU RUN THROUGH MAIN_MODULE.PY)

---

## 🗂 Project layout (current)

```
PIE_DV_NEW/
├─ main.py                       # Entry point (splash → main window)
├─ ARCHITECTURE.md               # This doc
├─ config/
│  ├─ paths.py                   # resource_path(), ICON_* constants, UI paths, helpers
│  ├─ styles.py                  # (optional) central QSS strings (if used)
│  └─ constants.py               # (optional) column names, regex, etc.
├─ ui/
│  ├─ landing.ui                 # splash
│  ├─ main_window.ui             # main window via Qt Designer
│  └─ icons/                     # SVG/PNG assets (arrows, watermark, etc.)
├─ app/
│  ├─ app.py                     # QApplication wrapper (splash + boot)
│  └─ main_window/
│     └─ window.py               # ✅ single robust main window implementation
├─ models/
│  └─ pandas_model.py            # PandasModel → QTableView adapter
├─ workers/
│  └─ pipe_loader.py             # PipeLoaderWorker (QThread)
├─ widgets/
│  ├─ midbar.py                  # MidBarSplitter + handle with tabbar
│  └─ loading.py                 # ModernLoadingDialog
├─ utils/
│  ├─ table.py                   # setup_table_scroll(), misc table helpers
│  └─ assests 
|  |_build_exe_client           #to create the exe ( changes may required )
├─ pages/                        # Dialogs/graphs/report tabs (About, ERF, Reports, etc.)
├─ backend/                      # Plot generation, html writers, etc.
└─ manual/                       # User manual
```

---

## 🚦 Runtime flow (high level)

1. **`main.py`**  
   - Sets env flags (QtWebEngine), configures Matplotlib backend, starts `MainApp`.
2. **`app/app.py` → `MainApp`**  
   - Shows splash (`ui/landing.ui`), then instantiates `MyMainWindow` (from `window.py`).
3. **`app/main_window/window.py` → `MyMainWindow`**  
   - Loads `ui/main_window.ui`.  
   - Wires actions, builds the mid splitter, overlays, and custom scrollbars.  
   - Project lifecycle (open/close), loads PKL + finds per-pipe HTML/assets via **`PipeLoaderWorker`**.  
   - Populates tables, syncs charts (QWebEngineView), guards tabs, digsheet logic.

---

## 🧠 Anatomy of `window.py` (what lives where)

> Tip: keep the file navigable by using obvious region comments like `# --- Project lifecycle ---`, `# --- Assets & tabs ---`, etc.

1. **Construction & UI wiring**
   - `self.ui = Form(); self.ui.setupUi(self)`
   - Hide disabled menu items, style menu bar, remove left toolbars.
   - Create **Load** / **Digsheet** buttons next to `comboBoxPipe`.
   - Install global `eventFilter` for:  
     - project gate popups (no project open),  
     - “digsheet” hint when disabled,  
     - mid-tabbar click interception.

2. **Splitter + custom top scrollbars**
   - `MidBarSplitter` (from `widgets/midbar.py`) contains:
     - Top: `QWebEngineView` (main chart) with **`mainTopBar`** (custom h scrollbar).
     - Bottom (stack):
       - Defect table page (`QTableWidget`) with **`tableTopBar`**,
       - Data table page (`QTableView`),
       - Proximity line chart (`QWebEngineView`) with **`proxTopBar`**.
   - Sync helpers map each top bar to its inner scroll area; keep right margin so scroll thumbs don’t overshoot.

3. **Overlays & empty states**
   - “Create Project” overlay (centered card).
   - “No Pipe Selected” overlay (when project open but no pipe chosen).
   - “No Defects Found” card (when the table filter leaves nothing).
   - Show/hide logic in `_show_*` helpers.

4. **Project lifecycle**
   - `open_project()` → choose folder, auto-load **pipe tally** from `pipetally_main`, discover PKLs in `pickle_data`, populate `comboBoxPipe`.  
   - `_update_project_actions()` enables/disables menus based on project/tally/report availability.
   - `close_project()` → tear down data/UI safely, restore overlays, reset tabs without firing guarded logic.

5. **Async loading (threads)**
   - `load_selected_pipe()` → kicks off `PipeLoaderWorker(pkl, project_root, pipe_idx)`.
   - Worker emits:
     - `progress_updated(int, str)` → updates `ModernLoadingDialog`,
     - `data_loaded(df)` → binds `PandasModel` to `tableView`,
     - `assets_loaded(dict)` → stores paths to heatmap/line/3D/prox HTML,
     - `table_data_ready(df)` → fills `tableWidgetDefect` (batched, non-blocking),
     - `error_occurred(str)` → modal error,
     - `finished()` → close dialog (guarded if batch fill not done), arm topbars, refresh views.

   ⚠️ **Rule:** worker thread must not touch Qt widgets directly.

6. **Assets & tabs**
   - `tab_switcher2()` renders the current tab’s HTML into the top `QWebEngineView`, possibly shows the proximity chart in the bottom.  
   - `_load_scrollable_chart()` wraps the file in an iframe + forces visible scrollbars.
   - `_arm_topbar()` / `_arm_main_topbar()` sync scrollbar ranges after content loads.

7. **Defect tables**
   - **Tally path**: `_populate_defect_table_from_tally()` filters to “Metal Loss”, normalizes columns, rounds numeric columns, and fills `QTableWidget` in batches.  
   - **CSV path**: `_populate_defect_table_from_csv()` maps candidate column names to the canonical table header.
   - `setup_table_scroll()` (from `utils/table.py`) applies smooth scrolling.

8. **Digsheet (ABS-based)**
   - Enabled only when: project open, in graph tabs (Heatmap/3D), and exactly one row with a valid **Abs. Distance** is selected.
   - `_get_selected_abs_distance_from_defect_table()` reads the value, `_dump_tally_to_temp()` passes tally to the digsheet script (works both dev & frozen).

9. **Guards, actions & events**
   - `_guarded_open_tab()` and `_on_middle_tab_changed()` prevent tab switches when no project is open (show a friendly popup).  
   - `eventFilter()` shows “project required” and “digsheet hint” popups contextually.  
   - `setup_actions()` connects menu entries to dialogs (About/Admin/ERF/Reports/Highlights/Schema/etc.).

10. **Status bar & timer**
    - `set_loading()/set_idle()` update the right-side elapsed time.
    - `_tick()` updates every 100ms while loading.

---

## 📦 Paths, resources & QSS (avoid common traps)

- Always resolve files via **`config.paths.resource_path()`** so dev & PyInstaller builds both work.
- For **QSS images**, pass a **file URL** (not a path) or a normalized forward-slash path.

**Safe pattern (recommended):**
```python
from PyQt6.QtCore import QUrl
from config.paths import ICON_ARROW_DOWN

arrow_url = QUrl.fromLocalFile(ICON_ARROW_DOWN).toString()  # e.g. "file:///C:/.../arrow_down.svg"
self.ui.comboBoxPipe.setStyleSheet(f"""
    QComboBox::down-arrow {{
        image: url("{arrow_url}");
        width: 12px; height: 12px;
    }}
""")
```

**Do not** prepend `file:` yourself if `arrow_url` already includes it (that causes
`.../file:/C:/...` errors). Also avoid mixing backslashes—Qt is happier with `/`.

- If you must use a plain path, normalize first:
  ```python
  path = ICON_ARROW_DOWN.replace("\\", "/")
  # then image: url("{path}")
  ```

---

## 🔗 Signals & threading (quick map)

- `PipeLoaderWorker` emits:
  - `progress_updated(int, str)`
  - `time_estimate(float)` *(display only; don’t rely on it for control flow)*
  - `data_loaded(pd.DataFrame)`
  - `assets_loaded(dict)`
  - `table_data_ready(pd.DataFrame|None)`
  - `error_occurred(str)`
  - `finished()`

- **Never** touch widgets in worker thread. UI updates belong in `on_*` slots in `window.py`.

---

## ✅ Coding guidelines (while staying single-file)

- Group related methods with big, obvious region comments. Example:

  ```python
  # ===============================
  #   Project lifecycle (open/close)
  # ===============================
  ```

- Keep helper methods **private** (`_name`) and short. If a method grows >150 lines, extract sub-helpers (still in the same file).
- Prefer **pure helpers** (no widget access) for data massaging.
- Any action that can block must:
  - run in a worker or
  - use batched UI updates (see `_start_fill_qtablewidget_batched`).

---

## 🧪 Local dev & quick tests

- Quick boot: run `python main.py`.  
- If a test harness (`test_main.py`) instantiates the window, **import `MyMainWindow` from `app.main_window.window`**, not from a copy.  
  - Common gotcha: editing `MyMainWindow` in a *different* test file and expecting the app to use it — it won’t.

---

## 🧰 Troubleshooting (greatest hits)

- **Combo arrow not showing**  
  Use `QUrl.fromLocalFile(ICON_ARROW_DOWN).toString()` and **don’t** add another `file:`.  
  Confirm with a debug print:  
  `DEBUG arrow_url: file:///F:/.../ui/icons/arrow_down.svg`

- **QSS “Could not parse stylesheet”**  
  Look for unescaped quotes or Windows backslashes. Normalize with `.replace("\\", "/")`.

- **Scrollbars “wrap” or jump**  
  Ensure `_arm_topbar()` / `_arm_main_topbar()` are called **after** the widget is visible and the inner scrollbar got a valid `rangeChanged`. There are `QTimer.singleShot(...)` nudges in place—keep them.

- **No defects state**  
  The hidden overlay must remain hidden while the defect table is visible. Use `_show_defects_table()` and `_show_no_defects_message()` consistently.

---

## 📄 Glossary

- **Tally**: The master pipe tally dataframe (from `/pipetally_main`).
- **PKL**: Per-pipe pickle with telemetry/defect data.
- **Assets**: Prebuilt HTML charts (heatmap/line/3D/proximity).
- **Graph tabs**: “Heatmap”, “LineChart”, “3D”.
