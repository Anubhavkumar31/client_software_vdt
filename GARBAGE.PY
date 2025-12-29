import sys
import pandas as pd
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout,
    QLineEdit, QTableView, QFrame, QHBoxLayout,
    QPushButton, QLabel, QFileDialog, QMessageBox
)
from PyQt6.QtGui import QStandardItemModel, QStandardItem, QFont
from PyQt6.QtCore import Qt, QSortFilterProxyModel


EXCEL_PATH = r"C:\Users\admin\Downloads\Pipe_Tally_8inch (1).xlsx"


# ================= ENTERPRISE TOOL BAR =================

class PipeTallyToolPanel(QFrame):
    def __init__(self, parent):
        super().__init__(parent)
        self.parent = parent
        self.setFixedHeight(56)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(18, 8, 18, 8)
        layout.setSpacing(12)

        title = QLabel("PIPE TALLY")
        title.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        title.setStyleSheet("color:#1e293b;")
        layout.addWidget(title)
        layout.addSpacing(10)

        layout.addWidget(self._vline())

        self.btn_search = self._tool_btn("Search")
        self.btn_export = self._tool_btn("Export Excel")
        self.btn_graph  = self._tool_btn("Graph")
        self.btn_filter = self._tool_btn("Filter")
        self.btn_more   = self._tool_btn("More")

        for b in (self.btn_search, self.btn_export, self.btn_graph, self.btn_filter, self.btn_more):
            layout.addWidget(b)

        layout.addStretch()

        self.setStyleSheet("""
            QFrame {
                background:qlineargradient(x1:0,y1:0,x2:0,y2:1,
                    stop:0 #ffffff, stop:1 #f1f5f9);
                border-bottom:1px solid #cbd5f5;
            }
        """)

        self.btn_export.clicked.connect(self.export_excel)
        self.btn_search.clicked.connect(self.parent.toggle_search_bar)

    def _tool_btn(self, text):
        b = QPushButton(text)
        b.setFixedHeight(34)
        b.setCursor(Qt.CursorShape.PointingHandCursor)
        b.setStyleSheet("""
            QPushButton {
                background:white;
                border:1px solid #cbd5f5;
                border-radius:8px;
                padding:6px 18px;
                color:#1e293b;
                font-weight:600;
            }
            QPushButton:hover { background:#e2e8f0; }
            QPushButton:pressed { background:#c7d2fe; }
        """)
        return b

    def _vline(self):
        l = QFrame()
        l.setFrameShape(QFrame.Shape.VLine)
        l.setStyleSheet("color:#cbd5f5;")
        return l

    def export_excel(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export Pipe Tally", "", "Excel (*.xlsx)")
        if not path:
            return
        self.parent.get_filtered_dataframe().to_excel(path, index=False)
        QMessageBox.information(self, "Exported", "Pipe tally exported successfully!")


# ================= MAIN VIEWER =================

class PipeTallyViewer(QMainWindow):
    def __init__(self, df):
        super().__init__()
        self.df = df

        self.setWindowTitle("Pipe Tally Viewer")
        self.resize(1500, 820)

        central = QWidget()
        self.setCentralWidget(central)

        layout = QVBoxLayout(central)
        layout.setSpacing(0)
        layout.setContentsMargins(0,0,0,0)

        # Toolbar
        self.tool_panel = PipeTallyToolPanel(self)
        layout.addWidget(self.tool_panel)

        # Search (hidden initially)
        self.search = QLineEdit()
        self.search.setPlaceholderText("Search Pipe Tally...")
        self.search.setFixedHeight(38)
        self.search.setStyleSheet("margin:6px 14px;")
        self.search.setVisible(False)
        layout.addWidget(self.search)

        # Table
        self.table = QTableView()
        layout.addWidget(self.table)

        # -------- Build Model --------
        model = QStandardItemModel()
        model.setRowCount(len(df))
        model.setColumnCount(len(df.columns))
        model.setHorizontalHeaderLabels(df.columns.astype(str).tolist())

        for r in range(len(df)):
            for c, col in enumerate(df.columns):
                val = df.iloc[r, c]

                if col == "Pipe Number" and not pd.isna(val):
                    item = QStandardItem()
                    item.setData(int(val), Qt.ItemDataRole.DisplayRole)
                elif pd.isna(val):
                    item = QStandardItem("")
                elif isinstance(val, (int, float)):
                    item = QStandardItem()
                    item.setData(float(val), Qt.ItemDataRole.DisplayRole)
                else:
                    item = QStandardItem(str(val))

                item.setEditable(False)
                model.setItem(r, c, item)

        # -------- Sorting + Filtering --------
        self.proxy = QSortFilterProxyModel()
        self.proxy.setSourceModel(model)
        self.proxy.setSortRole(Qt.ItemDataRole.DisplayRole)
        self.proxy.setFilterCaseSensitivity(Qt.CaseSensitivity.CaseInsensitive)

        self.table.setModel(self.proxy)
        self.table.setSortingEnabled(True)
        self.table.setAlternatingRowColors(True)
        self.table.horizontalHeader().setStretchLastSection(True)

        self.table.sortByColumn(
            df.columns.get_loc("Pipe Number"),
            Qt.SortOrder.AscendingOrder
        )

        self.search.textChanged.connect(self.proxy.setFilterFixedString)
        self.fit_columns_to_headers()

        # -------- Light Theme --------
        self.setStyleSheet("""
            QMainWindow { background:#f8fafc; }
            QTableView {
                background:white;
                gridline-color:#e2e8f0;
                font-size:13px;
                alternate-background-color:#f1f5f9;
                selection-background-color:#2563eb;
                selection-color:white;
            }
            QHeaderView::section {
                background:#e2e8f0;
                padding:7px;
                border:0;
                font-weight:600;
            }
            QLineEdit {
                background:white;
                border:1px solid #cbd5f5;
                border-radius:6px;
                padding:7px;
                font-size:13px;
            }
        """)

    def toggle_search_bar(self):
        self.search.setVisible(not self.search.isVisible())
        if self.search.isVisible():
            self.search.setFocus()

    def fit_columns_to_headers(self):
        fm = self.table.fontMetrics()
        for i in range(self.proxy.columnCount()):
            header_text = self.proxy.headerData(i, Qt.Orientation.Horizontal)
            w = fm.horizontalAdvance(str(header_text)) + 28
            self.table.setColumnWidth(i, w)
            self.table.resizeColumnToContents(i)

    def get_filtered_dataframe(self):
        rows = []
        for r in range(self.proxy.rowCount()):
            row = {}
            for c in range(self.proxy.columnCount()):
                idx = self.proxy.index(r, c)
                row[self.df.columns[c]] = idx.data()
            rows.append(row)
        return pd.DataFrame(rows)


if __name__ == "__main__":
    df = pd.read_excel(EXCEL_PATH)

    app = QApplication(sys.argv)
    win = PipeTallyViewer(df)
    win.show()
    sys.exit(app.exec())
