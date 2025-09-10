# # models/pandas_model.py
import pandas as pd
from PyQt6.QtCore import Qt, QAbstractTableModel, QModelIndex, QVariant

# class PandasModel(QAbstractTableModel):
#     """
#     A model to interface a Qt view with a pandas DataFrame.
#     """

#     def __init__(self, df: pd.DataFrame = pd.DataFrame(), parent=None):
#         super().__init__(parent)
#         self._df = df.copy()

#     # Required Qt methods
#     def rowCount(self, parent=QModelIndex()):
#         return 0 if parent.isValid() else len(self._df)

#     def columnCount(self, parent=QModelIndex()):
#         return 0 if parent.isValid() else len(self._df.columns)

#     def data(self, index, role=Qt.ItemDataRole.DisplayRole):
#         if not index.isValid():
#             return QVariant()
#         if role == Qt.ItemDataRole.DisplayRole:
#             value = self._df.iloc[index.row(), index.column()]
#             return "" if pd.isna(value) else str(value)
#         return QVariant()

#     # Header labels
#     def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
#         if role != Qt.ItemDataRole.DisplayRole:
#             return QVariant()
#         if orientation == Qt.Orientation.Horizontal:
#             try:
#                 return str(self._df.columns[section])
#             except IndexError:
#                 return QVariant()
#         else:
#             return str(section + 1)

#     # Helpers
#     def setDataFrame(self, df: pd.DataFrame):
#         self.beginResetModel()
#         self._df = df.copy()
#         self.endResetModel()

#     def dataFrame(self) -> pd.DataFrame:
#         return self._df



# --- Lightweight DataFrame model (no per-cell Qt items) ---
from PyQt6.QtCore import QAbstractTableModel, QVariant

class PandasModel(QAbstractTableModel):
    def __init__(self, df: pd.DataFrame, parent=None):
        super().__init__(parent)
        self._df = df

    def rowCount(self, _parent=None):
        return 0 if self._df is None else len(self._df)

    def columnCount(self, _parent=None):
        return 0 if self._df is None else self._df.shape[1]

    def data(self, index, role=Qt.ItemDataRole.DisplayRole):
        if not index.isValid():
            return QVariant()

        if role == Qt.ItemDataRole.DisplayRole:
            val = self._df.iat[index.row(), index.column()]
            if pd.isna(val):
                return ""
            # cheap formatting for floats
            if isinstance(val, float):
                return f"{val:.6g}"
            return str(val)

        return QVariant()

    def headerData(self, section, orientation, role=Qt.ItemDataRole.DisplayRole):
        if role == Qt.ItemDataRole.DisplayRole:
            if orientation == Qt.Orientation.Horizontal:
                return str(self._df.columns[section])
            return str(section + 1)
        elif role == Qt.ItemDataRole.FontRole:
            # Make headers bold
            from PyQt6.QtGui import QFont
            font = QFont()
            font.setBold(True)
            return font
        elif role == Qt.ItemDataRole.TextAlignmentRole:
            return Qt.AlignmentFlag.AlignCenter

        return QVariant()

    def flags(self, index):
        """Make all items non-editable"""
        if not index.isValid():
            return Qt.ItemFlag.NoItemFlags
        return Qt.ItemFlag.ItemIsEnabled | Qt.ItemFlag.ItemIsSelectable