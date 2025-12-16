from PyQt6 import QtWidgets
from PyQt6.QtWidgets import QTabBar

from main_section_view.build_main_section import _build_main_section


def setup_tab_system(self):
    """
    -------------------------------------------------------------
    CUSTOM TAB SYSTEM (MIDDLE TABBAR + SPLITTER)
    -------------------------------------------------------------
    Replaces the default QTabWidget tabs with a custom QTabBar.
    Features:
      • Syncs QTabBar ↔ TabWidget ↔ Dropdown
      • Applies guarded switching (blocks invalid tab jumps)
      • Installs event filters for control of clicks
      • Hides original tab bar for a clean UI
      • Builds the main splitter layout that holds
        the graph area + table area + tab area.
    -------------------------------------------------------------
    """
    QtWidgets.QApplication.instance().installEventFilter(self)

    self.mid_tabbar = QTabBar()
    self.mid_tabbar.setExpanding(False)

    for i in range(self.ui.tabWidgetM.count()):
        self.mid_tabbar.addTab(self.ui.tabWidgetM.tabText(i))

    # Sync in both directions
    self.mid_tabbar.currentChanged.connect(
        lambda i: [self.ui.tabWidgetM.setCurrentIndex(i),
                   _sync_dropdown_with_tabs(self, i)][0]
    )
    self.ui.tabWidgetM.currentChanged.connect(
        lambda i: [self.mid_tabbar.setCurrentIndex(i),
                   _sync_dropdown_with_tabs(self, i)][0]
    )

    self.mid_tabbar.installEventFilter(self)
    self.ui.tabWidgetM.hide()

    # self._build_splitter()
    # _build_main_section(self)


def _sync_dropdown_with_tabs(self, index: int):
    """Sync dropdown when tab changes from other sources"""
    try:
        # Block signals to prevent infinite loop
        self.tabSwitcherDropdown.blockSignals(True)

        # Update dropdown selection
        self.tabSwitcherDropdown.setCurrentIndex(index)

        # Unblock signals
        self.tabSwitcherDropdown.blockSignals(False)

    except Exception as e:
        print(f"Error syncing dropdown: {e}")