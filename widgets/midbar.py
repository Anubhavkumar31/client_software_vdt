

# widgets/midbar.py
from PyQt6.QtWidgets import QSplitter, QSplitterHandle, QFrame, QTabBar, QHBoxLayout as _QHBoxLayout
from PyQt6.QtCore import Qt
from config.constants import HANDLE_RADIUS
from typing import Optional
from PyQt6.QtCore import QEvent
from PyQt6.QtGui import QCursor

class MidBarHandle(QSplitterHandle):
    def __init__(self, orientation, parent, tabbar: QTabBar):
        super().__init__(orientation, parent)
        self.setObjectName("MidBarHandle")
        self.setCursor(Qt.CursorShape.SplitVCursor)

        self.frame = QFrame(self)
        self.frame.setObjectName("MidBarFrame")
        self.frame.setFrameShape(QFrame.Shape.NoFrame)
        self.frame.setCursor(Qt.CursorShape.SplitVCursor)

        self.tabbar = tabbar
        self.tabbar.setParent(self.frame)
        self.tabbar.setDrawBase(False)
        self.tabbar.setCursor(Qt.CursorShape.ArrowCursor)

        self.tabbar.setMouseTracking(True)
        self.tabbar.setAttribute(Qt.WidgetAttribute.WA_Hover, True)

        lay = _QHBoxLayout(self.frame)
        lay.setContentsMargins(8, 4, 8, 4)
        lay.addWidget(self.tabbar)

        self.tabbar.installEventFilter(self)

    def resizeEvent(self, ev):
        super().resizeEvent(ev)
        self.frame.setGeometry(0, 0, self.width(), self.height())

    def eventFilter(self, obj, ev):
        if obj is self.tabbar:
            t = ev.type()
            p = None
            if t in (QEvent.Type.MouseMove, QEvent.Type.HoverMove):
                if hasattr(ev, "position"):
                    p = ev.position().toPoint()
                elif hasattr(ev, "pos"):
                    p = ev.pos()
            elif t in (QEvent.Type.Enter, QEvent.Type.HoverEnter):
                p = self.tabbar.mapFromGlobal(QCursor.pos())
            elif t in (QEvent.Type.Leave, QEvent.Type.HoverLeave):
                self.tabbar.setCursor(Qt.CursorShape.ArrowCursor)
                return False

            if p is not None:
                idx = self.tabbar.tabAt(p)
                if idx != -1 and self.tabbar.isTabEnabled(idx):
                    self.tabbar.setCursor(Qt.CursorShape.PointingHandCursor)
                else:
                    self.tabbar.setCursor(Qt.CursorShape.ArrowCursor)
            return False

        return QSplitterHandle.eventFilter(self, obj, ev)


class MidBarSplitter(QSplitter):
    def __init__(self, parent=None, tabbar: Optional[QTabBar] = None):
        super().__init__(Qt.Orientation.Vertical, parent)
        self._tabbar = tabbar

    def createHandle(self):
        return MidBarHandle(self.orientation(), self, self._tabbar)
