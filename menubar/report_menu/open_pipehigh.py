import pandas as pd
from PyQt6.QtCore import QSize, Qt
from PyQt6.QtGui import QIcon
from PyQt6.QtWidgets import QMessageBox, QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel


def open_PipeHigh(self):
    """Open Pipeline Highlights embedded in the main window"""
    try:
        # Check if pipe_tally is loaded
        if not hasattr(self, 'pipe_tally') or not isinstance(self.pipe_tally,
                                                             pd.DataFrame) or self.pipe_tally.empty:
            QMessageBox.warning(
                self,
                "No Pipe Tally Data",
                "Please load a project with pipe tally data first.\n\n"
                "Steps to load data:\n"
                "1. Go to File → Create Project\n"
                "2. Select a folder containing pipe tally files\n"
                "3. Wait for the data to load\n"
                "4. Try opening Pipe Highlights again"
            )
            return

        # Check if Pipeline Highlights is already open
        if hasattr(self, '_central_pipeline') and self.centralWidget() is self._central_pipeline:
            return  # Already showing Pipeline Highlights

        # Save the original central widget
        if not hasattr(self, '_central_original') or self._central_original is None:
            self._central_original = self.centralWidget()

        print(f"🔍 Opening Pipeline Highlights with {len(self.pipe_tally)} rows of data")
        print(f"📊 Available columns: {list(self.pipe_tally.columns)}")

        # Import the embedded version
        from pages.Pipe_Highlights_Embedded import PipeHighlightEmbedded

        # Create container widget
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setContentsMargins(12, 12, 12, 12)
        layout.setSpacing(10)

        # Header with back button
        header_layout = QHBoxLayout()
        back_btn = QPushButton("Back")
        back_btn.setIcon(QIcon("ui/icons/arrow_left.svg"))  # replace with your arrow icon path
        back_btn.setIconSize(QSize(16, 16))
        back_btn.setCursor(Qt.CursorShape.PointingHandCursor)

        back_btn.setStyleSheet("""
               QPushButton {
                   background-color: #ffffff;
                   color: #000000;
                   border: 1.5px solid #000000;
                   border-radius: 8px;
                   padding: 5px 14px;
                   font-size: 13px;
                   font-weight: 500;
               }
               QPushButton:hover {
                   background-color: #f2f2f2;
               }
               QPushButton:pressed {
                   background-color: #e0e0e0;
               }
               QPushButton:disabled {
                   background-color: #f9f9f9;
                   color: #aaaaaa;
                   border: 1.5px solid #cccccc;
               }
           """)
        back_btn.clicked.connect(lambda : _close_pipeline_view(self))

        title_label = QLabel("")
        title_label.setStyleSheet("font-weight: 600; font-size: 16pt; color: #2c3e50;")

        header_layout.addWidget(back_btn)
        header_layout.addSpacing(20)
        header_layout.addWidget(title_label)
        header_layout.addStretch(1)

        layout.addLayout(header_layout)

        # Create and add the Pipeline Highlights widget
        self._pipeline_widget = PipeHighlightEmbedded(parent=container, pipe_tally_df=self.pipe_tally,
                                                      project_root=self.project_root)
        layout.addWidget(self._pipeline_widget, stretch=1)

        # Store reference and switch central widget
        self._central_pipeline = container

        # Switch to Pipeline Highlights view
        if self._central_original is not None and self._central_original.parent() is self:
            self.takeCentralWidget()
        self.setCentralWidget(container)

        print("✅ Pipeline Highlights opened successfully in embedded mode")

    except ImportError as e:
        self.open_Error(
            f"Could not import Pipeline Highlights module:\n{e}\n\nPlease check if the Pipe_Highlights_Embedded.py file exists in the pages folder.")
    except Exception as e:
        self.open_Error(f"Error running Pipeline Highlights:\n{e}")
        # Restore original view on error
        try:
            if hasattr(self, '_central_original') and self._central_original is not None:
                if self.centralWidget() is not self._central_original:
                    self.setCentralWidget(self._central_original)
        except Exception:
            pass


def _close_pipeline_view(self):
    """Close Pipeline Highlights and return to main view"""
    try:
        if self.centralWidget() is getattr(self, '_central_original', None):
            return  # Already showing original view

        # Take current widget and delete it
        pipeline_central = self.takeCentralWidget()
        if pipeline_central is not None:
            pipeline_central.deleteLater()

        # Restore original central widget
        if hasattr(self, '_central_original') and self._central_original is not None:
            if self._central_original.parent() is not self:
                self._central_original.setParent(self)
            self.setCentralWidget(self._central_original)

        # Clean up references
        if hasattr(self, '_pipeline_widget'):
            self._pipeline_widget = None
        if hasattr(self, '_central_pipeline'):
            self._central_pipeline = None

        print("✅ Returned to main view from Pipeline Highlights")

    except Exception as e:
        print(f"⚠️ Error closing Pipeline Highlights view: {e}")