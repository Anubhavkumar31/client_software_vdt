import sys
import os
import pandas as pd
import plotly.io as pio
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel,
    QComboBox, QSizePolicy, QMessageBox,QHBoxLayout
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import Qt, QUrl
from pages.graphs_func import plot_erf, plot_psafe, plot_depth, plot_orientation
from pages.Defects import (plot_metal_loss, plot_sensor_percentage, plot_temperature)# Metal Loss and Sensor Loss Plot functions

pio.kaleido.scope.default_format = "png"


class GraphApp(QWidget):
    def __init__(self, dataframe=None):
        super().__init__()
        
        self.df = dataframe

        # 🛠️ Create UI elements...
        self.setWindowTitle("Pipeline Defect Graph Viewer")
        self.setGeometry(200, 100, 1100, 800)

        self.layout = QVBoxLayout()
        self.button_layout = QHBoxLayout()

        self.file_label = QLabel("No file selected")
        self.layout.addWidget(self.file_label)

        # (... all your widget setup continues here ...)

        self.setLayout(self.layout)


        # Graph Type Dropdown
        self.graph_type_label = QLabel("Select Graph Type:")
        self.graph_type_label.setVisible(False)
        self.button_layout.addWidget(self.graph_type_label)

        self.graph_type = QComboBox()
        self.graph_type.setFixedWidth(150)
        self.graph_type.addItems(["", "Defects", "ERF", "Psafe", "Depth", "Orientation", "Temperature", "Sensor Loss"])
        self.graph_type.setVisible(False)
        self.graph_type.currentTextChanged.connect(self.on_graph_type_changed)
        self.button_layout.addWidget(self.graph_type)

        # Feature Identification
        self.feature_identification_label = QLabel("Select Feature Identification:")
        self.feature_identification_label.setVisible(False)
        self.button_layout.addWidget(self.feature_identification_label)

        self.feature_identification = QComboBox()
        self.feature_identification.setFixedWidth(150)
        self.feature_identification.addItems(["", "Corrosion", "MFG", "Both(Corrosion,MFG)"])
        self.feature_identification.setVisible(False)
        self.button_layout.addWidget(self.feature_identification)

        # Dimensional Classification
        self.dimension_classification_label = QLabel("Select Dimensional Classification:")
        self.dimension_classification_label.setVisible(False)
        self.button_layout.addWidget(self.dimension_classification_label)

        self.dimension_classification = QComboBox()
        self.dimension_classification.setFixedWidth(150)
        self.dimension_classification.addItems([
            "", "Pitting", "Axial Grooving", "Axial Slotting",
            "Circumferential Grooving", "Circumferential Slotting",
            "Pinhole", "General"
        ])
        self.dimension_classification.setVisible(False)
        self.button_layout.addWidget(self.dimension_classification)

        # Surface View
        self.view_type_label = QLabel("Select Surface View:")
        self.view_type_label.setVisible(False)
        self.button_layout.addWidget(self.view_type_label)

        self.feature_identification.currentTextChanged.connect(self.on_feature_identification_changed)

        self.view_type = QComboBox()
        self.view_type.setFixedWidth(150)
        self.view_type.addItems(["", "Internal", "External", "Both"])
        self.view_type.setVisible(False)
        self.button_layout.addWidget(self.view_type)

        # Plot Button
        self.plot_btn = QPushButton("Plot Graph")
        self.plot_btn.setFixedWidth(150)
        self.plot_btn.setVisible(False)
        self.plot_btn.clicked.connect(self.plot_graph)
        self.button_layout.addWidget(self.plot_btn)

        # Add some stretch for spacing (optional)
        self.button_layout.addStretch()

        # Add horizontal layout now — before graph view
        self.layout.addLayout(self.button_layout)

        # Save Button (optional: move it to horizontal row if desired)
        self.save_btn = QPushButton("Save Graph as PNG")
        self.save_btn.setStyleSheet("""
            QPushButton {
                background-color: #444;
                color: white;
                padding: 6px 15px;
                font-weight: bold;
                border-radius: 5px;
            }
            QPushButton:hover {
                background-color: #666;
            }
        """)
        self.save_btn.setFixedWidth(200)
        self.save_btn.setVisible(False)
        self.save_btn.clicked.connect(self.save_graph)
        self.layout.addWidget(self.save_btn)

        # Graph display
        self.browser = QWebEngineView()
        # self.browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        self.layout.addWidget(self.browser)

        # Set final layout
        self.setLayout(self.layout)

        if self.df is not None:
            self.file_label.setText("Pipe tally loaded from main app")
            self.graph_type_label.setVisible(True)
            self.graph_type.setVisible(True)
            self.plot_btn.setVisible(True)


    def on_graph_type_changed(self, text):
        # Clear previous selections when a new graph type is selected
        self.view_type.setCurrentIndex(0)  # Reset the surface view
        self.feature_identification.setCurrentIndex(0)  # Reset the feature identification
        self.dimension_classification.setCurrentIndex(0)  # Reset the dimension classification

        if text == "Defects":
            # When Metal Loss is selected, show feature identification and dimensional classification dropdowns
            self.feature_identification_label.setVisible(True)
            self.feature_identification.setVisible(True)
            self.dimension_classification_label.setVisible(True)
            self.dimension_classification.setVisible(True)
            self.view_type_label.setVisible(False)
            self.view_type.setVisible(False)
        elif text in ["ERF", "Psafe", "Orientation"]:
            # When any of these are selected, show the surface view dropdown
            self.feature_identification_label.setVisible(False)
            self.feature_identification.setVisible(False)
            self.dimension_classification_label.setVisible(False)
            self.dimension_classification.setVisible(False)
            self.view_type_label.setVisible(True)
            self.view_type.setVisible(True)
        elif text in ["Depth"]:
            self.feature_identification_label.setVisible(False)
            self.feature_identification.setVisible(False)
            self.dimension_classification_label.setVisible(False)
            self.dimension_classification.setVisible(False)
            self.view_type_label.setVisible(False)
            self.view_type.setVisible(False)

        elif text == "Sensor Loss":
            # When Sensor Loss is selected, hide the feature and dimension dropdowns and the surface view dropdown
            self.feature_identification_label.setVisible(False)
            self.feature_identification.setVisible(False)
            self.dimension_classification_label.setVisible(False)
            self.dimension_classification.setVisible(False)
            self.view_type_label.setVisible(False)
            self.view_type.setVisible(False)
        elif text == "Temperature":
            # When Temperature is selected, hide the feature and dimension dropdowns and the surface view dropdown
            self.feature_identification_label.setVisible(False)
            self.feature_identification.setVisible(False)
            self.dimension_classification_label.setVisible(False)
            self.dimension_classification.setVisible(False)
            self.view_type_label.setVisible(False)
            self.view_type.setVisible(False)
        else:
            # If no valid graph type is selected, hide all dropdowns and their titles
            self.feature_identification_label.setVisible(False)
            self.feature_identification.setVisible(False)
            self.dimension_classification_label.setVisible(False)
            self.dimension_classification.setVisible(False)
            self.view_type_label.setVisible(False)
            self.view_type.setVisible(False)

    def on_feature_identification_changed(self, text):
        # Reset the Dimensional Classification dropdown whenever Feature Identification changes
        self.dimension_classification.setCurrentIndex(0)  # Reset to the first option (empty)

        # No need to disable Dimensional Classification, keep it always enabled
        self.dimension_classification.setEnabled(True)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Excel File", "", "Excel Files (*.xlsx *.xls)")
        if path:
            self.df = pd.read_excel(path)
            self.df.columns = self.df.columns.str.strip()
            self.file_label.setText(f"Loaded: {os.path.basename(path)}")

            # After the file is loaded, show the Graph Type dropdown and its title
            self.graph_type_label.setVisible(True)
            self.graph_type.setVisible(True)
            self.plot_btn.setVisible(True)

    def plot_graph(self):
        try:
            graph_type = self.graph_type.currentText()
            feature = self.feature_identification.currentText()
            dimension = self.dimension_classification.currentText()
            view = self.view_type.currentText()

            if not graph_type:
                msg = QMessageBox()
                msg.setIcon(QMessageBox.Warning)
                msg.setWindowTitle("Graph Type Not Selected")
                msg.setText("Please select the graph type before plotting.")
                msg.setStandardButtons(QMessageBox.Ok)
                msg.exec()
                return

            if graph_type == "Defects":
                if not feature and dimension == "":
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Feature Identification or Dimensional Classification Not Selected")
                    msg.setText("Please select either Feature Identification or Dimensional Classification.")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec()
                    return

                feature_id = feature if feature else None
                dimension_class = dimension if dimension != "Both" else None

                # Plot Metal Loss
                fig, path = plot_metal_loss(self.df.copy(), feature_type=feature_id, dimension_class=dimension_class, return_fig=True)
                self.current_fig = fig
                self.browser.load(QUrl.fromLocalFile(path))
                self.save_btn.setVisible(True)

            elif graph_type in ["ERF", "Psafe", "Orientation"]:
                if not view:
                    msg = QMessageBox()
                    msg.setIcon(QMessageBox.Warning)
                    msg.setWindowTitle("Surface View Not Selected")
                    msg.setText("Please select Surface View before plotting.")
                    msg.setStandardButtons(QMessageBox.Ok)
                    msg.exec()
                    return

                if graph_type == "ERF":
                    fig, path = plot_erf(self.df.copy(), view, return_fig=True)
                elif graph_type == "Psafe":
                    fig, path = plot_psafe(self.df.copy(), view, return_fig=True)
                elif graph_type == "Orientation":
                    fig, path = plot_orientation(self.df.copy(), view, return_fig=True)

                self.current_fig = fig
                self.browser.load(QUrl.fromLocalFile(path))
                self.save_btn.setVisible(True)
            elif graph_type == "Depth":
                fig, path = plot_depth(self.df.copy(), view, return_fig=True)
                self.current_fig = fig
                self.browser.load(QUrl.fromLocalFile(path))
                self.save_btn.setVisible(True)

            elif graph_type == "Sensor Loss":
                # Plot Sensor Loss
                fig, path = plot_sensor_percentage(self.df.copy(), return_fig=True)  # Fix here by passing only return_fig
                self.current_fig = fig
                self.browser.load(QUrl.fromLocalFile(path))
                self.save_btn.setVisible(True)

            elif graph_type == "Temperature":
                # Plot Sensor Loss
                fig, path = plot_temperature(self.df.copy(), return_fig=True)  # Fix here by passing only return_fig
                self.current_fig = fig
                self.browser.load(QUrl.fromLocalFile(path))
                self.save_btn.setVisible(True)

            else:
                self.file_label.setText("Please select a graph type.")

        except Exception as e:
            self.file_label.setText(f"Plot failed: {str(e)}")
            self.current_fig = None

    def save_graph(self):
        if hasattr(self, 'current_fig') and self.current_fig is not None:
            downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
            default_file = os.path.join(downloads_folder, "graph.png")

            file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot as PNG", default_file, "PNG Files (*.png);;All Files (*)")
            if file_path:
                try:
                    if not file_path.lower().endswith(".png"):
                        file_path += ".png"

                    self.current_fig.write_image(file_path)
                    self.file_label.setText(f"Graph saved as: {file_path}")
                except Exception as e:
                    self.file_label.setText(f"Failed to save graph: {str(e)}")
            else:
                self.file_label.setText("No file path selected.")
        else:
            self.file_label.setText("No graph to save.")


if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = GraphApp()
    viewer.show()
    sys.exit(app.exec())
