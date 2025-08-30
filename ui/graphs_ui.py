# # import sys
# # import os
# # import pandas as pd
# # import plotly.io as pio
# # from PyQt6.QtWidgets import (
# #     QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel,
# #     QComboBox, QSizePolicy, QMessageBox,QHBoxLayout
# # )
# # from PyQt6.QtWebEngineWidgets import QWebEngineView
# # from PyQt6.QtCore import Qt, QUrl
# # from pages.graphs_func import plot_erf, plot_psafe, plot_depth, plot_orientation
# # from pages.Defects import (plot_metal_loss, plot_sensor_percentage, plot_temperature)# Metal Loss and Sensor Loss Plot functions

# # pio.kaleido.scope.default_format = "png"


# # class GraphApp(QWidget):
# #     def __init__(self, dataframe=None):
# #         super().__init__()
        
# #         self.df = dataframe

# #         # 🛠️ Create UI elements...
# #         self.setWindowTitle("Pipeline Defect Graph Viewer")
# #         self.setGeometry(200, 100, 1100, 800)

# #         self.layout = QVBoxLayout()
# #         self.button_layout = QHBoxLayout()

# #         self.file_label = QLabel("No file selected")
# #         self.layout.addWidget(self.file_label)

# #         # (... all your widget setup continues here ...)

# #         self.setLayout(self.layout)


# #         # Graph Type Dropdown
# #         self.graph_type_label = QLabel("Select Graph Type:")
# #         self.graph_type_label.setVisible(False)
# #         self.button_layout.addWidget(self.graph_type_label)

# #         self.graph_type = QComboBox()
# #         self.graph_type.setFixedWidth(150)
# #         self.graph_type.addItems(["", "Defects", "ERF", "Psafe", "Depth", "Orientation", "Temperature", "Sensor Loss"])
# #         self.graph_type.setVisible(False)
# #         self.graph_type.currentTextChanged.connect(self.on_graph_type_changed)
# #         self.button_layout.addWidget(self.graph_type)

# #         # Feature Identification
# #         self.feature_identification_label = QLabel("Select Feature Identification:")
# #         self.feature_identification_label.setVisible(False)
# #         self.button_layout.addWidget(self.feature_identification_label)

# #         self.feature_identification = QComboBox()
# #         self.feature_identification.setFixedWidth(150)
# #         self.feature_identification.addItems(["", "Corrosion", "MFG", "Both(Corrosion,MFG)"])
# #         self.feature_identification.setVisible(False)
# #         self.button_layout.addWidget(self.feature_identification)

# #         # Dimensional Classification
# #         self.dimension_classification_label = QLabel("Select Dimensional Classification:")
# #         self.dimension_classification_label.setVisible(False)
# #         self.button_layout.addWidget(self.dimension_classification_label)

# #         self.dimension_classification = QComboBox()
# #         self.dimension_classification.setFixedWidth(150)
# #         self.dimension_classification.addItems([
# #             "", "Pitting", "Axial Grooving", "Axial Slotting",
# #             "Circumferential Grooving", "Circumferential Slotting",
# #             "Pinhole", "General"
# #         ])
# #         self.dimension_classification.setVisible(False)
# #         self.button_layout.addWidget(self.dimension_classification)

# #         # Surface View
# #         self.view_type_label = QLabel("Select Surface View:")
# #         self.view_type_label.setVisible(False)
# #         self.button_layout.addWidget(self.view_type_label)

# #         self.feature_identification.currentTextChanged.connect(self.on_feature_identification_changed)

# #         self.view_type = QComboBox()
# #         self.view_type.setFixedWidth(150)
# #         self.view_type.addItems(["", "Internal", "External", "Both"])
# #         self.view_type.setVisible(False)
# #         self.button_layout.addWidget(self.view_type)

# #         # Plot Button
# #         self.plot_btn = QPushButton("Plot Graph")
# #         self.plot_btn.setFixedWidth(150)
# #         self.plot_btn.setVisible(False)
# #         self.plot_btn.clicked.connect(self.plot_graph)
# #         self.button_layout.addWidget(self.plot_btn)

# #         # Add some stretch for spacing (optional)
# #         self.button_layout.addStretch()

# #         # Add horizontal layout now — before graph view
# #         self.layout.addLayout(self.button_layout)

# #         # Save Button (optional: move it to horizontal row if desired)
# #         self.save_btn = QPushButton("Save Graph as PNG")
# #         self.save_btn.setStyleSheet("""
# #             QPushButton {
# #                 background-color: #444;
# #                 color: white;
# #                 padding: 6px 15px;
# #                 font-weight: bold;
# #                 border-radius: 5px;
# #             }
# #             QPushButton:hover {
# #                 background-color: #666;
# #             }
# #         """)
# #         self.save_btn.setFixedWidth(200)
# #         self.save_btn.setVisible(False)
# #         self.save_btn.clicked.connect(self.save_graph)
# #         self.layout.addWidget(self.save_btn)

# #         # Graph display
# #         self.browser = QWebEngineView()
# #         # self.browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
# #         self.browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

# #         self.layout.addWidget(self.browser)

# #         # Set final layout
# #         self.setLayout(self.layout)

# #         if self.df is not None:
# #             self.file_label.setText("Pipe tally loaded from main app")
# #             self.graph_type_label.setVisible(True)
# #             self.graph_type.setVisible(True)
# #             self.plot_btn.setVisible(True)


# #     def on_graph_type_changed(self, text):
# #         # Clear previous selections when a new graph type is selected
# #         self.view_type.setCurrentIndex(0)  # Reset the surface view
# #         self.feature_identification.setCurrentIndex(0)  # Reset the feature identification
# #         self.dimension_classification.setCurrentIndex(0)  # Reset the dimension classification

# #         if text == "Defects":
# #             # When Metal Loss is selected, show feature identification and dimensional classification dropdowns
# #             self.feature_identification_label.setVisible(True)
# #             self.feature_identification.setVisible(True)
# #             self.dimension_classification_label.setVisible(True)
# #             self.dimension_classification.setVisible(True)
# #             self.view_type_label.setVisible(False)
# #             self.view_type.setVisible(False)
# #         elif text in ["ERF", "Psafe", "Orientation"]:
# #             # When any of these are selected, show the surface view dropdown
# #             self.feature_identification_label.setVisible(False)
# #             self.feature_identification.setVisible(False)
# #             self.dimension_classification_label.setVisible(False)
# #             self.dimension_classification.setVisible(False)
# #             self.view_type_label.setVisible(True)
# #             self.view_type.setVisible(True)
# #         elif text in ["Depth"]:
# #             self.feature_identification_label.setVisible(False)
# #             self.feature_identification.setVisible(False)
# #             self.dimension_classification_label.setVisible(False)
# #             self.dimension_classification.setVisible(False)
# #             self.view_type_label.setVisible(False)
# #             self.view_type.setVisible(False)

# #         elif text == "Sensor Loss":
# #             # When Sensor Loss is selected, hide the feature and dimension dropdowns and the surface view dropdown
# #             self.feature_identification_label.setVisible(False)
# #             self.feature_identification.setVisible(False)
# #             self.dimension_classification_label.setVisible(False)
# #             self.dimension_classification.setVisible(False)
# #             self.view_type_label.setVisible(False)
# #             self.view_type.setVisible(False)
# #         elif text == "Temperature":
# #             # When Temperature is selected, hide the feature and dimension dropdowns and the surface view dropdown
# #             self.feature_identification_label.setVisible(False)
# #             self.feature_identification.setVisible(False)
# #             self.dimension_classification_label.setVisible(False)
# #             self.dimension_classification.setVisible(False)
# #             self.view_type_label.setVisible(False)
# #             self.view_type.setVisible(False)
# #         else:
# #             # If no valid graph type is selected, hide all dropdowns and their titles
# #             self.feature_identification_label.setVisible(False)
# #             self.feature_identification.setVisible(False)
# #             self.dimension_classification_label.setVisible(False)
# #             self.dimension_classification.setVisible(False)
# #             self.view_type_label.setVisible(False)
# #             self.view_type.setVisible(False)

# #     def on_feature_identification_changed(self, text):
# #         # Reset the Dimensional Classification dropdown whenever Feature Identification changes
# #         self.dimension_classification.setCurrentIndex(0)  # Reset to the first option (empty)

# #         # No need to disable Dimensional Classification, keep it always enabled
# #         self.dimension_classification.setEnabled(True)

# #     def load_file(self):
# #         path, _ = QFileDialog.getOpenFileName(self, "Select Excel File", "", "Excel Files (*.xlsx *.xls)")
# #         if path:
# #             self.df = pd.read_excel(path)
# #             self.df.columns = self.df.columns.str.strip()
# #             self.file_label.setText(f"Loaded: {os.path.basename(path)}")

# #             # After the file is loaded, show the Graph Type dropdown and its title
# #             self.graph_type_label.setVisible(True)
# #             self.graph_type.setVisible(True)
# #             self.plot_btn.setVisible(True)

# #     def plot_graph(self):
# #         try:
# #             graph_type = self.graph_type.currentText()
# #             feature = self.feature_identification.currentText()
# #             dimension = self.dimension_classification.currentText()
# #             view = self.view_type.currentText()

# #             if not graph_type:
# #                 msg = QMessageBox()
# #                 msg.setIcon(QMessageBox.Warning)
# #                 msg.setWindowTitle("Graph Type Not Selected")
# #                 msg.setText("Please select the graph type before plotting.")
# #                 msg.setStandardButtons(QMessageBox.Ok)
# #                 msg.exec()
# #                 return

# #             if graph_type == "Defects":
# #                 if not feature and dimension == "":
# #                     msg = QMessageBox()
# #                     msg.setIcon(QMessageBox.Warning)
# #                     msg.setWindowTitle("Feature Identification or Dimensional Classification Not Selected")
# #                     msg.setText("Please select either Feature Identification or Dimensional Classification.")
# #                     msg.setStandardButtons(QMessageBox.Ok)
# #                     msg.exec()
# #                     return

# #                 feature_id = feature if feature else None
# #                 dimension_class = dimension if dimension != "Both" else None

# #                 # Plot Metal Loss
# #                 fig, path = plot_metal_loss(self.df.copy(), feature_type=feature_id, dimension_class=dimension_class, return_fig=True)
# #                 self.current_fig = fig
# #                 self.browser.load(QUrl.fromLocalFile(path))
# #                 self.save_btn.setVisible(True)

# #             elif graph_type in ["ERF", "Psafe", "Orientation"]:
# #                 if not view:
# #                     msg = QMessageBox()
# #                     msg.setIcon(QMessageBox.Warning)
# #                     msg.setWindowTitle("Surface View Not Selected")
# #                     msg.setText("Please select Surface View before plotting.")
# #                     msg.setStandardButtons(QMessageBox.Ok)
# #                     msg.exec()
# #                     return

# #                 if graph_type == "ERF":
# #                     fig, path = plot_erf(self.df.copy(), view, return_fig=True)
# #                 elif graph_type == "Psafe":
# #                     fig, path = plot_psafe(self.df.copy(), view, return_fig=True)
# #                 elif graph_type == "Orientation":
# #                     fig, path = plot_orientation(self.df.copy(), view, return_fig=True)

# #                 self.current_fig = fig
# #                 self.browser.load(QUrl.fromLocalFile(path))
# #                 self.save_btn.setVisible(True)
# #             elif graph_type == "Depth":
# #                 fig, path = plot_depth(self.df.copy(), view, return_fig=True)
# #                 self.current_fig = fig
# #                 self.browser.load(QUrl.fromLocalFile(path))
# #                 self.save_btn.setVisible(True)

# #             elif graph_type == "Sensor Loss":
# #                 # Plot Sensor Loss
# #                 fig, path = plot_sensor_percentage(self.df.copy(), return_fig=True)  # Fix here by passing only return_fig
# #                 self.current_fig = fig
# #                 self.browser.load(QUrl.fromLocalFile(path))
# #                 self.save_btn.setVisible(True)

# #             elif graph_type == "Temperature":
# #                 # Plot Sensor Loss
# #                 fig, path = plot_temperature(self.df.copy(), return_fig=True)  # Fix here by passing only return_fig
# #                 self.current_fig = fig
# #                 self.browser.load(QUrl.fromLocalFile(path))
# #                 self.save_btn.setVisible(True)

# #             else:
# #                 self.file_label.setText("Please select a graph type.")

# #         except Exception as e:
# #             self.file_label.setText(f"Plot failed: {str(e)}")
# #             self.current_fig = None

# #     def save_graph(self):
# #         if hasattr(self, 'current_fig') and self.current_fig is not None:
# #             downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
# #             default_file = os.path.join(downloads_folder, "graph.png")

# #             file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot as PNG", default_file, "PNG Files (*.png);;All Files (*)")
# #             if file_path:
# #                 try:
# #                     if not file_path.lower().endswith(".png"):
# #                         file_path += ".png"

# #                     self.current_fig.write_image(file_path)
# #                     self.file_label.setText(f"Graph saved as: {file_path}")
# #                 except Exception as e:
# #                     self.file_label.setText(f"Failed to save graph: {str(e)}")
# #             else:
# #                 self.file_label.setText("No file path selected.")
# #         else:
# #             self.file_label.setText("No graph to save.")


# # if __name__ == "__main__":
# #     app = QApplication(sys.argv)
# #     viewer = GraphApp()
# #     viewer.show()
# #     sys.exit(app.exec())


# import sys
# import os
# import pandas as pd
# import plotly.io as pio
# from PyQt6.QtWidgets import (
#     QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel,
#     QComboBox, QSizePolicy, QMessageBox,QHBoxLayout
# )
# from PyQt6.QtWidgets import QMessageBox
# import plotly.graph_objects as go

# from PyQt6.QtWebEngineWidgets import QWebEngineView
# from PyQt6.QtCore import Qt, QUrl
# from pages.graphs_func import plot_erf, plot_psafe, plot_depth, plot_orientation
# from pages.Defects import (plot_metal_loss, plot_sensor_percentage, plot_temperature)# Metal Loss and Sensor Loss Plot functions

# pio.kaleido.scope.default_format = "png"


# class GraphApp(QWidget):
#     def __init__(self, dataframe=None):
#         super().__init__()
        
#         self.df = dataframe

#         # 🛠️ Create UI elements...
#         self.setWindowTitle("Pipeline Defect Graph Viewer")
#         self.setGeometry(200, 100, 1100, 800)

#         self.layout = QVBoxLayout()
#         self.button_layout = QHBoxLayout()

#         self.file_label = QLabel("No file selected")
#         self.layout.addWidget(self.file_label)

#         # (... all your widget setup continues here ...)

#         self.setLayout(self.layout)


#         # Graph Type Dropdown
#         self.graph_type_label = QLabel("Select Graph Type:")
#         self.graph_type_label.setVisible(False)
#         self.button_layout.addWidget(self.graph_type_label)

#         self.graph_type = QComboBox()
#         self.graph_type.setFixedWidth(150)
#         self.graph_type.addItems(["", "Defects", "ERF", "Psafe", "Depth", "Orientation", "Temperature", "Sensor Loss"])
#         self.graph_type.setVisible(False)
#         self.graph_type.currentTextChanged.connect(self.on_graph_type_changed)
#         self.button_layout.addWidget(self.graph_type)

#         # Feature Identification
#         self.feature_identification_label = QLabel("Select Feature Identification:")
#         self.feature_identification_label.setVisible(False)
#         self.button_layout.addWidget(self.feature_identification_label)

#         self.feature_identification = QComboBox()
#         self.feature_identification.setFixedWidth(150)
#         self.feature_identification.addItems(["", "Corrosion", "MFG", "Both(Corrosion,MFG)"])
#         self.feature_identification.setVisible(False)
#         self.button_layout.addWidget(self.feature_identification)

#         # Dimensional Classification
#         self.dimension_classification_label = QLabel("Select Dimensional Classification:")
#         self.dimension_classification_label.setVisible(False)
#         self.button_layout.addWidget(self.dimension_classification_label)

#         self.dimension_classification = QComboBox()
#         self.dimension_classification.setFixedWidth(150)
#         self.dimension_classification.addItems([
#             "", "Pitting", "Axial Grooving", "Axial Slotting",
#             "Circumferential Grooving", "Circumferential Slotting",
#             "Pinhole", "General"
#         ])
#         self.dimension_classification.setVisible(False)
#         self.button_layout.addWidget(self.dimension_classification)

#         # Surface View
#         self.view_type_label = QLabel("Select Surface View:")
#         self.view_type_label.setVisible(False)
#         self.button_layout.addWidget(self.view_type_label)

#         self.feature_identification.currentTextChanged.connect(self.on_feature_identification_changed)

#         self.view_type = QComboBox()
#         self.view_type.setFixedWidth(150)
#         self.view_type.addItems(["", "Internal", "External", "Both"])
#         self.view_type.setVisible(False)
#         self.button_layout.addWidget(self.view_type)

#         # Plot Button
#         self.plot_btn = QPushButton("Plot Graph")
#         self.plot_btn.setFixedWidth(150)
#         self.plot_btn.setVisible(False)
#         self.plot_btn.clicked.connect(self.plot_graph)
#         self.button_layout.addWidget(self.plot_btn)

#         # Add some stretch for spacing (optional)
#         self.button_layout.addStretch()

#         # Add horizontal layout now — before graph view
#         self.layout.addLayout(self.button_layout)

#         # Save Button (optional: move it to horizontal row if desired)
#         self.save_btn = QPushButton("Save Graph as PNG")
#         self.save_btn.setStyleSheet("""
#             QPushButton {
#                 background-color: #444;
#                 color: white;
#                 padding: 6px 15px;
#                 font-weight: bold;
#                 border-radius: 5px;
#             }
#             QPushButton:hover {
#                 background-color: #666;
#             }
#         """)
#         self.save_btn.setFixedWidth(200)
#         self.save_btn.setVisible(False)
#         self.save_btn.clicked.connect(self.save_graph)
#         self.layout.addWidget(self.save_btn)

#         # Graph display
#         self.browser = QWebEngineView()
#         # self.browser.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
#         self.browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

#         self.layout.addWidget(self.browser)

#         # Set final layout
#         self.setLayout(self.layout)

#         if self.df is not None:
#             self.file_label.setText("Pipe tally loaded from main app")
#             self.graph_type_label.setVisible(True)
#             self.graph_type.setVisible(True)
#             self.plot_btn.setVisible(True)


#     def on_graph_type_changed(self, text):
#         # Clear previous selections when a new graph type is selected
#         self.view_type.setCurrentIndex(0)  # Reset the surface view
#         self.feature_identification.setCurrentIndex(0)  # Reset the feature identification
#         self.dimension_classification.setCurrentIndex(0)  # Reset the dimension classification

#         if text == "Defects":
#             # When Metal Loss is selected, show feature identification and dimensional classification dropdowns
#             self.feature_identification_label.setVisible(True)
#             self.feature_identification.setVisible(True)
#             self.dimension_classification_label.setVisible(True)
#             self.dimension_classification.setVisible(True)
#             self.view_type_label.setVisible(False)
#             self.view_type.setVisible(False)
#         elif text in ["ERF", "Psafe", "Orientation"]:
#             # When any of these are selected, show the surface view dropdown
#             self.feature_identification_label.setVisible(False)
#             self.feature_identification.setVisible(False)
#             self.dimension_classification_label.setVisible(False)
#             self.dimension_classification.setVisible(False)
#             self.view_type_label.setVisible(True)
#             self.view_type.setVisible(True)
#         elif text in ["Depth"]:
#             self.feature_identification_label.setVisible(False)
#             self.feature_identification.setVisible(False)
#             self.dimension_classification_label.setVisible(False)
#             self.dimension_classification.setVisible(False)
#             self.view_type_label.setVisible(False)
#             self.view_type.setVisible(False)

#         elif text == "Sensor Loss":
#             # When Sensor Loss is selected, hide the feature and dimension dropdowns and the surface view dropdown
#             self.feature_identification_label.setVisible(False)
#             self.feature_identification.setVisible(False)
#             self.dimension_classification_label.setVisible(False)
#             self.dimension_classification.setVisible(False)
#             self.view_type_label.setVisible(False)
#             self.view_type.setVisible(False)
#         elif text == "Temperature":
#             # When Temperature is selected, hide the feature and dimension dropdowns and the surface view dropdown
#             self.feature_identification_label.setVisible(False)
#             self.feature_identification.setVisible(False)
#             self.dimension_classification_label.setVisible(False)
#             self.dimension_classification.setVisible(False)
#             self.view_type_label.setVisible(False)
#             self.view_type.setVisible(False)
#         else:
#             # If no valid graph type is selected, hide all dropdowns and their titles
#             self.feature_identification_label.setVisible(False)
#             self.feature_identification.setVisible(False)
#             self.dimension_classification_label.setVisible(False)
#             self.dimension_classification.setVisible(False)
#             self.view_type_label.setVisible(False)
#             self.view_type.setVisible(False)

#     def on_feature_identification_changed(self, text):
#         # Reset the Dimensional Classification dropdown whenever Feature Identification changes
#         self.dimension_classification.setCurrentIndex(0)  # Reset to the first option (empty)

#         # No need to disable Dimensional Classification, keep it always enabled
#         self.dimension_classification.setEnabled(True)

#     def load_file(self):
#         path, _ = QFileDialog.getOpenFileName(self, "Select Excel File", "", "Excel Files (*.xlsx *.xls)")
#         if path:
#             self.df = pd.read_excel(path)
#             self.df.columns = self.df.columns.str.strip()
#             self.file_label.setText(f"Loaded: {os.path.basename(path)}")

#             # After the file is loaded, show the Graph Type dropdown and its title
#             self.graph_type_label.setVisible(True)
#             self.graph_type.setVisible(True)
#             self.plot_btn.setVisible(True)

#     def plot_graph(self):
#         try:
#             graph_type = self.graph_type.currentText()
#             feature = self.feature_identification.currentText()
#             dimension = self.dimension_classification.currentText()
#             view = self.view_type.currentText()

#             if not graph_type:
#                 msg = QMessageBox()
#                 msg.setIcon(QMessageBox.Warning)
#                 msg.setWindowTitle("Graph Type Not Selected")
#                 msg.setText("Please select the graph type before plotting.")
#                 msg.setStandardButtons(QMessageBox.Ok)
#                 msg.exec()
#                 return

#             if graph_type == "Defects":
#                 if not feature and dimension == "":
#                     msg = QMessageBox()
#                     msg.setIcon(QMessageBox.Warning)
#                     msg.setWindowTitle("Feature Identification or Dimensional Classification Not Selected")
#                     msg.setText("Please select either Feature Identification or Dimensional Classification.")
#                     msg.setStandardButtons(QMessageBox.Ok)
#                     msg.exec()
#                     return

#                 feature_id = feature if feature else None
#                 dimension_class = dimension if dimension != "Both" else None

#                 # Plot Metal Loss
#                 fig, path = plot_metal_loss(self.df.copy(), feature_type=feature_id, dimension_class=dimension_class, return_fig=True)
#                 self.current_fig = fig
#                 self.browser.load(QUrl.fromLocalFile(path))
#                 self.save_btn.setVisible(True)

#             elif graph_type in ["ERF", "Psafe", "Orientation"]:
#                 if not view:
#                     msg = QMessageBox()
#                     msg.setIcon(QMessageBox.Warning)
#                     msg.setWindowTitle("Surface View Not Selected")
#                     msg.setText("Please select Surface View before plotting.")
#                     msg.setStandardButtons(QMessageBox.Ok)
#                     msg.exec()
#                     return

#                 if graph_type == "ERF":
#                     fig, path = plot_erf(self.df.copy(), view, return_fig=True)
#                 elif graph_type == "Psafe":
#                     fig, path = plot_psafe(self.df.copy(), view, return_fig=True)
#                 elif graph_type == "Orientation":
#                     fig, path = plot_orientation(self.df.copy(), view, return_fig=True)

#                 self.current_fig = fig
#                 self.browser.load(QUrl.fromLocalFile(path))
#                 self.save_btn.setVisible(True)

#             elif graph_type == "Depth":
#                 fig, path = plot_depth(self.df.copy(), return_fig=True)
#                 self.current_fig = fig
#                 if path.endswith(".png"):
#                     self.browser.load(QUrl.fromLocalFile(path))
#                 else:
#                     self.browser.load(QUrl.fromLocalFile(path))

#                 self.save_btn.setVisible(True)


#             elif graph_type == "Sensor Loss":
#                 # Plot Sensor Loss
#                 fig, path = plot_sensor_percentage(self.df.copy(), return_fig=True)  # Fix here by passing only return_fig
#                 self.current_fig = fig
#                 self.browser.load(QUrl.fromLocalFile(path))
#                 self.save_btn.setVisible(True)

#             elif graph_type == "Temperature":
#                 # Plot Sensor Loss
#                 fig, path = plot_temperature(self.df.copy(), return_fig=True)  # Fix here by passing only return_fig
#                 self.current_fig = fig
#                 self.browser.load(QUrl.fromLocalFile(path))
#                 self.save_btn.setVisible(True)

#             else:
#                 self.file_label.setText("Please select a graph type.")

#         except Exception as e:
#             self.file_label.setText(f"Plot failed: {str(e)}")
#             self.current_fig = None



#     def save_graph(self):
#         if hasattr(self, 'current_fig') and self.current_fig is not None:
#             downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
#             default_file = os.path.join(downloads_folder, "graph.png")

#             file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot as PNG", default_file, "PNG Files (*.png);;All Files (*)")
#             if file_path:
#                 try:
#                     if not file_path.lower().endswith(".png"):
#                         file_path += ".png"

#                     self.current_fig.write_image(file_path)
#                     self.file_label.setText(f"Graph saved as: {file_path}")
#                 except Exception as e:
#                     self.file_label.setText(f"Failed to save graph: {str(e)}")
#             else:
#                 self.file_label.setText("No file path selected.")
#         else:
#             self.file_label.setText("No graph to save.")


# if __name__ == "__main__":
#     app = QApplication(sys.argv)
#     viewer = GraphApp()
#     viewer.show()
#     sys.exit(app.exec())

import sys
import os
import pandas as pd
import plotly.io as pio
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QPushButton, QFileDialog, QLabel,
    QComboBox, QSizePolicy, QMessageBox, QHBoxLayout, QProgressDialog
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtCore import Qt, QUrl
from pages.graphs_func import plot_erf, plot_psafe, plot_depth, plot_orientation
from pages.Defects import (plot_metal_loss, plot_sensor_percentage, plot_temperature)  # Plot functions
import time 

pio.kaleido.scope.default_format = "png"


class GraphApp(QWidget):
    def __init__(self, dataframe=None):
        super().__init__()
        self.df = dataframe

        # Window properties
        self.setWindowTitle("Pipeline Defect Graph Viewer")
        self.setGeometry(200, 100, 1100, 800)

        # Apply global stylesheet
        self.setStyleSheet("""
            QWidget {
                background-color: #f8f9fa;
                font-family: Segoe UI, Arial;
                font-size: 11pt;
                color: #2c3e50;
            }
            QLabel {
                font-weight: 500;
            }
            QComboBox {
                padding: 5px;
                border: 1px solid #ccc;
                border-radius: 5px;
                min-width: 150px;
                background-color: #fff;
            }
            QComboBox:hover {
                border: 1px solid #3498db;
            }
            QPushButton {
                background-color: #3498db;
                color: white;
                padding: 6px 12px;
                border-radius: 5px;
                font-weight: 600;
            }
            QPushButton:hover {
                background-color: #2980b9;
            }
        """)

        # --- Main Layout ---
        self.layout = QVBoxLayout()

        # File label
        self.file_label = QLabel("No file selected")
        self.layout.addWidget(self.file_label)

        # --- Top toolbar layout ---
        top_bar = QHBoxLayout()
        top_bar.setSpacing(12)

        # Graph Type
        self.graph_type_label = QLabel("Graph Type:")
        self.graph_type = QComboBox()
        self.graph_type.addItems(["", "Defects", "ERF", "Psafe", "Depth", "Orientation", "Temperature", "Sensor Loss"])
        self.graph_type.currentTextChanged.connect(self.on_graph_type_changed)
        self.graph_type.setToolTip("Choose the type of graph to display")
        top_bar.addWidget(self.graph_type_label)
        top_bar.addWidget(self.graph_type)

        # Feature Identification
        self.feature_identification_label = QLabel("Feature Type:")
        self.feature_identification = QComboBox()
        self.feature_identification.addItems(["", "Corrosion", "MFG", "Both(Corrosion,MFG)"])
        self.feature_identification.setToolTip("Filter by defect feature (Corrosion, MFG, Both)")
        self.feature_identification.currentTextChanged.connect(self.on_feature_identification_changed)
        top_bar.addWidget(self.feature_identification_label)
        top_bar.addWidget(self.feature_identification)

        # Dimensional Classification
        self.dimension_classification_label = QLabel("Dimension Classification:")
        self.dimension_classification = QComboBox()
        self.dimension_classification.addItems([
            "", "Pitting", "Axial Grooving", "Axial Slotting",
            "Circumferential Grooving", "Circumferential Slotting",
            "Pinhole", "General"
        ])
        self.dimension_classification.setToolTip("Classify defect shape (Pitting, Slotting, etc.)")
        top_bar.addWidget(self.dimension_classification_label)
        top_bar.addWidget(self.dimension_classification)

        # Surface View
        self.view_type_label = QLabel("Surface View:")
        self.view_type = QComboBox()
        self.view_type.addItems(["", "Internal", "External", "Both"])
        self.view_type.setToolTip("Select whether to view Internal, External, or Both surfaces")
        top_bar.addWidget(self.view_type_label)
        top_bar.addWidget(self.view_type)

        # Plot Button
        self.plot_btn = QPushButton("Plot Graph")
        self.plot_btn.setToolTip("Click to generate the selected graph")
        self.plot_btn.clicked.connect(self.plot_graph)
        self.plot_btn.setEnabled(False)
        top_bar.addWidget(self.plot_btn)

        # Save Button
        self.save_btn = QPushButton("Save Graph as PNG")
        self.save_btn.setToolTip("Save the plotted graph as PNG")
        self.save_btn.clicked.connect(self.save_graph)
        self.save_btn.setEnabled(False)
        top_bar.addWidget(self.save_btn)

        top_bar.addStretch()
        self.layout.addLayout(top_bar)

        # --- Graph display ---
        self.browser = QWebEngineView()
        self.browser.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.browser, stretch=1)

        # Welcome message in browser
        welcome_html = """
        <html>
        <head><style>
        body { font-family: Segoe UI, Arial; color:#555; text-align:center; margin-top:100px; }
        h2 { color:#2c3e50; }
        </style></head>
        <body>
        <h2>📊 Pipeline Graph Viewer</h2>
        <p>Please select a graph type and click <b>Plot Graph</b> to begin.</p>
        </body></html>
        """
        self.browser.setHtml(welcome_html)

        # --- Status label at bottom ---
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet("color: #888; font-size: 9pt; margin-top: 5px;")
        self.layout.addWidget(self.status_label)

        # Final layout
        self.setLayout(self.layout)

        # Initially hide all optional controls
        self._hide_all_optionals()

        if self.df is not None:
            self.file_label.setText("Pipe tally loaded from main app")
            self.plot_btn.setEnabled(True)

    # ------------------------------------
    # Helpers
    # ------------------------------------
    def _hide_all_optionals(self):
        """Hide feature/dimension/view controls + labels"""
        self.feature_identification.hide()
        self.feature_identification_label.hide()
        self.dimension_classification.hide()
        self.dimension_classification_label.hide()
        self.view_type.hide()
        self.view_type_label.hide()

    def _set_feature_visible(self, visible=True):
        self.feature_identification.setVisible(visible)
        self.feature_identification_label.setVisible(visible)

    def _set_dimension_visible(self, visible=True):
        self.dimension_classification.setVisible(visible)
        self.dimension_classification_label.setVisible(visible)

    def _set_view_visible(self, visible=True):
        self.view_type.setVisible(visible)
        self.view_type_label.setVisible(visible)

    # ------------------------------------
    # UI Logic
    # ------------------------------------
    def on_graph_type_changed(self, text):
        self.plot_btn.setEnabled(bool(text))

        # Reset values
        self.view_type.setCurrentIndex(0)
        self.feature_identification.setCurrentIndex(0)
        self.dimension_classification.setCurrentIndex(0)

        # Hide everything first
        self._hide_all_optionals()

        # Show only what’s needed
        if text == "Defects":
            self._set_feature_visible(True)
            self._set_dimension_visible(True)
        elif text in ["ERF", "Psafe", "Orientation"]:
            self._set_view_visible(True)
        elif text in ["Depth", "Temperature", "Sensor Loss"]:
            pass  # no extra controls

    def on_feature_identification_changed(self, text):
        self.dimension_classification.setCurrentIndex(0)

    def load_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Select Excel File", "", "Excel Files (*.xlsx *.xls)")
        if path:
            self.df = pd.read_excel(path)
            self.df.columns = self.df.columns.str.strip()
            self.file_label.setText(f"Loaded: {os.path.basename(path)}")
            self.plot_btn.setEnabled(True)

    def plot_graph(self):
        try:
            graph_type = self.graph_type.currentText()
            feature = self.feature_identification.currentText()
            dimension = self.dimension_classification.currentText()
            view = self.view_type.currentText()

            if not graph_type:
                QMessageBox.warning(self, "Missing Selection", "Please select a graph type before plotting.")
                return

            if graph_type == "Defects":
                if not feature and dimension == "":
                    QMessageBox.warning(self, "Missing Selection", "Select Feature Identification or Dimensional Classification.")
                    return
                fig, path = plot_metal_loss(self.df.copy(), feature_type=feature or None,
                                            dimension_class=dimension or None, return_fig=True)
                self.current_fig = fig
                self.browser.load(QUrl.fromLocalFile(path))

            elif graph_type in ["ERF", "Psafe", "Orientation"]:
                if not view:
                    QMessageBox.warning(self, "Missing Selection", "Select Surface View before plotting.")
                    return
                if graph_type == "ERF":
                    fig, path = plot_erf(self.df.copy(), view, return_fig=True)
                elif graph_type == "Psafe":
                    fig, path = plot_psafe(self.df.copy(), view, return_fig=True)
                else:
                    fig, path = plot_orientation(self.df.copy(), view, return_fig=True)
                self.current_fig = fig
                self.browser.load(QUrl.fromLocalFile(path))

            elif graph_type == "Depth":
                fig, path = plot_depth(self.df.copy(), return_fig=True)
                self.current_fig = fig
                self.browser.load(QUrl.fromLocalFile(path))

            elif graph_type == "Sensor Loss":
                fig, path = plot_sensor_percentage(self.df.copy(), return_fig=True)
                self.current_fig = fig
                self.browser.load(QUrl.fromLocalFile(path))

            elif graph_type == "Temperature":
                fig, path = plot_temperature(self.df.copy(), return_fig=True)
                self.current_fig = fig
                self.browser.load(QUrl.fromLocalFile(path))

            else:
                self.file_label.setText("Invalid selection.")

            # Enable save + status update
            self.save_btn.setEnabled(True)
            self.status_label.setText(f"✅ {graph_type} graph plotted successfully.")

        except Exception as e:
            self.status_label.setText(f"❌ Plot failed: {str(e)}")
            self.current_fig = None

    # def save_graph(self):
    #     if hasattr(self, 'current_fig') and self.current_fig is not None:
    #         downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    #         default_file = os.path.join(downloads_folder, "graph.png")
    #         file_path, _ = QFileDialog.getSaveFileName(self, "Save Plot as PNG", default_file, "PNG Files (*.png)")
    #         if file_path:
    #             try:
    #                 if not file_path.lower().endswith(".png"):
    #                     file_path += ".png"
    #                 self.current_fig.write_image(file_path)
    #                 self.status_label.setText(f"💾 Graph saved at {file_path}")
    #             except Exception as e:
    #                 self.status_label.setText(f"❌ Failed to save graph: {str(e)}")
    #     else:
    #         self.status_label.setText("No graph to save.")

    # def save_graph(self):
    #     if hasattr(self, 'current_fig') and self.current_fig is not None:
    #         downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
    #         default_file = os.path.join(downloads_folder, "graph.png")
    #         file_path, _ = QFileDialog.getSaveFileName(
    #             self, "Save Plot as PNG", default_file, "PNG Files (*.png)"
    #         )
    #         if file_path:
    #             try:
    #                 if not file_path.lower().endswith(".png"):
    #                     file_path += ".png"

    #                 self.current_fig.write_image(file_path)
    #                 self.status_label.setText(f"💾 Graph saved at {file_path}")

    #                 # EXACT-style popup (title "Saved!", info icon, OK button)
    #                 QMessageBox.information(
    #                     self,
    #                     "Saved!",
    #                     f"File saved successfully:\n{file_path}",
    #                     QMessageBox.StandardButton.Ok
    #                 )

    #             except Exception as e:
    #                 self.status_label.setText(f"❌ Failed to save graph: {str(e)}")
    #                 QMessageBox.critical(self, "Save Failed", f"Could not save the file.\n\nError: {e}")
    #     else:
    #         self.status_label.setText("No graph to save.")
    #         QMessageBox.information(self, "Nothing to Save", "No graph is currently plotted.")



    def save_graph(self):
        if hasattr(self, 'current_fig') and self.current_fig is not None:
            downloads_folder = os.path.join(os.path.expanduser("~"), "Downloads")
            default_file = os.path.join(downloads_folder, "graph.png")
            file_path, _ = QFileDialog.getSaveFileName(
                self, "Save Plot as PNG", default_file, "PNG Files (*.png)"
            )
            if file_path:
                try:
                    if not file_path.lower().endswith(".png"):
                        file_path += ".png"

                    # --- Progress dialog ---
                    progress = QProgressDialog("Saving graph...", None, 0, 100, self)
                    progress.setWindowTitle("Saving...")
                    progress.setWindowModality(Qt.WindowModality.ApplicationModal)
                    progress.setMinimumDuration(0)
                    progress.setValue(0)

                    # Simulate save steps (for UX)
                    for i in range(1, 101, 25):
                        time.sleep(0.1)  # tiny delay for effect
                        progress.setValue(i)
                        QApplication.processEvents()

                    # Actual save
                    self.current_fig.write_image(file_path)
                    progress.setValue(100)

                    self.status_label.setText(f"💾 Graph saved at {file_path}")

                    # Success popup
                    QMessageBox.information(
                        self,
                        "Saved!",
                        f"File saved successfully:\n{file_path}",
                        QMessageBox.StandardButton.Ok
                    )

                except Exception as e:
                    self.status_label.setText(f"❌ Failed to save graph: {str(e)}")
                    QMessageBox.critical(self, "Save Failed", f"Could not save the file.\n\nError: {e}")
        else:
            self.status_label.setText("No graph to save.")
            QMessageBox.information(self, "Nothing to Save", "No graph is currently plotted.")




if __name__ == "__main__":
    app = QApplication(sys.argv)
    viewer = GraphApp()
    viewer.show()
    sys.exit(app.exec())