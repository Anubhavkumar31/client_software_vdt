<<<<<<< HEAD
# PIE_software
=======
# Data-Pipeline-System(v1.2.1)

## Purpose

Pipe, Inspect, Extract (PIE) is a data visualization system designed for analyzing pipe sensor data to detect defects using various plots and heatmaps. It provides a user-friendly interface powered by PyQt6 and integrates multiple data visualization libraries.


## Features

- **Data Loading:** Load data from Excel (.xlsx) or CSV files.
- **Visualization:** Generate custom plots, telemetry plots, anomalies distribution plots, and heatmaps.
- **User Interface:** Utilizes PyQt6 for the graphical user interface (GUI) with customizable widgets.
- **Interactive Plots:** Ability to interactively plot data based on user selections.
- **File Management:** Open, close, and manage project files seamlessly.


## Installation

1. **Clone the repository:**
```
git clone <repo-url>
cd <repo-name>
```

2. **Install dependencies:**
```
pip install -r requirements.txt
```

3. **Run the application:**

## Requirements

- PyQt6 == 6.2.3
- openpyxl == 3.1.3
- pyqtgraph == 0.13.7
- PySide == 1.2.4
- matplotlib == 3.9.1
- seaborn == 0.13.2
- pandas == 2.2.2
- numpy == 2.0.1
- plotly == 5.22.0
- mpld3 == 0.5.10
- PyQt6-WebEngine == 6.7.0
- kaleido == 0.2.1.post1

## Usage

- **Open Project:** Use the File menu to open an Excel or CSV file containing pipe sensor data.
- **View Data:** Data is displayed in a sortable table view.
- **Generate Plots:**
    - Custom Plot: Visualize data based on selected columns.
    - Telemetry Plot: Display telemetry data.
    - Anomalies Distribution Plot: Show distribution of anomalies.
    - Heatmap: Visual representation of defects (internal or external).
    - Multi Line plot: Sensors signals mapped by odometer range

## References

* PyQt6 Docs
* Qt Designer
* Plotly Docs

## Authors

### Aksshat Govind and Abhishek Gupta, 2024
>>>>>>> master
