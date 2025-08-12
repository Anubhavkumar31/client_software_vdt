from PyQt6.QtWidgets import QApplication, QMainWindow
from PyQt6.uic import loadUi
import sys

class MainWindow(QMainWindow):
    def __init__(self):
        super(MainWindow, self).__init__()
        
        # Load the .ui file
        loadUi("ui/pipe_tally.ui", self)  # Replace with the path to your .ui file

    # Example function for button click (if you have buttons in the UI)
    def on_button_clicked(self):
        print("Button clicked!")

if __name__ == "__main__":
    app = QApplication(sys.argv)
    
    # Create an instance of your main window class
    window = MainWindow()
    
    # Show the window
    window.show()
    
    # Start the application loop
    sys.exit(app.exec())




