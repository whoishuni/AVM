import sys

# --- Dependency Check ---
try:
    from PyQt5.QtWidgets import QApplication, QMessageBox
except ImportError as e:
    # A simple fallback for when even PyQt5 is missing.
    # We can't use QMessageBox without a QApplication.
    print(f"CRITICAL: Missing required Python library: {e.name}")
    print("Please install it using: 'pip install numpy opencv-python scikit-image PyQt5 scikit-learn'")
    sys.exit(1)

# --- Main Application Execution ---
if __name__ == '__main__':
    try:
        # Check for other libraries before creating the main window
        import cv2
        import skimage
        import numpy

        # If all checks pass, import the main app and run it
        from src.app import VesselTracerApp

        app = QApplication(sys.argv)
        main_window = VesselTracerApp()
        main_window.show()
        sys.exit(app.exec_())

    except ImportError as e:
        # This block will catch missing cv2, skimage, etc.
        app = QApplication([]) # We know this works from the first check
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Critical)
        msg_box.setText(f"Missing required Python library: {e.name}")
        msg_box.setInformativeText("Please install it using: 'pip install numpy opencv-python scikit-image PyQt5 scikit-learn'")
        msg_box.setWindowTitle("Dependency Error")
        msg_box.exec_()
        sys.exit(1)
    except Exception as e:
        # Catch any other unexpected errors during initialization
        QMessageBox.critical(None, "Fatal Error", f"The application encountered an unrecoverable error on startup: {e}")
        sys.exit(1)
