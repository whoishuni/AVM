import sys
import os
from typing import Optional

# Add the 'src' directory to the Python path to allow for absolute imports
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from PyQt6.QtWidgets import QApplication, QMessageBox, QDialog
from gui.main_window import YC_VesselTracerApp
from gui.language_dialog import YC_LanguageSelectionDialog

def main():
    """
    The main entry point for the YC_VesselTracer application.
    """
    def exception_hook(exctype, value, traceback):
        print(f"Uncaught exception: {exctype.__name__}, {value}")
        import traceback
        traceback.print_exception(exctype, value, traceback)
        QMessageBox.critical(
            None,
            "Application Error",
            f"An unexpected error occurred:\n\n{value}\n\nPlease see the console for more details.",
            QMessageBox.StandardButton.Ok
        )
        sys.exit(1)

    sys.excepthook = exception_hook

    try:
        app = QApplication(sys.argv)

        lang_dialog = YC_LanguageSelectionDialog()
        if lang_dialog.exec() == QDialog.DialogCode.Accepted:
            language = lang_dialog.get_selected_language()
        else:
            language = "en"

        main_window = YC_VesselTracerApp(language=language)
        main_window.show()
        sys.exit(app.exec())
    except ImportError as e:
        app = QApplication([])
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setText(f"Missing Required Python Library: {e.name}")
        msg_box.setInformativeText(
            "Please ensure all required libraries are installed. You can typically install them using:\n"
            "'pip install -r requirements.txt'"
        )
        msg_box.setWindowTitle("Dependency Error")
        msg_box.exec()
        sys.exit(1)
    except Exception as e:
        print(f"Fatal error during application startup: {e}")
        app = QApplication.instance() or QApplication([])
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Fatal Startup Error")
        msg_box.setText("The application failed to start because a Qt platform plugin could not be initialized.")
        msg_box.setInformativeText(
            f"This is often due to a system configuration issue or a conflict between libraries.\n\n"
            "Potential solutions:\n"
            "1. Ensure you are using 'opencv-python-headless' instead of 'opencv-python'.\n"
            "2. On Linux, try installing required system libraries like 'libxcb-cursor0'.\n\n"
            f"Original Error: {e}"
        )
        msg_box.exec()
        sys.exit(1)

if __name__ == '__main__':
    main()