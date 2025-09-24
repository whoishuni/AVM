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

    app = QApplication(sys.argv)

    try:
        # Attempt to create and show the language selection dialog first
        lang_dialog = YC_LanguageSelectionDialog()
        if lang_dialog.exec() == QDialog.DialogCode.Accepted:
            language = lang_dialog.get_selected_language()
        else:
            # Exit if the user closes the language dialog
            sys.exit(0)

        main_window = YC_VesselTracerApp(language=language)
        main_window.show()
        sys.exit(app.exec())

    except ImportError as e:
        # This handles missing critical libraries like PyQt6
        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setText(f"Missing Required Python Library: {e.name}")
        msg_box.setInformativeText(
            "A required library is missing. Please install all dependencies to run the application.\n"
            "You can typically install them using the following command in your terminal:\n\n"
            "pip install -r requirements.txt"
        )
        msg_box.setWindowTitle("Dependency Error")
        msg_box.exec()
        sys.exit(1)

    except Exception as e:
        # This is a catch-all for other potential errors during initialization
        print(f"Fatal error during application startup: {e}")
        import traceback
        traceback.print_exc()

        msg_box = QMessageBox()
        msg_box.setIcon(QMessageBox.Icon.Critical)
        msg_box.setWindowTitle("Fatal Startup Error")
        msg_box.setText("The application failed to start due to an unexpected error.")
        msg_box.setInformativeText(
            "A critical error occurred that prevented the application from launching. "
            "This could be due to a variety of reasons, such as a corrupted installation, "
            "a conflict with system libraries, or a bug in the application.\n\n"
            "Please check the console output for a detailed error message (traceback).\n\n"
            f"Error details: {e}"
        )
        msg_box.exec()
        sys.exit(1)

if __name__ == '__main__':
    main()