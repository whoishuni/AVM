from PyQt6.QtWidgets import QProgressDialog, QApplication

class ProgressUpdater:
    """
    A helper class to safely update a QProgressDialog from a background thread.

    This class provides a simple and clean interface to update a progress dialog's
    value and check for user cancellation without passing the dialog object
    directly into the processing function. This decouples the processing logic
    from the UI components.
    """

    def __init__(self, progress_dialog: QProgressDialog):
        """
        Initializes the ProgressUpdater.

        Args:
            progress_dialog: The QProgressDialog instance to be controlled.
        """
        self.dialog = progress_dialog
        # The 'is_running' flag is crucial for signaling cancellation to the background thread.
        self.is_running = True
        # Set the dialog to be modal, so the user cannot interact with the main window
        # while the process is running.
        self.dialog.setModal(True)

    def update(self, value: int, total: int, message: str = ""):
        """
        Updates the progress dialog and checks for cancellation.

        If the user clicks the "Cancel" button on the dialog, the `is_running`
        flag is set to False, which the background process should check regularly.

        Args:
            value: The current progress value.
            total: The maximum progress value.
            message: An optional message to display on the progress dialog.
        """
        # Check if the user has cancelled the operation.
        if self.dialog.wasCanceled():
            self.is_running = False
            return

        # Update the dialog's properties.
        self.dialog.setMaximum(total)
        self.dialog.setValue(value)
        if message:
            self.dialog.setLabelText(message)

        # Process UI events to ensure the dialog updates immediately and remains responsive.
        QApplication.processEvents()

    def finish(self):
        """Closes the progress dialog."""
        self.dialog.close()