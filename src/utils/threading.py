from PyQt6.QtWidgets import QProgressDialog, QApplication

class ProgressUpdater:
    """
    A helper class to safely update a QProgressDialog.
    It can operate directly on the dialog's range or be configured to map
    a task's progress to a sub-range (e.g., 0-80%) of the dialog's total value.
    """

    def __init__(self, progress_dialog: QProgressDialog, min_val: int = None, max_val: int = None):
        """
        Initializes the ProgressUpdater.
        Args:
            progress_dialog: The QProgressDialog instance to be controlled.
            min_val: The value on the dialog that corresponds to 0% progress for this task.
            max_val: The value on the dialog that corresponds to 100% progress for this task.
        """
        self.dialog = progress_dialog
        self.is_running = True
        self.dialog.setModal(True)

        # Check if this updater should scale its progress to a sub-range of the dialog
        self.use_scaling = min_val is not None and max_val is not None
        if self.use_scaling:
            self.min_val = min_val
            self.max_val = max_val
            self.range = max_val - min_val
        else:
            self.min_val, self.max_val, self.range = 0, 0, 0

    def update(self, value: int, total: int, message: str = ""):
        """
        Updates the progress dialog and checks for cancellation.
        """
        if self.dialog.wasCanceled():
            self.is_running = False
            return

        if self.use_scaling:
            # Scale the task's progress (value/total) to the specified sub-range
            # of the main dialog (e.g., if range is 0-80, a 50% value becomes 40).
            scaled_value = self.min_val + int((value / total) * self.range) if total > 0 else self.min_val
            self.dialog.setValue(scaled_value)
        else:
            # Original behavior: direct 1-to-1 update of the dialog's value and max.
            self.dialog.setMaximum(total)
            self.dialog.setValue(value)

        if message:
            self.dialog.setLabelText(message)

        QApplication.processEvents()

    def finish(self):
        """
        Finalizes the progress update.
        If not using scaling, it closes the dialog (original behavior for standalone tasks).
        If using scaling, it just sets the progress to the max of its range and does NOT close it.
        """
        if self.use_scaling:
            self.dialog.setValue(self.max_val)
            QApplication.processEvents()
        else:
            self.dialog.close()