from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox, QWidget, QGridLayout
from PyQt6.QtGui import QKeySequence
from PyQt6.QtCore import Qt

class YC_HelpDialog(QDialog):
    """A dialog that displays a list of all available keyboard shortcuts and controls."""
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YC Controls and Shortcuts")
        self.setMinimumWidth(400)

        self.layout = QVBoxLayout(self)

        description = QLabel("This guide lists the keyboard and mouse controls for navigating and using the application.")
        description.setWordWrap(True)
        self.layout.addWidget(description)

        grid_widget = QWidget()
        grid_layout = QGridLayout(grid_widget)
        grid_layout.setContentsMargins(0, 10, 0, 10)
        grid_layout.setHorizontalSpacing(20)
        grid_layout.setVerticalSpacing(10)

        shortcuts = [
            ("Open Folder", QKeySequence(QKeySequence.StandardKey.Open).toString()),
            ("Reset Application", "Ctrl+R"),
            ("Exit Application", QKeySequence(QKeySequence.StandardKey.Quit).toString()),
            ("Open Parameters", "Ctrl+P"),
            ("Zoom In", QKeySequence(QKeySequence.StandardKey.ZoomIn).toString()),
            ("Zoom Out", QKeySequence(QKeySequence.StandardKey.ZoomOut).toString()),
            ("Reset Zoom", "Ctrl+0"),
            ("Show Controls (This Window)", "F1"),
            ("Next Frame", "D"),
            ("Previous Frame", "A"),
            ("Zoom In/Out (Alternative)", "Ctrl + Mouse Wheel"),
            ("Pan Image", "Middle Mouse Button + Drag"),
        ]

        key_style = "background-color: #555; color: #EEE; padding: 2px 6px; border-radius: 4px; font-weight: bold;"

        for i, (desc, key) in enumerate(shortcuts):
            desc_label = QLabel(desc)
            key_label = QLabel(key)
            key_label.setStyleSheet(key_style)
            key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)

            grid_layout.addWidget(desc_label, i, 0)
            grid_layout.addWidget(key_label, i, 1)

        self.layout.addWidget(grid_widget)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)