from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox, QWidget, QGridLayout, QScrollArea, QFrame
from PyQt6.QtGui import QKeySequence
from PyQt6.QtCore import Qt

class YC_HelpDialog(QDialog):
    """
    A dialog that displays a list of all available keyboard shortcuts, controls,
    and explanations for algorithm parameters.
    """
    def __init__(self, params_meta: dict, parent=None):
        super().__init__(parent)
        self.params_meta = params_meta
        self.setWindowTitle("YC Help")
        self.setMinimumSize(600, 700)

        self.layout = QVBoxLayout(self)

        scroll_area = QScrollArea()
        scroll_area.setWidgetResizable(True)
        self.layout.addWidget(scroll_area)

        main_widget = QWidget()
        scroll_area.setWidget(main_widget)

        main_layout = QVBoxLayout(main_widget)

        # --- Controls Section ---
        controls_group = QWidget()
        controls_layout = QGridLayout(controls_group)
        controls_layout.setContentsMargins(10, 10, 10, 10)
        controls_layout.setHorizontalSpacing(20)
        controls_layout.setVerticalSpacing(10)

        controls_label = QLabel("<h2>Controls & Shortcuts</h2>")
        main_layout.addWidget(controls_label)

        shortcuts = [
            ("Open Folder", QKeySequence(QKeySequence.StandardKey.Open).toString()),
            ("Reset Application", "Ctrl+R"),
            ("Exit Application", QKeySequence(QKeySequence.StandardKey.Quit).toString()),
            ("Open Parameters", "Ctrl+P"),
            ("Zoom In", QKeySequence(QKeySequence.StandardKey.ZoomIn).toString()),
            ("Zoom Out", QKeySequence(QKeySequence.StandardKey.ZoomOut).toString()),
            ("Reset Zoom", "Ctrl+0"),
            ("Show Help (This Window)", "F1"),
            ("Next Frame", "D"),
            ("Previous Frame", "A"),
            ("Zoom In/Out (Alternative)", "Ctrl + Mouse Wheel"),
            ("Pan Image", "Middle Mouse Button + Drag"),
        ]

        key_style = "background-color: #555; color: #EEE; padding: 2px 6px; border-radius: 4px; font-weight: bold;"
        for i, (desc, key) in enumerate(shortcuts):
            controls_layout.addWidget(QLabel(desc), i, 0)
            key_label = QLabel(key)
            key_label.setStyleSheet(key_style)
            key_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            controls_layout.addWidget(key_label, i, 1)

        main_layout.addWidget(controls_group)

        # --- Separator ---
        separator = QFrame()
        separator.setFrameShape(QFrame.Shape.HLine)
        separator.setFrameShadow(QFrame.Shadow.Sunken)
        main_layout.addWidget(separator)

        # --- Parameters Section ---
        params_group = QWidget()
        params_layout = QGridLayout(params_group)
        params_layout.setContentsMargins(10, 10, 10, 10)
        params_layout.setVerticalSpacing(10)

        params_label = QLabel("<h2>Algorithm Parameters</h2>")
        main_layout.addWidget(params_label)

        param_name_style = "font-weight: bold; color: #00A0A0;"
        for i, (name, meta) in enumerate(self.params_meta.items()):
            desc, _, _, _ = meta
            name_label = QLabel(name)
            name_label.setStyleSheet(param_name_style)
            desc_label = QLabel(desc)
            desc_label.setWordWrap(True)
            params_layout.addWidget(name_label, i, 0)
            params_layout.addWidget(desc_label, i, 1)

        main_layout.addWidget(params_group)

        # --- Close Button ---
        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)