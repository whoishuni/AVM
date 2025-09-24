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
        self.setWindowTitle(parent.tr("help_title"))
        self.setMinimumSize(600, 700)
        self.set_stylesheet()

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

        tr = self.parent().tr
        controls_label = QLabel(f"<h2>{tr('help_controls_title')}</h2>")
        main_layout.addWidget(controls_label)

        shortcuts = [
            (tr("help_control_open"), QKeySequence(QKeySequence.StandardKey.Open).toString()),
            (tr("help_control_reset"), "Ctrl+R"),
            (tr("help_control_exit"), QKeySequence(QKeySequence.StandardKey.Quit).toString()),
            (tr("help_control_params"), "Ctrl+P"),
            (tr("help_control_zoom_in"), QKeySequence(QKeySequence.StandardKey.ZoomIn).toString()),
            (tr("help_control_zoom_out"), QKeySequence(QKeySequence.StandardKey.ZoomOut).toString()),
            (tr("help_control_reset_zoom"), "Ctrl+0"),
            (tr("help_control_help"), "F1"),
            (tr("help_control_next_frame"), "D"),
            (tr("help_control_prev_frame"), "A"),
            (tr("help_control_alt_zoom"), "Ctrl + Mouse Wheel"),
            (tr("help_control_pan"), "Middle Mouse Button + Drag"),
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

        params_label = QLabel(f"<h2>{tr('help_params_title')}</h2>")
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

    def set_stylesheet(self):
        style = """
            QDialog {
                background-color: #383838;
                color: #D0D0D0;
            }
            QScrollArea {
                border: none;
            }
            QLabel {
                color: #D0D0D0;
                font-size: 13px;
            }
            h2 {
                color: #00A0A0;
                font-size: 16px;
                font-weight: bold;
                border-bottom: 1px solid #555;
                padding-bottom: 5px;
                margin-top: 10px;
            }
            QFrame {
                border: 1px solid #555;
            }
            QPushButton {
                background-color: #555555;
                color: #EEEEEE;
                border: 1px solid #666666;
                padding: 5px 15px;
                border-radius: 4px;
                font-size: 13px;
            }
            QPushButton:hover {
                background-color: #686868;
                border: 1px solid #777777;
            }
            QPushButton:pressed {
                background-color: #4A4A4A;
            }
        """
        self.setStyleSheet(style)