from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QDialogButtonBox, QWidget, QFormLayout,
    QLabel, QSpinBox, QDoubleSpinBox, QToolTip
)
from PyQt6.QtCore import Qt

class YC_ParameterDialog(QDialog):
    """A dialog for viewing and editing the application's core parameters."""
    def __init__(self, current_params: dict, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YC Algorithm Parameters")
        self.setMinimumWidth(450)

        self.params = current_params.copy()
        self.widgets = {}

        self.layout = QVBoxLayout(self)

        form_widget = QWidget()
        form_layout = QFormLayout(form_widget)
        form_layout.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapAllRows)
        form_layout.setLabelAlignment(Qt.AlignmentFlag.AlignLeft)

        param_meta = {
            "BG_REMOVAL_THRESHOLD_OFFSET": ("Offset from background color to detect bright areas.", int, 0, 100),
            "BG_REMOVAL_KERNEL_SIZE": ("Size of the kernel for removing bright areas (must be odd).", int, 1, 51),
            "MAX_NODE_SEARCH_RADIUS": ("Max distance (px) to find a graph node near a click.", int, 5, 200),
            "MAX_GAP_BRIDGE_DISTANCE": ("Max distance (px) to connect broken vessel segments in the mask.", int, 5, 100),
            "MAX_TEMPORAL_LINKING_DISTANCE": ("Max distance (px) to link vessel nodes between consecutive frames.", int, 5, 100),
        }

        for name, value in self.params.items():
            if name not in param_meta:
                continue

            description, param_type, min_val, max_val = param_meta[name]
            label = QLabel(f"{name.replace('_', ' ').title()}:")
            label.setToolTip(description)

            if param_type is int:
                widget = QSpinBox()
                widget.setRange(min_val, max_val)
                if "KERNEL_SIZE" in name:
                    widget.setSingleStep(2)
                    if value % 2 == 0: value += 1
                widget.setValue(int(value))
            else:
                widget = QDoubleSpinBox()
                widget.setRange(min_val, max_val)
                widget.setDecimals(2)
                if value > 1000:
                    widget.setDecimals(0)
                widget.setValue(float(value))

            self.widgets[name] = widget
            form_layout.addRow(label, widget)

        self.layout.addWidget(form_widget)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.button_box.accepted.connect(self.apply_changes)
        self.button_box.rejected.connect(self.reject)
        self.layout.addWidget(self.button_box)

    def apply_changes(self):
        for name, widget in self.widgets.items():
            self.params[name] = widget.value()
        self.accept()

    def get_parameters(self) -> dict:
        return self.params