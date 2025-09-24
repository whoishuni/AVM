from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QPushButton, QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QPixmap

class YC_StepViewerDialog(QDialog):
    """A dialog for viewing the sequential steps of image processing and analysis."""

    def __init__(self, steps: list, main_window, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YC Processing Step Viewer")
        self.setMinimumSize(800, 600)
        self.steps = steps
        self.main_window = main_window
        self.current_step_index = 0

        self.layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.image_label)

        self.description_label = QLabel()
        self.description_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.description_label.font()
        font.setPointSize(12)
        self.description_label.setFont(font)
        self.layout.addWidget(self.description_label)

        btn_layout = QHBoxLayout()
        self.prev_button = QPushButton("Previous")
        self.replay_button = QPushButton("Replay Animation")
        self.replay_button.hide()
        self.next_button = QPushButton("Next")
        btn_layout.addWidget(self.prev_button)
        btn_layout.addWidget(self.replay_button)
        btn_layout.addWidget(self.next_button)
        self.layout.addLayout(btn_layout)

        self.prev_button.clicked.connect(self.prev_step)
        self.next_button.clicked.connect(self.next_step)
        self.replay_button.clicked.connect(self.trigger_replay)
        self.update_view()

    def update_view(self):
        if not self.steps or not (0 <= self.current_step_index < len(self.steps)):
            return

        step_info = self.steps[self.current_step_index]
        pixmap, description = step_info[0], step_info[1]

        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)
        self.description_label.setText(f"Step {self.current_step_index + 1}/{len(self.steps)}: {description}")

        self.prev_button.setEnabled(self.current_step_index > 0)
        self.next_button.setEnabled(self.current_step_index < len(self.steps) - 1)

        if len(step_info) > 2 and isinstance(step_info[2], dict) and step_info[2].get("type") == "animation":
            self.replay_button.show()
        else:
            self.replay_button.hide()

    def trigger_replay(self):
        step_info = self.steps[self.current_step_index]
        if len(step_info) > 2 and self.main_window and hasattr(self.main_window, 'replay_path_animation'):
            anim_data = step_info[2]
            self.main_window.replay_path_animation(anim_data)

    def prev_step(self):
        if self.current_step_index > 0:
            self.current_step_index -= 1
            self.update_view()

    def next_step(self):
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            self.update_view()

    def resizeEvent(self, event):
        self.update_view()
        super().resizeEvent(event)