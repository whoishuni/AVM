import cv2
import numpy as np
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QLabel, QSlider, QDialogButtonBox,
    QSizePolicy
)
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QImage, QPixmap

class YC_SmoothingPreviewDialog(QDialog):
    """A dialog for interactively previewing Gaussian blur smoothing."""

    def __init__(self, base_image: np.ndarray, initial_level: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YC Smoothing Effect Preview")
        self.setMinimumSize(600, 500)
        self.base_image = base_image
        self.smoothing_level = initial_level

        self.layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.layout.addWidget(self.image_label)

        control_layout = QHBoxLayout()
        self.slider_label = QLabel(f"Smoothing: {self.smoothing_level}")
        self.slider = QSlider(Qt.Orientation.Horizontal)
        self.slider.setRange(0, 10)
        self.slider.setValue(self.smoothing_level)
        control_layout.addWidget(self.slider_label)
        control_layout.addWidget(self.slider)
        self.layout.addLayout(control_layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Ok | QDialogButtonBox.StandardButton.Cancel)
        self.layout.addWidget(self.button_box)

        self.slider.valueChanged.connect(self.update_preview)
        self.button_box.accepted.connect(self.accept)
        self.button_box.rejected.connect(self.reject)
        self.update_preview()

    def update_preview(self):
        self.smoothing_level = self.slider.value()
        kernel_size = self.smoothing_level * 2 + 1

        if self.smoothing_level == 0:
            self.slider_label.setText("Smoothing: 0 (None)")
            processed_image = self.base_image
        else:
            self.slider_label.setText(f"Smoothing: {self.smoothing_level} (Kernel: {kernel_size}x{kernel_size})")
            processed_image = cv2.GaussianBlur(self.base_image, (kernel_size, kernel_size), 0)

        q_image = QImage(
            processed_image.data,
            processed_image.shape[1],
            processed_image.shape[0],
            processed_image.strides[0],
            QImage.Format.Format_Grayscale8
        )
        pixmap = QPixmap.fromImage(q_image)

        scaled_pixmap = pixmap.scaled(
            self.image_label.size(),
            Qt.AspectRatioMode.KeepAspectRatio,
            Qt.TransformationMode.SmoothTransformation
        )
        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.update_preview()
        super().resizeEvent(event)

    def get_selected_level(self) -> int:
        return self.smoothing_level