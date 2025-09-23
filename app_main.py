import sys
import os
import re
import numpy as np
import heapq
import math
from collections import deque
from enum import Enum, auto
from typing import Optional, List, Tuple, Dict, Any
from vessel_graph import VesselMemory, build_vessel_memory, find_path_on_graph, VesselNode, visualize_vessel_memory

# Attempt to import necessary libraries, provide guidance on failure
try:
    from PyQt5.QtWidgets import (QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
                                 QFileDialog, QLabel, QStatusBar, QMainWindow, QMessageBox,
                                 QSizePolicy, QProgressDialog, QSlider, QDialog, QDialogButtonBox,
                                 QGroupBox, QStyle)
    from PyQt5.QtGui import (QPixmap, QImage, QPainter, QPen, QColor, QBrush, QFont)
    from PyQt5.QtCore import (Qt, QPoint, pyqtSignal, QThread, QRect, QSize)
    import cv2
    from skimage.morphology import skeletonize
    from skimage.filters import frangi, sato, meijering
except ImportError as e:
    # If a library is missing, create a simple QApplication to show an error message
    app = QApplication([])
    msg_box = QMessageBox()
    msg_box.setIcon(QMessageBox.Critical)
    msg_box.setText(f"Missing required Python library: {e.name}")
    msg_box.setInformativeText("Please install it using: 'pip install numpy opencv-python scikit-image PyQt5 scikit-learn'")
    msg_box.setWindowTitle("Dependency Error")
    msg_box.exec_()
    sys.exit(1)


# --- Global Helper Functions ---

def natural_sort_key(s: str) -> list:
    """Provides a key for natural sorting of filenames."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
    """Loads an image sequence from a folder, sorted naturally."""
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    try:
        filenames = os.listdir(folder_path)
        filenames.sort(key=natural_sort_key)
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                img_path = os.path.join(folder_path, filename)
                img_array = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
        return images
    except Exception as e:
        print(f"Error loading images: {e}")
        return []


def get_most_frequent_color(image: np.ndarray) -> int:
    """Gets the most frequent color in the image, usually the background."""
    if image is None: return 255
    unique, counts = np.unique(image, return_counts=True)
    return unique[np.argmax(counts)]


def bridge_gaps_in_mask(mask: np.ndarray, max_distance: int = 15) -> np.ndarray:
    """
    Intelligently connects separated vessel segments by finding endpoints on the skeleton.
    This method is more accurate and efficient than brute-force distance calculations between contour points.
    """
    if mask is None or np.sum(mask) == 0:
        return mask.copy() if mask is not None else np.array([])

    bridged_mask = mask.copy()

    # 1. Filter out small contours/noise and create a mask with only valid contours
    min_contour_area = 5
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    if not contours:
        return bridged_mask

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]

    if len(valid_contours) < 2:
        return bridged_mask  # No need to connect if there are fewer than two contours

    valid_contours_mask = np.zeros_like(mask)
    cv2.drawContours(valid_contours_mask, valid_contours, -1, 255, -1)

    # 2. Skeletonize the filtered mask
    skeleton = skeletonize(valid_contours_mask / 255).astype(np.uint8) * 255

    # 3. Find endpoints of the skeleton (points with only one neighbor)
    # Use convolution to quickly count neighbors for each pixel.
    # In the kernel, the center is 10 and neighbors are 1. After convolution, a skeleton point with a value of 11 (1*10 + 1*1) is an endpoint.
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    convolved = cv2.filter2D(skeleton, -1, kernel)
    endpoints_map = np.zeros_like(skeleton)
    endpoints_map[(convolved == 11) & (skeleton > 0)] = 255

    # 4. Associate endpoints with their original contours
    endpoint_coords = np.argwhere(endpoints_map > 0)
    if len(endpoint_coords) < 2:
        return bridged_mask  # Not enough endpoints to connect

    # Create a list of endpoints with their coordinates and parent contour index
    endpoints_with_contour_info = []
    for y, x in endpoint_coords:
        for i, cnt in enumerate(valid_contours):
            # Check if the point is inside or on the edge of the contour
            if cv2.pointPolygonTest(cnt, (int(x), int(y)), False) >= 0:
                endpoints_with_contour_info.append({'point': (y, x), 'contour_idx': i})
                break

    # 5. Find the closest pair of endpoints from different contours and connect them
    for i in range(len(endpoints_with_contour_info)):
        for j in range(i + 1, len(endpoints_with_contour_info)):
            ep1 = endpoints_with_contour_info[i]
            ep2 = endpoints_with_contour_info[j]

            # Ensure the two endpoints belong to different contours
            if ep1['contour_idx'] != ep2['contour_idx']:
                p1_yx = ep1['point']
                p2_yx = ep2['point']

                dist = np.linalg.norm(np.array(p1_yx) - np.array(p2_yx))

                if dist < max_distance:
                    # cv2.line needs (x, y) format
                    p1_xy = (int(p1_yx[1]), int(p1_yx[0]))
                    p2_xy = (int(p2_yx[1]), int(p2_yx[0]))
                    # Draw a line on the copy of the original mask to bridge the gap
                    cv2.line(bridged_mask, p1_xy, p2_xy, 255, 1)

    return bridged_mask


def remove_large_bright_areas(image: np.ndarray, bg_color: int, threshold_offset: int = 15,
                              kernel_size: int = 15) -> np.ndarray:
    """Removes large bright areas from the image to reduce background interference."""
    threshold_value = max(0, bg_color - threshold_offset)
    _, bright_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    large_areas_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    processed_image = image.copy()
    processed_image[large_areas_mask == 255] = bg_color
    return processed_image


def create_enhanced_vessel_masks(images: List[np.ndarray], noise_rois: List[QRect], bg_color: int,
                                 app_instance: 'VesselTracerApp', smoothing_level: int,
                                 worker_thread: Optional['ProgressUpdater']) -> Optional[List[np.ndarray]]:
    """Generates a sequence of enhanced vessel masks."""
    masks = []
    total_images = len(images)
    bg_color_int = int(bg_color)
    max_gap_dist = app_instance.MAX_GAP_BRIDGE_DISTANCE

    # Define sigmas for the filters to detect vessels of different thicknesses
    filter_sigmas = range(1, 6, 2)

    for i, img in enumerate(images):
        if worker_thread and not worker_thread.is_running:
            return None

        processed_img = img.copy()
        processed_img = remove_large_bright_areas(processed_img, bg_color_int, app_instance.BG_REMOVAL_THRESHOLD_OFFSET,
                                                  app_instance.BG_REMOVAL_KERNEL_SIZE)

        if smoothing_level > 0:
            kernel_size = smoothing_level * 2 + 1
            processed_img = cv2.GaussianBlur(processed_img, (kernel_size, kernel_size), 0)

        for roi in noise_rois:
            x, y, w, h = roi.x(), roi.y(), roi.width(), roi.height()
            processed_img[y:y + h, x:x + w] = bg_color_int

        # Invert image: filters enhance bright structures on a dark background
        inverted_img = cv2.bitwise_not(processed_img)

        # --- New pipeline: Apply vessel enhancement filters ---
        # black_ridges=False because vessels are now white
        frangi_img = frangi(inverted_img, sigmas=filter_sigmas, black_ridges=False)
        sato_img = sato(inverted_img, sigmas=filter_sigmas, black_ridges=False)
        meijering_img = meijering(inverted_img, sigmas=filter_sigmas, black_ridges=False)

        # Combine results by taking the maximum response
        combined_response = np.maximum.reduce([frangi_img, sato_img, meijering_img])

        # Normalize the result to a 0-255 uint8 range for thresholding
        normalized_response = cv2.normalize(combined_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        # Threshold the result to get a binary mask. 30 is a reasonable starting value.
        _, binary_mask = cv2.threshold(normalized_response, 30, 255, cv2.THRESH_BINARY)
        # --- End of new pipeline ---

        bridged_mask = bridge_gaps_in_mask(binary_mask, max_gap_dist)
        masks.append(bridged_mask)

        if worker_thread:
            worker_thread.progress_updated(i + 1, total_images)

    return masks


def create_maximum_intensity_projection(images: List[np.ndarray]) -> Optional[np.ndarray]:
    """Creates a Maximum Intensity Projection image."""
    if not images: return None
    return np.max(np.stack(images, axis=0), axis=0)


# This function is now deprecated in favor of the graph-based approach
def create_temporal_cost_map(masks: List[np.ndarray], obstacle_cost: float) -> Optional[np.ndarray]:
    return None


# This function is now deprecated in favor of the graph-based approach
def create_vessel_layers(mask: np.ndarray, original_mip: np.ndarray) -> Optional[np.ndarray]:
    return None


# --- State Management Enums ---

class AppState(Enum):
    IDLE = auto()
    LOADED = auto()
    MARKING_PATH = auto()
    RANGE_CONFIRMED = auto()
    PROCESSING = auto()
    DONE = auto()


class DrawingMode(Enum):
    NOISE_ROI = auto()


# --- PyQt5 Components ---

class ImageLabel(QLabel):
    """Custom image label supporting mouse clicks and ROI drawing."""
    point_clicked = pyqtSignal(QPoint)
    roi_drawn = pyqtSignal(QRect)

    def __init__(self, parent: 'VesselTracerApp'):
        super().__init__(parent)
        self.main_window = parent
        self.current_pixmap: Optional[QPixmap] = None
        self.current_drawing_roi: Optional[QRect] = None
        self.is_drawing_roi: bool = False

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.setMinimumSize(400, 400)
        self.setAlignment(Qt.AlignCenter)

    def setPixmap(self, pixmap: QPixmap):
        self.current_pixmap = pixmap
        self.update_scaled_pixmap()

    def update_scaled_pixmap(self):
        if self.current_pixmap and not self.current_pixmap.isNull():
            super().setPixmap(self.current_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        self.update_scaled_pixmap()
        super().resizeEvent(event)

    def get_image_coords(self, event_pos: QPoint) -> Optional[QPoint]:
        """Converts window coordinates to original image coordinates."""
        if not self.current_pixmap or self.current_pixmap.isNull():
            return None

        scaled_pixmap = self.current_pixmap.scaled(self.size(), Qt.KeepAspectRatio)
        offset_x = (self.width() - scaled_pixmap.width()) // 2
        offset_y = (self.height() - scaled_pixmap.height()) // 2

        if not (offset_x <= event_pos.x() < offset_x + scaled_pixmap.width() and \
                offset_y <= event_pos.y() < offset_y + scaled_pixmap.height()):
            return None

        if scaled_pixmap.width() == 0 or scaled_pixmap.height() == 0:
            return None

        img_x = (event_pos.x() - offset_x) * self.current_pixmap.width() / scaled_pixmap.width()
        img_y = (event_pos.y() - offset_y) * self.current_pixmap.height() / scaled_pixmap.height()

        return QPoint(int(img_x), int(img_y))

    def mousePressEvent(self, event):
        if event.button() == Qt.LeftButton:
            if self.main_window.app_state == AppState.MARKING_PATH:
                image_coords = self.get_image_coords(event.pos())
                if image_coords:
                    self.point_clicked.emit(image_coords)
            elif self.main_window.drawing_mode is not None:
                start_pos = self.get_image_coords(event.pos())
                if start_pos:
                    self.is_drawing_roi = True
                    self.current_drawing_roi = QRect(start_pos, start_pos)
                    self.update()

    def mouseMoveEvent(self, event):
        if self.is_drawing_roi:
            end_pos = self.get_image_coords(event.pos())
            if end_pos and self.current_drawing_roi:
                self.current_drawing_roi.setBottomRight(end_pos)
                self.update()

    def mouseReleaseEvent(self, event):
        if self.is_drawing_roi and event.button() == Qt.LeftButton:
            self.is_drawing_roi = False
            if self.current_drawing_roi and self.current_drawing_roi.width() > 5 and self.current_drawing_roi.height() > 5:
                self.roi_drawn.emit(self.current_drawing_roi.normalized())
            self.current_drawing_roi = None
            self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.is_drawing_roi or not self.current_drawing_roi:
            return

        painter = QPainter(self)
        if self.main_window.drawing_mode == DrawingMode.NOISE_ROI:
            pen = QPen(QColor(255, 0, 0, 255), 2, Qt.DashLine)
            brush = QBrush(QColor(255, 0, 0, 70))
        else:
            return

        painter.setPen(pen)
        painter.setBrush(brush)

        scaled_pixmap = self.current_pixmap.scaled(self.size(), Qt.KeepAspectRatio)
        offset_x = (self.width() - scaled_pixmap.width()) // 2
        offset_y = (self.height() - scaled_pixmap.height()) // 2

        if self.current_pixmap.width() == 0 or self.current_pixmap.height() == 0: return

        scale_ratio_w = scaled_pixmap.width() / self.current_pixmap.width()
        scale_ratio_h = scaled_pixmap.height() / self.current_pixmap.height()

        display_roi = QRect(
            int(self.current_drawing_roi.x() * scale_ratio_w + offset_x),
            int(self.current_drawing_roi.y() * scale_ratio_h + offset_y),
            int(self.current_drawing_roi.width() * scale_ratio_w),
            int(self.current_drawing_roi.height() * scale_ratio_h)
        )
        painter.drawRect(display_roi.normalized())


class SmoothingPreviewDialog(QDialog):
    """Dialog for previewing the smoothing effect."""

    def __init__(self, base_image: np.ndarray, initial_level: int, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Smoothing Effect Preview")
        self.setMinimumSize(600, 500)
        self.base_image = base_image
        self.smoothing_level = initial_level

        self.layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.image_label)

        control_layout = QHBoxLayout()
        self.slider_label = QLabel(f"Smoothing: {self.smoothing_level}")
        self.slider = QSlider(Qt.Horizontal)
        self.slider.setRange(0, 10)
        self.slider.setValue(self.smoothing_level)
        control_layout.addWidget(self.slider_label)
        control_layout.addWidget(self.slider)
        self.layout.addLayout(control_layout)

        self.button_box = QDialogButtonBox(QDialogButtonBox.Ok | QDialogButtonBox.Cancel)
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

        q_image = QImage(processed_image.data, processed_image.shape[1], processed_image.shape[0],
                         processed_image.shape[1], QImage.Format_Grayscale8)
        pixmap = QPixmap.fromImage(q_image)
        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
        self.image_label.setPixmap(scaled_pixmap)

    def resizeEvent(self, event):
        self.update_preview()
        super().resizeEvent(event)


class StepViewerDialog(QDialog):
    """Dialog to show the image processing and analysis steps."""

    def __init__(self, steps: list, main_window: 'VesselTracerApp', parent=None):
        super().__init__(parent)
        self.setWindowTitle("Processing Step Viewer")
        self.setMinimumSize(800, 600)
        self.steps = steps
        self.main_window = main_window
        self.current_step_index = 0

        self.layout = QVBoxLayout(self)
        self.image_label = QLabel()
        self.image_label.setAlignment(Qt.AlignCenter)
        self.image_label.setSizePolicy(QSizePolicy.Expanding, QSizePolicy.Expanding)
        self.layout.addWidget(self.image_label)

        self.description_label = QLabel()
        self.description_label.setAlignment(Qt.AlignCenter)
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
        step_info = self.steps[self.current_step_index]
        pixmap, description = step_info[0], step_info[1]

        scaled_pixmap = pixmap.scaled(self.image_label.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation)
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
        if len(step_info) > 2 and self.main_window:
            anim_data = step_info[2]
            # This is now handled by the new analysis steps
            pass

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


class ProgressUpdater:
    """Helper class to update a progress dialog from a background thread."""

    def __init__(self, progress_dialog: QProgressDialog):
        self.dialog = progress_dialog
        self.is_running = True

    def progress_updated(self, value: int, total: int):
        if self.dialog.wasCanceled():
            self.is_running = False
            return
        self.dialog.setMaximum(total)
        self.dialog.setValue(value)
        QApplication.processEvents()


# --- Main Application ---

class VesselTracerApp(QMainWindow):
    """The main application window."""
    # --- Tunable Parameters ---
    BG_REMOVAL_THRESHOLD_OFFSET = 15
    BG_REMOVAL_KERNEL_SIZE = 15
    MAX_NODE_SEARCH_RADIUS = 50
    MAX_GAP_BRIDGE_DISTANCE = 30
    GRAPH_LINK_THRESHOLD = 15 # New parameter for graph building

    def __init__(self):
        super().__init__()
        self.setWindowTitle("YC血管追蹤3D (Graph-based)")
        self.setGeometry(100, 100, 1280, 960)
        self.set_stylesheet()

        # --- State Variables ---
        self.images: List[np.ndarray] = []
        self.global_background_color: int = 255
        self.vessel_masks: Optional[List[np.ndarray]] = None
        self.noise_rois: List[QRect] = []
        self.drawing_mode: Optional[DrawingMode] = None
        self.active_thread: Optional[QThread] = None
        self.final_paths: Optional[List[List[VesselNode]]] = None
        self.final_path_image: Optional[np.ndarray] = None
        self.base_mask_projection: Optional[np.ndarray] = None
        self.current_frame_index: int = 0
        self.path_points_info: List[Dict[str, Any]] = []
        self.app_state: AppState = AppState.IDLE
        self.smoothing_level: int = 4
        self.vessel_memory: VesselMemory = VesselMemory()

        self.init_ui()
        self.connect_signals()
        self.update_ui_for_state()

    def set_stylesheet(self):
        """Sets the QSS dark theme style for the application."""
        style = """
            QMainWindow {
                background-color: #2E2E2E;
            }
            QGroupBox {
                background-color: #3C3C3C;
                border: 1px solid #555;
                border-radius: 5px;
                margin-top: 1ex;
                font-size: 14px;
                font-weight: bold;
                color: #E0E0E0;
            }
            QGroupBox::title {
                subcontrol-origin: margin;
                subcontrol-position: top center;
                padding: 0 3px;
                background-color: #3C3C3C;
            }
            QLabel, QStatusBar {
                color: #D0D0D0;
                font-size: 12px;
            }
            QPushButton {
                background-color: #555555;
                color: #EEEEEE;
                border: 1px solid #666666;
                padding: 8px 16px;
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
            QPushButton:disabled {
                background-color: #404040;
                color: #888888;
                border-color: #555555;
            }
            QSlider::groove:horizontal {
                border: 1px solid #4A4A4A;
                height: 8px;
                background: #404040;
                margin: 2px 0;
                border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00A0A0;
                border: 1px solid #00A0A0;
                width: 18px;
                margin: -5px 0;
                border-radius: 9px;
            }
            QSlider::handle:horizontal:disabled {
                background: #777;
                border-color: #777;
            }
            QDialog {
                background-color: #383838;
            }
        """
        self.setStyleSheet(style)

    def init_ui(self):
        """Initializes the UI components."""
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        self.layout = QVBoxLayout(self.central_widget)

        # --- Layout Optimization: Use GroupBoxes for sections ---
        main_controls_layout = QHBoxLayout()

        # Group 1: Load
        group1 = QGroupBox("Step 1: Load Images")
        group1_layout = QHBoxLayout(group1)
        self.btn_select_folder = QPushButton("Select Image Folder")
        group1_layout.addWidget(self.btn_select_folder)
        main_controls_layout.addWidget(group1)

        # Group 2: Settings
        group2 = QGroupBox("Step 2: Mark & Configure")
        group2_layout = QHBoxLayout(group2)
        self.btn_add_noise_roi = QPushButton("Draw Noise Area")
        self.btn_smoothing_preview = QPushButton("Adjust Smoothing")
        group2_layout.addWidget(self.btn_add_noise_roi)
        group2_layout.addWidget(self.btn_smoothing_preview)
        main_controls_layout.addWidget(group2)
        self.group_tools = group2

        # Group 3: Execute
        group3 = QGroupBox("Step 3: Execute")
        group3_layout = QHBoxLayout(group3)
        self.btn_main_action = QPushButton("Start Marking Path")
        group3_layout.addWidget(self.btn_main_action)
        main_controls_layout.addWidget(group3)

        # Group 4: Tools
        group4 = QGroupBox("View & Reset")
        group4_layout = QHBoxLayout(group4)
        self.btn_show_path = QPushButton("Preview Mask")
        self.btn_view_memory = QPushButton("View Memory Graph")
        self.btn_step_view = QPushButton("View Steps")
        self.btn_reset = QPushButton("Reset All")
        group4_layout.addWidget(self.btn_show_path)
        group4_layout.addWidget(self.btn_view_memory)
        group4_layout.addWidget(self.btn_step_view)
        group4_layout.addWidget(self.btn_reset)
        main_controls_layout.addWidget(group4)

        self.layout.addLayout(main_controls_layout)

        # --- Image Frame Navigation ---
        frame_nav_layout = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_info_label = QLabel("Frame: -- / --")
        frame_nav_layout.addWidget(self.frame_slider)
        frame_nav_layout.addWidget(self.frame_info_label)
        self.layout.addLayout(frame_nav_layout)

        # --- Image Display Area ---
        self.image_label = ImageLabel(self)
        self.layout.addWidget(self.image_label, 1)  # Allow image area to take more space

        # --- Status/Info Label ---
        self.info_label = QLabel("Please load an image folder to begin.")
        self.info_label.setAlignment(Qt.AlignCenter)
        font = self.info_label.font()
        font.setPointSize(14)
        self.info_label.setFont(font)
        self.layout.addWidget(self.info_label)

        self.setStatusBar(QStatusBar(self))

        # --- Add Icons ---
        self.btn_select_folder.setIcon(self.style().standardIcon(QStyle.SP_DirOpenIcon))
        self.btn_reset.setIcon(self.style().standardIcon(QStyle.SP_DialogResetButton))
        self.btn_main_action.setIcon(self.style().standardIcon(QStyle.SP_DialogApplyButton))
        self.btn_add_noise_roi.setIcon(self.style().standardIcon(QStyle.SP_CustomBase))  # Placeholder icon
        self.btn_smoothing_preview.setIcon(self.style().standardIcon(QStyle.SP_CustomBase))
        self.btn_view_memory.setIcon(self.style().standardIcon(QStyle.SP_CustomBase))

    def connect_signals(self):
        """Connects all signals and slots."""
        self.btn_select_folder.clicked.connect(self.select_folder)
        self.btn_main_action.clicked.connect(self.handle_main_action)
        self.btn_reset.clicked.connect(self.reset_system)
        self.btn_add_noise_roi.clicked.connect(self.add_noise_roi_mode)
        self.btn_smoothing_preview.clicked.connect(self.open_smoothing_preview)
        self.btn_show_path.clicked.connect(self.show_segmented_path_preview)
        self.btn_step_view.clicked.connect(self.show_step_viewer)
        self.btn_view_memory.clicked.connect(self.show_memory_view)
        self.frame_slider.valueChanged.connect(self.slider_value_changed)
        self.image_label.point_clicked.connect(self.handle_point_selection)
        self.image_label.roi_drawn.connect(self.handle_roi_drawn)

    def show_memory_view(self):
        """Visualizes the entire vessel memory graph."""
        if self.app_state not in [AppState.RANGE_CONFIRMED, AppState.DONE]:
            QMessageBox.information(self, "Info", "Please confirm path points first to build the memory graph.")
            return

        if not self.vessel_memory or not self.vessel_memory.nodes:
            QMessageBox.information(self, "Info", "Vessel memory is empty. Run analysis to build it.")
            return

        self.statusBar().showMessage("Generating memory visualization...")
        QApplication.processEvents()

        if self.images:
            h, w = self.images[0].shape[:2]
            vis_image = visualize_vessel_memory(self.vessel_memory, (h, w))
            self.display_image(vis_image)
            self.statusBar().showMessage("Displaying vessel memory graph.")
        else:
            self.statusBar().showMessage("No image data available for shape.", 3000)

    def update_ui_for_state(self):
        """Updates the UI based on the current application state."""
        is_interactive = self.app_state != AppState.PROCESSING

        # Define UI configurations for each state
        state_configs = {
            AppState.IDLE: {
                "main_action_text": "Start Marking Path", "main_action_enabled": False,
                "info_text": "Click 'Select Image Folder' to begin.", "status_text": "Ready",
                "tools_visible": False, "slider_enabled": False, "select_folder_enabled": True,
                "memory_view_enabled": False
            },
            AppState.LOADED: {
                "main_action_text": "Start Marking Path", "main_action_enabled": True,
                "info_text": "Images loaded. Click on the image to mark start, middle, and end points.",
                "status_text": f"{len(self.images)} images loaded.",
                "tools_visible": False, "slider_enabled": True, "select_folder_enabled": True,
                "memory_view_enabled": False
            },
            AppState.MARKING_PATH: {
                "main_action_text": "Confirm Points", "main_action_enabled": len(self.path_points_info) >= 2,
                "info_text": f"Marked {len(self.path_points_info)} points. Use 'A' and 'D' to switch frames.",
                "status_text": "Marking path...",
                "tools_visible": False, "slider_enabled": True, "select_folder_enabled": False,
                "memory_view_enabled": False
            },
            AppState.RANGE_CONFIRMED: {
                "main_action_text": "Run Full Analysis", "main_action_enabled": True,
                "info_text": f"Points confirmed. Smoothing: {self.smoothing_level}. Draw noise areas or start analysis.",
                "status_text": "Ready for analysis...",
                "tools_visible": True, "slider_enabled": False, "select_folder_enabled": False,
                "memory_view_enabled": True
            },
            AppState.PROCESSING: {
                "main_action_text": "Processing...", "main_action_enabled": False,
                "info_text": "Running analysis, please wait...", "status_text": "Processing...",
                "tools_visible": False, "slider_enabled": False, "select_folder_enabled": False,
                "memory_view_enabled": False
            },
            AppState.DONE: {
                "main_action_text": "Analysis Complete", "main_action_enabled": False,
                "info_text": "Path analysis is complete! Reset to start a new analysis.",
                "status_text": "Done",
                "tools_visible": True, "slider_enabled": False, "select_folder_enabled": False,
                "memory_view_enabled": True
            }
        }

        config = state_configs.get(self.app_state, state_configs[AppState.IDLE])

        self.btn_main_action.setText(config["main_action_text"])
        self.btn_main_action.setEnabled(config["main_action_enabled"] and is_interactive)
        self.info_label.setText(config["info_text"])
        self.statusBar().showMessage(config["status_text"])

        self.group_tools.setVisible(config["tools_visible"])
        self.btn_show_path.setVisible(config["tools_visible"])
        self.btn_step_view.setVisible(config["tools_visible"])
        self.btn_view_memory.setVisible(config["tools_visible"])
        self.btn_view_memory.setEnabled(config.get("memory_view_enabled", False))


        self.frame_slider.setEnabled(config["slider_enabled"])
        self.btn_select_folder.setEnabled(config["select_folder_enabled"] and is_interactive)
        self.btn_reset.setEnabled(is_interactive)

        if self.drawing_mode == DrawingMode.NOISE_ROI and self.app_state == AppState.RANGE_CONFIRMED:
            self.info_label.setText("Drag the mouse on the image to draw a noise area to exclude.")

    def keyPressEvent(self, event):
        """Handles keyboard events (A/D to switch frames)."""
        if not self.images or self.app_state not in [AppState.LOADED, AppState.MARKING_PATH]:
            super().keyPressEvent(event)
            return

        current_idx = self.current_frame_index
        new_idx = -1
        if event.key() == Qt.Key_D:
            new_idx = min(len(self.images) - 1, current_idx + 1)
        elif event.key() == Qt.Key_A:
            new_idx = max(0, current_idx - 1)

        if new_idx != -1 and new_idx != current_idx:
            self.frame_slider.setValue(new_idx)
        else:
            super().keyPressEvent(event)

    def select_folder(self):
        """Selects and loads an image folder."""
        path = QFileDialog.getExistingDirectory(self, "Select Image Folder")
        if path:
            self.reset_system()
            self.images = load_images_from_folder(path)
            if not self.images:
                QMessageBox.warning(self, "Error", "Could not load any images from the selected folder.")
                self.reset_system()
                return

            self.global_background_color = get_most_frequent_color(self.images[0])
            self.current_frame_index = 0
            self.frame_slider.setRange(0, len(self.images) - 1)
            self.frame_slider.setValue(0)
            self.update_frame_display(0)
            self.app_state = AppState.LOADED
            self.update_ui_for_state()

    def slider_value_changed(self, value: int):
        self.update_frame_display(value)
        self.update_ui_for_state()

    def update_frame_display(self, frame_index: int):
        """Updates the displayed image frame."""
        if not self.images or not (0 <= frame_index < len(self.images)):
            return

        self.current_frame_index = frame_index
        self.frame_info_label.setText(f"Frame: {frame_index + 1}/{len(self.images)}")

        base_img = self.images[frame_index].copy()
        display_img = self.get_overlayed_display_image(base_img, frame_index)
        self.display_image(display_img)

    def draw_path_points_on_image(self, image_bgr: np.ndarray, frame_index: int, detailed_color: bool) -> np.ndarray:
        """Draws marked points on the image."""
        for i, p_info in enumerate(self.path_points_info):
            pt = p_info["point"]
            radius = 6
            color = (0, 255, 255)  # Yellow
            thickness = -1  # Solid

            is_on_current_frame = p_info["frame"] == frame_index

            if detailed_color:
                thickness = -1 if is_on_current_frame else 2
                if i == 0:
                    color = (0, 0, 255)  # Start: Red
                elif i == len(self.path_points_info) - 1 and self.app_state != AppState.MARKING_PATH:
                    color = (255, 100, 0)  # End: Blue

            # Outline
            cv2.circle(image_bgr, (pt.x(), pt.y()), radius + 1, (0, 0, 0), -1)
            # Inner circle
            cv2.circle(image_bgr, (pt.x(), pt.y()), radius, color, thickness)
        return image_bgr

    def get_overlayed_display_image(self, base_image_gray: np.ndarray, frame_index: Optional[int]) -> np.ndarray:
        """Gets the display image with overlays."""
        if frame_index is None:
            frame_index = self.current_frame_index

        display_img_bgr = cv2.cvtColor(base_image_gray, cv2.COLOR_GRAY2BGR)
        display_img_bgr = self.draw_path_points_on_image(display_img_bgr, frame_index, detailed_color=True)

        for r in self.noise_rois:
            cv2.rectangle(display_img_bgr, (r.x(), r.y()), (r.x() + r.width(), r.y() + r.height()), (0, 0, 255), 2)

        return display_img_bgr

    def handle_main_action(self):
        """Handles clicks on the main action button."""
        if self.app_state == AppState.LOADED:
            self.app_state = AppState.MARKING_PATH
        elif self.app_state == AppState.MARKING_PATH:
            if not self.path_points_info:
                QMessageBox.warning(self, "Info", "Please mark at least a starting point.")
                return
            if len(self.path_points_info) < 2:
                QMessageBox.warning(self, "Info", "Please mark at least a start and an end point.")
                return
            self.app_state = AppState.RANGE_CONFIRMED
            self.update_range_view()
        elif self.app_state == AppState.RANGE_CONFIRMED:
            self.start_analysis()
        self.update_ui_for_state()

    def handle_point_selection(self, point: QPoint):
        """Handles point clicks on the image (for path marking)."""
        if self.app_state == AppState.MARKING_PATH:
            self.path_points_info.append({"point": point, "frame": self.current_frame_index})
            self.path_points_info.sort(key=lambda p: p['frame'])
            self.update_frame_display(self.current_frame_index)
            self.update_ui_for_state()

    def _get_frame_range(self, for_processing: bool = False) -> Optional[Tuple[int, int]]:
        """
        Gets the frame range defined by marked points.
        - for_processing=True: For backend analysis, always starts from frame 0.
        - for_processing=False: For UI preview, uses the actual range marked by the user.
        """
        if not self.path_points_info:
            return None

        all_frames = [p["frame"] for p in self.path_points_info]
        if not all_frames:
            return None

        end_f = max(all_frames)
        start_f = 0 if for_processing else min(all_frames)

        if start_f > end_f:
            return None

        return start_f, end_f

    def update_range_view(self):
        """Updates to show the Maximum Intensity Projection of the marked range."""
        frame_range = self._get_frame_range(for_processing=False)  # For UI preview, show user-selected range
        if not frame_range: return
        start_f, end_f = frame_range

        range_pip = create_maximum_intensity_projection(self.images[start_f: end_f + 1])
        if range_pip is not None:
            img_with_overlays = self.get_overlayed_display_image(range_pip, -1)  # -1 means don't highlight points from any specific frame
            self.display_image(img_with_overlays)

    def add_noise_roi_mode(self):
        """Enters the mode for drawing noise ROIs."""
        self.drawing_mode = DrawingMode.NOISE_ROI
        self.update_ui_for_state()

    def open_smoothing_preview(self):
        """Opens the smoothing preview dialog."""
        if self.app_state != AppState.RANGE_CONFIRMED:
            return

        frame_range = self._get_frame_range(for_processing=False)  # For UI preview, show user-selected range
        if not frame_range:
            QMessageBox.warning(self, "Error", "Please mark points first to define a preview range.")
            return
        start_f, end_f = frame_range
        pip_image = create_maximum_intensity_projection(self.images[start_f:end_f + 1])

        if pip_image is None:
            QMessageBox.warning(self, "Error", "Could not create a preview image.")
            return

        dialog = SmoothingPreviewDialog(pip_image, self.smoothing_level, self)
        if dialog.exec_() == QDialog.Accepted:
            new_level = dialog.smoothing_level
            if new_level != self.smoothing_level:
                self.smoothing_level = new_level
                # Clear cached mask data as smoothing parameter has changed
                self.vessel_masks = None
                self.base_mask_projection = None
                self.vessel_memory.clear()
                self.layered_vessel_mask = None
                self.statusBar().showMessage(f"Smoothing level set to: {self.smoothing_level}")
                self.update_ui_for_state()

    def handle_roi_drawn(self, roi: QRect):
        """Handles the completion of an ROI drawing."""
        if self.drawing_mode == DrawingMode.NOISE_ROI:
            if self.app_state == AppState.RANGE_CONFIRMED:
                self.noise_rois.append(roi)
                self.info_label.setText(f"Defined {len(self.noise_rois)} noise area(s).")
                self.drawing_mode = None
                # Clear cache
                self.vessel_masks = None
                self.base_mask_projection = None
                self.vessel_memory.clear()
                self.layered_vessel_mask = None
                self.update_range_view()
                self.update_ui_for_state()

    def show_segmented_path_preview(self):
        """Shows a preview of the segmented path."""
        if self.app_state != AppState.RANGE_CONFIRMED: return

        if self.base_mask_projection is not None:
            self.display_image(self.overlay_points_on_image(self.base_mask_projection))
            self.statusBar().showMessage("Showing cached vessel mask.")
            return

        if self.prepare_and_generate_masks():
            self.display_image(self.overlay_points_on_image(self.base_mask_projection))
            self.statusBar().showMessage("Vessel mask generated and displayed.")
        else:
            QMessageBox.warning(self, "Error", "Failed to generate vessel mask.")

    def prepare_and_generate_masks(self) -> bool:
        """Prepares and generates vessel masks."""
        # Core requirement: processing range always starts from frame 0
        frame_range = self._get_frame_range(for_processing=True)
        if not frame_range:
            return False
        start_f, end_f = frame_range

        images_subset = self.images[start_f: end_f + 1]

        progress = QProgressDialog("Generating vessel masks...", "Cancel", 0, len(images_subset), self)
        progress.setWindowModality(Qt.WindowModal)
        updater = ProgressUpdater(progress)

        dominant_bg_color = self.global_background_color

        masks = create_enhanced_vessel_masks(images_subset, self.noise_rois, dominant_bg_color, self,
                                             self.smoothing_level, updater)
        progress.close()

        if masks and updater.is_running:
            self.vessel_masks = masks
            self.base_mask_projection = np.max(np.stack(self.vessel_masks, axis=0), axis=0)

            self.statusBar().showMessage("Building vessel memory graph...")
            QApplication.processEvents()
            self.vessel_memory = build_vessel_memory(self.vessel_masks, self.GRAPH_LINK_THRESHOLD)
            self.statusBar().showMessage("Vessel memory built.")

            return True
        else:
            self.vessel_masks = None
            self.base_mask_projection = None
            self.vessel_memory.clear()
            return False

    def generate_mask_steps(self, image: np.ndarray, smoothing_level: int) -> List[Tuple[np.ndarray, str]]:
        """Generates detailed steps for mask creation on a single image for visualization."""
        if image is None:
            return []
        steps = []
        img = image.copy()
        steps.append((img, "1. Original Image (MIP)"))

        img = remove_large_bright_areas(img, int(self.global_background_color), self.BG_REMOVAL_THRESHOLD_OFFSET,
                                        self.BG_REMOVAL_KERNEL_SIZE)
        steps.append((img, "2. Remove Bright Background Areas"))

        if smoothing_level > 0:
            kernel_size = smoothing_level * 2 + 1
            img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
            steps.append((img, f"3. Gaussian Smoothing (Kernel: {kernel_size}x{kernel_size})"))

        inverted_img = cv2.bitwise_not(img)

        # --- New pipeline: Show filter steps ---
        filter_sigmas = range(1, 6, 2)

        def normalize_for_display(float_img):
            return cv2.normalize(float_img, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        frangi_img = frangi(inverted_img, sigmas=filter_sigmas, black_ridges=False)
        steps.append((normalize_for_display(frangi_img), "4a. Frangi Filter Response"))

        sato_img = sato(inverted_img, sigmas=filter_sigmas, black_ridges=False)
        steps.append((normalize_for_display(sato_img), "4b. Sato Filter Response"))

        meijering_img = meijering(inverted_img, sigmas=filter_sigmas, black_ridges=False)
        steps.append((normalize_for_display(meijering_img), "4c. Meijering Filter Response"))

        combined_response = np.maximum.reduce([frangi_img, sato_img, meijering_img])
        normalized_response = normalize_for_display(combined_response)
        steps.append((normalized_response, "5. Combined Max Filter Response"))

        _, binary_mask = cv2.threshold(normalized_response, 30, 255, cv2.THRESH_BINARY)
        steps.append((binary_mask, "6. Binarization (Fixed Threshold)"))

        bridged_mask = bridge_gaps_in_mask(binary_mask, self.MAX_GAP_BRIDGE_DISTANCE)
        steps.append((bridged_mask, f"7. Gap Bridging (Max Dist: {self.MAX_GAP_BRIDGE_DISTANCE}px)"))
        # --- End of new pipeline ---

        return steps

    def show_step_viewer(self):
        """Shows the step viewer dialog."""
        if self.app_state not in [AppState.RANGE_CONFIRMED, AppState.DONE]: return
        self.statusBar().showMessage("Preparing step viewer...", 5000)
        QApplication.processEvents()

        # For preview, we use the user-selected range, which is more intuitive
        frame_range = self._get_frame_range(for_processing=False)
        if not frame_range: return
        start_f, end_f = frame_range
        image_to_process = create_maximum_intensity_projection(self.images[start_f: end_f + 1])

        step_data = self.generate_mask_steps(image_to_process, self.smoothing_level)

        qt_steps = [(self.convert_np_to_pixmap(img), desc) for img, desc in step_data]
        dialog = StepViewerDialog(qt_steps, self)
        dialog.exec_()
        self.statusBar().showMessage("Ready")

    def overlay_points_on_image(self, base_image: np.ndarray) -> np.ndarray:
        """Overlays points on a base image."""
        bgr_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
        # Use a -1 frame index, meaning don't specially highlight any points
        bgr_image = self.draw_path_points_on_image(bgr_image, -1, detailed_color=False)
        return bgr_image

    def start_analysis(self):
        """Starts the full analysis process using the vessel memory graph."""
        self.app_state = AppState.PROCESSING
        self.update_ui_for_state()

        if not self.vessel_memory or not self.vessel_memory.nodes:
            if not self.prepare_and_generate_masks() or not self.vessel_memory.nodes:
                QMessageBox.warning(self, "Analysis Aborted", "Failed to build vessel memory graph. Cannot continue analysis.")
                self.app_state = AppState.RANGE_CONFIRMED
                self.update_ui_for_state()
                return

        # 1. Snap user-marked points to the nearest nodes in the graph
        path_nodes = []
        all_nodes_found = True
        for p_info in self.path_points_info:
            point = p_info["point"]
            frame = p_info["frame"]
            node = self.vessel_memory.find_closest_node(point.y(), point.x(), frame)
            if node:
                path_nodes.append(node)
            else:
                all_nodes_found = False
                break

        if not all_nodes_found or len(path_nodes) < 2:
            QMessageBox.warning(self, "Pathfinding Failed",
                                "Could not locate all marked points on the vessel graph.\n\n"
                                "Please try:\n"
                                "- Adjusting the smoothing level\n"
                                "- Re-marking points to be closer to vessel centers")
            self.app_state = AppState.RANGE_CONFIRMED
            self.update_ui_for_state()
            return

        # 2. Find path segments between the snapped nodes
        self.statusBar().showMessage("Executing graph pathfinding...", 5000)
        QApplication.processEvents()

        full_path = []
        path_found_for_all_segments = True
        for i in range(len(path_nodes) - 1):
            start_node = path_nodes[i]
            end_node = path_nodes[i+1]

            segment = find_path_on_graph(start_node, end_node)

            if segment is None:
                path_found_for_all_segments = False
                break

            # Add segment to the full path, avoiding duplicate nodes
            full_path.extend(segment if i == 0 else segment[1:])

        self.final_paths = [full_path] if path_found_for_all_segments and full_path else []

        if not self.final_paths:
            QMessageBox.warning(self, "Pathfinding Failed", "Could not find a continuous path between all marked points on the graph.")
            self.app_state = AppState.RANGE_CONFIRMED
        else:
            self.generate_final_path_image()
            self.display_image(self.final_path_image)
            self.app_state = AppState.DONE

        self.update_ui_for_state()

    def generate_final_path_image(self):
        """Generates the final result image with the path found on the graph."""
        frame_range = self._get_frame_range(for_processing=False)
        if not frame_range or not self.images:
            # Fallback shape if images are not loaded
            h, w = (512, 512)
        else:
            h, w = self.images[0].shape[:2]
            start_f, end_f = frame_range
            base_original_pip = create_maximum_intensity_projection(self.images[start_f: end_f + 1])
            self.final_path_image = cv2.cvtColor(base_original_pip, cv2.COLOR_GRAY2BGR)

        if self.final_path_image is None:
             self.final_path_image = np.zeros((h, w, 3), dtype=np.uint8)


        if not self.final_paths or not self.final_paths[0]: return

        path_nodes = self.final_paths[0]

        # Draw the path segments
        for i in range(len(path_nodes) - 1):
            p1 = path_nodes[i]
            p2 = path_nodes[i+1]
            # cv2.line wants (x,y)
            p1_xy = (p1.x, p1.y)
            p2_xy = (p2.x, p2.y)

            # Draw with a black border for visibility
            cv2.line(self.final_path_image, p1_xy, p2_xy, (0, 0, 0), thickness=4)
            cv2.line(self.final_path_image, p1_xy, p2_xy, (50, 255, 50), thickness=2)

        # Redraw marked points on top for clarity
        for i, p_info in enumerate(self.path_points_info):
            pt = p_info["point"]
            radius = 6
            color = (0, 255, 255)  # Yellow
            if i == 0:
                color = (0, 0, 255)  # Red
            elif i == len(self.path_points_info) - 1:
                color = (255, 100, 0)  # Blue
            cv2.circle(self.final_path_image, (pt.x(), pt.y()), radius + 2, (0, 0, 0), -1)
            cv2.circle(self.final_path_image, (pt.x(), pt.y()), radius, color, -1)

    def reset_system(self):
        """Resets all states and data."""
        self.images = []
        self.global_background_color = 255
        self.vessel_masks = None
        self.noise_rois = []
        self.drawing_mode = None
        self.final_paths = None
        self.final_path_image = None
        self.base_mask_projection = None
        self.current_frame_index = 0
        self.path_points_info = []
        self.smoothing_level = 4
        if self.vessel_memory:
            self.vessel_memory.clear()

        self.image_label.setPixmap(QPixmap())
        self.frame_slider.setRange(0, 0)
        self.frame_info_label.setText("Frame: -- / --")

        self.app_state = AppState.IDLE
        self.update_ui_for_state()

    def display_image(self, image_data: np.ndarray):
        """Displays an image on the UI."""
        pixmap = self.convert_np_to_pixmap(image_data)
        self.image_label.setPixmap(pixmap)

    def convert_np_to_pixmap(self, image_data: np.ndarray) -> QPixmap:
        """Converts a NumPy array to a QPixmap."""
        if image_data is None: return QPixmap()

        if image_data.dtype != np.uint8:
            image_data = np.clip(image_data, 0, 255).astype(np.uint8)

        image_data = np.ascontiguousarray(image_data)
        q_image = None

        if len(image_data.shape) == 2:  # Grayscale
            h, w = image_data.shape
            q_image = QImage(image_data.data, w, h, w, QImage.Format_Grayscale8)
        elif len(image_data.shape) == 3:
            h, w, ch = image_data.shape
            if ch == 3:  # BGR (OpenCV) to RGB
                image_data_rgb = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
                q_image = QImage(image_data_rgb.data, w, h, w * ch, QImage.Format_RGB888)
            elif ch == 4:  # BGRA to RGBA
                image_data_rgba = cv2.cvtColor(image_data, cv2.COLOR_BGRA2RGBA)
                q_image = QImage(image_data_rgba.data, w, h, w * ch, QImage.Format_RGBA8888)

        return QPixmap.fromImage(q_image) if q_image else QPixmap()

    def closeEvent(self, event):
        """Cleans up before closing the application."""
        self.reset_system()
        event.accept()


if __name__ == '__main__':
    try:
        app = QApplication(sys.argv)
        main_window = VesselTracerApp()
        main_window.show()
        sys.exit(app.exec_())
    except Exception as e:
        QMessageBox.critical(None, "Fatal Error", f"The application encountered an unrecoverable error: {e}")
