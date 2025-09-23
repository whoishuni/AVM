import sys
import os
import re
import numpy as np
import heapq
import math
from collections import deque
from enum import Enum, auto
from typing import Optional, List, Tuple, Dict, Any

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
    """Provides a key for natural sorting of filenames (e.g., 'img10.jpg' after 'img2.jpg').

    Args:
        s: The string to generate a sort key for.

    Returns:
        A list of strings and integers used for sorting.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
    """Loads a sequence of grayscale images from a folder, sorted naturally.

    Args:
        folder_path: The path to the directory containing the images.

    Returns:
        A list of images as NumPy arrays, or an empty list if an error occurs.
    """
    images = []
    valid_extensions = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff"}
    try:
        filenames = os.listdir(folder_path)
        filenames.sort(key=natural_sort_key)
        for filename in filenames:
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                img_path = os.path.join(folder_path, filename)
                # Use np.fromfile to handle non-ASCII paths, then imdecode
                img_array = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
        return images
    except Exception as e:
        print(f"Error loading images: {e}")
        return []


def get_most_frequent_color(image: np.ndarray) -> int:
    """Gets the most frequent pixel value in an image, assumed to be the background.

    Args:
        image: The input image as a NumPy array.

    Returns:
        The most frequent pixel value (0-255). Returns 255 if the image is None.
    """
    if image is None: return 255
    unique, counts = np.unique(image, return_counts=True)
    return unique[np.argmax(counts)]


def bridge_gaps_in_mask(mask: np.ndarray, max_distance: int = 15) -> np.ndarray:
    """Intelligently connects separated vessel segments in a binary mask.

    This method is more accurate than simple dilation. It skeletonizes the mask,
    finds endpoints of the skeleton lines, and connects the closest pair of
    endpoints that belong to different contours, provided they are within
    `max_distance`.

    Args:
        mask: The binary (0 or 255) vessel mask as a NumPy array.
        max_distance: The maximum pixel distance to bridge between two endpoints.

    Returns:
        A new mask with gaps bridged, or a copy of the original if no bridging occurs.
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
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    convolved = cv2.filter2D(skeleton, -1, kernel)
    endpoints_map = np.zeros_like(skeleton)
    endpoints_map[(convolved == 11) & (skeleton > 0)] = 255

    # 4. Associate endpoints with their original contours
    endpoint_coords = np.argwhere(endpoints_map > 0)
    if len(endpoint_coords) < 2:
        return bridged_mask  # Not enough endpoints to connect

    endpoints_with_contour_info = []
    for y, x in endpoint_coords:
        for i, cnt in enumerate(valid_contours):
            if cv2.pointPolygonTest(cnt, (int(x), int(y)), False) >= 0:
                endpoints_with_contour_info.append({'point': (y, x), 'contour_idx': i})
                break

    # 5. Find the closest pair of endpoints from different contours and connect them
    for i in range(len(endpoints_with_contour_info)):
        for j in range(i + 1, len(endpoints_with_contour_info)):
            ep1 = endpoints_with_contour_info[i]
            ep2 = endpoints_with_contour_info[j]

            if ep1['contour_idx'] != ep2['contour_idx']:
                p1_yx = ep1['point']
                p2_yx = ep2['point']
                dist = np.linalg.norm(np.array(p1_yx) - np.array(p2_yx))

                if dist < max_distance:
                    p1_xy = (int(p1_yx[1]), int(p1_yx[0]))
                    p2_xy = (int(p2_yx[1]), int(p2_yx[0]))
                    cv2.line(bridged_mask, p1_xy, p2_xy, 255, 1)

    return bridged_mask


def remove_large_bright_areas(image: np.ndarray, bg_color: int, threshold_offset: int = 15,
                              kernel_size: int = 15) -> np.ndarray:
    """Removes large bright areas from the image to reduce background interference.

    This is useful for removing large, non-vessel structures (like catheters or
    bone) that are brighter than the background but not part of the vasculature.

    Args:
        image: The input grayscale image.
        bg_color: The background color of the image.
        threshold_offset: Value subtracted from `bg_color` to set the brightness threshold.
        kernel_size: The size of the morphological kernel used to identify large areas.

    Returns:
        The processed image with large bright areas replaced by `bg_color`.
    """
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
    """Generates a sequence of enhanced binary vessel masks from raw images.

    This function applies a multi-stage pipeline to each image:
    1. Removes large, bright background areas.
    2. Applies Gaussian smoothing.
    3. Masks out user-defined noise regions.
    4. Inverts the image and applies Frangi, Sato, and Meijering vessel enhancement filters.
    5. Combines the filter responses and thresholds the result into a binary mask.
    6. Bridges small gaps in the final mask.

    Args:
        images: A list of the raw grayscale images.
        noise_rois: A list of QRects defining areas to exclude from processing.
        bg_color: The dominant background color of the images.
        app_instance: The main application instance to access parameters.
        smoothing_level: The level of Gaussian blur to apply (0 for none).
        worker_thread: An optional updater for reporting progress to the UI.

    Returns:
        A list of binary vessel masks, or None if the process was canceled.
    """
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
    """Creates a Maximum Intensity Projection (MIP) image from a sequence.

    The MIP is an image where each pixel takes the maximum intensity value from
    the corresponding pixels across all images in the sequence.

    Args:
        images: A list of images as NumPy arrays.

    Returns:
        A single MIP image, or None if the input list is empty.
    """
    if not images: return None
    return np.max(np.stack(images, axis=0), axis=0)


def create_temporal_cost_map(masks: List[np.ndarray], obstacle_cost: float) -> Optional[np.ndarray]:
    """Creates a cost map where the cost is related to the frame number.

    This map is used in pathfinding to penalize paths that jump between frames
    that are far apart in time. The cost of a vessel pixel is its frame index,
    encouraging the path to stay within vessels that appear early and persist.

    Args:
        masks: A list of binary vessel masks for the sequence.
        obstacle_cost: The high cost value to assign to non-vessel pixels.

    Returns:
        A 2D cost map, or None if the input list is empty.
    """
    if not masks: return None
    h, w = masks[0].shape
    cost_map = np.full((h, w), obstacle_cost, dtype=np.float32)

    for frame_idx, mask in enumerate(masks):
        vessel_pixels = mask > 0
        # Cost is proportional to the frame number, encouraging the path to stay in similar frames
        cost_map[vessel_pixels] = np.minimum(cost_map[vessel_pixels], frame_idx)

    return cost_map


def create_vessel_layers(mask: np.ndarray, original_mip: np.ndarray) -> Optional[np.ndarray]:
    """Divides the vessel mask into 10 layers based on original image brightness.

    Brighter vessels in the original MIP are assigned to higher layers (e.g.,
    layer 10), while dimmer vessels are in lower layers. This can help to
    separate overlapping vessels based on their intensity.

    Args:
        mask: The binary vessel mask.
        original_mip: The Maximum Intensity Projection of the original images.

    Returns:
        A layered mask where pixel values (1-10) correspond to the brightness
        layer, or None if the inputs are invalid.
    """
    if mask is None or original_mip is None or np.sum(mask) == 0:
        return None

    layered_mask = np.zeros_like(mask, dtype=np.uint8)
    vessel_locations = mask > 0

    # Get brightness values from the vessel regions
    brightness_values = original_mip[vessel_locations]
    if brightness_values.size == 0:
        return layered_mask

    # Divide vessels into 10 layers (1 to 10) based on brightness
    # Use percentiles for more robust binning, avoiding outliers
    bins = np.percentile(brightness_values, np.linspace(0, 100, 11))
    bins[-1] += 1  # Ensure the maximum value is included

    # Assign pixels to layers based on brightness
    for i in range(10):
        # Find pixels within the current brightness bin
        layer_pixels_indices = (original_mip >= bins[i]) & (original_mip < bins[i + 1]) & vessel_locations
        layered_mask[layer_pixels_indices] = i + 1

    # Clean each layer using morphological operations to remove small noise
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_layered_mask = np.zeros_like(layered_mask)
    for i in range(1, 11):
        layer_mask = np.where(layered_mask == i, 255, 0).astype(np.uint8)
        opened_mask = cv2.morphologyEx(layer_mask, cv2.MORPH_OPEN, kernel)
        cleaned_layered_mask[opened_mask == 255] = i

    return cleaned_layered_mask


def generate_path_coherence_map(start_node: Tuple[int, int], vessel_mask: np.ndarray, original_mip: np.ndarray, app_instance: 'VesselTracerApp') -> np.ndarray:
    """Generates a map where each pixel's value represents path coherence from a start node.

    This is done using a Dijkstra-like search where the 'cost' is a measure of
    'incoherence', penalizing turns and changes in brightness. The final map
    is an inverse of these costs, so high values mean high coherence.

    Args:
        start_node: The (y, x) starting point for the exploration.
        vessel_mask: The binary mask of all vessels.
        original_mip: The original MIP image, used for brightness checks.
        app_instance: The main application instance to access parameters.

    Returns:
        A float32 NumPy array representing the coherence map.
    """
    if vessel_mask[start_node] == 0:
        return np.zeros(vessel_mask.shape, dtype=np.float32)

    costs = np.full(vessel_mask.shape, np.inf, dtype=np.float32)
    costs[start_node] = 0
    pq = [(0, start_node)]  # (cost, (y, x))
    came_from = {}

    while pq:
        cost, current = heapq.heappop(pq)

        if cost > costs[current]:
            continue

        parent = came_from.get(current)

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue

                neighbor = (current[0] + dr, current[1] + dc)

                if not (0 <= neighbor[0] < vessel_mask.shape[0] and 0 <= neighbor[1] < vessel_mask.shape[1]) or \
                   vessel_mask[neighbor] == 0:
                    continue

                # Calculate incoherence cost for this step
                incoherence = 0

                # 1. Turn penalty
                if parent:
                    v_in = (current[0] - parent[0], current[1] - parent[1])
                    v_out = (neighbor[0] - current[0], neighbor[1] - current[1])
                    mag_in = math.sqrt(v_in[0] ** 2 + v_in[1] ** 2)
                    mag_out = math.sqrt(v_out[0] ** 2 + v_out[1] ** 2)
                    if mag_in > 0 and mag_out > 0:
                        dot = v_in[0] * v_out[0] + v_in[1] * v_out[1]
                        cosine_similarity = dot / (mag_in * mag_out)
                        incoherence += app_instance.COHERENCE_TURN_PENALTY * (1.0 - cosine_similarity)

                # 2. Color/Intensity change penalty
                color_diff = abs(int(original_mip[current]) - int(original_mip[neighbor]))
                incoherence += app_instance.COHERENCE_COLOR_CHANGE_PENALTY * (color_diff / 255.0)

                # 3. Movement cost
                move_cost = math.sqrt(dr**2 + dc**2)

                new_cost = costs[current] + move_cost + incoherence
                if new_cost < costs[neighbor]:
                    costs[neighbor] = new_cost
                    came_from[neighbor] = current
                    heapq.heappush(pq, (new_cost, neighbor))

    # Convert costs to a coherence map (inverse relationship, handle inf)
    coherence_map = np.zeros_like(costs)
    valid_costs = costs[costs != np.inf]
    if len(valid_costs) > 0:
        max_cost = np.max(valid_costs)
        # Invert cost to get coherence, adding epsilon to avoid division by zero
        coherence_map[costs != np.inf] = max_cost - costs[costs != np.inf]

    # Normalize to 0-1 range
    min_val, max_val = np.min(coherence_map), np.max(coherence_map)
    if max_val > min_val:
        coherence_map = (coherence_map - min_val) / (max_val - min_val)

    return coherence_map


def identify_main_vessels(mask: np.ndarray, thickness_threshold: int) -> np.ndarray:
    """Identifies main vessel trunks based on their thickness.

    Args:
        mask: A binary mask of the entire vessel structure.
        thickness_threshold: The minimum radius a pixel must have to be
                             considered part of a main vessel.

    Returns:
        A binary mask highlighting only the main vessels.
    """
    if mask is None or np.sum(mask) == 0:
        return np.zeros_like(mask)

    # Use distance transform to get the radius of the vessel at each point
    dist_transform = cv2.distanceTransform(mask, cv2.DIST_L2, 5)

    # Threshold the distance map to find areas of sufficient thickness
    _, main_vessels_mask = cv2.threshold(dist_transform, thickness_threshold, 255, cv2.THRESH_BINARY)

    return main_vessels_mask.astype(np.uint8)

def build_vessel_identity_map(masks: List[np.ndarray], main_vessel_mask: np.ndarray) -> Optional[np.ndarray]:
    """
    Builds a map that assigns a unique, persistent ID to each vessel segment across frames.

    This function tracks vessel segments from frame to frame. If a segment in one frame
    overlaps with a segment in the next, they are considered the same vessel and share
    the same ID. This creates a "memory" of the vessel structure's growth.

    A key feature is the natural branching rule: new, independent vessels are only
    allowed to form if they originate from a pre-identified "main vessel" trunk.

    Args:
        masks: A list of binary vessel masks, one for each frame in the sequence.
        main_vessel_mask: A binary mask identifying the thickest "trunk" vessels.

    Returns:
        An optional 2D NumPy array of the same dimensions as the input masks. Each
        pixel corresponding to a vessel is assigned an integer ID unique to that
        vessel structure across all frames. Returns None if the input is empty.
    """
    if not masks:
        return None

    h, w = masks[0].shape
    identity_map = np.zeros((h, w), dtype=np.int32)
    next_vessel_id = 1

    # Process the first frame to initialize vessel identities
    if np.any(masks[0]):
        num_labels, labels = cv2.connectedComponents(masks[0])
        for label_idx in range(1, num_labels): # Skip background label 0
            component_mask = (labels == label_idx)
            identity_map[component_mask] = next_vessel_id
            next_vessel_id += 1

    # Process subsequent frames
    for i in range(1, len(masks)):
        # Get the new components from the current frame that are not in the previous one
        new_growth_mask = cv2.subtract(masks[i], masks[i-1])

        if not np.any(new_growth_mask):
            continue

        num_labels, labels = cv2.connectedComponents(new_growth_mask)

        for label_idx in range(1, num_labels):
            component_mask = (labels == label_idx)

            # To link a new component, we check its boundary against the existing identity map
            kernel = np.ones((3,3), np.uint8)
            eroded_component = cv2.erode(component_mask.astype(np.uint8), kernel, iterations=1)
            boundary_mask = component_mask & ~eroded_component.astype(bool)

            overlap_pixels = identity_map[boundary_mask]
            overlapping_ids = np.unique(overlap_pixels[overlap_pixels > 0])

            if len(overlapping_ids) > 0:
                # This component is a continuation of an existing vessel.
                unique_ids, counts = np.unique(overlapping_ids, return_counts=True)
                chosen_id = unique_ids[np.argmax(counts)]
                identity_map[component_mask] = chosen_id
            else:
                # This is a new, disjoint vessel. Apply the natural branching rule.
                # It's only a valid new branch if it originates from a main vessel trunk.
                dilated_component = cv2.dilate(component_mask.astype(np.uint8), kernel, iterations=1)
                if np.any((dilated_component > 0) & (main_vessel_mask > 0)):
                    identity_map[component_mask] = next_vessel_id
                    next_vessel_id += 1
                # Otherwise, this component is considered noise and is not given an ID.

    return identity_map

# --- State Management Enums ---

class AppState(Enum):
    """Defines the possible states of the application's finite state machine."""
    IDLE = auto()             # Application is waiting for images to be loaded.
    LOADED = auto()           # Images are loaded, ready for user interaction.
    MARKING_PATH = auto()     # User is actively marking points on the image.
    RANGE_CONFIRMED = auto()  # User has confirmed points, ready for configuration or analysis.
    PROCESSING = auto()       # Application is busy with a background task (e.g., mask generation).
    DONE = auto()             # Analysis is complete and results are shown.


class DrawingMode(Enum):
    """Defines the available drawing modes for the user."""
    NOISE_ROI = auto()        # User is drawing a rectangle to define a noise area.


class AnalysisWorker(QThread):
    """A QThread worker for running analysis tasks in the background."""
    analysis_complete = pyqtSignal(dict)

    def __init__(self, app_instance, start_point):
        super().__init__()
        self.app = app_instance
        self.start_point = start_point
        self.is_running = True

    def run(self):
        """Runs the analysis pipeline."""
        results = {"success": False}
        # Call prepare_and_generate_masks without a worker_thread to prevent UI creation
        if self.app.base_mask_projection is None:
            # Pass worker_thread=None to prevent UI creation from background thread
            if not self.app.prepare_and_generate_masks(worker_thread=None):
                results["error"] = "Mask generation was canceled or failed during pre-analysis."
                self.analysis_complete.emit(results)
                return

        start_node = self.app.find_closest_pixel_on_mask(self.start_point, self.app.base_mask_projection)
        if not start_node:
            results["error"] = "Point Not on Vessel"
            self.analysis_complete.emit(results)
            return

        full_range_mip = create_maximum_intensity_projection(self.app.images)
        coherence_map = generate_path_coherence_map(start_node, self.app.base_mask_projection, full_range_mip, self.app)

        results["success"] = True
        results["coherence_map"] = coherence_map
        results["start_node"] = start_node
        self.analysis_complete.emit(results)

    def stop(self):
        self.is_running = False


# --- PyQt5 Components ---

class ImageLabel(QLabel):
    """A custom QLabel for displaying images with interactive capabilities.

    This label handles scaling pixmaps to fit its size, converting mouse click
    coordinates from window space to image space, and managing the drawing of
    rectangular regions of interest (ROIs).

    Signals:
        point_clicked (pyqtSignal): Emitted when the user clicks on the image,
                                    providing the QPoint in image coordinates.
        roi_drawn (pyqtSignal): Emitted when the user finishes drawing a
                                rectangular ROI, providing the QRect.
    """
    point_clicked = pyqtSignal(QPoint)
    roi_drawn = pyqtSignal(QRect)

    def __init__(self, parent: 'VesselTracerApp'):
        """Initializes the ImageLabel.

        Args:
            parent: The parent widget, typically the main application window.
        """
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
        """Sets the pixmap to be displayed and triggers a rescale.

        Args:
            pixmap: The QPixmap to display in the label.
        """
        self.current_pixmap = pixmap
        self.update_scaled_pixmap()

    def update_scaled_pixmap(self):
        """Rescales the current pixmap to fit the label size while maintaining aspect ratio."""
        if self.current_pixmap and not self.current_pixmap.isNull():
            super().setPixmap(self.current_pixmap.scaled(self.size(), Qt.KeepAspectRatio, Qt.SmoothTransformation))

    def resizeEvent(self, event):
        """Handles the widget's resize event to rescale the pixmap.

        Args:
            event: The QResizeEvent.
        """
        self.update_scaled_pixmap()
        super().resizeEvent(event)

    def get_image_coords(self, event_pos: QPoint) -> Optional[QPoint]:
        """Converts window coordinates to original, unscaled image coordinates.

        Args:
            event_pos: The QPoint of the mouse event in widget coordinates.

        Returns:
            A QPoint in the original image's coordinate system, or None if the
            click was outside the displayed pixmap area.
        """
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
        """Handles the start of a mouse click or ROI drawing.

        Args:
            event: The QMouseEvent.
        """
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
        """Handles mouse movement during ROI drawing to update the rectangle.

        Args:
            event: The QMouseEvent.
        """
        if self.is_drawing_roi:
            end_pos = self.get_image_coords(event.pos())
            if end_pos and self.current_drawing_roi:
                self.current_drawing_roi.setBottomRight(end_pos)
                self.update()

    def mouseReleaseEvent(self, event):
        """Handles the end of an ROI drawing, emitting the final rectangle.

        Args:
            event: The QMouseEvent.
        """
        if self.is_drawing_roi and event.button() == Qt.LeftButton:
            self.is_drawing_roi = False
            if self.current_drawing_roi and self.current_drawing_roi.width() > 5 and self.current_drawing_roi.height() > 5:
                self.roi_drawn.emit(self.current_drawing_roi.normalized())
            self.current_drawing_roi = None
            self.update()

    def paintEvent(self, event):
        """Draws the ROI rectangle on top of the image during creation.

        Args:
            event: The QPaintEvent.
        """
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
    """A dialog window for interactively previewing Gaussian blur smoothing.

    This dialog shows a preview of an image with a variable amount of Gaussian
    blur applied, controlled by a slider. This allows the user to choose an
    appropriate smoothing level before running the full analysis.
    """

    def __init__(self, base_image: np.ndarray, initial_level: int, parent=None):
        """Initializes the smoothing preview dialog.

        Args:
            base_image: The original image (as a NumPy array) to apply smoothing to.
            initial_level: The starting value for the smoothing slider (0-10).
            parent: The parent widget.
        """
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
        """Updates the image preview when the slider value changes."""
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
        """Handles the resize event to ensure the preview image scales correctly.

        Args:
            event: The QResizeEvent.
        """
        self.update_preview()
        super().resizeEvent(event)


class StepViewerDialog(QDialog):
    """A dialog for viewing the sequential steps of image processing and analysis.

    This window displays a series of images, each representing a step in the
    processing pipeline (e.g., filtering, masking, pathfinding). The user can
    navigate back and forth through these steps.
    """

    def __init__(self, steps: list, main_window: 'VesselTracerApp', parent=None):
        """Initializes the step viewer dialog.

        Args:
            steps: A list of tuples, where each tuple contains a QPixmap for the
                   step's image and a string description.
            main_window: A reference to the main application window, used for
                         triggering animation replays.
            parent: The parent widget.
        """
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
        """Updates the displayed image and description to the current step."""
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
        """Triggers the pathfinding animation replay in the main window."""
        step_info = self.steps[self.current_step_index]
        if len(step_info) > 2 and self.main_window:
            anim_data = step_info[2]
            self.main_window.replay_path_animation(anim_data)

    def prev_step(self):
        """Navigates to the previous step."""
        if self.current_step_index > 0:
            self.current_step_index -= 1
            self.update_view()

    def next_step(self):
        """Navigates to the next step."""
        if self.current_step_index < len(self.steps) - 1:
            self.current_step_index += 1
            self.update_view()

    def resizeEvent(self, event):
        """Handles the resize event to scale the currently displayed step image.

        Args:
            event: The QResizeEvent.
        """
        self.update_view()
        super().resizeEvent(event)


class ProgressUpdater:
    """A helper class to update a QProgressDialog from a background thread.

    This class provides a simple interface to update a progress dialog's value
    and check for cancellation without directly passing the dialog object into
    the processing function, which can be cleaner.
    """

    def __init__(self, progress_dialog: QProgressDialog):
        """Initializes the ProgressUpdater.

        Args:
            progress_dialog: The QProgressDialog instance to be controlled.
        """
        self.dialog = progress_dialog
        self.is_running = True

    def progress_updated(self, value: int, total: int):
        """Updates the progress dialog and checks for cancellation.

        If the dialog has been canceled by the user, the `is_running` flag is
        set to False.

        Args:
            value: The current progress value.
            total: The maximum progress value.
        """
        if self.dialog.wasCanceled():
            self.is_running = False
            return
        self.dialog.setMaximum(total)
        self.dialog.setValue(value)
        QApplication.processEvents()


# --- Main Application ---

class VesselTracerApp(QMainWindow):
    """The main application window for 2D vessel tracing.

    This class manages the application's state, UI, and the core logic for
    image processing and path analysis. It follows a state machine pattern
    defined by the AppState enum.

    Attributes:
        BG_REMOVAL_THRESHOLD_OFFSET (int): Parameter for background removal.
        BG_REMOVAL_KERNEL_SIZE (int): Kernel size for background removal morphology.
        MAX_NODE_SEARCH_RADIUS (int): Max distance to search for a vessel pixel near a click.
        MAX_GAP_BRIDGE_DISTANCE (int): Max distance for bridging gaps in vessel masks.
        FORBIDDEN_ZONE_RADIUS (int): Radius to prevent A* from immediately backtracking.
        TIME_COST_WEIGHT (float): Weight for the temporal cost in A* pathfinding.
        PATHFINDING_OBSTACLE_COST (float): High cost for pixels not on the vessel mask.
        TURN_PENALTY_WEIGHT (float): Weight for the turn penalty in A* pathfinding.
        CROSS_VESSEL_PENALTY (float): High penalty for jumping between different vessels.
    """
    # --- Tunable Parameters ---
    BG_REMOVAL_THRESHOLD_OFFSET = 15
    BG_REMOVAL_KERNEL_SIZE = 15
    MAX_NODE_SEARCH_RADIUS = 50
    MAX_GAP_BRIDGE_DISTANCE = 30
    FORBIDDEN_ZONE_RADIUS = 30
    TIME_COST_WEIGHT = 1.0
    PATHFINDING_OBSTACLE_COST = 1e9
    TURN_PENALTY_WEIGHT = 50.0
    CROSS_VESSEL_PENALTY = 1e6
    MAIN_VESSEL_THICKNESS_THRESHOLD = 5
    # --- Coherence Map Parameters ---
    COHERENCE_TURN_PENALTY = 5.0
    COHERENCE_COLOR_CHANGE_PENALTY = 10.0
    COHERENCE_MAP_WEIGHT = 50.0

    def __init__(self):
        """Initializes the main application window, state variables, and UI."""
        super().__init__()
        self.setWindowTitle("YC血管追蹤2D")
        self.setGeometry(100, 100, 1280, 960)
        self.set_stylesheet()

        # --- State Variables ---
        self.images: List[np.ndarray] = []
        self.global_background_color: int = 255
        self.vessel_masks: Optional[List[np.ndarray]] = None
        self.layered_vessel_mask: Optional[np.ndarray] = None
        self.vessel_identity_map: Optional[np.ndarray] = None
        self.main_vessel_mask: Optional[np.ndarray] = None
        self.path_coherence_map: Optional[np.ndarray] = None
        self.noise_rois: List[QRect] = []
        self.drawing_mode: Optional[DrawingMode] = None
        self.active_thread: Optional[QThread] = None
        self.final_paths: Optional[List[List[Tuple[int, int]]]] = None
        self.final_path_image: Optional[np.ndarray] = None
        self.base_mask_projection: Optional[np.ndarray] = None
        self.temporal_cost_map: Optional[np.ndarray] = None
        self.current_frame_index: int = 0
        self.path_points_info: List[Dict[str, Any]] = []
        self.app_state: AppState = AppState.IDLE
        self.smoothing_level: int = 4

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
        """Initializes and arranges all UI components in the main window."""
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
        self.btn_step_view = QPushButton("View Steps")
        self.btn_reset = QPushButton("Reset All")
        group4_layout.addWidget(self.btn_show_path)
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

    def connect_signals(self):
        """Connects all widget signals to their corresponding slots."""
        self.btn_select_folder.clicked.connect(self.select_folder)
        self.btn_main_action.clicked.connect(self.handle_main_action)
        self.btn_reset.clicked.connect(self.reset_system)
        self.btn_add_noise_roi.clicked.connect(self.add_noise_roi_mode)
        self.btn_smoothing_preview.clicked.connect(self.open_smoothing_preview)
        self.btn_show_path.clicked.connect(self.show_segmented_path_preview)
        self.btn_step_view.clicked.connect(self.show_step_viewer)
        self.frame_slider.valueChanged.connect(self.slider_value_changed)
        self.image_label.point_clicked.connect(self.handle_point_selection)
        self.image_label.roi_drawn.connect(self.handle_roi_drawn)

    def update_ui_for_state(self):
        """Updates the UI element states (text, enabled/disabled) based on the current AppState."""
        is_interactive = self.app_state != AppState.PROCESSING

        # Define UI configurations for each state
        state_configs = {
            AppState.IDLE: {
                "main_action_text": "Start Marking Path", "main_action_enabled": False,
                "info_text": "Click 'Select Image Folder' to begin.", "status_text": "Ready",
                "tools_visible": False, "slider_enabled": False, "select_folder_enabled": True
            },
            AppState.LOADED: {
                "main_action_text": "Start Marking Path", "main_action_enabled": True,
                "info_text": "Images loaded. Click on the image to mark start, middle, and end points.",
                "status_text": f"{len(self.images)} images loaded.",
                "tools_visible": False, "slider_enabled": True, "select_folder_enabled": True
            },
            AppState.MARKING_PATH: {
                "main_action_text": "Confirm Points", "main_action_enabled": len(self.path_points_info) >= 2,
                "info_text": (f"Start point set. Pre-analysis complete. Please mark your end point." if len(self.path_points_info) == 1 else f"Marked {len(self.path_points_info)} points. Click 'Confirm Points' when done."),
                "status_text": "Marking path...",
                "tools_visible": False, "slider_enabled": True, "select_folder_enabled": False
            },
            AppState.RANGE_CONFIRMED: {
                "main_action_text": "Run Full Analysis", "main_action_enabled": True,
                "info_text": f"Points confirmed. Smoothing: {self.smoothing_level}. Draw noise areas or start analysis.",
                "status_text": "Ready for analysis...",
                "tools_visible": True, "slider_enabled": False, "select_folder_enabled": False
            },
            AppState.PROCESSING: {
                "main_action_text": "Processing...", "main_action_enabled": False,
                "info_text": "Running analysis, please wait...", "status_text": "Processing...",
                "tools_visible": False, "slider_enabled": False, "select_folder_enabled": False
            },
            AppState.DONE: {
                "main_action_text": "Analysis Complete", "main_action_enabled": False,
                "info_text": "Path analysis is complete! Reset to start a new analysis.",
                "status_text": "Done",
                "tools_visible": True, "slider_enabled": False, "select_folder_enabled": False
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

        self.frame_slider.setEnabled(config["slider_enabled"])
        self.btn_select_folder.setEnabled(config["select_folder_enabled"] and is_interactive)
        self.btn_reset.setEnabled(is_interactive)

        if self.drawing_mode == DrawingMode.NOISE_ROI and self.app_state == AppState.RANGE_CONFIRMED:
            self.info_label.setText("Drag the mouse on the image to draw a noise area to exclude.")

    def keyPressEvent(self, event):
        """Handles keyboard events for frame navigation.

        'A' moves to the previous frame, 'D' moves to the next frame.

        Args:
            event: The QKeyEvent.
        """
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
        """Opens a dialog to select an image folder and loads the images."""
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
        """Slot for the frame slider's valueChanged signal.

        Args:
            value: The new slider value (frame index).
        """
        self.update_frame_display(value)
        self.update_ui_for_state()

    def update_frame_display(self, frame_index: int):
        """Updates the main image display to show a specific frame.

        Args:
            frame_index: The index of the image to display.
        """
        if not self.images or not (0 <= frame_index < len(self.images)):
            return

        self.current_frame_index = frame_index
        self.frame_info_label.setText(f"Frame: {frame_index + 1}/{len(self.images)}")

        base_img = self.images[frame_index].copy()
        display_img = self.get_overlayed_display_image(base_img, frame_index)
        self.display_image(display_img)

    def draw_path_points_on_image(self, image_bgr: np.ndarray, frame_index: int, detailed_color: bool) -> np.ndarray:
        """Draws marked path points (start, middle, end) onto an image.

        Args:
            image_bgr: The BGR image (NumPy array) to draw on.
            frame_index: The index of the current frame being displayed.
            detailed_color: If True, uses different colors for start/end points
                            and highlights points on the current frame.

        Returns:
            The image with points drawn on it.
        """
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
        """Creates a display image by overlaying points and ROIs on a base image.

        Args:
            base_image_gray: The base grayscale image.
            frame_index: The current frame index, used for highlighting points.
                         If None, uses the currently stored frame index.

        Returns:
            A BGR image with all overlays drawn.
        """
        if frame_index is None:
            frame_index = self.current_frame_index

        display_img_bgr = cv2.cvtColor(base_image_gray, cv2.COLOR_GRAY2BGR)
        display_img_bgr = self.draw_path_points_on_image(display_img_bgr, frame_index, detailed_color=True)

        for r in self.noise_rois:
            cv2.rectangle(display_img_bgr, (r.x(), r.y()), (r.x() + r.width(), r.y() + r.height()), (0, 0, 255), 2)

        return display_img_bgr

    def handle_main_action(self):
        """Handles clicks on the main action button, progressing the application's state."""
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
        """Handles a point selection event from the ImageLabel.

        If it's the first point, it triggers the coherence map pre-analysis
        in a background thread. Otherwise, it just adds the point to the list.

        Args:
            point: The QPoint where the user clicked, in image coordinates.
        """
        if self.app_state != AppState.MARKING_PATH or self.active_thread is not None:
            return

        # --- New Workflow: Pre-analysis on first point ---
        if not self.path_points_info:
            self.app_state = AppState.PROCESSING
            self.update_ui_for_state()
            self.info_label.setText("First point marked. Running background pre-analysis of vessel structure...")

            # Store the point temporarily so the worker can access it
            self.path_points_info.append({"point": point, "frame": self.current_frame_index})

            self.active_thread = AnalysisWorker(self, point)
            self.active_thread.analysis_complete.connect(self.on_pre_analysis_complete)
            self.active_thread.start()
        else:
            # This is a subsequent point (middle or end).
            self.path_points_info.append({"point": point, "frame": self.current_frame_index})
            self.path_points_info.sort(key=lambda p: p['frame'])
            self.update_frame_display(self.current_frame_index)
            self.update_ui_for_state()

    def on_pre_analysis_complete(self, results: dict):
        """Handles the completion of the background pre-analysis task."""
        self.active_thread = None

        if not results.get("success"):
            error_msg = results.get("error", "An unknown error occurred during pre-analysis.")
            QMessageBox.warning(self, "Pre-analysis Failed", error_msg)
            # Clear the bad start point
            self.path_points_info = []
            self.app_state = AppState.MARKING_PATH # Return to marking state
            self.update_ui_for_state()
            return

        # Store the results from the worker
        self.path_coherence_map = results["coherence_map"]
        start_node = results["start_node"]

        # The original point is the first one in the list.
        original_point = self.path_points_info[0]['point']
        original_frame = self.path_points_info[0]['frame']

        # Now that pre-analysis is done, update the first point's info with the snapped node
        self.path_points_info = [{"point": original_point, "frame": original_frame, "node": start_node}]

        self.app_state = AppState.MARKING_PATH
        self.update_ui_for_state()
        self.info_label.setText("Start point set. Pre-analysis complete. Please mark your end point.")
        self.update_frame_display(self.current_frame_index)


    def _get_frame_range(self, for_processing: bool = False) -> Optional[Tuple[int, int]]:
        """Gets the frame range defined by the earliest and latest marked points.

        Args:
            for_processing: If True, the start frame is always 0 for analysis.
                            If False, uses the actual earliest marked frame for UI previews.

        Returns:
            A tuple of (start_frame, end_frame), or None if no points are marked.
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
        """Updates the image display to show the MIP of the selected frame range."""
        frame_range = self._get_frame_range(for_processing=False)  # For UI preview, show user-selected range
        if not frame_range: return
        start_f, end_f = frame_range

        range_pip = create_maximum_intensity_projection(self.images[start_f: end_f + 1])
        if range_pip is not None:
            img_with_overlays = self.get_overlayed_display_image(range_pip, -1)  # -1 means don't highlight points from any specific frame
            self.display_image(img_with_overlays)

    def add_noise_roi_mode(self):
        """Enters the mode for drawing noise ROIs on the image."""
        self.drawing_mode = DrawingMode.NOISE_ROI
        self.update_ui_for_state()

    def open_smoothing_preview(self):
        """Opens the smoothing preview dialog to adjust the smoothing level."""
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
                self.temporal_cost_map = None
                self.layered_vessel_mask = None
                self.vessel_identity_map = None
                self.statusBar().showMessage(f"Smoothing level set to: {self.smoothing_level}")
                self.update_ui_for_state()

    def handle_roi_drawn(self, roi: QRect):
        """Handles the completion of an ROI drawing from the ImageLabel.

        Adds the ROI to the list of noise areas and clears cached data.

        Args:
            roi: The QRect of the drawn noise area.
        """
        if self.drawing_mode == DrawingMode.NOISE_ROI:
            if self.app_state == AppState.RANGE_CONFIRMED:
                self.noise_rois.append(roi)
                self.info_label.setText(f"Defined {len(self.noise_rois)} noise area(s).")
                self.drawing_mode = None
                # Clear cache
                self.vessel_masks = None
                self.base_mask_projection = None
                self.temporal_cost_map = None
                self.layered_vessel_mask = None
                self.vessel_identity_map = None
                self.update_range_view()
                self.update_ui_for_state()

    def show_segmented_path_preview(self):
        """Shows a preview of the generated vessel mask, with a progress dialog."""
        if self.app_state != AppState.RANGE_CONFIRMED: return

        if self.base_mask_projection is not None:
            self.display_image(self.overlay_points_on_image(self.base_mask_projection))
            self.statusBar().showMessage("Showing cached vessel mask.")
            return

        # This is a user-facing action, so create a progress dialog.
        frame_range = self._get_frame_range(for_processing=True)
        if not frame_range: return
        num_images = frame_range[1] - frame_range[0] + 1

        progress = QProgressDialog("Generating vessel masks...", "Cancel", 0, num_images, self)
        progress.setWindowModality(Qt.WindowModal)
        updater = ProgressUpdater(progress)

        was_successful = self.prepare_and_generate_masks(worker_thread=updater)
        progress.close() # Ensure dialog is closed regardless of outcome

        if was_successful:
            self.display_image(self.overlay_points_on_image(self.base_mask_projection))
            self.statusBar().showMessage("Vessel mask generated and displayed.")
        elif updater.is_running: # Don't show error if user canceled
            QMessageBox.warning(self, "Error", "Failed to generate vessel mask.")

    def prepare_and_generate_masks(self, worker_thread: Optional['ProgressUpdater'] = None) -> bool:
        """Prepares and generates all derived data like masks and cost maps.

        This function runs the main pre-processing pipeline. It can be run with
        a ProgressUpdater to show a dialog, or without for silent background processing.

        Args:
            worker_thread: An optional updater to report progress to a UI dialog.

        Returns:
            True if mask generation was successful and not canceled, False otherwise.
        """
        frame_range = self._get_frame_range(for_processing=True)
        if not frame_range:
            return False
        start_f, end_f = frame_range

        images_subset = self.images[start_f: end_f + 1]
        dominant_bg_color = self.global_background_color

        masks = create_enhanced_vessel_masks(images_subset, self.noise_rois, dominant_bg_color, self,
                                             self.smoothing_level, worker_thread)

        if masks and (worker_thread is None or worker_thread.is_running):
            self.vessel_masks = masks
            self.base_mask_projection = np.max(np.stack(self.vessel_masks, axis=0), axis=0)
            self.main_vessel_mask = identify_main_vessels(self.base_mask_projection, self.MAIN_VESSEL_THICKNESS_THRESHOLD)
            self.temporal_cost_map = create_temporal_cost_map(self.vessel_masks, self.PATHFINDING_OBSTACLE_COST)
            self.vessel_identity_map = build_vessel_identity_map(self.vessel_masks, self.main_vessel_mask)
            return True
        else:
            # Clear caches if generation fails or is canceled
            self.vessel_masks = None
            self.base_mask_projection = None
            self.temporal_cost_map = None
            self.vessel_identity_map = None
            self.main_vessel_mask = None
            return False

    def generate_mask_steps(self, image: np.ndarray, smoothing_level: int) -> List[Tuple[np.ndarray, str]]:
        """Generates a list of (image, description) tuples for visualizing the mask creation process.

        Args:
            image: The input image to process (typically a MIP).
            smoothing_level: The smoothing level to use for the demonstration.

        Returns:
            A list of tuples, where each is (step_image, step_description).
        """
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
        """Shows the step-by-step processing viewer dialog."""
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
        """Overlays marked path points onto a given base image.

        Args:
            base_image: The grayscale base image.

        Returns:
            A BGR image with the points drawn on it.
        """
        bgr_image = cv2.cvtColor(base_image, cv2.COLOR_GRAY2BGR)
        # Use a -1 frame index, meaning don't specially highlight any points
        bgr_image = self.draw_path_points_on_image(bgr_image, -1, detailed_color=False)
        return bgr_image

    def start_analysis(self):
        """Starts the full analysis pipeline after user confirmation."""
        self.app_state = AppState.PROCESSING
        self.update_ui_for_state()

        if self.base_mask_projection is None or self.temporal_cost_map is None:
            # This is a user-facing action, so create a progress dialog.
            frame_range = self._get_frame_range(for_processing=True)
            if not frame_range:
                self.app_state = AppState.RANGE_CONFIRMED; self.update_ui_for_state(); return
            num_images = frame_range[1] - frame_range[0] + 1

            progress = QProgressDialog("Generating vessel masks for analysis...", "Cancel", 0, num_images, self)
            progress.setWindowModality(Qt.WindowModal)
            updater = ProgressUpdater(progress)

            was_successful = self.prepare_and_generate_masks(worker_thread=updater)
            progress.close()

            if not was_successful:
                if updater.is_running: # Don't show error if user canceled
                    QMessageBox.warning(self, "Analysis Aborted", "Failed to generate vessel mask. Cannot continue analysis.")
                self.app_state = AppState.RANGE_CONFIRMED
                self.update_ui_for_state()
                return

        final_mask = self.base_mask_projection
        cost_map = self.temporal_cost_map

        if np.sum(final_mask) == 0:
            QMessageBox.warning(self, "Analysis Aborted", "The generated vessel mask is empty. No path can be found.")
            self.app_state = AppState.RANGE_CONFIRMED
            self.update_ui_for_state()
            return

        # The start node was already found during pre-analysis.
        mask_pixels = [self.path_points_info[0]["node"]]
        all_pixels_found = True
        # Find subsequent points
        for p_info in self.path_points_info[1:]:
            pixel = self.find_closest_pixel_on_mask(p_info["point"], final_mask)
            if pixel:
                mask_pixels.append(pixel)
            else:
                all_pixels_found = False
                break

        if not all_pixels_found or not mask_pixels or len(mask_pixels) < 2:
            QMessageBox.warning(self, "Pathfinding Failed",
                                "Could not locate all marked points on the vessel mask.\n\n"
                                "Please try:\n"
                                "- Adjusting the smoothing level\n"
                                "- Re-marking points to be closer to vessel centers\n"
                                "- Drawing noise areas to exclude interference")
            self.app_state = AppState.RANGE_CONFIRMED
            self.update_ui_for_state()
            return

        pathfinding_costmap = np.full(cost_map.shape, self.PATHFINDING_OBSTACLE_COST, dtype=np.float32)
        pathfinding_costmap[final_mask > 0] = 0  # Base cost is 0
        pathfinding_costmap += cost_map  # Add time cost

        self.show_full_analysis_steps(final_mask, mask_pixels, pathfinding_costmap)

    def replay_path_animation(self, anim_data: dict):
        """Replays the A* pathfinding search animation from the step viewer.

        Args:
            anim_data: A dictionary containing the data needed for the animation,
                       including the costmap, start/end pixels, and base image.
        """
        self.statusBar().showMessage("Replaying pathfinding animation...")
        cost_map = anim_data["costmap"]
        pixels = anim_data["pixels"]
        base_image = anim_data["baseimage"]
        identity_map = anim_data["identity_map"]
        coherence_map = anim_data["coherence_map"]

        def update_visualization(visited):
            temp_img = base_image.copy()
            for node in visited:
                temp_img[node[0], node[1]] = (100, 0, 0)  # Blue
            self.display_image(temp_img)
            QApplication.processEvents()

        full_path = []
        for i in range(len(pixels) - 1):
            start_node = pixels[i]
            end_node = pixels[i + 1]

            current_costmap = cost_map.copy()
            if i > 0:  # Add a forbidden zone to prevent the path from going backward
                prev_node = pixels[i - 1]
                cv2.circle(current_costmap, (prev_node[1], prev_node[0]), self.FORBIDDEN_ZONE_RADIUS,
                           self.PATHFINDING_OBSTACLE_COST, -1)

            segment = self.find_path_astar(current_costmap, start_node, end_node, identity_map, coherence_map,
                                           viz_callback=update_visualization)
            if segment is None:
                # If not found with forbidden zone, try again without it
                segment = self.find_path_astar(cost_map, start_node, end_node, identity_map, coherence_map,
                                               viz_callback=update_visualization)

            if segment:
                full_path.extend(segment if i == 0 else segment[1:])

        path_points = np.array(full_path, dtype=np.int32).reshape(-1, 1, 2)
        path_points = path_points[:, :, ::-1]  # (y,x) to (x,y)
        cv2.polylines(base_image, [path_points], isClosed=False, color=(50, 255, 50), thickness=2)
        self.display_image(base_image)
        self.statusBar().showMessage("Animation replay finished.", 3000)

    def show_full_analysis_steps(self, final_mask, mask_pixels, pathfinding_costmap):
        """Prepares and shows the final step-by-step analysis results dialog.

        This method assembles all visualization steps, runs the final A*
        pathfinding, and displays the results in a StepViewerDialog.

        Args:
            final_mask: The final binary vessel mask.
            mask_pixels: The list of user-marked points, snapped to the mask.
            pathfinding_costmap: The cost map for the A* algorithm.
        """
        self.statusBar().showMessage("Preparing full analysis steps...", 5000)
        QApplication.processEvents()

        steps = []
        # For visualization, we use the user-selected range, which is more intuitive
        frame_range = self._get_frame_range(for_processing=False)
        if not frame_range:
            self.statusBar().showMessage("Error: Could not determine frame range")
            self.app_state = AppState.RANGE_CONFIRMED
            self.update_ui_for_state()
            return
        start_f, end_f = frame_range
        base_original_pip = create_maximum_intensity_projection(self.images[start_f: end_f + 1])

        # 1. Mask Generation Steps
        mask_steps_data = self.generate_mask_steps(base_original_pip, self.smoothing_level)
        for img, desc in mask_steps_data:
            steps.append((self.convert_np_to_pixmap(img), f"Mask Generation - {desc}"))

        # Added: Vessel Layering Step
        self.layered_vessel_mask = create_vessel_layers(final_mask, base_original_pip)
        if self.layered_vessel_mask is not None:
            # Normalize to 0-255 for color mapping
            normalized_layers = cv2.normalize(self.layered_vessel_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            layer_heatmap = cv2.applyColorMap(normalized_layers, cv2.COLORMAP_JET)
            layer_heatmap[self.layered_vessel_mask == 0] = [0, 0, 0]  # Set background to black
            steps.append((self.convert_np_to_pixmap(layer_heatmap), "Vessel Layering (Bright=Top, Dark=Bottom)"))

        # NEW: Vessel Identity Map Visualization
        if self.vessel_identity_map is not None:
            max_id = np.max(self.vessel_identity_map)
            if max_id > 0:
                # Use a colormap to give each vessel ID a unique color
                norm_ids = cv2.normalize(self.vessel_identity_map, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
                identity_heatmap = cv2.applyColorMap(norm_ids, cv2.COLORMAP_JET)
                identity_heatmap[self.vessel_identity_map == 0] = [0, 0, 0]  # Set background to black
                steps.append((self.convert_np_to_pixmap(identity_heatmap), "Vessel Identity Map (Memory)"))


        # 2. Cost Map Heatmap
        display_costmap = pathfinding_costmap.copy()
        valid_pixels = display_costmap < self.PATHFINDING_OBSTACLE_COST
        if np.any(valid_pixels):
            min_val = np.min(display_costmap[valid_pixels])
            max_val = np.max(display_costmap[valid_pixels])
            if max_val > min_val:
                normalized_map = 255 * (display_costmap - min_val) / (max_val - min_val)
                normalized_map[~valid_pixels] = 0
                heatmap = cv2.applyColorMap(normalized_map.astype(np.uint8), cv2.COLORMAP_JET)
                heatmap[~valid_pixels] = [0, 0, 0]
                steps.append((self.convert_np_to_pixmap(heatmap), "Temporal Cost Map (Blue = Lower Cost)"))

        # 3. Marked Points and Path Search Animation
        path_base_image = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
        for i, p in enumerate(mask_pixels):
            color = (0, 255, 255)  # Yellow
            if i == 0:
                color = (0, 0, 255)  # Red
            elif i == len(mask_pixels) - 1:
                color = (255, 100, 0)  # Blue
            cv2.circle(path_base_image, (p[1], p[0]), 5, color, -1)
        steps.append((self.convert_np_to_pixmap(path_base_image), "Located Marked Points on Mask"))

        self.statusBar().showMessage("Executing pathfinding...", 5000)
        QApplication.processEvents()

        exploration_img = path_base_image.copy()

        full_path = []
        path_found_for_all_segments = True
        for i in range(len(mask_pixels) - 1):
            start_node = mask_pixels[i]
            end_node = mask_pixels[i + 1]

            current_costmap = pathfinding_costmap.copy()
            if i > 0:
                prev_node = mask_pixels[i - 1]
                cv2.circle(current_costmap, (prev_node[1], prev_node[0]), self.FORBIDDEN_ZONE_RADIUS,
                           self.PATHFINDING_OBSTACLE_COST, -1)

            # Don't visualize in real-time during the loop to speed things up
            segment = self.find_path_astar(current_costmap, start_node, end_node, self.vessel_identity_map, self.path_coherence_map, viz_callback=None)

            if segment is None:
                # Try again without the forbidden zone
                segment = self.find_path_astar(pathfinding_costmap, start_node, end_node, self.vessel_identity_map, self.path_coherence_map,
                                               viz_callback=None)

            if segment is None:
                path_found_for_all_segments = False
                break

            full_path.extend(segment if i == 0 else segment[1:])

        self.final_paths = [full_path] if path_found_for_all_segments and full_path else []

        if not self.final_paths:
            QMessageBox.warning(self, "Pathfinding Failed",
                                "Could not find a continuous path between all marked points.\n\n"
                                "<b>Recommended Actions:</b>\n"
                                "1. <b>Adjust Smoothing</b>: Change this in 'Mark & Configure' to alter mask connectivity.\n"
                                "2. <b>Use Noise Areas</b>: If there's background interference, use 'Draw Noise Area' to exclude it.\n"
                                "3. <b>Check Marked Points</b>: Ensure points are within clear vessel structures.")

            path_img = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
            steps.append((self.convert_np_to_pixmap(path_img), "Pathfinding Failed"))
            dialog = StepViewerDialog(steps, self)
            dialog.exec_()
            self.app_state = AppState.RANGE_CONFIRMED
            self.update_ui_for_state()
            self.info_label.setText("Pathfinding failed. Please adjust parameters and try again.")
            return

        # 4. Display Path Search Result
        path_points_yx = np.array(full_path, dtype=np.int32).reshape(-1, 1, 2)
        path_points_xy = path_points_yx[:, :, ::-1]
        cv2.polylines(exploration_img, [path_points_xy], isClosed=False, color=(50, 255, 50), thickness=2)

        anim_data = {"type": "animation", "costmap": pathfinding_costmap, "pixels": mask_pixels,
                     "baseimage": path_base_image, "identity_map": self.vessel_identity_map,
                     "coherence_map": self.path_coherence_map}
        steps.append((self.convert_np_to_pixmap(exploration_img), "A* Algorithm Search Result (Click Replay)", anim_data))

        # 5. Final Result
        self.generate_final_path_image(base_original_pip)
        steps.append((self.convert_np_to_pixmap(self.final_path_image), "Final Result"))

        dialog = StepViewerDialog(steps, self)
        dialog.exec_()

        self.display_image(self.final_path_image)
        self.app_state = AppState.DONE
        self.update_ui_for_state()

    def generate_final_path_image(self, base_original_pip: np.ndarray):
        """Generates the final result image with the path drawn on the original MIP.

        Args:
            base_original_pip: The Maximum Intensity Projection of the original images.
        """
        if base_original_pip is None:
            self.final_path_image = np.zeros((512, 512, 3), dtype=np.uint8)
        else:
            # Use the original MIP directly without histogram equalization for a more authentic look
            self.final_path_image = cv2.cvtColor(base_original_pip, cv2.COLOR_GRAY2BGR)

        if not self.final_paths or not self.final_paths[0]: return

        path = self.final_paths[0]
        path_points_yx = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
        path_points_xy = path_points_yx[:, :, ::-1]

        # Draw the path (with a black border for visibility)
        cv2.polylines(self.final_path_image, [path_points_xy], isClosed=False, color=(0, 0, 0), thickness=4)
        cv2.polylines(self.final_path_image, [path_points_xy], isClosed=False, color=(50, 255, 50), thickness=2)

        # Redraw marked points on top
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

    def find_path_astar(self, cost_map, start, end, vessel_identity_map, coherence_map, viz_callback=None):
        """Finds the optimal path between two points using the A* algorithm.

        This implementation is heavily guided by a pre-computed path coherence map,
        which encodes penalties for turns and brightness changes relative to the
        start point. It also includes costs for distance, time, and jumping
        between different vessel structures.

        Args:
            cost_map: The base cost map (incorporating temporal cost).
            start: The starting (y, x) coordinate tuple.
            end: The ending (y, x) coordinate tuple.
            vessel_identity_map: The map assigning a unique ID to each vessel.
            coherence_map: A map of path coherence scores from the start point.
            viz_callback: An optional function to call for visualizing the search.

        Returns:
            A list of (y, x) tuples representing the path, or None if no path is found.
        """
        if cost_map[start] >= self.PATHFINDING_OBSTACLE_COST or cost_map[end] >= self.PATHFINDING_OBSTACLE_COST:
            return None

        def heuristic(p1, p2):
            return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        open_set = [(heuristic(start, end), 0, start)]  # f_cost, g_cost, pos
        g_costs = {start: 0}
        came_from = {}
        closed_set = set()

        node_counter = 0
        viz_interval = 50

        while open_set:
            _, g_cost, current = heapq.heappop(open_set)

            if current == end:
                if viz_callback:
                    viz_callback(list(g_costs.keys()))
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start)
                path.reverse()
                return path

            closed_set.add(current)

            node_counter += 1
            if viz_callback and node_counter % viz_interval == 0:
                viz_callback(list(g_costs.keys()))


            for dr in [-1, 0, 1]:
                for dc in [-1, 0, 1]:
                    if dr == 0 and dc == 0: continue
                    neighbor = (current[0] + dr, current[1] + dc)

                    if not (0 <= neighbor[0] < cost_map.shape[0] and 0 <= neighbor[1] < cost_map.shape[1]) or \
                            cost_map[neighbor] >= self.PATHFINDING_OBSTACLE_COST or \
                            neighbor in closed_set:
                        continue

                    # --- Cost Calculation ---
                    # 1. Base movement cost
                    move_cost = np.sqrt(dr**2 + dc**2)

                    # 2. Temporal cost from pre-calculated map
                    time_cost = self.TIME_COST_WEIGHT * cost_map[neighbor]

                    # 3. Vessel identity penalty
                    cross_vessel_penalty = 0
                    if vessel_identity_map is not None:
                        current_id = vessel_identity_map[current]
                        neighbor_id = vessel_identity_map[neighbor]
                        if current_id > 0 and neighbor_id > 0 and current_id != neighbor_id:
                            cross_vessel_penalty = self.CROSS_VESSEL_PENALTY

                    # 4. Coherence cost (higher coherence = lower cost)
                    # This now implicitly handles turn penalties relative to the start point.
                    coherence_cost = self.COHERENCE_MAP_WEIGHT * (1.0 - coherence_map[neighbor])

                    # 5. Standard Turn Penalty (still useful for local smoothness)
                    turn_penalty = 0
                    parent = came_from.get(current)
                    if parent:
                        v_in = (current[0] - parent[0], current[1] - parent[1])
                        v_out = (neighbor[0] - current[0], neighbor[1] - current[1])
                        mag_in = math.sqrt(v_in[0] ** 2 + v_in[1] ** 2)
                        mag_out = math.sqrt(v_out[0] ** 2 + v_out[1] ** 2)
                        if mag_in > 0 and mag_out > 0:
                            dot = v_in[0] * v_out[0] + v_in[1] * v_out[1]
                            # Clamp cosine to avoid math domain errors with float precision
                            cosine_similarity = min(1.0, max(-1.0, dot / (mag_in * mag_out)))
                            turn_penalty = self.TURN_PENALTY_WEIGHT * (1.0 - cosine_similarity)

                    new_g_cost = g_costs[current] + move_cost + time_cost + cross_vessel_penalty + coherence_cost + turn_penalty

                    if new_g_cost < g_costs.get(neighbor, np.inf):
                        g_costs[neighbor] = new_g_cost
                        f_cost = new_g_cost + heuristic(neighbor, end)
                        heapq.heappush(open_set, (f_cost, new_g_cost, neighbor))
                        came_from[neighbor] = current

        return None

    def find_closest_pixel_on_mask(self, point: QPoint, mask_img: np.ndarray) -> Optional[Tuple[int, int]]:
        """Finds the closest white pixel on a binary mask to a given point.

        The search is limited to a maximum radius defined by `MAX_NODE_SEARCH_RADIUS`.

        Args:
            point: The QPoint (in image coordinates) to search from.
            mask_img: The binary mask to search within.

        Returns:
            A tuple (y, x) of the closest mask pixel, or None if no pixel is
            found within the search radius.
        """
        if mask_img is None or np.sum(mask_img) == 0: return None

        valid_points = np.argwhere(mask_img > 0)
        if valid_points.size == 0: return None

        point_coords = np.array([point.y(), point.x()])
        distances = np.linalg.norm(valid_points - point_coords, axis=1)
        min_dist_idx = np.argmin(distances)

        if distances[min_dist_idx] <= self.MAX_NODE_SEARCH_RADIUS:
            return tuple(valid_points[min_dist_idx])
        else:
            return None

    def reset_system(self):
        """Resets all application states, data, and UI elements to their initial values."""
        self.images = []
        self.global_background_color = 255
        self.vessel_masks = None
        self.layered_vessel_mask = None
        self.vessel_identity_map = None
        self.main_vessel_mask = None
        self.path_coherence_map = None
        self.noise_rois = []
        self.drawing_mode = None
        self.final_paths = None
        self.final_path_image = None
        self.base_mask_projection = None
        self.temporal_cost_map = None
        self.current_frame_index = 0
        self.path_points_info = []
        self.smoothing_level = 4  # Reset to default smoothing level

        self.image_label.setPixmap(QPixmap())
        self.frame_slider.setRange(0, 0)
        self.frame_info_label.setText("Frame: -- / --")

        self.app_state = AppState.IDLE
        self.update_ui_for_state()

    def display_image(self, image_data: np.ndarray):
        """Displays a NumPy array image in the main image label.

        Args:
            image_data: The image to display.
        """
        pixmap = self.convert_np_to_pixmap(image_data)
        self.image_label.setPixmap(pixmap)

    def convert_np_to_pixmap(self, image_data: np.ndarray) -> QPixmap:
        """Converts a NumPy array image to a QPixmap for display.

        Handles grayscale, BGR, and BGRA images.

        Args:
            image_data: The input image as a NumPy array.

        Returns:
            The converted QPixmap, or an empty QPixmap if conversion fails.
        """
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
        """Handles the application close event to ensure clean shutdown.

        Args:
            event: The QCloseEvent.
        """
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