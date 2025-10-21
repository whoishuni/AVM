import os
import re
import numpy as np
import cv2
from enum import Enum, auto
from typing import List, Optional, Tuple

# --- PyQt6 Imports ---
from PyQt6.QtGui import QPixmap, QImage, QIcon
from PyQt6.QtCore import QPoint

# --- Enums for State Management ---

class AppState(Enum):
    """Defines the possible states of the application's finite state machine."""
    IDLE = auto()             # Waiting for images.
    LOADED = auto()           # Images loaded, ready for interaction.
    ANALYZING_FLOW = auto()   # Analyzing the flow animation.
    FLOW_ANALYZED = auto()    # Flow is analyzed, ready to play animation or mark points.
    MARKING_PATH = auto()     # User is actively marking points.
    RANGE_CONFIRMED = auto()  # Points confirmed, ready for analysis configuration.
    PROCESSING = auto()       # Busy with a background task.
    DONE = auto()             # Analysis complete, results are shown.

class DrawingMode(Enum):
    """Defines the available drawing modes for the user."""
    MARKING = auto()          # Marking points on the vessel.
    NOISE_ROI = auto()        # Drawing a rectangle to define a noise area.
    PAN = auto()              # Panning the image.

# --- Image and File I/O Functions ---

def natural_sort_key(s: str) -> list:
    """
    Provides a key for natural sorting of filenames (e.g., 'img10.jpg' after 'img2.jpg').
    This is essential for correctly ordering image sequences.
    """
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]

def load_images_from_folder(folder_path: str) -> List[np.ndarray]:
    """
    Loads a sequence of grayscale images from a folder, sorted naturally.

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
                # Use np.fromfile and imdecode to handle non-ASCII paths correctly.
                img_array = np.fromfile(img_path, dtype=np.uint8)
                img = cv2.imdecode(img_array, cv2.IMREAD_GRAYSCALE)
                if img is not None:
                    images.append(img)
        return images
    except Exception as e:
        print(f"Error loading images from '{folder_path}': {e}")
        return []

# --- Image Analysis and Conversion ---

def get_most_frequent_color(image: np.ndarray) -> int:
    """
    Gets the most frequent pixel value in an image, assumed to be the background.

    Args:
        image: The input image as a NumPy array.

    Returns:
        The most frequent pixel value (0-255). Returns 255 if the image is invalid.
    """
    if image is None:
        return 255
    unique, counts = np.unique(image, return_counts=True)
    return unique[np.argmax(counts)]

def convert_np_to_pixmap(image_data: np.ndarray) -> QPixmap:
    """
    Converts a NumPy array image to a QPixmap for display in PyQt.

    Handles grayscale, BGR, and BGRA images, ensuring correct format conversion.

    Args:
        image_data: The input image as a NumPy array.

    Returns:
        The converted QPixmap, or an empty QPixmap if conversion fails.
    """
    if image_data is None:
        return QPixmap()

    # Ensure the data type is uint8, which is required by QImage.
    if image_data.dtype != np.uint8:
        image_data = np.clip(image_data, 0, 255).astype(np.uint8)

    # Ensure data is contiguous in memory for QImage.
    image_data = np.ascontiguousarray(image_data)
    h, w = image_data.shape[:2]

    if len(image_data.shape) == 2:  # Grayscale
        q_image = QImage(image_data.data, w, h, w, QImage.Format.Format_Grayscale8)
    elif len(image_data.shape) == 3 and image_data.shape[2] == 3:  # BGR (from OpenCV)
        # Convert BGR to RGB for correct display in PyQt.
        rgb_image = cv2.cvtColor(image_data, cv2.COLOR_BGR2RGB)
        q_image = QImage(rgb_image.data, w, h, w * 3, QImage.Format.Format_RGB888)
    elif len(image_data.shape) == 3 and image_data.shape[2] == 4: # BGRA
        rgba_image = cv2.cvtColor(image_data, cv2.COLOR_BGRA2RGBA)
        q_image = QImage(rgba_image.data, w, h, w * 4, QImage.Format.Format_RGBA8888)
    else:
        return QPixmap()

    return QPixmap.fromImage(q_image)

# --- Pathfinding and Geometry ---

def find_closest_pixel_on_mask(point: QPoint, mask_img: np.ndarray, max_radius: int) -> Optional[Tuple[int, int]]:
    """
    Finds the closest white pixel on a binary mask to a given point.

    The search is limited to a maximum radius to prevent incorrect matches.

    Args:
        point: The QPoint (in image coordinates) to search from.
        mask_img: The binary mask to search within.
        max_radius: The maximum pixel distance to search.

    Returns:
        A tuple (y, x) of the closest mask pixel, or None if no pixel is
        found within the search radius.
    """
    if mask_img is None or np.sum(mask_img) == 0:
        return None

    # Get coordinates of all non-zero pixels in the mask.
    valid_points = np.argwhere(mask_img > 0)
    if valid_points.size == 0:
        return None

    # Calculate the distance from the clicked point to all valid mask points.
    point_coords = np.array([point.y(), point.x()])
    distances = np.linalg.norm(valid_points - point_coords, axis=1)
    min_dist_idx = np.argmin(distances)

    # Return the point only if it's within the allowed radius.
    if distances[min_dist_idx] <= max_radius:
        return tuple(valid_points[min_dist_idx])
    else:
        return None

def create_yc_icon() -> QIcon:
    """Creates a dynamic QIcon with the letters 'YC'."""
    try:
        from PIL import Image, ImageDraw, ImageFont

        # Create a 256x256 image with a transparent background
        image = Image.new("RGBA", (256, 256), (0, 0, 0, 0))
        draw = ImageDraw.Draw(image)

        # Define colors and font
        bg_color = (30, 144, 255)  # Dodger Blue
        text_color = (255, 255, 255) # White

        # Draw a rounded rectangle background
        draw.rounded_rectangle(
            (10, 10, 246, 246),
            radius=40,
            fill=bg_color
        )

        # Select a font
        try:
            # Use a common, modern-looking font
            font = ImageFont.truetype("arialbd.ttf", 150)
        except IOError:
            # Fallback to default font if Arial Bold is not found
            font = ImageFont.load_default()

        # Calculate text position to center it
        text = "YC"
        # Use textbbox for more accurate centering
        bbox = draw.textbbox((0, 0), text, font=font)
        text_width = bbox[2] - bbox[0]
        text_height = bbox[3] - bbox[1]
        position = ((256 - text_width) / 2, (256 - text_height) / 2 - 15) # Minor vertical adjustment

        # Draw the text
        draw.text(position, text, font=font, fill=text_color)

        # Convert PIL image to QImage
        q_image = QImage(image.tobytes("raw", "RGBA"), image.width, image.height, QImage.Format.Format_RGBA8888)

        return QIcon(QPixmap.fromImage(q_image))

    except ImportError:
        # Return an empty QIcon if Pillow is not installed
        return QIcon()