import os
import cv2
import numpy as np
from skimage.morphology import skeletonize
from skimage.filters import frangi, sato, meijering
from typing import List, Optional, TYPE_CHECKING

from src.utils import natural_sort_key

if TYPE_CHECKING:
    from PyQt5.QtCore import QRect
    from src.app import VesselTracerApp, ProgressUpdater


# --- Image Loading and Pre-processing ---

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


# --- Vessel Enhancement and Mask Generation ---

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
        return bridged_mask

    valid_contours_mask = np.zeros_like(mask)
    cv2.drawContours(valid_contours_mask, valid_contours, -1, 255, -1)

    # 2. Skeletonize the filtered mask
    skeleton = skeletonize(valid_contours_mask / 255).astype(np.uint8) * 255

    # 3. Find endpoints of the skeleton
    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    convolved = cv2.filter2D(skeleton, -1, kernel)
    endpoints_map = np.zeros_like(skeleton)
    endpoints_map[(convolved == 11) & (skeleton > 0)] = 255

    # 4. Associate endpoints with their original contours
    endpoint_coords = np.argwhere(endpoints_map > 0)
    if len(endpoint_coords) < 2:
        return bridged_mask

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


def create_enhanced_vessel_masks(images: List[np.ndarray], noise_rois: List['QRect'], bg_color: int,
                                 app_instance: 'VesselTracerApp', smoothing_level: int,
                                 worker_thread: Optional['ProgressUpdater']) -> Optional[List[np.ndarray]]:
    """Generates a sequence of enhanced vessel masks."""
    masks = []
    total_images = len(images)
    bg_color_int = int(bg_color)
    max_gap_dist = app_instance.MAX_GAP_BRIDGE_DISTANCE

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

        inverted_img = cv2.bitwise_not(processed_img)

        frangi_img = frangi(inverted_img, sigmas=filter_sigmas, black_ridges=False)
        sato_img = sato(inverted_img, sigmas=filter_sigmas, black_ridges=False)
        meijering_img = meijering(inverted_img, sigmas=filter_sigmas, black_ridges=False)

        combined_response = np.maximum.reduce([frangi_img, sato_img, meijering_img])
        normalized_response = cv2.normalize(combined_response, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)

        _, binary_mask = cv2.threshold(normalized_response, 30, 255, cv2.THRESH_BINARY)

        bridged_mask = bridge_gaps_in_mask(binary_mask, max_gap_dist)
        masks.append(bridged_mask)

        if worker_thread:
            worker_thread.progress_updated(i + 1, total_images)

    return masks


def create_maximum_intensity_projection(images: List[np.ndarray]) -> Optional[np.ndarray]:
    """Creates a Maximum Intensity Projection image."""
    if not images: return None
    return np.max(np.stack(images, axis=0), axis=0)


def create_temporal_cost_map(masks: List[np.ndarray], obstacle_cost: float) -> Optional[np.ndarray]:
    """Creates a cost map with a temporal dimension."""
    if not masks: return None
    h, w = masks[0].shape
    cost_map = np.full((h, w), obstacle_cost, dtype=np.float32)

    for frame_idx, mask in enumerate(masks):
        vessel_pixels = mask > 0
        cost_map[vessel_pixels] = np.minimum(cost_map[vessel_pixels], frame_idx)

    return cost_map


def create_vessel_layers(mask: np.ndarray, original_mip: np.ndarray) -> Optional[np.ndarray]:
    """
    Divides the vessel mask into 10 layers based on the brightness of the original image.
    """
    if mask is None or original_mip is None or np.sum(mask) == 0:
        return None

    layered_mask = np.zeros_like(mask, dtype=np.uint8)
    vessel_locations = mask > 0

    brightness_values = original_mip[vessel_locations]
    if brightness_values.size == 0:
        return layered_mask

    bins = np.percentile(brightness_values, np.linspace(0, 100, 11))
    bins[-1] += 1

    for i in range(10):
        layer_pixels_indices = (original_mip >= bins[i]) & (original_mip < bins[i + 1]) & vessel_locations
        layered_mask[layer_pixels_indices] = i + 1

    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (3, 3))
    cleaned_layered_mask = np.zeros_like(layered_mask)
    for i in range(1, 11):
        layer_mask = np.where(layered_mask == i, 255, 0).astype(np.uint8)
        opened_mask = cv2.morphologyEx(layer_mask, cv2.MORPH_OPEN, kernel)
        cleaned_layered_mask[opened_mask == 255] = i

    return cleaned_layered_mask
