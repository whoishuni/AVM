import numpy as np
import cv2
from typing import List, Optional, Tuple

# --- Library Imports ---
try:
    from skimage.morphology import skeletonize
    from skimage.filters import frangi, sato, meijering
except ImportError as e:
    raise ImportError(
        f"Scikit-image library not found. Please install it using 'pip install scikit-image'. Original error: {e}"
    )

# --- Local Project Imports ---
from utils.threading import ProgressUpdater
from .graph_builder import build_graph_from_skeleton, Graph

# --- Image Enhancement and Segmentation Functions ---

def remove_large_bright_areas(
    image: np.ndarray, bg_color: int, threshold_offset: int = 15, kernel_size: int = 15
) -> np.ndarray:
    """Removes large bright areas from an image to reduce background interference."""
    threshold_value = max(0, bg_color - threshold_offset)
    _, bright_mask = cv2.threshold(image, threshold_value, 255, cv2.THRESH_BINARY)
    kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (kernel_size, kernel_size))
    large_areas_mask = cv2.morphologyEx(bright_mask, cv2.MORPH_OPEN, kernel, iterations=2)
    processed_image = image.copy()
    processed_image[large_areas_mask == 255] = bg_color
    return processed_image


def bridge_gaps_in_mask(mask: np.ndarray, max_distance: int = 15) -> np.ndarray:
    """Intelligently connects separated vessel segments in a binary mask."""
    if mask is None or np.sum(mask) == 0:
        return mask.copy() if mask is not None else np.array([])

    bridged_mask = mask.copy()
    min_contour_area = 5
    contours, _ = cv2.findContours(mask, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    if not contours or len(contours) < 2:
        return bridged_mask

    valid_contours = [cnt for cnt in contours if cv2.contourArea(cnt) > min_contour_area]
    if len(valid_contours) < 2:
        return bridged_mask

    valid_contours_mask = np.zeros_like(mask)
    cv2.drawContours(valid_contours_mask, valid_contours, -1, 255, -1)
    skeleton = skeletonize(valid_contours_mask / 255).astype(np.uint8) * 255

    kernel = np.array([[1, 1, 1], [1, 10, 1], [1, 1, 1]], dtype=np.uint8)
    convolved = cv2.filter2D(skeleton, -1, kernel)
    endpoints_map = np.zeros_like(skeleton)
    endpoints_map[(convolved == 11) & (skeleton > 0)] = 255

    endpoint_coords = np.argwhere(endpoints_map > 0)
    if len(endpoint_coords) < 2:
        return bridged_mask

    endpoints_with_contour_info = []
    for y, x in endpoint_coords:
        for i, cnt in enumerate(valid_contours):
            if cv2.pointPolygonTest(cnt, (int(x), int(y)), False) >= 0:
                endpoints_with_contour_info.append({'point': (y, x), 'contour_idx': i})
                break

    min_dist_found = float('inf')
    best_pair = None
    for i in range(len(endpoints_with_contour_info)):
        for j in range(i + 1, len(endpoints_with_contour_info)):
            ep1 = endpoints_with_contour_info[i]
            ep2 = endpoints_with_contour_info[j]
            if ep1['contour_idx'] != ep2['contour_idx']:
                dist = np.linalg.norm(np.array(ep1['point']) - np.array(ep2['point']))
                if dist < max_distance and dist < min_dist_found:
                    min_dist_found = dist
                    best_pair = (ep1['point'], ep2['point'])

    if best_pair:
        p1_yx, p2_yx = best_pair
        p1_xy = (int(p1_yx[1]), int(p1_yx[0]))
        p2_xy = (int(p2_yx[1]), int(p2_yx[0]))
        cv2.line(bridged_mask, p1_xy, p2_xy, 255, 1)

    return bridged_mask

def create_enhanced_vessel_masks(
    images: List[np.ndarray], noise_rois: List, bg_color: int,
    params: dict, smoothing_level: int, worker_thread: Optional[ProgressUpdater]
) -> Optional[List[np.ndarray]]:
    """Generates a sequence of enhanced binary vessel masks from raw images."""
    masks = []
    total_images = len(images)
    bg_color_int = int(bg_color)
    max_gap_dist = params["MAX_GAP_BRIDGE_DISTANCE"]
    filter_sigmas = range(1, 6, 2)

    for i, img in enumerate(images):
        if worker_thread and not worker_thread.is_running:
            return None

        processed_img = img.copy()
        processed_img = remove_large_bright_areas(
            processed_img, bg_color_int, params["BG_REMOVAL_THRESHOLD_OFFSET"],
            params["BG_REMOVAL_KERNEL_SIZE"]
        )

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
            worker_thread.update(i + 1, total_images, f"Processing frame {i+1}/{total_images}")

    return masks


def create_maximum_intensity_projection(images: List[np.ndarray]) -> Optional[np.ndarray]:
    """Creates a Maximum Intensity Projection (MIP) from an image sequence."""
    if not images: return None
    return np.max(np.stack(images, axis=0), axis=0)


def create_temporal_cost_map(masks: List[np.ndarray], obstacle_cost: float) -> Optional[np.ndarray]:
    """Creates a cost map where the cost is related to the frame number (time)."""
    if not masks: return None
    h, w = masks[0].shape
    cost_map = np.full((h, w), obstacle_cost, dtype=np.float32)

    for frame_idx, mask in enumerate(masks):
        vessel_pixels = mask > 0
        cost_map[vessel_pixels] = np.minimum(cost_map[vessel_pixels], frame_idx)

    return cost_map


def create_vessel_layers(mask: np.ndarray, original_mip: np.ndarray) -> Optional[np.ndarray]:
    """Divides the vessel mask into 10 layers based on original image brightness."""
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


def _get_component_orientation(mask: np.ndarray) -> Optional[np.ndarray]:
    """Calculates the orientation vector of a binary mask component."""
    coords = np.argwhere(mask)
    if len(coords) < 5:  # Need minimum points to determine orientation
        return None
    coords = coords.astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(coords, mean=None)
    return eigenvectors[0]

def build_vessel_identity_map(
    masks: List[np.ndarray],
    images: List[np.ndarray],
    params: dict
) -> Optional[np.ndarray]:
    """
    Builds a map assigning a unique, persistent ID to each vessel segment,
    considering overlap, orientation, width, and intensity similarity.
    """
    if not masks or not images: return None
    if len(masks) != len(images):
        print("Warning: Mismatch between number of masks and images.")
        return None

    h, w = masks[0].shape
    identity_map = np.zeros((h, w), dtype=np.int32)
    next_vessel_id = 1

    vessel_orientations = {}
    vessel_avg_widths = {}
    vessel_avg_intensities = {}

    width_maps = [cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5) * 2 for m in masks]

    if np.any(masks[0]):
        num_labels, labels = cv2.connectedComponents(masks[0])
        current_image = images[0]
        current_width_map = width_maps[0]

        for label_idx in range(1, num_labels):
            component_mask = (labels == label_idx)
            identity_map[component_mask] = next_vessel_id

            orientation = _get_component_orientation(component_mask)
            if orientation is not None:
                vessel_orientations[next_vessel_id] = orientation

            vessel_avg_widths[next_vessel_id] = np.mean(current_width_map[component_mask])
            vessel_avg_intensities[next_vessel_id] = np.mean(current_image[component_mask])
            next_vessel_id += 1

    for i in range(1, len(masks)):
        prev_mask = masks[i-1]
        current_mask = masks[i]
        current_image = images[i]
        current_width_map = width_maps[i]

        new_growth_mask = cv2.subtract(current_mask, prev_mask)
        if not np.any(new_growth_mask):
            continue

        num_labels, labels = cv2.connectedComponents(new_growth_mask)
        for label_idx in range(1, num_labels):
            component_mask = (labels == label_idx)

            component_orientation = _get_component_orientation(component_mask)
            component_avg_width = np.mean(current_width_map[component_mask])
            component_avg_intensity = np.mean(current_image[component_mask])

            overlap_region_mask = component_mask & (prev_mask > 0)

            if not np.any(overlap_region_mask):
                chosen_id = next_vessel_id
                next_vessel_id += 1
            else:
                overlapping_ids_pixels = identity_map[overlap_region_mask]
                unique_ids, counts = np.unique(overlapping_ids_pixels[overlapping_ids_pixels > 0], return_counts=True)

                if len(unique_ids) == 0:
                    chosen_id = next_vessel_id
                    next_vessel_id += 1
                else:
                    best_score = -float('inf')
                    chosen_id = -1

                    overlap_scores = {uid: count for uid, count in zip(unique_ids, counts)}

                    for uid in unique_ids:
                        overlap_score = overlap_scores.get(uid, 0) / np.sum(counts)

                        orientation_similarity = 0
                        if component_orientation is not None and uid in vessel_orientations:
                            vec1 = component_orientation
                            vec2 = vessel_orientations[uid]
                            orientation_similarity = abs(np.dot(vec1, vec2))

                        prev_width = vessel_avg_widths.get(uid, component_avg_width)
                        width_increase_penalty = max(0, (component_avg_width - prev_width) / prev_width if prev_width > 0 else 0)

                        prev_intensity = vessel_avg_intensities.get(uid, component_avg_intensity)
                        brightness_increase_penalty = max(0, (component_avg_intensity - prev_intensity) / prev_intensity if prev_intensity > 0 else 0)

                        total_score = (
                            overlap_score * params.get("OVERLAP_WEIGHT", 2.0) +
                            orientation_similarity * params.get("DIRECTION_SIMILARITY_WEIGHT", 1.0) -
                            width_increase_penalty * params.get("MAX_WIDTH_INCREASE_WEIGHT", 10.0) -
                            brightness_increase_penalty * params.get("MAX_BRIGHTNESS_INCREASE_WEIGHT", 10.0)
                        )

                        if total_score > best_score:
                            best_score = total_score
                            chosen_id = uid

            identity_map[component_mask] = chosen_id

            if component_orientation is not None:
                vessel_orientations[chosen_id] = component_orientation
            vessel_avg_widths[chosen_id] = component_avg_width
            vessel_avg_intensities[chosen_id] = component_avg_intensity

    return identity_map

def create_combined_identity_map(
    vessel_identity_map: Optional[np.ndarray],
    layered_vessel_mask: Optional[np.ndarray]
) -> Optional[np.ndarray]:
    """Combines temporal and layer-based identity maps into a single, more robust map."""
    if vessel_identity_map is None or layered_vessel_mask is None:
        return vessel_identity_map

    combined_map = np.zeros_like(vessel_identity_map)
    unique_pairs = {}
    next_new_id = 1

    vessel_pixels = np.argwhere(vessel_identity_map > 0)
    for y, x in vessel_pixels:
        vessel_id = vessel_identity_map[y, x]
        layer_id = layered_vessel_mask[y, x]
        if layer_id > 0:
            pair = (vessel_id, layer_id)
            if pair not in unique_pairs:
                unique_pairs[pair] = next_new_id
                next_new_id += 1
            combined_map[y, x] = unique_pairs[pair]

    return combined_map

def process_mip_and_build_graph(mip_image: np.ndarray, params: dict) -> Tuple[Optional[np.ndarray], Optional[Graph]]:
    """
    Processes the Maximum Intensity Projection image to create a skeleton and build a graph.

    Args:
        mip_image: The MIP image.
        params: A dictionary of tuning parameters.

    Returns:
        A tuple containing the skeletonized image and the vessel graph.
    """
    if mip_image is None:
        return None, None

    # This is a simplified processing chain for the MIP
    # It assumes the MIP is already enhanced. In a real scenario,
    # some of the enhancement steps from create_enhanced_vessel_masks would be applied here.

    _, binary_mask = cv2.threshold(mip_image, 30, 255, cv2.THRESH_BINARY)
    bridged_mask = bridge_gaps_in_mask(binary_mask, params.get("MAX_GAP_BRIDGE_DISTANCE", 15))

    # Skeletonization
    skeleton = skeletonize(bridged_mask / 255).astype(np.uint8) * 255
    if not np.any(skeleton):
        return skeleton, None

    # Graph building
    vessel_graph = build_graph_from_skeleton(skeleton)

    return skeleton, vessel_graph


def generate_mask_steps(image: np.ndarray, smoothing_level: int, params: dict, bg_color: int) -> List[Tuple[np.ndarray, str]]:
    """Generates a list of (image, description) tuples for visualizing the mask creation process."""
    if image is None: return []
    steps = []
    img = image.copy()
    steps.append((img, "1. Original Image (MIP)"))

    img = remove_large_bright_areas(img, bg_color, params["BG_REMOVAL_THRESHOLD_OFFSET"], params["BG_REMOVAL_KERNEL_SIZE"])
    steps.append((img, "2. Remove Bright Background Areas"))

    if smoothing_level > 0:
        kernel_size = smoothing_level * 2 + 1
        img = cv2.GaussianBlur(img, (kernel_size, kernel_size), 0)
        steps.append((img, f"3. Gaussian Smoothing (Kernel: {kernel_size}x{kernel_size})"))

    inverted_img = cv2.bitwise_not(img)
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

    bridged_mask = bridge_gaps_in_mask(binary_mask, params["MAX_GAP_BRIDGE_DISTANCE"])
    steps.append((bridged_mask, f"7. Gap Bridging (Max Dist: {params['MAX_GAP_BRIDGE_DISTANCE']}px)"))

    return steps