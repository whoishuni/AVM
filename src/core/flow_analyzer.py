import numpy as np
import cv2
from typing import List, Tuple, Dict, Optional
from collections import defaultdict
from skimage.measure import label, regionprops


def _get_component_orientation(pixels: np.ndarray) -> Optional[np.ndarray]:
    """Calculates the orientation vector of a set of pixels."""
    if len(pixels) < 5:  # Need minimum points to determine orientation
        return None
    coords = pixels.astype(np.float32)
    mean, eigenvectors = cv2.PCACompute(coords, mean=None)
    return eigenvectors[0]


class FlowNode:
    """Represents a node in the directed flow graph, corresponding to a vessel component."""
    def __init__(self, component_id: int, frame_index: int, pixels: np.ndarray, bbox: Tuple[int, int, int, int]):
        self.id = f"{frame_index}_{component_id}"
        self.frame_index = frame_index
        self.component_id = component_id
        self.pixels = pixels
        self.bbox = bbox
        self.children: List['FlowNode'] = []
        self.parent: Optional['FlowNode'] = None
        self.orientation: Optional[np.ndarray] = None
        self.avg_width: float = 0.0
        self.avg_intensity: float = 0.0

    def __repr__(self):
        return f"FlowNode(id={self.id}, parent={self.parent.id if self.parent else None})"

class FlowAnalyzer:
    """Analyzes a sequence of vessel masks to build a directed graph representing blood flow."""
    def __init__(self, vessel_masks: List[np.ndarray], images: List[np.ndarray], params: Dict):
        self.vessel_masks = vessel_masks
        self.images = images
        self.params = params
        self.flow_graph: Dict[str, FlowNode] = {}
        self.nodes_by_frame: Dict[int, List[FlowNode]] = defaultdict(list)
        self.width_maps = [cv2.distanceTransform(m.astype(np.uint8), cv2.DIST_L2, 5) * 2 for m in self.vessel_masks]


    def analyze(self):
        """
        Builds the directed flow graph by processing the vessel masks frame by frame.
        """
        if not self.vessel_masks:
            return None

        # --- Step 1: Identify all components (nodes) in each frame ---
        for i, mask in enumerate(self.vessel_masks):
            labeled_mask = label(mask, connectivity=2, background=0)
            regions = regionprops(labeled_mask)
            current_image = self.images[i]
            current_width_map = self.width_maps[i]

            for region in regions:
                node = FlowNode(
                    component_id=region.label,
                    frame_index=i,
                    pixels=region.coords,
                    bbox=region.bbox
                )
                # Calculate and store advanced properties
                component_mask = (labeled_mask == region.label)
                node.orientation = _get_component_orientation(node.pixels)
                node.avg_width = np.mean(current_width_map[component_mask])
                node.avg_intensity = np.mean(current_image[component_mask])

                self.flow_graph[node.id] = node
                self.nodes_by_frame[i].append(node)

        # --- Step 2: Establish parent-child relationships between frames ---
        for frame_idx in range(1, len(self.vessel_masks)):
            current_nodes = self.nodes_by_frame.get(frame_idx, [])
            prev_nodes = self.nodes_by_frame.get(frame_idx - 1, [])

            if not current_nodes or not prev_nodes:
                continue

            for child_node in current_nodes:
                best_parent = self._find_best_parent(child_node, prev_nodes)
                if best_parent:
                    child_node.parent = best_parent
                    best_parent.children.append(child_node)

        return self.flow_graph

    def _find_best_parent(self, child_node: FlowNode, potential_parents: List[FlowNode]) -> Optional[FlowNode]:
        """
        Finds the most likely parent for a child node based on a comprehensive score.
        """
        best_parent = None
        best_score = -float('inf')

        # Create a small mask for the child component for efficient overlap calculation
        min_r, min_c, max_r, max_c = child_node.bbox
        child_mask_local = np.zeros((max_r - min_r, max_c - min_c), dtype=np.uint8)
        local_coords = child_node.pixels - np.array([min_r, min_c])
        child_mask_local[local_coords[:, 0], local_coords[:, 1]] = 1

        for parent_node in potential_parents:
            # Bbox intersection check
            p_min_r, p_min_c, p_max_r, p_max_c = parent_node.bbox
            if not (max_r > p_min_r and min_r < p_max_r and max_c > p_min_c and min_c < p_max_c):
                continue

            # --- 1. Overlap Score ---
            parent_pixels_in_bbox = parent_node.pixels[
                (parent_node.pixels[:, 0] >= min_r) & (parent_node.pixels[:, 0] < max_r) &
                (parent_node.pixels[:, 1] >= min_c) & (parent_node.pixels[:, 1] < max_c)
            ]
            overlap = 0
            if parent_pixels_in_bbox.size > 0:
                parent_overlap_mask = np.zeros_like(child_mask_local, dtype=np.uint8)
                local_parent_coords = parent_pixels_in_bbox - np.array([min_r, min_c])
                parent_overlap_mask[local_parent_coords[:, 0], local_parent_coords[:, 1]] = 1
                overlap = np.sum(child_mask_local & parent_overlap_mask)

            overlap_score = overlap / len(child_node.pixels) if len(child_node.pixels) > 0 else 0

            # --- 2. Orientation Similarity ---
            orientation_similarity = 0.0
            if child_node.orientation is not None and parent_node.orientation is not None:
                orientation_similarity = abs(np.dot(child_node.orientation, parent_node.orientation))

            # --- 3. Width Increase Penalty ---
            width_increase_penalty = max(0, (child_node.avg_width - parent_node.avg_width) / parent_node.avg_width if parent_node.avg_width > 0 else 0)

            # --- 4. Brightness Increase Penalty ---
            brightness_increase_penalty = max(0, (child_node.avg_intensity - parent_node.avg_intensity) / parent_node.avg_intensity if parent_node.avg_intensity > 0 else 0)

            # --- 5. Total Score ---
            total_score = (
                overlap_score * self.params.get("OVERLAP_WEIGHT", 2.0) +
                orientation_similarity * self.params.get("DIRECTION_SIMILARITY_WEIGHT", 1.0) -
                width_increase_penalty * self.params.get("MAX_WIDTH_INCREASE_WEIGHT", 10.0) -
                brightness_increase_penalty * self.params.get("MAX_BRIGHTNESS_INCREASE_WEIGHT", 10.0)
            )

            if total_score > best_score:
                best_score = total_score
                best_parent = parent_node

        return best_parent

    def visualize(self) -> Optional[np.ndarray]:
        """
        Generates a visualization of the flow graph.

        Returns:
            A BGR numpy array representing the visualization, or None if no graph.
        """
        if not self.flow_graph:
            return None

        # Find the overall dimensions needed for the visualization canvas
        max_h, max_w = 0, 0
        if self.vessel_masks:
            max_h, max_w = self.vessel_masks[0].shape

        if max_h == 0 or max_w == 0:
            return np.zeros((100, 100, 3), dtype=np.uint8) # Return a blank image if no dimensions

        # Create a colormap for different frames
        num_frames = max(node.frame_index for node in self.flow_graph.values()) + 1
        # Create a visually distinct colormap (e.g., HSV)
        colors = (np.array([ (h * 180 / num_frames, 255, 255) for h in range(num_frames) ], dtype=np.uint8)).reshape(-1, 1, 3)
        colors = cv2.cvtColor(colors, cv2.COLOR_HSV2BGR)

        # Create the canvas
        canvas = np.zeros((max_h, max_w, 3), dtype=np.uint8)

        # Draw each node's pixels with a color corresponding to its frame
        for node in self.flow_graph.values():
            color = colors[node.frame_index][0].tolist()
            canvas[node.pixels[:, 0], node.pixels[:, 1]] = color

        # Draw arrows from parent to child centroids
        for node in self.flow_graph.values():
            if not node.children:
                continue

            py, px = np.mean(node.pixels, axis=0).astype(int)

            for child in node.children:
                cy, cx = np.mean(child.pixels, axis=0).astype(int)
                # Draw a white arrow
                cv2.arrowedLine(canvas, (px, py), (cx, cy), (255, 255, 255), 1, tipLength=0.2)

        return canvas
