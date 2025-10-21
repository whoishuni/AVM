import numpy as np
import cv2
import heapq
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

        # --- Step 2: Establish parent-child relationships using a forward-tracking approach ---
        for frame_idx in range(len(self.vessel_masks) - 1):
            parent_nodes = self.nodes_by_frame.get(frame_idx, [])
            child_nodes = self.nodes_by_frame.get(frame_idx + 1, [])

            if not parent_nodes or not child_nodes:
                continue

            for parent_node in parent_nodes:
                candidate_children = []
                for child_node in child_nodes:
                    score = self._calculate_connection_score(parent_node, child_node)
                    if score > self.params.get("MIN_CONNECTION_SCORE", 0.5):
                        candidate_children.append((score, child_node))

                # Sort candidates by score in descending order
                candidate_children.sort(key=lambda x: x[0], reverse=True)

                # Connect to the best N children (e.g., for branching)
                max_children = self.params.get("MAX_CHILDREN_PER_NODE", 1)
                for i in range(min(max_children, len(candidate_children))):
                    score, child_to_connect = candidate_children[i]
                    parent_node.children.append(child_to_connect)
                    # In this model, a child can have multiple parents (merging)
                    # but we only track the primary parent for simplicity in some algos.
                    # For now, we allow multiple parents and don't set a single 'child.parent'
                    # child_to_connect.parent = parent_node # This would overwrite for merging vessels

        return self.flow_graph

    def _calculate_connection_score(self, parent_node: FlowNode, child_node: FlowNode) -> float:
        """
        Calculates a comprehensive score for a potential parent-child connection.
        """
        # --- 1. Bbox Overlap / Proximity ---
        p_min_r, p_min_c, p_max_r, p_max_c = parent_node.bbox
        c_min_r, c_min_c, c_max_r, c_max_c = child_node.bbox

        # Check for bbox intersection. If they don't intersect, calculate distance.
        if not (p_max_r > c_min_r and p_min_r < c_max_r and p_max_c > c_min_c and p_min_c < c_max_c):
            # No intersection, penalize based on distance
            parent_centroid = np.mean(parent_node.pixels, axis=0)
            child_centroid = np.mean(child_node.pixels, axis=0)
            distance = np.linalg.norm(parent_centroid - child_centroid)

            max_allowed_dist = self.params.get("MAX_NODE_DISTANCE", 30) # pixels
            if distance > max_allowed_dist:
                return -1.0 # Instant disqualification

            # Penalize non-overlapping but close nodes
            distance_penalty = (distance / max_allowed_dist) ** 2
            overlap_score = 1.0 - distance_penalty # Score is higher for closer nodes
        else:
            # There is an intersection, calculate IoU (Intersection over Union) or simple overlap
            # For simplicity, we can use a basic pixel overlap score for now
            # A more robust implementation might use a shared mask for efficiency
            overlap_score = 1.0 # Placeholder for actual overlap calculation if needed.
                                # The distance check handles the primary filtering.

        # --- 2. Orientation Similarity ---
        orientation_similarity = 0.0
        if parent_node.orientation is not None and child_node.orientation is not None:
            orientation_similarity = abs(np.dot(parent_node.orientation, child_node.orientation))

        # --- 3. Width Similarity ---
        # Penalize large differences in width. A small increase is expected.
        width_diff = abs(child_node.avg_width - parent_node.avg_width)
        max_width_diff = parent_node.avg_width * self.params.get("MAX_WIDTH_DIFFERENCE_RATIO", 0.5)
        width_similarity = max(0, 1.0 - (width_diff / max_width_diff)) if max_width_diff > 0 else 1.0

        # --- 4. Brightness Similarity ---
        # Penalize large differences in intensity.
        intensity_diff = abs(child_node.avg_intensity - parent_node.avg_intensity)
        max_intensity_diff = parent_node.avg_intensity * self.params.get("MAX_INTENSITY_DIFFERENCE_RATIO", 0.5)
        intensity_similarity = max(0, 1.0 - (intensity_diff / max_intensity_diff)) if max_intensity_diff > 0 else 1.0

        # --- 5. Total Score ---
        # Weights determine the importance of each factor
        total_score = (
            overlap_score * self.params.get("OVERLAP_WEIGHT", 1.5) +
            orientation_similarity * self.params.get("DIRECTION_SIMILARITY_WEIGHT", 1.0) +
            width_similarity * self.params.get("WIDTH_SIMILARITY_WEIGHT", 0.8) +
            intensity_similarity * self.params.get("INTENSITY_SIMILARITY_WEIGHT", 0.5)
        )

        return total_score

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

    def find_path(self, start_coords: Tuple[int, int], end_coords: Tuple[int, int]) -> Tuple[Optional[List[FlowNode]], float]:
        """
        Finds the best path from a start coordinate to an end coordinate using the flow graph.
        Uses the A* algorithm.
        """
        if not self.flow_graph:
            return None, 0.0

        # --- 1. Find the start and end nodes closest to the coordinates ---
        start_node = self._find_node_at(start_coords)
        end_node = self._find_node_at(end_coords)

        if not start_node or not end_node:
            print("Error: Could not find start or end node.")
            return None, 0.0

        if start_node == end_node:
            return [start_node], 0.0

        # --- 2. A* Algorithm Setup ---
        open_set = [(0, start_node.id)]  # (priority, node_id)
        came_from: Dict[str, str] = {}
        g_costs: Dict[str, float] = {node_id: float('inf') for node_id in self.flow_graph}
        g_costs[start_node.id] = 0

        # Heuristic function (Euclidean distance between centroids)
        def heuristic(node_a_id: str, node_b_id: str) -> float:
            node_a = self.flow_graph[node_a_id]
            node_b = self.flow_graph[node_b_id]
            a_centroid = np.mean(node_a.pixels, axis=0)
            b_centroid = np.mean(node_b.pixels, axis=0)
            return np.linalg.norm(a_centroid - b_centroid)

        # --- 3. A* Main Loop ---
        while open_set:
            _, current_id = heapq.heappop(open_set)

            if current_id == end_node.id:
                # Path found, reconstruct it
                path = []
                total_cost = g_costs[current_id]
                while current_id in came_from:
                    path.append(self.flow_graph[current_id])
                    current_id = came_from[current_id]
                path.append(start_node)
                return path[::-1], total_cost

            current_node = self.flow_graph[current_id]

            # Explore neighbors (children in the directed graph)
            for neighbor_node in current_node.children:
                neighbor_id = neighbor_node.id

                # Cost calculation
                move_cost = heuristic(current_id, neighbor_id) # Distance is the base cost
                time_cost = abs(current_node.frame_index - neighbor_node.frame_index) * self.params.get("TEMPORAL_COST_WEIGHT", 10)

                # Turn penalty
                turn_penalty = 0
                if current_id in came_from:
                    parent_id = came_from[current_id]
                    parent_node = self.flow_graph[parent_id]
                    if parent_node.orientation is not None and current_node.orientation is not None:
                         # cosine similarity: 1 is straight, 0 is 90 deg turn
                        cos_sim = abs(np.dot(parent_node.orientation, current_node.orientation))
                        turn_penalty = (1 - cos_sim) * self.params.get("TURN_PENALTY_WEIGHT", 50)


                new_g_cost = g_costs[current_id] + move_cost + time_cost + turn_penalty

                if new_g_cost < g_costs[neighbor_id]:
                    g_costs[neighbor_id] = new_g_cost
                    f_cost = new_g_cost + heuristic(neighbor_id, end_node.id)
                    heapq.heappush(open_set, (f_cost, neighbor_id))
                    came_from[neighbor_id] = current_id

        return None, 0.0 # No path found

    def _find_node_at(self, coords: Tuple[int, int], max_dist: int = 20) -> Optional[FlowNode]:
        """
        Finds the closest node to the given coordinates (y, x) within a maximum search distance.
        This is more robust than requiring an exact pixel match.
        """
        y, x = coords
        click_point = np.array([y, x])

        best_candidate: Optional[FlowNode] = None
        min_dist = float('inf')

        # To optimize, we can first find the frame with the closest node, then the node itself,
        # but for simplicity and robustness, we check all nodes.
        for node in self.flow_graph.values():
            # A quick check using bounding box to discard distant nodes
            min_r, min_c, max_r, max_c = node.bbox
            if not (min_r - max_dist <= y <= max_r + max_dist and \
                    min_c - max_dist <= x <= max_c + max_dist):
                continue

            # For nodes that are potentially close, calculate the precise minimum distance
            # from the click point to any pixel in the node.
            distances = np.linalg.norm(node.pixels - click_point, axis=1)
            dist = np.min(distances)

            if dist < min_dist:
                min_dist = dist
                best_candidate = node

        # Return the best node found, but only if it's within the allowed distance
        if min_dist <= max_dist:
            print(f"Found closest node '{best_candidate.id}' at a distance of {min_dist:.2f} pixels.")
            return best_candidate

        print(f"No node found within {max_dist} pixels. Closest was {min_dist:.2f} pixels away.")
        return None
