import numpy as np
import heapq
import math
from typing import Optional, List, Tuple, Dict, Any, Callable

import cv2
from skimage.morphology import skeletonize

# --- Constants moved from VesselTracerApp ---
# These will be configurable via argparse later
PATHFINDING_OBSTACLE_COST = 1e9
TIME_COST_WEIGHT = 1.0
TURN_PENALTY_WEIGHT = 50.0
CROSS_VESSEL_PENALTY = 1e6

class InteractivePathfinder:
    """
    Manages the state and logic for iterative, user-guided pathfinding.
    """
    def __init__(self, mask: np.ndarray, cost_map: np.ndarray, identity_map: np.ndarray):
        """
        Initializes the pathfinder with the necessary data maps.

        Args:
            mask: The binary vessel mask where pathfinding is allowed.
            cost_map: The temporal cost map.
            identity_map: The map assigning unique IDs to each vessel segment.
        """
        self.mask = mask
        self.cost_map = cost_map
        self.identity_map = identity_map
        self.skeleton = None
        self.junctions = []

        self.full_path: List[Tuple[int, int]] = []
        self.branch_options: List[List[Tuple[int, int]]] = []
        self.current_position: Optional[Tuple[int, int]] = None

        self._prepare_skeleton_and_junctions()

    def _prepare_skeleton_and_junctions(self):
        """Generates the skeleton of the vessel mask and finds the junction points."""
        print("Preparing skeleton and finding junctions...")
        # Ensure mask is binary (0 or 1) for skeletonize
        binary_mask = self.mask > 0
        self.skeleton = skeletonize(binary_mask).astype(np.uint8)
        self.junctions = find_junctions(self.skeleton)
        print(f"Found {len(self.junctions)} junctions.")

    def start_pathfinding(self, start_point: Tuple[int, int]):
        """
        Starts a new pathfinding session from a given point.

        Args:
            start_point: The (y, x) coordinates of the starting point.
        """
        self.full_path = [start_point]
        self.current_position = start_point
        self.branch_options = []
        print(f"Pathfinding started at: {start_point}")
        self.find_next_branches()

    def find_next_branches(self):
        """
        Finds the next set of possible branches from the current position.
        """
        if not self.current_position:
            self.branch_options = []
            return

        print(f"Finding next branches from {self.current_position}...")
        self.branch_options = []

        # Find the 5 nearest junctions that are not already in the path
        path_nodes = set(self.full_path)
        valid_junctions = [j for j in self.junctions if j not in path_nodes]

        if not valid_junctions:
            print("No valid junctions left to explore.")
            return

        # Calculate distances from the current position to all valid junctions
        current_pos_arr = np.array(self.current_position)
        junction_coords = np.array(valid_junctions)
        distances = np.linalg.norm(junction_coords - current_pos_arr, axis=1)

        # Get the indices of the 5 closest junctions
        num_to_find = min(5, len(valid_junctions))
        closest_junction_indices = np.argsort(distances)[:num_to_find]

        found_branches = []
        for idx in closest_junction_indices:
            junction = tuple(junction_coords[idx])
            print(f"  -> Attempting to find path to junction: {junction}")

            # Use A* to find a path from the current position to the junction
            segment = find_path_astar(
                cost_map=self.cost_map,
                start=self.current_position,
                end=junction,
                vessel_identity_map=self.identity_map
            )

            if segment:
                print(f"     ... Found path of length {len(segment)}")
                found_branches.append(segment)

        self.branch_options = found_branches
        print(f"Found {len(self.branch_options)} potential branches.")

    def select_branch(self, branch_index: int):
        """
        Selects a branch, appends it to the main path, and finds the next set of branches.

        Args:
            branch_index: The index of the chosen branch in self.branch_options.
        """
        if not (0 <= branch_index < len(self.branch_options)):
            print(f"Error: Invalid branch index {branch_index}")
            return

        selected_segment = self.branch_options[branch_index]
        # Avoid duplicating the connection point
        self.full_path.extend(selected_segment[1:])
        self.current_position = self.full_path[-1]

        print(f"Branch {branch_index} selected. New path length: {len(self.full_path)}")
        self.find_next_branches()

# --- A* Pathfinding Algorithm (Moved from baseline.py) ---

def find_junctions(skeleton: np.ndarray) -> List[Tuple[int, int]]:
    """
    Finds junction points in a skeletonized image.

    A junction is a pixel with more than two neighbors in a 3x3 window.

    Args:
        skeleton: A binary, 1-pixel-wide skeleton image (np.uint8).

    Returns:
        A list of (y, x) coordinates of the junction points.
    """
    if skeleton is None or skeleton.dtype != np.uint8:
        return []

    # This kernel counts the number of neighbors of a pixel.
    # The result of the convolution at a pixel will be the number of non-zero
    # neighbors. We are looking for pixels where this count is > 2.
    kernel = np.array([[1, 1, 1],
                       [1, 0, 1],
                       [1, 1, 1]], dtype=np.uint8)

    # Convolve the image with the kernel
    neighbor_count = cv2.filter2D(skeleton, -1, kernel, borderType=cv2.BORDER_CONSTANT)

    # A junction is a skeleton pixel with more than 2 neighbors.
    junction_mask = (neighbor_count > 2) & (skeleton > 0)
    junction_points = np.argwhere(junction_mask)

    return [tuple(point) for point in junction_points]


def find_path_astar(cost_map: np.ndarray,
                    start: Tuple[int, int],
                    end: Tuple[int, int],
                    vessel_identity_map: Optional[np.ndarray],
                    viz_callback: Optional[Callable] = None) -> Optional[List[Tuple[int, int]]]:
    """
    Finds the optimal path between two points using the A* algorithm.

    Args:
        cost_map: The base cost map (incorporating temporal cost).
        start: The starting (y, x) coordinate tuple.
        end: The ending (y, x) coordinate tuple.
        vessel_identity_map: The map assigning a unique ID to each vessel.
        viz_callback: An optional function to call for visualizing the search.

    Returns:
        A list of (y, x) tuples representing the path, or None if no path is found.
    """
    if cost_map[start] >= PATHFINDING_OBSTACLE_COST or cost_map[end] >= PATHFINDING_OBSTACLE_COST:
        return None

    def heuristic(p1, p2):
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    open_set = [(heuristic(start, end), 0, start)]  # f_cost, g_cost, pos
    came_from = {}
    g_costs = {start: 0}
    closed_set = set()

    node_counter = 0
    viz_interval = 50

    while open_set:
        _, g_cost, current = heapq.heappop(open_set)

        if current == end:
            if viz_callback:
                viz_callback(list(closed_set))

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
            viz_callback(list(closed_set))

        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0: continue
                neighbor = (current[0] + dr, current[1] + dc)

                if not (0 <= neighbor[0] < cost_map.shape[0] and 0 <= neighbor[1] < cost_map.shape[1]) or \
                        cost_map[neighbor] >= PATHFINDING_OBSTACLE_COST or \
                        neighbor in closed_set:
                    continue

                cross_vessel_penalty = 0
                if vessel_identity_map is not None:
                    current_id = vessel_identity_map[current]
                    neighbor_id = vessel_identity_map[neighbor]
                    if current_id > 0 and neighbor_id > 0 and current_id != neighbor_id:
                        cross_vessel_penalty = CROSS_VESSEL_PENALTY

                turn_penalty = 0
                parent = came_from.get(current)
                if parent:
                    v1 = (current[0] - parent[0], current[1] - parent[1])
                    v2 = (neighbor[0] - current[0], neighbor[1] - current[1])

                    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                    mag1 = math.sqrt(v1[0] ** 2 + v1[1] ** 2)
                    mag2 = math.sqrt(v2[0] ** 2 + v2[1] ** 2)

                    if mag1 > 0 and mag2 > 0:
                        cosine_similarity = dot_product / (mag1 * mag2)
                        turn_penalty = TURN_PENALTY_WEIGHT * (1.0 - cosine_similarity)

                move_cost = np.sqrt(dr ** 2 + dc ** 2)
                time_cost = TIME_COST_WEIGHT * cost_map[neighbor]
                new_g_cost = g_costs[current] + move_cost + time_cost + turn_penalty + cross_vessel_penalty

                if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                    g_costs[neighbor] = new_g_cost
                    f_cost = new_g_cost + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_cost, new_g_cost, neighbor))
                    came_from[neighbor] = current

    return None
