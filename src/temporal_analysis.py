import numpy as np
import cv2
import heapq
import math
from typing import List, Dict, Tuple, Set, Optional

# Node in the graph can be represented by its (y, x) coordinate tuple.
NodeType = Tuple[int, int]

# The graph can be a dictionary where keys are nodes and values are lists of connected nodes.
GraphType = Dict[NodeType, List[NodeType]]

class VesselHistory:
    """
    Analyzes a sequence of vessel masks to build a temporally coherent graph
    of the vascular structure, representing its growth over time.
    """
    def __init__(self):
        self.graph: GraphType = {}
        self.node_to_frame: Dict[NodeType, int] = {}
        self.frames: List[np.ndarray] = []

    def build(self, masks: List[np.ndarray]):
        """
        Builds the temporal graph from a sequence of binary vessel masks.
        This method correctly processes only new pixels at each frame, ensuring
        vessel segments are internally connected and linked to the parent structure.
        """
        if not masks:
            return

        self.frames = masks
        h, w = masks[0].shape
        cumulative_mask = np.zeros_like(masks[0], dtype=np.uint8)

        for frame_idx, current_mask in enumerate(masks):
            new_pixels_mask = cv2.subtract(current_mask, cumulative_mask)
            new_pixel_coords = np.argwhere(new_pixels_mask > 0)

            # Pass 1: Add new nodes to the graph
            for y, x in new_pixel_coords:
                node = (y, x)
                if node not in self.graph:
                    self.graph[node] = []
                    self.node_to_frame[node] = frame_idx

            # Pass 2: Connect new nodes to existing nodes and to each other
            for y, x in new_pixel_coords:
                node = (y, x)
                # Check 8-connectivity neighborhood
                for dr in [-1, 0, 1]:
                    for dc in [-1, 0, 1]:
                        if dr == 0 and dc == 0:
                            continue

                        neighbor_y, neighbor_x = y + dr, x + dc
                        neighbor_node = (neighbor_y, neighbor_x)

                        # Skip if neighbor is out of bounds
                        if not (0 <= neighbor_y < h and 0 <= neighbor_x < w):
                            continue

                        # A connection is valid if the neighbor is in the current mask
                        # (which includes old pixels and other new pixels)
                        if current_mask[neighbor_y, neighbor_x] > 0:
                            # To avoid duplicate edges, only add edge from the "smaller" node
                            if hash(node) < hash(neighbor_node):
                                if neighbor_node in self.graph:
                                    self.graph[node].append(neighbor_node)
                                    self.graph[neighbor_node].append(node)

            # Update the cumulative mask for the next frame
            cumulative_mask = cv2.bitwise_or(cumulative_mask, current_mask)

    def find_path(self, start_coord: NodeType, end_coord: NodeType, turn_penalty_weight: float = 50.0) -> Optional[List[NodeType]]:
        """
        Finds a path between two nodes in the temporally constructed graph using A*.
        The cost function includes distance, a turn penalty, and a temporal penalty.
        """
        if start_coord not in self.graph or end_coord not in self.graph:
            return None

        def heuristic(p1: NodeType, p2: NodeType) -> float:
            return math.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

        open_set: List[Tuple[float, float, NodeType]] = [(heuristic(start_coord, end_coord), 0, start_coord)]  # (f_cost, g_cost, pos)
        came_from: Dict[NodeType, NodeType] = {}
        g_costs: Dict[NodeType, float] = {start_coord: 0}

        while open_set:
            _, g_cost, current = heapq.heappop(open_set)

            if current == end_coord:
                path = []
                while current in came_from:
                    path.append(current)
                    current = came_from[current]
                path.append(start_coord)
                path.reverse()
                return path

            for neighbor in self.graph.get(current, []):
                # --- Cost Calculation ---
                move_cost = heuristic(current, neighbor)  # Distance between adjacent pixels

                # Turn Penalty
                turn_penalty = 0
                parent = came_from.get(current)
                if parent:
                    v1 = (current[0] - parent[0], current[1] - parent[1])
                    v2 = (neighbor[0] - current[0], neighbor[1] - current[1])
                    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                    if mag1 > 0 and mag2 > 0:
                        cosine_similarity = dot_product / (mag1 * mag2)
                        turn_penalty = turn_penalty_weight * (1.0 - cosine_similarity)

                # Temporal Penalty (simple version: penalize moving to a much later frame)
                temporal_penalty = abs(self.node_to_frame[current] - self.node_to_frame[neighbor]) * 0.1

                new_g_cost = g_costs[current] + move_cost + turn_penalty + temporal_penalty

                if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                    g_costs[neighbor] = new_g_cost
                    f_cost = new_g_cost + heuristic(neighbor, end_coord)
                    heapq.heappush(open_set, (f_cost, new_g_cost, neighbor))
                    came_from[neighbor] = current

        return None  # Path not found

    def get_graph_for_visualization(self) -> Tuple[List[NodeType], List[Tuple[NodeType, NodeType]]]:
        """
        Returns the nodes and edges for easy visualization.
        """
        nodes = list(self.graph.keys())
        edges = []
        for node, neighbors in self.graph.items():
            for neighbor in neighbors:
                # To avoid duplicate edges, only add if the hash of the first node is smaller
                if hash(node) < hash(neighbor):
                    edges.append((node, neighbor))
        return nodes, edges
