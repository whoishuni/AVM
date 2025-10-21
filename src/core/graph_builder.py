import numpy as np
import cv2
from typing import List, Tuple, Dict, Set
from collections import defaultdict

class Node:
    """Represents a node in the vessel graph (junctions or endpoints)."""
    def __init__(self, y: int, x: int, node_type: str):
        self.y = y
        self.x = x
        self.type = node_type  # 'junction' or 'endpoint'
        self.id = f"{y},{x}"

    def __repr__(self):
        return f"Node({self.id}, {self.type})"

class Edge:
    """Represents an edge in the vessel graph (a segment of a vessel)."""
    def __init__(self, node1: Node, node2: Node, pixels: List[Tuple[int, int]]):
        self.node1 = node1
        self.node2 = node2
        self.pixels = pixels
        self.length = len(pixels)
        self.vector = self._calculate_vector()

    def _calculate_vector(self) -> Tuple[float, float]:
        """Calculate the direction vector of the edge."""
        if self.length == 0:
            return (0, 0)

        # For short edges, use the direct start-to-end vector
        if self.length < 5:
            dy = self.pixels[-1][0] - self.pixels[0][0]
            dx = self.pixels[-1][1] - self.pixels[0][1]
        # For longer edges, use a more robust calculation based on a subset of points
        else:
            num_points = min(self.length, 10)
            start_point = np.array(self.pixels[0])
            end_point = np.array(self.pixels[-1])
            dy = end_point[0] - start_point[0]
            dx = end_point[1] - start_point[1]

        norm = np.sqrt(dx**2 + dy**2)
        return (dy / norm, dx / norm) if norm > 0 else (0, 0)

    def __repr__(self):
        return f"Edge({self.node1.id} -> {self.node2.id}, len={self.length})"


class Graph:
    """Represents the vascular network as a graph."""
    def __init__(self):
        self.nodes: Dict[str, Node] = {}
        self.edges: List[Edge] = []
        self.adjacency: Dict[str, List[Edge]] = defaultdict(list)

    def add_node(self, node: Node):
        if node.id not in self.nodes:
            self.nodes[node.id] = node

    def add_edge(self, edge: Edge):
        self.edges.append(edge)
        self.adjacency[edge.node1.id].append(edge)
        self.adjacency[edge.node2.id].append(edge)

def build_graph_from_skeleton(skeleton: np.ndarray) -> Graph:
    """
    Builds a graph representation from a skeletonized vessel image.

    Args:
        skeleton: A 2D numpy array where non-zero pixels represent the vessel skeleton.

    Returns:
        A Graph object representing the vascular network.
    """
    if skeleton is None or not np.any(skeleton):
        return Graph()

    graph = Graph()
    h, w = skeleton.shape

    # 1. Find all node candidates (junctions and endpoints)
    node_pixels_map, node_mask = _find_node_pixels(skeleton)

    # 2. Create Node objects
    for (y, x), node_type in node_pixels_map.items():
        node = Node(y, x, node_type)
        graph.add_node(node)

    # 3. Trace paths (edges) between nodes
    visited = set()
    for node_id, start_node in graph.nodes.items():
        start_y, start_x = start_node.y, start_node.x

        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue

                ny, nx = start_y + dy, start_x + dx

                if not (0 <= ny < h and 0 <= nx < w) or skeleton[ny, nx] == 0:
                    continue

                path, end_node_id = _trace_path(skeleton, (ny, nx), start_node.id, graph.nodes, visited)

                if end_node_id and end_node_id != start_node.id:
                    end_node = graph.nodes[end_node_id]

                    # To avoid duplicate edges, use a canonical representation
                    # (e.g., always from the node with the smaller ID)
                    if start_node.id < end_node.id:
                        edge_pixels = [(start_y, start_x)] + path
                        edge = Edge(start_node, end_node, edge_pixels)
                        graph.add_edge(edge)

                        # Add the pixels of the traced path to the visited set
                        for p in path:
                            visited.add(p)

    return graph


def _find_node_pixels(skeleton: np.ndarray) -> Tuple[Dict[Tuple[int, int], str], np.ndarray]:
    """Finds pixels that are junctions or endpoints and returns their type."""
    kernel = np.array([[1, 1, 1],
                       [1, 10, 1],
                       [1, 1, 1]], dtype=np.uint8)

    convolved = cv2.filter2D(skeleton, -1, kernel)

    endpoints_mask = ((convolved == 11) & (skeleton > 0))
    junctions_mask = ((convolved > 12) & (skeleton > 0))

    node_pixels = {}
    for y, x in np.argwhere(endpoints_mask):
        node_pixels[(y, x)] = 'endpoint'
    for y, x in np.argwhere(junctions_mask):
        node_pixels[(y, x)] = 'junction'

    return node_pixels, (endpoints_mask | junctions_mask)


def _trace_path(skeleton: np.ndarray, start_pixel: Tuple[int, int],
                start_node_id: str, nodes: Dict[str, Node],
                visited: Set[Tuple[int, int]]) -> Tuple[List[Tuple[int, int]], str]:
    """Traces a path along the skeleton from a start pixel until a node is found."""
    path = []
    current_y, current_x = start_pixel
    prev_y, prev_x = map(int, start_node_id.split(','))

    while True:
        pixel_id = f"{current_y},{current_x}"
        if pixel_id in nodes:
            return path, pixel_id

        if (current_y, current_x) in visited:
            return [], ""

        path.append((current_y, current_x))

        found_next = False
        for dy in [-1, 0, 1]:
            for dx in [-1, 0, 1]:
                if dy == 0 and dx == 0:
                    continue

                ny, nx = current_y + dy, current_x + dx

                if ny == prev_y and nx == prev_x:
                    continue

                if 0 <= ny < skeleton.shape[0] and 0 <= nx < skeleton.shape[1] and skeleton[ny, nx] > 0:
                    prev_y, prev_x = current_y, current_x
                    current_y, current_x = ny, nx
                    found_next = True
                    break
            if found_next:
                break

        if not found_next:
            # Reached a dead end that wasn't classified as a node
            return [], ""
