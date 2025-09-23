import numpy as np
import heapq
import cv2
from typing import Optional, List, Tuple, Dict
from skimage.morphology import skeletonize
from scipy.spatial import cKDTree

class VesselNode:
    """Represents a single point in the vessel structure across time."""
    _id_counter = 0

    def __init__(self, y: int, x: int, z: int):
        self.y, self.x, self.z = y, x, z  # y, x for array coords, z for frame index
        self.parent: Optional['VesselNode'] = None
        self.children: List['VesselNode'] = []
        self.id = VesselNode._id_counter
        VesselNode._id_counter += 1

    @property
    def pos(self) -> Tuple[int, int, int]:
        return self.y, self.x, self.z

    def __repr__(self) -> str:
        return f"Node(id={self.id}, pos=({self.y}, {self.x}, {self.z}))"

    def __lt__(self, other):
        # heapq needs a way to compare nodes if priorities are equal
        return self.id < other.id


class VesselMemory:
    """A graph-based data structure to hold the temporal memory of vessel formation."""

    def __init__(self):
        self.nodes: Dict[int, VesselNode] = {}  # All nodes by their unique ID
        self.frame_to_nodes: Dict[int, List[VesselNode]] = {}  # Nodes organized by frame
        # A map to quickly find a node if it already exists for a given coordinate
        self._coord_to_node_id: Dict[Tuple[int, int, int], int] = {}

    def add_node(self, y: int, x: int, z: int) -> VesselNode:
        """Creates a new node and adds it to the memory. If a node already exists at the given coordinates, returns the existing node."""
        coord = (y, x, z)
        if coord in self._coord_to_node_id:
            return self.nodes[self._coord_to_node_id[coord]]

        new_node = VesselNode(y, x, z)
        self.nodes[new_node.id] = new_node
        if z not in self.frame_to_nodes:
            self.frame_to_nodes[z] = []
        self.frame_to_nodes[z].append(new_node)
        self._coord_to_node_id[coord] = new_node.id
        return new_node

    def link_nodes(self, parent_node: VesselNode, child_node: VesselNode):
        """Creates a parent-child link between two nodes."""
        if child_node and parent_node and child_node not in parent_node.children:
            child_node.parent = parent_node
            parent_node.children.append(child_node)

    def get_nodes_in_frame(self, frame_index: int) -> List[VesselNode]:
        """Returns all nodes for a specific frame."""
        return self.frame_to_nodes.get(frame_index, [])

    def find_closest_node(self, y: int, x: int, z_frame: int, max_dist: int = 100) -> Optional[VesselNode]:
        """Finds the closest node in a specific frame to a given 2D coordinate."""
        nodes_in_frame = self.get_nodes_in_frame(z_frame)
        if not nodes_in_frame:
            return None

        node_coords = np.array([[node.y, node.x] for node in nodes_in_frame])
        point_coord = np.array([y, x])

        kdtree = cKDTree(node_coords)
        dist, idx = kdtree.query(point_coord)

        if dist <= max_dist:
            return nodes_in_frame[idx]
        return None

    def clear(self):
        """Resets the memory."""
        self.nodes.clear()
        self.frame_to_nodes.clear()
        self._coord_to_node_id.clear()
        VesselNode._id_counter = 0

def build_vessel_memory(masks: List[np.ndarray], link_threshold: int = 10) -> VesselMemory:
    """
    Builds a temporal graph representation of the vessel structure from a series of masks.
    """
    memory = VesselMemory()
    skeletons = [skeletonize(mask > 0) for mask in masks]

    # First frame: Add all skeleton points as initial nodes
    z = 0
    if z < len(skeletons):
        initial_points = np.argwhere(skeletons[0])
        for y_coord, x_coord in initial_points:
            memory.add_node(y_coord, x_coord, z)

    # Process subsequent frames
    for z in range(1, len(skeletons)):
        prev_points = np.argwhere(skeletons[z-1])
        curr_points = np.argwhere(skeletons[z])

        if len(prev_points) == 0 or len(curr_points) == 0:
            continue

        # Use a KD-tree for efficient nearest neighbor search
        kdtree = cKDTree(prev_points)
        distances, indices = kdtree.query(curr_points)

        # Link points that are close enough
        for i, dist in enumerate(distances):
            if dist <= link_threshold:
                # Current point
                cy, cx = curr_points[i]
                child_node = memory.add_node(cy, cx, z)

                # Corresponding parent point
                py, px = prev_points[indices[i]]
                parent_node = memory.add_node(py, px, z - 1)

                memory.link_nodes(parent_node, child_node)

    return memory

def find_path_on_graph(start_node: VesselNode, end_node: VesselNode) -> Optional[List[VesselNode]]:
    """
    Performs A* search on the VesselMemory graph.
    """
    if not start_node or not end_node:
        return None

    def heuristic(node_a: VesselNode, node_b: VesselNode) -> float:
        """Calculates the 3D Euclidean distance between two nodes."""
        pos_a = np.array(node_a.pos)
        pos_b = np.array(node_b.pos)
        return np.linalg.norm(pos_a - pos_b)

    open_set = [(0, start_node)]  # (f_cost, node)
    came_from: Dict[VesselNode, VesselNode] = {}
    g_costs: Dict[VesselNode, float] = {start_node: 0}

    while open_set:
        _, current_node = heapq.heappop(open_set)

        if current_node == end_node:
            # Reconstruct path
            path = []
            while current_node in came_from:
                path.append(current_node)
                current_node = came_from[current_node]
            path.append(start_node)
            return path[::-1]

        # Get neighbors (parent and children)
        neighbors = current_node.children[:]
        if current_node.parent:
            neighbors.append(current_node.parent)

        for neighbor in neighbors:
            # The cost to move from current to neighbor is the 3D distance
            move_cost = heuristic(current_node, neighbor)
            new_g_cost = g_costs[current_node] + move_cost

            if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                g_costs[neighbor] = new_g_cost
                f_cost = new_g_cost + heuristic(neighbor, end_node)
                heapq.heappush(open_set, (f_cost, neighbor))
                came_from[neighbor] = current_node

    return None # Path not found

def visualize_vessel_memory(memory: VesselMemory, shape: Tuple[int, int]) -> np.ndarray:
    """
    Creates an image visualizing the entire vessel graph, colored by frame index.
    """
    vis_image = np.zeros((shape[0], shape[1], 3), dtype=np.uint8)
    if not memory.nodes:
        return vis_image

    # Find the min and max frame index for normalization
    frame_indices = [node.z for node in memory.nodes.values()]
    min_z, max_z = min(frame_indices), max(frame_indices)

    # Create a colormap
    colormap = cv2.applyColorMap(np.arange(256, dtype=np.uint8), cv2.COLORMAP_JET)

    for node in memory.nodes.values():
        if node.parent:
            # Normalize the z value to 0-255 for the colormap
            if max_z > min_z:
                norm_z = int(255 * (node.z - min_z) / (max_z - min_z))
            else:
                norm_z = 0

            color = tuple(int(c) for c in colormap[norm_z][0])

            p1_xy = (node.x, node.y)
            p2_xy = (node.parent.x, node.parent.y)
            cv2.line(vis_image, p1_xy, p2_xy, color, 1)

    return vis_image
