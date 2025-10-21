# --- Standard Library Imports ---
from typing import List, Optional, Tuple

# --- Third-Party Library Imports ---
import networkx as nx
import numpy as np
from skimage.morphology import skeletonize
import cv2

# --- Local Project Imports ---
from utils.threading import ProgressUpdater


class VesselGraph:
    """
    A class to represent the vascular structure as a directed graph (DiGraph),
    capturing vessel connectivity, directionality, and temporal evolution.
    """

    def __init__(self):
        """Initializes an empty VesselGraph."""
        self.graph = nx.DiGraph()
        # Node data will store attributes like:
        # - pos: (y, x) coordinates
        # - frame: the frame number (time)
        # - type: 'junction', 'endpoint', or 'body'

        # Edge data will store attributes like:
        # - pixels: a list of (y, x) coordinates forming the vessel segment
        # - length: the length of the segment in pixels
        # - direction_vector: a vector representing the segment's orientation

    def build_from_sequence(
        self,
        masks: List[np.ndarray],
        params: dict,
        roi_mask: Optional[np.ndarray] = None,
        worker_thread: Optional[ProgressUpdater] = None
    ) -> None:
        """
        Constructs the spatio-temporal vessel graph from a sequence of binary masks.

        Args:
            masks: A list of binary vessel masks, one for each frame.
            roi_mask: An optional mask to confine the processing to a specific region.
            worker_thread: An optional handle to update a progress bar in the UI.
        """
        if not masks:
            return

        self.clear()
        total_frames = len(masks)

        prev_graph = None

        for i, mask in enumerate(masks):
            if worker_thread and not worker_thread.is_running:
                return

            # Apply ROI mask if provided
            current_mask = mask
            if roi_mask is not None:
                current_mask = cv2.bitwise_and(mask, mask, mask=roi_mask)

            frame_graph = self._build_graph_for_frame(current_mask, i)

            # Merge the nodes and edges from the frame-specific graph into the main graph
            # We need to relabel nodes to ensure they are unique across frames
            node_mapping = {n: len(self.graph) + n for n in frame_graph.nodes()}
            nx.relabel_nodes(frame_graph, node_mapping, copy=False)
            self.graph.add_nodes_from(frame_graph.nodes(data=True))
            self.graph.add_edges_from(frame_graph.edges(data=True))

            if prev_graph:
                self._link_graphs_temporally(prev_graph, frame_graph, params)

            prev_graph = frame_graph

            if worker_thread:
                worker_thread.update(i + 1, total_frames, f"Building graph for frame {i+1}/{total_frames}")

    def _build_graph_for_frame(self, mask: np.ndarray, frame_index: int) -> nx.DiGraph:
        """
        Builds a vessel graph for a single frame.

        Args:
            mask: The binary vessel mask for the current frame.
            frame_index: The index of the current frame.

        Returns:
            A NetworkX DiGraph representing the vessels in the single frame.
        """
        if mask is None or not np.any(mask):
            return nx.DiGraph()

        frame_graph = nx.DiGraph()

        # 1. Skeletonize the mask to get vessel centerlines
        skeleton = skeletonize(mask / 255).astype(np.uint8) * 255

        # 2. Find junctions and endpoints (nodes)
        node_pixels = self._find_node_pixels(skeleton)

        # 3. Add nodes to the graph
        node_id_counter = 0
        pixel_to_node_map = {}
        for (y, x), node_type in node_pixels.items():
            frame_graph.add_node(
                node_id_counter,
                pos=(y, x),
                frame=frame_index,
                type=node_type
            )
            pixel_to_node_map[(y, x)] = node_id_counter
            node_id_counter += 1

        # 4. Trace the paths between nodes (edges)
        self._trace_and_add_edges(frame_graph, skeleton, pixel_to_node_map)

        return frame_graph

    def _trace_and_add_edges(self, graph: nx.DiGraph, skeleton: np.ndarray, pixel_to_node_map: dict):
        """
        Traces paths between nodes in the skeleton and adds them as edges to the graph.
        """
        visited_pixels = set(pixel_to_node_map.keys())

        for start_pixel, start_node_id in pixel_to_node_map.items():
            # For each neighbor of the start pixel
            for neighbor in self._get_neighbors(start_pixel, skeleton):
                if neighbor in visited_pixels:
                    continue

                # Start tracing a path
                path = [start_pixel, neighbor]
                prev_pixel = start_pixel
                current_pixel = neighbor

                while True:
                    visited_pixels.add(current_pixel)

                    # Check if we've reached another node
                    if current_pixel in pixel_to_node_map:
                        end_node_id = pixel_to_node_map[current_pixel]
                        # Add a bidirectional edge for now; directionality will be determined later
                        graph.add_edge(start_node_id, end_node_id, pixels=path)
                        graph.add_edge(end_node_id, start_node_id, pixels=path[::-1])
                        break

                    next_neighbors = [p for p in self._get_neighbors(current_pixel, skeleton) if p != prev_pixel]

                    if len(next_neighbors) == 1:
                        # Continue the path
                        prev_pixel = current_pixel
                        current_pixel = next_neighbors[0]
                        path.append(current_pixel)
                    else:
                        # Path dead-ends without finding another node (should be rare)
                        break

    def _get_neighbors(self, pixel: Tuple[int, int], image: np.ndarray) -> List[Tuple[int, int]]:
        """
        Gets the 8-connected neighbors of a pixel that are part of the skeleton.
        """
        y, x = pixel
        neighbors = []
        rows, cols = image.shape
        for i in range(max(0, y - 1), min(rows, y + 2)):
            for j in range(max(0, x - 1), min(cols, x + 2)):
                if (i, j) == (y, x):
                    continue
                if image[i, j] > 0:
                    neighbors.append((i, j))
        return neighbors

    def _find_node_pixels(self, skeleton: np.ndarray) -> dict:
        """
        Finds junction and endpoint pixels in a skeleton image.

        Args:
            skeleton: A binary, 1-pixel-wide skeleton image.

        Returns:
            A dictionary mapping pixel coordinates (y, x) to their type ('junction' or 'endpoint').
        """
        # Use a 3x3 kernel to count neighbors
        kernel = np.array([[1, 1, 1],
                           [1, 10, 1],
                           [1, 1, 1]], dtype=np.uint8)

        # We divide by 255 to convert the skeleton to a 0/1 image for convolution
        convolved = cv2.filter2D(skeleton // 255, -1, kernel)

        # Endpoints have 1 neighbor (value will be 11: 10 for the center + 1 for the neighbor)
        endpoints = np.argwhere((convolved == 11) & (skeleton > 0))
        # Junctions have > 2 neighbors (value > 12: 10 for center + >2 for neighbors)
        junctions = np.argwhere((convolved > 12) & (skeleton > 0))

        node_pixels = {}
        for y, x in endpoints:
            node_pixels[(y, x)] = 'endpoint'
        for y, x in junctions:
            node_pixels[(y, x)] = 'junction'

        return node_pixels

    def _link_graphs_temporally(self, prev_graph: nx.DiGraph, current_graph: nx.DiGraph, params: dict) -> None:
        """
        Links nodes between the graphs of two consecutive frames to represent blood flow.

        This is a critical step that establishes the direction of edges in the final graph.

        Args:
            prev_graph: The graph from the previous frame (t-1).
            current_graph: The graph from the current frame (t).
            params: A dictionary of tuning parameters.
        """
        max_linking_distance = params.get("MAX_TEMPORAL_LINKING_DISTANCE", 20)

        for u, u_data in current_graph.nodes(data=True):
            min_dist = float('inf')
            best_candidate = None

            u_pos = np.array(u_data['pos'])

            for v, v_data in prev_graph.nodes(data=True):
                v_pos = np.array(v_data['pos'])
                dist = np.linalg.norm(u_pos - v_pos)

                if dist < min_dist:
                    min_dist = dist
                    best_candidate = v

            if best_candidate is not None and min_dist < max_linking_distance:
                # Add a temporal link from the node in the previous frame to the current one
                self.graph.add_edge(best_candidate, u, type='temporal')

    def get_downstream_subgraph(self, start_node: int) -> nx.DiGraph:
        """
        Returns the subgraph containing all nodes and edges reachable from a given start node.

        Args:
            start_node: The ID of the node from which to start the traversal.

        Returns:
            A new DiGraph representing the downstream vascular structure.
        """
        if start_node not in self.graph:
            return nx.DiGraph()

        # Use a breadth-first search or depth-first search to find all reachable nodes
        downstream_nodes = nx.descendants(self.graph, start_node)
        downstream_nodes.add(start_node)

        return self.graph.subgraph(downstream_nodes)

    def clear(self) -> None:
        """Resets the graph to an empty state."""
        self.graph.clear()
