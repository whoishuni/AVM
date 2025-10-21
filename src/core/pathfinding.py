import numpy as np
import heapq
import math
from typing import Optional, List, Tuple, Callable, Dict
from collections import deque
import networkx as nx

from .flow_analyzer import FlowNode
from .vessel_graph import VesselGraph


def find_path_on_graph(
    vessel_graph: VesselGraph,
    start_pixel_qpoint: any,  # Actually a QPoint
    start_frame: int,
    end_pixel_qpoint: any,    # Actually a QPoint
    end_frame: int,
    max_search_radius: int
) -> Optional[List[Tuple[int, int]]]:
    """
    Finds a path in the vessel graph from a start point to an end point using A*.
    """
    start_pixel = (start_pixel_qpoint.y(), start_pixel_qpoint.x())
    end_pixel = (end_pixel_qpoint.y(), end_pixel_qpoint.x())

    # 1. Find the closest graph nodes to the start and end pixels
    start_node = _find_closest_graph_node(vessel_graph, start_pixel, start_frame, max_search_radius)
    end_node = _find_closest_graph_node(vessel_graph, end_pixel, end_frame, max_search_radius)

    if start_node is None:
        print(f"Error: Could not map start point {start_pixel} in frame {start_frame} to a graph node.")
        return None
    if end_node is None:
        print(f"Error: Could not map end point {end_pixel} in frame {end_frame} to a graph node.")
        return None

    # 2. Define cost and heuristic functions for A*
    def cost_func(u, v, edge_data):
        # The cost of traversing an edge is its length in pixels
        return len(edge_data.get('pixels', []))

    def heuristic_func(u, v):
        # Use Euclidean distance as the heuristic
        pos1 = np.array(vessel_graph.graph.nodes[u]['pos'])
        pos2 = np.array(vessel_graph.graph.nodes[v]['pos'])
        return np.linalg.norm(pos1 - pos2)

    # 3. Run A* on the graph
    try:
        node_path = nx.astar_path(
            vessel_graph.graph,
            source=start_node,
            target=end_node,
            heuristic=heuristic_func,
            weight=cost_func
        )
    except nx.NetworkXNoPath:
        print("No path found between the specified points in the vessel graph.")
        return None

    # 4. Reconstruct the pixel path from the node path
    if not node_path or len(node_path) < 2:
        return None

    pixel_path = []
    for i in range(len(node_path) - 1):
        u = node_path[i]
        v = node_path[i+1]
        edge_data = vessel_graph.graph.get_edge_data(u, v)
        if 'pixels' in edge_data:
            # Ensure the pixel path flows in the correct direction
            segment_pixels = edge_data['pixels']
            if tuple(segment_pixels[0]) == vessel_graph.graph.nodes[u]['pos']:
                pixel_path.extend(segment_pixels)
            else:
                pixel_path.extend(segment_pixels[::-1])

    return pixel_path

from scipy.spatial import KDTree

def _find_closest_graph_node(
    vessel_graph: VesselGraph,
    pixel: Tuple[int, int],
    frame_index: int,
    max_dist: int
) -> Optional[int]:
    """
    Finds the closest node in the VesselGraph to a pixel within a given frame
    using a KDTree for efficient searching.
    """
    # Filter nodes belonging to the specific frame
    frame_nodes = [
        (node_id, data['pos'])
        for node_id, data in vessel_graph.graph.nodes(data=True)
        if data.get('frame') == frame_index
    ]

    if not frame_nodes:
        return None

    node_ids, positions_yx = zip(*frame_nodes)

    # KDTree works with (x, y) coordinates, so we need to swap them
    positions_xy = [(pos[1], pos[0]) for pos in positions_yx]
    pixel_xy = (pixel[1], pixel[0])

    # Create a KDTree from the node positions (x, y)
    kdtree = KDTree(positions_xy)

    # Query the KDTree for the nearest neighbor using the (x, y) pixel
    distance, index = kdtree.query(pixel_xy)

    # Check if the found node is within the maximum allowed distance
    if distance <= max_dist:
        return node_ids[index]

    return None


def find_path_in_flow_graph(
    flow_graph: Dict[str, FlowNode],
    start_pixel_qpoint: any, # Actually a QPoint
    start_frame: int,
    end_pixel_qpoint: any, # Actually a QPoint
    end_frame: int,
    max_search_radius: int
) -> Optional[List[Tuple[int, int]]]:
    """
    Finds a path in the directed flow graph from a start point to an end point.
    The search is constrained to follow the parent-to-child links.
    """
    start_pixel = (start_pixel_qpoint.y(), start_pixel_qpoint.x())
    end_pixel = (end_pixel_qpoint.y(), end_pixel_qpoint.x())

    start_node = _find_closest_node_to_pixel(flow_graph, start_pixel, start_frame, max_search_radius)
    end_node = _find_closest_node_to_pixel(flow_graph, end_pixel, end_frame, max_search_radius)

    if not start_node:
        print(f"Error: Could not map start point {start_pixel} in frame {start_frame} to a flow node.")
        return None
    if not end_node:
        print(f"Error: Could not map end point {end_pixel} in frame {end_frame} to a flow node.")
        return None

    # Handle case where start and end are the same node
    if start_node.id == end_node.id:
        return [tuple(p) for p in start_node.pixels]

    # --- Search forward from start_node's branch to find a path to end_node ---
    queue = deque([[start_node]])
    visited = {start_node.id}
    while queue:
        path = queue.popleft()
        current_node = path[-1]

        if current_node.id == end_node.id:
            # Direct path found
            return _reconstruct_pixel_path(path)

        for child_node in current_node.children:
            if child_node.id not in visited:
                new_path = list(path)
                new_path.append(child_node)
                queue.append(new_path)
                visited.add(child_node.id)

    # --- If no direct forward path, they might be on different branches or end is an ancestor of start ---
    # To handle this, find the common ancestor.
    start_ancestors = {n.id: n for n in _get_ancestors(start_node)}
    end_ancestors = _get_ancestors(end_node)

    common_ancestor_path = None
    for ancestor in end_ancestors:
        if ancestor.id in start_ancestors:
            # Found the youngest common ancestor. Now construct the path:
            # Path = (start -> ancestor) + (ancestor -> end)
            path_start_to_ancestor = _trace_path_to_ancestor(start_node, ancestor.id)
            path_ancestor_to_end = _trace_path_to_ancestor(end_node, ancestor.id)

            if path_start_to_ancestor and path_ancestor_to_end:
                 # The path from start to ancestor should be reversed
                common_ancestor_path = list(reversed(path_start_to_ancestor)) + path_ancestor_to_end[1:] # Exclude duplicate ancestor
                break

    if common_ancestor_path:
        return _reconstruct_pixel_path(common_ancestor_path)

    print("No path found between the specified points following the flow.")
    return None

def _reconstruct_pixel_path(node_path: List[FlowNode]) -> List[Tuple[int, int]]:
    """Converts a path of FlowNodes into a list of pixel coordinates."""
    pixel_path = []
    for node in node_path:
        pixel_path.extend([tuple(p) for p in node.pixels])
    return pixel_path

def _get_ancestors(node: FlowNode) -> List[FlowNode]:
    """Traces back from a node to the root, returning the list of ancestors."""
    path = []
    curr = node
    while curr:
        path.append(curr)
        curr = curr.parent
    return path

def _trace_path_to_ancestor(start_node: FlowNode, ancestor_id: str) -> Optional[List[FlowNode]]:
    """Returns the path from a start_node up to a specific ancestor."""
    path = []
    curr = start_node
    while curr:
        path.append(curr)
        if curr.id == ancestor_id:
            return path
        curr = curr.parent
    return None # Ancestor not found

def _find_closest_node_to_pixel(
    flow_graph: Dict[str, FlowNode],
    pixel: Tuple[int, int],
    frame_index: int,
    max_dist: int
) -> Optional[FlowNode]:
    """Finds the closest FlowNode to a pixel within a given frame and search radius."""
    py, px = pixel
    closest_node = None
    min_dist_sq = max_dist ** 2

    for node in flow_graph.values():
        if node.frame_index != frame_index:
            continue

        # Check if pixel is within expanded bbox first for efficiency
        min_r, min_c, max_r, max_c = node.bbox
        if not (min_r - max_dist <= py < max_r + max_dist and min_c - max_dist <= px < max_c + max_dist):
            continue

        # Find the squared distance to the closest pixel in the node
        dist_sq = np.min(np.sum((node.pixels - np.array([py, px]))**2, axis=1))

        if dist_sq < min_dist_sq:
            min_dist_sq = dist_sq
            closest_node = node

    return closest_node


def find_path_astar(
    cost_map: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    identity_map: Optional[np.ndarray],
    width_map: Optional[np.ndarray],
    main_vessel_width: float,
    params: dict,
    viz_callback: Optional[Callable[[List[Tuple[int, int]]], None]] = None
) -> Optional[List[Tuple[int, int]]]:
    """
    Finds the optimal path between two points using a modified A* algorithm.

    This implementation includes costs for distance, time (from cost_map),
    and dynamic penalties for path curvature and crossing vessel boundaries.
    The penalties are adjusted based on whether the path is on a 'main' or
    'side' vessel, determined by vessel width.

    Args:
        cost_map: The base cost map (incorporating temporal cost).
        start: The starting (y, x) coordinate tuple.
        end: The ending (y, x) coordinate tuple.
        identity_map: Map assigning a unique ID to each vessel segment.
        width_map: Map where pixel values correspond to vessel width.
        main_vessel_width: The characteristic width of the main vessel.
        params: A dictionary of tuning parameters for the algorithm.
        viz_callback: An optional function to call for visualizing the search.

    Returns:
        A list of (y, x) tuples representing the path, or None if no path is found.
    """
    obstacle_cost = params["PATHFINDING_OBSTACLE_COST"]
    if cost_map[start] >= obstacle_cost or cost_map[end] >= obstacle_cost:
        return None

    # The open set is a priority queue storing (f_cost, g_cost, position).
    # f_cost is the estimated total cost (g_cost + heuristic), but here we use g_cost
    # as the priority, making it closer to Dijkstra's algorithm.
    open_set = [(0, 0, start)]
    came_from = {}
    g_costs = {start: 0}
    closed_set = set()

    # --- Visualization variables ---
    node_counter = 0
    viz_interval = 50  # Update visualization every 50 nodes processed

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
            return path[::-1]  # Return reversed path

        closed_set.add(current)
        node_counter += 1
        if viz_callback and node_counter % viz_interval == 0:
            viz_callback(list(closed_set))

        parent = came_from.get(current)

        # --- Process Neighbors ---
        for dr in [-1, 0, 1]:
            for dc in [-1, 0, 1]:
                if dr == 0 and dc == 0:
                    continue

                neighbor = (current[0] + dr, current[1] + dc)

                # --- Boundary and Obstacle Checks ---
                if not (0 <= neighbor[0] < cost_map.shape[0] and 0 <= neighbor[1] < cost_map.shape[1]) \
                   or cost_map[neighbor] >= obstacle_cost or neighbor in closed_set:
                    continue

                # --- Calculate Penalties and Costs ---
                cosine_similarity = 1.0
                if parent:
                    v1 = (current[0] - parent[0], current[1] - parent[1])
                    v2 = (neighbor[0] - current[0], neighbor[1] - current[1])
                    dot_product = v1[0] * v2[0] + v1[1] * v2[1]
                    mag1 = math.sqrt(v1[0]**2 + v1[1]**2)
                    mag2 = math.sqrt(v2[0]**2 + v2[1]**2)
                    if mag1 > 0 and mag2 > 0:
                        cosine_similarity = min(1.0, max(-1.0, dot_product / (mag1 * mag2)))

                # Dynamic Penalties based on Vessel Type (Main vs. Side Branch)
                is_on_main_vessel = True
                if width_map is not None and main_vessel_width > 0:
                    neighbor_width = 2 * width_map[neighbor]
                    is_on_main_vessel = abs(neighbor_width - main_vessel_width) <= (main_vessel_width * params["MAIN_VESSEL_WIDTH_TOLERANCE"])

                # 1. Vessel Crossing Penalty
                cross_vessel_penalty = 0
                if identity_map is not None:
                    current_id = identity_map[current]
                    neighbor_id = identity_map[neighbor]
                    if current_id > 0 and neighbor_id > 0 and current_id != neighbor_id:
                        cross_vessel_penalty = params["CROSS_VESSEL_PENALTY"]

                # 2. Turn Penalty
                turn_penalty = params["TURN_PENALTY_WEIGHT"] * (1.0 - cosine_similarity)
                if not is_on_main_vessel:
                    turn_penalty *= params["SIDE_BRANCH_TURN_PENALTY_MULTIPLIER"]

                # 3. Movement and Time Costs
                # Penalize non-straight moves more heavily
                dynamic_cost_multiplier = 1.0 + params["DYNAMIC_COST_WEIGHT"] * (1.0 - cosine_similarity)
                move_cost = (np.sqrt(dr**2 + dc**2)) * dynamic_cost_multiplier
                time_cost = (params["TIME_COST_WEIGHT"] * cost_map[neighbor]) * dynamic_cost_multiplier

                # --- Total Cost Calculation ---
                new_g_cost = g_costs.get(current, float('inf')) + move_cost + time_cost + turn_penalty + cross_vessel_penalty

                if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                    g_costs[neighbor] = new_g_cost
                    f_cost = new_g_cost  # Using g-cost as priority
                    heapq.heappush(open_set, (f_cost, new_g_cost, neighbor))
                    came_from[neighbor] = current

    return None  # No path found
