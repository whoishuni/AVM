import numpy as np
import heapq
import math
from typing import Optional, List, Tuple, Callable

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

def find_path_astar_3d(
    mask_volume: np.ndarray,
    start_node: Tuple[int, int, int],
    end_node: Tuple[int, int, int],
    flow_vectors: Optional[np.ndarray] = None,
    flow_points: Optional[np.ndarray] = None,
    flow_weight: float = 10.0
) -> Optional[List[Tuple[int, int, int]]]:
    """
    Finds a path in a 3D volume using the A* algorithm, guided by a flow field.

    Args:
        mask_volume: A 3D numpy array where non-zero values are traversable.
        start_node: The (z, y, x) starting coordinate.
        end_node: The (z, y, x) ending coordinate.
        flow_vectors: An (N, 3) array of flow vectors (vz, vy, vx).
        flow_points: An (N, 3) array of coordinates (z, y, x) for the flow vectors.
        flow_weight: A multiplier for the cost/reward of following the flow.

    Returns:
        A list of (z, y, x) tuples representing the path, or None if no path is found.
    """
    if mask_volume[start_node] == 0 or mask_volume[end_node] == 0:
        return None

    # Helper to find the nearest flow vector for a given point
    flow_kdtree = None
    if flow_points is not None:
        from scipy.spatial import cKDTree
        flow_kdtree = cKDTree(flow_points)

    def get_flow_vector_at(point: Tuple[int, int, int]) -> np.ndarray:
        if flow_kdtree is None:
            return np.array([1, 0, 0]) # Default flow: forward in time
        dist, idx = flow_kdtree.query(point)
        if idx < len(flow_vectors):
            return flow_vectors[idx]
        return np.array([1, 0, 0])

    def heuristic(a: Tuple[int, int, int], b: Tuple[int, int, int]) -> float:
        return np.linalg.norm(np.array(a) - np.array(b))

    open_set = [(0, start_node)]  # (f_cost, node)
    came_from = {}
    g_costs = {start_node: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if current == end_node:
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            return path[::-1]

        # Explore 26 neighbors in 3D
        for dz in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dz == 0 and dy == 0 and dx == 0:
                        continue

                    neighbor = (current[0] + dz, current[1] + dy, current[2] + dx)

                    # Check boundaries and if the neighbor is in a vessel
                    if not (0 <= neighbor[0] < mask_volume.shape[0] and
                            0 <= neighbor[1] < mask_volume.shape[1] and
                            0 <= neighbor[2] < mask_volume.shape[2] and
                            mask_volume[neighbor] > 0):
                        continue

                    move_vector = np.array([dz, dy, dx])
                    move_dist = np.linalg.norm(move_vector)

                    # Cost for moving against the flow
                    flow_cost = 0
                    if flow_vectors is not None:
                        flow_vec = get_flow_vector_at(current)
                        # Normalize move vector
                        move_vector_norm = move_vector / move_dist
                        # Cosine similarity: 1 if aligned, -1 if opposite
                        cosine_sim = np.dot(move_vector_norm, flow_vec)
                        # Cost is high when moving against the flow (cosine_sim is negative)
                        flow_cost = flow_weight * (1 - cosine_sim)

                    new_g_cost = g_costs.get(current, float('inf')) + move_dist + flow_cost

                    if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                        g_costs[neighbor] = new_g_cost
                        f_cost = new_g_cost + heuristic(neighbor, end_node)
                        heapq.heappush(open_set, (f_cost, neighbor))
                        came_from[neighbor] = current

    return None # No path found