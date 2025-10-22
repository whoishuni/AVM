import numpy as np
import heapq
import math
from typing import Optional, List, Tuple, Callable, Dict, Any
import cv2

def create_guidance_map(shape: Tuple[int, int], annotations: Dict[str, Any], obstacle_cost: float) -> Optional[np.ndarray]:
    if not annotations or "polygons" not in annotations:
        return None

    guidance_map = np.full(shape, obstacle_cost, dtype=np.float32)

    for frame_idx_str, polygons in annotations["polygons"].items():
        for poly in polygons:
            polygon = np.array(poly, dtype=np.int32)
            cv2.fillPoly(guidance_map, [polygon], 0)

    if "paths" in annotations and annotations["paths"]:
        for i in range(0, len(annotations["paths"]), 2):
            start_info = annotations["paths"][i]
            end_info = annotations["paths"][i+1]
            # This is a simplified reward. A more complex implementation could
            # draw a line or use a distance transform.
            cv2.circle(guidance_map, (start_info["point"][0], start_info["point"][1]), 10, -100, -1)
            cv2.circle(guidance_map, (end_info["point"][0], end_info["point"][1]), 10, -100, -1)

    return guidance_map

def find_path_astar(
    cost_map: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    identity_map: Optional[np.ndarray],
    width_map: Optional[np.ndarray],
    main_vessel_width: float,
    params: dict,
    annotations: Optional[Dict[str, Any]] = None,
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
    guidance_map = create_guidance_map(cost_map.shape, annotations, obstacle_cost)

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
                guidance_cost = guidance_map[neighbor] if guidance_map is not None else 0
                new_g_cost = g_costs.get(current, float('inf')) + move_cost + time_cost + turn_penalty + cross_vessel_penalty + guidance_cost

                if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                    g_costs[neighbor] = new_g_cost
                    f_cost = new_g_cost  # Using g-cost as priority
                    heapq.heappush(open_set, (f_cost, new_g_cost, neighbor))
                    came_from[neighbor] = current

    return None  # No path found