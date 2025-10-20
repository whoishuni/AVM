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
    vessel_masks: List[np.ndarray],
    start_info: dict,
    end_info: dict,
    params: dict
) -> Optional[List[Tuple[int, int]]]:
    """
    Finds a path using a 3D A* algorithm, where the dimensions are (y, x, time).

    This algorithm searches through the sequence of vessel masks, ensuring that
    the resulting path is temporally contiguous.

    Args:
        vessel_masks: A list of binary numpy arrays, where each array is a frame.
        start_info: A dictionary with "point" (y, x) and "frame" index for the start.
        end_info: A dictionary with "point" (y, x) and "frame" index for the end.
        params: A dictionary of tuning parameters.

    Returns:
        A list of (y, x) tuples for the path projection on the 2D plane, or None.
    """
    start_node = (start_info["point"][0], start_info["point"][1], start_info["frame"])
    end_node_2d = (end_info["point"][0], end_info["point"][1])
    num_frames = len(vessel_masks)

    obstacle_cost = params.get("PATHFINDING_OBSTACLE_COST", 1e9)
    time_cost_weight = params.get("TIME_COST_WEIGHT", 1.0)
    turn_penalty_weight = params.get("TURN_PENALTY_WEIGHT", 50.0)

    # Check if start or end points are valid
    if not (0 <= start_node[2] < num_frames and vessel_masks[start_node[2]][start_node[0], start_node[1]] > 0):
        return None
    if not (0 <= end_info["frame"] < num_frames and vessel_masks[end_info["frame"]][end_info["point"][0], end_info["point"][1]] > 0):
         # Even if the exact end point is not on a vessel (due to clicking error),
         # the search should still proceed and find the nearest valid vessel point.
         pass


    open_set = [(0, start_node)]  # (f_cost, node)
    came_from = {}
    g_costs = {start_node: 0}

    def heuristic(node):
        # 3D Euclidean distance
        dist_y = abs(node[0] - end_node_2d[0])
        dist_x = abs(node[1] - end_node_2d[1])
        dist_t = abs(node[2] - end_info["frame"])
        return np.sqrt(dist_y**2 + dist_x**2 + (dist_t * time_cost_weight)**2)

    while open_set:
        _, current = heapq.heappop(open_set)
        current_y, current_x, current_t = current

        # Goal check: If we are at or past the target frame, we can consider this a potential end point
        if current_t >= end_info["frame"]:
            # Check if this point is the closest we've found so far in the target frame
            dist_to_end = np.linalg.norm(np.array(current[:2]) - np.array(end_node_2d))

            # A simple greedy approach: if we hit the exact target, we are done
            if dist_to_end == 0:
                break
            # Otherwise, we continue searching for a potentially better path that ends closer
            # A more complex implementation could store the best path found so far and prune searches
            # that are already more expensive.

        # --- Process Neighbors ---
        # Neighbors are in the current frame and the next frame
        for dt in [0, 1]:
            next_t = current_t + dt
            if not (0 <= next_t < num_frames):
                continue

            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dy == 0 and dx == 0 and dt == 0:
                        continue

                    neighbor_y, neighbor_x = current_y + dy, current_x + dx
                    neighbor = (neighbor_y, neighbor_x, next_t)

                    # Boundary and obstacle check
                    if not (0 <= neighbor_y < vessel_masks[0].shape[0] and 0 <= neighbor_x < vessel_masks[0].shape[1]) \
                       or vessel_masks[next_t][neighbor_y, neighbor_x] == 0:
                        continue

                    # --- Cost Calculation ---
                    move_cost = np.sqrt(dy**2 + dx**2 + (dt * time_cost_weight)**2)

                    # Turn penalty
                    turn_penalty = 0
                    if current in came_from:
                        parent = came_from[current]
                        v1 = (current_y - parent[0], current_x - parent[1], (current_t - parent[2]))
                        v2 = (neighbor_y - current_y, neighbor_x - current_x, (next_t - current_t))

                        dot_product = v1[0] * v2[0] + v1[1] * v2[1] + v1[2] * v2[2]
                        mag1 = np.sqrt(v1[0]**2 + v1[1]**2 + v1[2]**2)
                        mag2 = np.sqrt(v2[0]**2 + v2[1]**2 + v2[2]**2)

                        if mag1 > 0 and mag2 > 0:
                            cosine_similarity = dot_product / (mag1 * mag2)
                            turn_penalty = turn_penalty_weight * (1.0 - cosine_similarity)


                    new_g_cost = g_costs.get(current, float('inf')) + move_cost + turn_penalty

                    if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                        g_costs[neighbor] = new_g_cost
                        f_cost = new_g_cost + heuristic(neighbor)
                        heapq.heappush(open_set, (f_cost, neighbor))
                        came_from[neighbor] = current

    # After the search, reconstruct the path that ends closest to the target
    best_end_node = None
    min_dist = float('inf')

    # Find the node in the closed set that is in the target frame and closest to the end point
    for node in g_costs:
        if node[2] >= end_info["frame"]:
            dist = np.linalg.norm(np.array(node[:2]) - np.array(end_node_2d))
            if dist < min_dist:
                min_dist = dist
                best_end_node = node

    if best_end_node:
        path = []
        current = best_end_node
        while current in came_from:
            path.append((current[0], current[1])) # Project to 2D
            current = came_from[current]
        path.append((start_node[0], start_node[1]))
        return path[::-1]

    return None # No path found