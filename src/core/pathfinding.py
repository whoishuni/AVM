import numpy as np
import heapq
import math
from typing import Optional, List, Tuple, Callable
from .graph_builder import Graph, Node, Edge

def find_path_astar(
    cost_map: np.ndarray,
    start: Tuple[int, int],
    end: Tuple[int, int],
    identity_map: Optional[np.ndarray],
    width_map: Optional[np.ndarray],
    main_vessel_width: float,
    params: dict,
    vessel_graph: Optional[Graph] = None,
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


def find_path_astar_graph(
    graph: Graph,
    start_node: Node,
    end_node: Node,
    width_map: np.ndarray,
    identity_map: np.ndarray,
    params: dict
) -> Optional[List[Tuple[int, int]]]:
    """
    Finds the optimal path between two nodes in a graph using A*.
    This version incorporates a direction inertia penalty.
    """
    # Pre-calculate edge properties if they don't exist
    if not hasattr(graph, '_edge_properties_calculated'):
        for edge in graph.edges:
            pixel_coords = np.array(edge.pixels)
            y_coords, x_coords = pixel_coords[:, 0], pixel_coords[:, 1]

            edge.avg_width = np.mean(width_map[y_coords, x_coords])

            ids = identity_map[y_coords, x_coords]
            # Find the most frequent non-zero ID
            unique_ids, counts = np.unique(ids[ids > 0], return_counts=True)
            edge.vessel_id = unique_ids[np.argmax(counts)] if len(unique_ids) > 0 else 0
        graph._edge_properties_calculated = True


    open_set = [(0, start_node.id, None)]  # (f_cost, node_id, incoming_edge_id)
    came_from = {}
    g_costs = {start_node.id: 0}

    while open_set:
        _, current_id, incoming_edge_repr = heapq.heappop(open_set)

        if current_id == end_node.id:
            return _reconstruct_graph_path(graph, came_from, current_id, width_map, identity_map)

        current_node = graph.nodes[current_id]
        incoming_edge = next((e for e in graph.edges if repr(e) == incoming_edge_repr), None)

        for edge in graph.adjacency[current_id]:
            neighbor_node = edge.node2 if edge.node1.id == current_id else edge.node1

            # --- Cost Calculation ---
            move_cost = edge.length

            # --- Penalties ---
            direction_penalty = 0
            cross_vessel_penalty = 0

            if incoming_edge:
                # 1. Direction Inertia Penalty
                incoming_vector = np.array(incoming_edge.vector) * (-1 if incoming_edge.node2.id != current_id else 1)
                outgoing_vector = np.array(edge.vector) * (-1 if edge.node1.id != current_id else 1)
                dot_product = np.dot(incoming_vector, outgoing_vector)
                cosine_similarity = min(1.0, max(-1.0, dot_product))
                direction_penalty = params.get("DIRECTION_INERTIA_WEIGHT", 100.0) * (1.0 - cosine_similarity)

                # 2. Cross Vessel Penalty
                if hasattr(incoming_edge, 'vessel_id') and hasattr(edge, 'vessel_id'):
                    if incoming_edge.vessel_id != edge.vessel_id:
                        cross_vessel_penalty = params.get("CROSS_VESSEL_PENALTY", 1e6)


            new_g_cost = g_costs.get(current_id, float('inf')) + move_cost + direction_penalty + cross_vessel_penalty

            if neighbor_node.id not in g_costs or new_g_cost < g_costs[neighbor_node.id]:
                g_costs[neighbor_node.id] = new_g_cost
                h_cost = np.linalg.norm(np.array((neighbor_node.y, neighbor_node.x)) - np.array((end_node.y, end_node.x)))
                f_cost = new_g_cost + h_cost

                heapq.heappush(open_set, (f_cost, neighbor_node.id, repr(edge)))
                came_from[neighbor_node.id] = (current_id, repr(edge))

    return None # No path found


def _reconstruct_graph_path(graph: Graph, came_from: dict, current_id: str, width_map: np.ndarray, identity_map: np.ndarray) -> List[Tuple[int, int]]:
    """Reconstructs the path from the came_from dictionary, returning a list of pixels."""
    total_path = []

    # Add the pixels of the final node
    current_node = graph.nodes[current_id]
    total_path.append((current_node.y, current_node.x))

    while current_id in came_from:
        prev_id, edge_repr = came_from[current_id]

        edge = next((e for e in graph.edges if repr(e) == edge_repr), None)

        if edge:
            edge_pixels = list(edge.pixels)
            # If the path is from node2 to node1, reverse the pixels
            if edge.node2.id == current_id:
                edge_pixels.reverse()

            # Prepend the pixels to the total path
            total_path = edge_pixels + total_path

        current_id = prev_id

    return total_path