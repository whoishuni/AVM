import numpy as np
import heapq
import math
from typing import Optional, List, Tuple, Callable, Set, TYPE_CHECKING

if TYPE_CHECKING:
    from PyQt5.QtCore import QPoint


def find_closest_pixel_on_mask(point: 'QPoint', mask_img: np.ndarray, max_radius: int) -> Optional[Tuple[int, int]]:
    """Finds the closest pixel on a mask to a given point within a radius."""
    if mask_img is None or np.sum(mask_img) == 0:
        return None

    valid_points = np.argwhere(mask_img > 0)
    if valid_points.size == 0:
        return None

    point_coords = np.array([point.y(), point.x()])
    distances = np.linalg.norm(valid_points - point_coords, axis=1)
    min_dist_idx = np.argmin(distances)

    if distances[min_dist_idx] <= max_radius:
        return tuple(valid_points[min_dist_idx])
    else:
        return None


def find_path_astar(cost_map: np.ndarray,
                    start: Tuple[int, int],
                    end: Tuple[int, int],
                    obstacle_cost: float,
                    time_cost_weight: float,
                    turn_penalty_weight: float,
                    viz_callback: Optional[Callable[[List[Tuple[int, int]]], None]] = None
                    ) -> Optional[List[Tuple[int, int]]]:
    """A* pathfinding algorithm with a turn penalty."""
    if cost_map[start] >= obstacle_cost or cost_map[end] >= obstacle_cost:
        return None

    def heuristic(p1: Tuple[int, int], p2: Tuple[int, int]) -> float:
        return np.sqrt((p1[0] - p2[0]) ** 2 + (p1[1] - p2[1]) ** 2)

    open_set: List[Tuple[float, float, Tuple[int, int]]] = [(heuristic(start, end), 0, start)]  # (f_cost, g_cost, pos)
    came_from: dict[Tuple[int, int], Tuple[int, int]] = {}
    g_costs: dict[Tuple[int, int], float] = {start: 0}
    closed_set: Set[Tuple[int, int]] = set()

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
                if dr == 0 and dc == 0:
                    continue
                neighbor = (current[0] + dr, current[1] + dc)

                if not (0 <= neighbor[0] < cost_map.shape[0] and 0 <= neighbor[1] < cost_map.shape[1]) or \
                        cost_map[neighbor] >= obstacle_cost or \
                        neighbor in closed_set:
                    continue

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
                        turn_penalty = turn_penalty_weight * (1.0 - cosine_similarity)

                move_cost = np.sqrt(dr ** 2 + dc ** 2)
                time_cost = time_cost_weight * cost_map[neighbor]
                new_g_cost = g_costs[current] + move_cost + time_cost + turn_penalty

                if neighbor not in g_costs or new_g_cost < g_costs[neighbor]:
                    g_costs[neighbor] = new_g_cost
                    f_cost = new_g_cost + heuristic(neighbor, end)
                    heapq.heappush(open_set, (f_cost, new_g_cost, neighbor))
                    came_from[neighbor] = current

    return None
