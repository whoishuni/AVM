import numpy as np
from typing import Optional, Tuple, List
import heapq

def find_path_in_3d_volume(
    vessel_mask_volume: np.ndarray,
    start_point_2d: Tuple[int, int],
    end_point_2d: Tuple[int, int],
    time_penalty_weight: float = 1.5
) -> Optional[List[Tuple[int, int, int]]]:
    """
    Finds a path in a 3D (Y, X, Time) volume using A* algorithm.

    Args:
        vessel_mask_volume: A 3D numpy array where non-zero values represent the vessel.
        start_point_2d: A tuple (y, x) for the start point.
        end_point_2d: A tuple (y, x) for the end point.
        time_penalty_weight: Weight to penalize moving backward in time.

    Returns:
        A list of (y, x, t) tuples representing the path, or None if no path is found.
    """
    # Find the first time frame where the start and end points appear
    start_t = np.argmax(vessel_mask_volume[start_point_2d[0], start_point_2d[1], :] > 0)
    end_t = np.argmax(vessel_mask_volume[end_point_2d[0], end_point_2d[1], :] > 0)

    if not (vessel_mask_volume[start_point_2d[0], start_point_2d[1], start_t] and
            vessel_mask_volume[end_point_2d[0], end_point_2d[1], end_t]):
        return None  # Start or end point not in the vessel mask

    start_node = (start_point_2d[0], start_point_2d[1], start_t)
    end_node = (end_point_2d[0], end_point_2d[1], end_t)

    open_set = [(0, start_node)]
    came_from = {}
    g_score = {start_node: 0}

    while open_set:
        _, current = heapq.heappop(open_set)

        if (current[0], current[1]) == (end_node[0], end_node[1]):
            path = []
            while current in came_from:
                path.append(current)
                current = came_from[current]
            path.append(start_node)
            return path[::-1]

        y, x, t = current
        for dt in [-1, 0, 1]:
            for dy in [-1, 0, 1]:
                for dx in [-1, 0, 1]:
                    if dt == 0 and dy == 0 and dx == 0:
                        continue

                    neighbor_t, neighbor_y, neighbor_x = t + dt, y + dy, x + dx

                    if not (0 <= neighbor_y < vessel_mask_volume.shape[0] and
                            0 <= neighbor_x < vessel_mask_volume.shape[1] and
                            0 <= neighbor_t < vessel_mask_volume.shape[2]):
                        continue

                    if vessel_mask_volume[neighbor_y, neighbor_x, neighbor_t] == 0:
                        continue

                    move_cost = np.sqrt(dx**2 + dy**2 + dt**2)

                    # Penalize moving backward in time
                    if dt < 0:
                        move_cost *= time_penalty_weight

                    tentative_g_score = g_score[current] + move_cost

                    neighbor_node = (neighbor_y, neighbor_x, neighbor_t)
                    if neighbor_node not in g_score or tentative_g_score < g_score[neighbor_node]:
                        g_score[neighbor_node] = tentative_g_score
                        f_score = tentative_g_score + np.linalg.norm(np.array(end_node) - np.array(neighbor_node))
                        heapq.heappush(open_set, (f_score, neighbor_node))
                        came_from[neighbor_node] = current
    return None
