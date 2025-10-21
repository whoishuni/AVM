
import sys
import os
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

import numpy as np
from PyQt6.QtCore import QPoint

from src.core.image_processing import create_enhanced_vessel_masks, create_roi_mask
from src.core.vessel_graph import VesselGraph
from src.core.pathfinding import find_path_on_graph
from src.utils.helpers import load_images_from_folder, get_most_frequent_color

def run_verification():
    """
    An offline script to verify the vessel graph construction and temporal
    pathfinding without running the full GUI.
    """
    print("Starting offline verification...")

    # --- 1. Load Images ---
    data_folder = 'data/1'
    print(f"Loading images from: {data_folder}")
    images = load_images_from_folder(data_folder)
    if not images:
        print("Error: Could not load images.")
        return

    # --- 2. Set Parameters ---
    params = {
        "BG_REMOVAL_THRESHOLD_OFFSET": 15,
        "BG_REMOVAL_KERNEL_SIZE": 15,
        "MAX_NODE_SEARCH_RADIUS": 100,
        "MAX_GAP_BRIDGE_DISTANCE": 30,
        "MAX_TEMPORAL_LINKING_DISTANCE": 20,
    }
    smoothing_level = 4
    noise_rois = []
    global_background_color = get_most_frequent_color(images[0])

    print("Parameters set.")

    # --- 3. Generate Masks ---
    print("Generating vessel masks...")
    roi_mask = create_roi_mask(images)
    masks = create_enhanced_vessel_masks(
        images, noise_rois, global_background_color, params, smoothing_level, None
    )
    if not masks:
        print("Error: Failed to generate vessel masks.")
        return
    print(f"Generated {len(masks)} masks.")

    # --- 4. Build Vessel Graph ---
    print("Building vessel graph...")
    vessel_graph = VesselGraph()
    vessel_graph.build_from_sequence(masks, params, roi_mask, None)
    if not vessel_graph.graph:
        print("Error: Failed to build vessel graph.")
        return
    print(f"Vessel graph built with {len(vessel_graph.graph.nodes)} nodes and {len(vessel_graph.graph.edges)} edges.")

    # --- 5. Define Start and End Points for Pathfinding ---
    # These points are chosen based on manual inspection of the 'data/1' sequence
    start_point = QPoint(250, 400) # An early point in the vessel
    start_frame = 2
    end_point = QPoint(350, 150)   # A later point in the vessel
    end_frame = 8

    print(f"Attempting to find path from frame {start_frame} at {start_point} to frame {end_frame} at {end_point}.")

    # --- 6. Run Pathfinding ---
    path = find_path_on_graph(
        vessel_graph,
        start_point, start_frame,
        end_point, end_frame,
        params["MAX_NODE_SEARCH_RADIUS"]
    )

    # --- 7. Debug Visualization ---
    debug_frame_index = 2
    debug_mask = masks[debug_frame_index]
    debug_image = np.stack([debug_mask, debug_mask, debug_mask], axis=-1)

    frame_nodes = [
        data['pos']
        for node_id, data in vessel_graph.graph.nodes(data=True)
        if data.get('frame') == debug_frame_index
    ]

    for y, x in frame_nodes:
        debug_image[y, x] = [255, 0, 0] # Draw nodes in red

    import cv2
    cv2.imwrite("verify_temporal_path_debug.png", debug_image)
    print(f"Saved debug image to verify_temporal_path_debug.png")


    # --- 8. Report Results ---
    if path:
        print(f"SUCCESS: Path found with {len(path)} points.")
        # You could extend this to save the path or visualize it
    else:
        print("FAILURE: No path was found between the specified points.")

if __name__ == "__main__":
    run_verification()
