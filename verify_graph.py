import os
import cv2
import numpy as np
import glob

# Add the 'src' directory to the Python path
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'src')))

from core.image_processing import create_enhanced_vessel_masks
from core.flow_analyzer import FlowAnalyzer
from gui.main_window import YC_VesselTracerApp

def run_offline_verification():
    """
    Runs the full pipeline offline: image processing, flow graph generation,
    pathfinding, and saves a visualization of the final path.
    """
    print("Starting offline path verification...")

    # --- 1. Load Data ---
    data_path = 'data/1/'
    image_files = sorted(glob.glob(os.path.join(data_path, '*.jpg')))
    image_files = [f for f in image_files if "範例" not in f]
    if not image_files:
        print(f"Error: No images found in {data_path}")
        return
    print(f"Found {len(image_files)} images to process.")
    images = [cv2.imread(f, cv2.IMREAD_GRAYSCALE) for f in image_files]
    if any(img is None for img in images):
        print("Error: Failed to load one or more images.")
        return

    # --- 2. Get Default Parameters ---
    params = YC_VesselTracerApp.DEFAULT_PARAMS.copy()
    print("Using default parameters for processing.")

    # --- 3. Process Images ---
    print("Generating vessel masks...")
    vessel_masks = create_enhanced_vessel_masks(images=images, noise_rois=[], bg_color=240, params=params, smoothing_level=4, worker_thread=None)
    if not vessel_masks:
        print("Error: Failed to generate vessel masks.")
        return
    print(f"Successfully generated {len(vessel_masks)} vessel masks.")

    # --- 4. Analyze Flow ---
    print("Initializing FlowAnalyzer...")
    flow_analyzer = FlowAnalyzer(vessel_masks, images, params)
    print("Building flow graph...")
    flow_graph = flow_analyzer.analyze()
    if not flow_graph:
        print("Error: FlowAnalyzer failed to build the graph.")
        return
    print("Flow graph built successfully.")

    # --- 5. Find Path ---
    # SWAPPED: Start with the point that resolved to an earlier frame
    # and end with the point that resolved to a later frame.
    start_point = (450, 385)
    end_point = (186, 126)
    print(f"Finding path from {start_point} to {end_point}...")

    path_segments, cost = flow_analyzer.find_path(start_point, end_point)

    if not path_segments:
        print("Error: Failed to find a path.")
        return
    print(f"Path found with {len(path_segments)} segments and cost {cost}.")

    # --- 6. Reconstruct and Visualize Path ---
    print("Reconstructing path for visualization...")

    # Create a blank mask to draw the combined path segments
    h, w = images[0].shape
    path_mask = np.zeros((h, w), dtype=np.uint8)

    # Combine the masks of all segments in the path
    for segment in path_segments:
        path_mask = np.maximum(path_mask, segment.mask)

    # Skeletonize the combined mask to get a 1-pixel wide centerline
    _, binary_path_mask = cv2.threshold(path_mask, 0, 255, cv2.THRESH_BINARY)
    skeleton = cv2.ximgproc.thinning(binary_path_mask)

    # Prepare the first image for drawing
    visualization_image = cv2.cvtColor(images[0], cv2.COLOR_GRAY2BGR)

    # Draw the skeleton path in green
    visualization_image[skeleton > 0] = [0, 255, 0] # Green color for the path

    # Draw start and end points for context (use original logic for color)
    cv2.circle(visualization_image, (end_point[1], end_point[0]), 5, [0, 0, 255], -1) # Red circle for start
    cv2.circle(visualization_image, (start_point[1], start_point[0]), 5, [255, 0, 0], -1) # Blue circle for end

    output_path = "path_verification.png"
    cv2.imwrite(output_path, visualization_image)
    print(f"Successfully saved final path visualization to {os.path.abspath(output_path)}")

if __name__ == '__main__':
    run_offline_verification()
