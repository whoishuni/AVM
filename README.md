# 2D Vessel Tracer

This application is a desktop tool for semi-automatically tracing paths in 2D angiography image sequences. It's designed to help identify and delineate vessel structures, like blood vessels, through a series of images captured over time.

The key feature of this tool is its "vessel memory" system. Instead of just looking at a single flattened image, the software analyzes the image sequence frame by frame to understand how vessels grow and connect. This allows for more accurate pathfinding, preventing the algorithm from making incorrect jumps at intersections where one vessel overlaps another.

## Features

- **Image Sequence Loading**: Load a folder of images (JPG, PNG, etc.) that represent a time series.
- **Interactive Path Marking**: Click on the image to define start, middle, and end points for the vessel path you want to trace.
- **Vessel Memory Pathfinding**: A sophisticated A* pathfinding algorithm that uses temporal information to trace the most likely vessel path, avoiding jumps to unrelated overlapping vessels.
- **Noise Reduction**:
    - Interactively draw rectangular areas to exclude noise from the analysis.
    - Automatically removes large, bright areas that are not part of the vessel structure.
- **Adjustable Smoothing**: Interactively preview and set the level of Gaussian smoothing to apply before analysis.
- **Step-by-Step Visualization**: View the entire image processing and analysis pipeline, from filtering to the final path, to understand how the result was generated.

## Technical Details

This tool employs a sophisticated, multi-stage process to accurately identify and trace vessel paths. For developers looking to understand the core logic, the key components are outlined below.

### 1. Vessel Enhancement and Masking

Before pathfinding can occur, the vessels must be clearly segmented from the background. This is achieved through a pipeline designed to handle noise and enhance tubular structures:

- **Preprocessing**: Large, bright, non-vessel artifacts (like catheters or bone structures) are identified using morphological operations and removed from the image.
- **Vessel Filtering**: The core of vessel enhancement relies on a combination of three specialized filters from `scikit-image`:
    - **Frangi**: Detects vessel-like structures based on the Hessian matrix eigenvalues. It's excellent at identifying vessels of varying sizes.
    - **Sato**: Another Hessian-based filter that is also effective for detecting lines and tubes.
    - **Meijering**: A filter that is less sensitive to noise and provides a good baseline response.
- **Response Combination**: The responses from all three filters are combined by taking the pixel-wise maximum. This creates a robust "vesselness" map that leverages the strengths of each filter.
- **Binarization and Post-processing**: The combined response map is thresholded to create a binary mask. Finally, a custom gap-bridging algorithm connects small, disconnected segments by finding and linking the endpoints of their skeletons.

### 2. Pathfinding with a Custom A* Algorithm

The pathfinding is not a simple search on an image. It uses a custom A* algorithm with a unique cost function to find the most "natural" path between user-defined points. The total cost to move to a neighboring pixel is a weighted sum of several factors:

- **Dynamic Junction Scouting**: When the A* search encounters a junction (a point with multiple branching paths), it performs a "look-ahead" scout down each branch for a short distance. It calculates a "straightness score" for each branch based on how much it turns. The path that continues most directly forward is prioritized, and the other, turning paths are penalized. This allows the algorithm to make intelligent, local decisions at intersections.
- **Temporal Cost**: The cost of a pixel is proportional to the frame number in which it appears. This encourages the path to stay within vessels that appear early and persist through the image sequence, preventing it from jumping to vessels that appear in much later frames.
- **Vessel Identity Penalty**: The application builds a **Vessel Identity Map** that assigns a unique ID to each continuous vessel segment across all frames. The A* algorithm incurs a very high penalty for jumping from a pixel with one ID to a pixel with a different ID, effectively forcing it to stay within a single, connected vessel structure.
- **Local Turn Penalty**: In addition to the junction scouting, a standard local turn penalty based on the cosine similarity between the incoming and outgoing vectors is still used to ensure overall path smoothness between junctions.

This multi-faceted approach allows the algorithm to make intelligent decisions at complex intersections, preferring to follow a single, coherent vessel through time rather than simply taking the shortest spatial path.

## Setup and Installation

This application is built with Python and requires several common scientific and GUI libraries.

### Prerequisites

- Python 3.6+

### Dependencies

The required Python libraries are:
- `PyQt5`: For the graphical user interface.
- `opencv-python`: For core image processing functions.
- `numpy`: For numerical operations and image array manipulation.
- `scikit-image`: For advanced image filtering and morphology (Frangi, Sato, skeletonize).
- `scikit-learn`: (Implicit dependency, good to have).

You can install all required dependencies using pip:

```bash
pip install PyQt5 opencv-python numpy scikit-image scikit-learn
```

## How to Use

1.  **Run the Application**: Execute the `baseline.py` script from your terminal:
    ```bash
    python baseline.py
    ```

2.  **Step 1: Load Images**
    - Click the **"Select Image Folder"** button.
    - Navigate to and select the directory containing your image sequence. The images will be loaded and sorted automatically.
    - The first frame will be displayed. You can navigate through the frames using the **slider** or the **'A' (previous) and 'D' (next) keys**.

3.  **Step 2: Mark Path Points**
    - The application will now say "Start Marking Path". Click this button to enter marking mode.
    - Click on the image to place points along the vessel you wish to trace. You must define at least a start and an end point.
    - You can place points on different frames by navigating with the slider or A/D keys.
    - Once you have at least two points, click **"Confirm Points"**. The view will update to a Maximum Intensity Projection of the frame range you selected.

4.  **Step 3: Configure and Analyze**
    - After confirming points, a new set of tools will appear.
    - **(Optional) Adjust Smoothing**: Click **"Adjust Smoothing"** to open a live preview dialog. Use the slider to find a setting that makes the vessels clear without losing detail.
    - **(Optional) Draw Noise Area**: Click **"Draw Noise Area"**. Then, click and drag on the image to draw a red box around any area you want the algorithm to ignore. You can add multiple noise areas.
    - **Run Analysis**: When you are ready, click **"Run Full Analysis"**.

5.  **Step 4: View Results**
    - A progress bar will appear while the analysis runs.
    - After processing, a **"Processing Step Viewer"** dialog will automatically open.
    - Use the **"Next"** and **"Previous"** buttons to see each step of the analysis, including:
        - Filtering and masking.
        - The colored **Vessel Identity Map**, which shows how the "memory" feature has grouped vessel segments.
        - The final A* path drawn on the vessel mask.
        - The final result with the path overlayed on the original image.
    - Close the step viewer to see the final image in the main window.

6.  **Reset**
    - To start a new analysis, click the **"Reset All"** button. This will clear all data and return the application to its initial state.
