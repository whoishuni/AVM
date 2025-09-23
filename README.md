# Vessel Tracing Application

This application performs 3D vessel tracing from a sequence of 2D images using a graph-based approach.

## How to Run

The main application is contained in `app_main.py`. To run the application, execute the following command in your terminal:

```bash
python app_main.py
```

## File Notes

*   `app_main.py`: This is the main, updated application script. It contains all the new features, including the graph-based memory model and visualization.
*   `vessel_graph.py`: This module contains the data structures (`VesselNode`, `VesselMemory`) and core logic for building, traversing, and visualizing the vessel graph.
*   `穩定2DAVM.py`: This is the original script. Due to technical limitations in the development environment regarding the handling of non-ASCII filenames, this file could not be modified or deleted. Please ignore this file and use `app_main.py` instead.
