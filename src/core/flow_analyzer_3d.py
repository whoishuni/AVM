import numpy as np
from scipy.ndimage import center_of_mass
from typing import Optional, Tuple

class FlowAnalyzer3D:
    """
    Analyzes a 3D vessel mask (x, y, time) to determine blood flow direction.
    """
    def __init__(self, mask_volume: np.ndarray):
        """
        Initializes the analyzer with a 3D mask volume.

        Args:
            mask_volume: A 3D numpy array where non-zero values represent vessels.
                         Shape: (height, width, num_frames).
        """
        if mask_volume is None or mask_volume.ndim != 3:
            raise ValueError("A valid 3D mask volume is required.")
        self.mask = mask_volume
        self.height, self.width, self.num_frames = self.mask.shape

    def analyze_flow(self, downsample_factor: int = 4) -> Optional[Tuple[np.ndarray, np.ndarray]]:
        """
        Analyzes the flow by calculating the temporal gradient and centroid shifts.
        This simplified approach generates a vector field representing the dominant flow.

        Args:
            downsample_factor: The factor by which to downsample the points for
                               which flow vectors are calculated, to improve performance.

        Returns:
            A tuple containing:
            - points (np.ndarray): An array of (x, y, z) coordinates for the vectors.
            - vectors (np.ndarray): An array of (vx, vy, vz) flow vectors.
            Returns None if the mask is empty.
        """
        if np.sum(self.mask) == 0:
            return None

        # 1. Calculate temporal gradient: where the signal appears first
        # We use np.gradient along the time axis (axis=2)
        # A positive gradient means the signal is increasing (contrast arriving)
        grad_z = np.gradient(self.mask.astype(float), axis=2)

        # We are interested in the first significant arrival of contrast
        # Let's find the frame index of the max gradient for each (x, y) pixel
        first_arrival_time = np.argmax(grad_z, axis=2).astype(float)

        # Suppress gradients where there is no vessel
        first_arrival_time[np.sum(self.mask, axis=2) == 0] = np.nan

        # 2. Calculate the gradient of this arrival time map.
        # The gradient of the arrival time will point in the direction of flow.
        grad_y, grad_x = np.gradient(first_arrival_time)

        # 3. Create the vector field
        points = []
        vectors = []

        # Subsample the points to avoid overcrowding the plot
        for z in range(0, self.num_frames, downsample_factor):
            # Get vessel points at this frame
            frame_mask = self.mask[:, :, z]
            vessel_points_yx = np.argwhere(frame_mask > 0)

            # Subsample points within the frame as well
            for y, x in vessel_points_yx[::downsample_factor**2]:
                if not np.isnan(grad_x[y, x]) and not np.isnan(grad_y[y, x]):
                    # The vector points "downhill" on the arrival time map
                    vx, vy = -grad_x[y, x], -grad_y[y, x]
                    # The z-component of the vector should always point forward in time
                    vz = 1.0

                    # Normalize the vector
                    norm = np.linalg.norm([vx, vy, vz])
                    if norm > 0:
                        vx, vy, vz = (vx/norm, vy/norm, vz/norm)

                        points.append([x, y, z])
                        vectors.append([vx, vy, vz])

        if not points:
            return None

        return np.array(points), np.array(vectors)
