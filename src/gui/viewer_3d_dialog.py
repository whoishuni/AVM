import numpy as np
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QProgressBar
from PyQt6.QtCore import Qt, QThread, pyqtSignal
import pyvista as pv
from pyvistaqt import QtInteractor
from typing import Optional, Tuple, List

class VolumeRenderThread(QThread):
    """A thread to render the volume to avoid freezing the GUI."""
    finished = pyqtSignal()
    progress = pyqtSignal(int)

    def __init__(self, plotter, data, mask_volume=None, flow_data=None, path_data=None, parent=None):
        super().__init__(parent)
        self.plotter = plotter
        self.data = data
        self.mask_volume = mask_volume
        self.flow_data = flow_data
        self.path_data = path_data

    def run(self):
        # --- Render Base Volume ---
        self.progress.emit(10)
        grid = pv.ImageData(
            dimensions=self.data.shape[::-1],
            origin=(0, 0, 0),
            spacing=(1, 1, 1)
        )
        grid.point_data["values"] = self.data.flatten(order="F")
        self.progress.emit(25)

        self.plotter.add_volume(grid, cmap="bone", opacity="linear", shade=False)
        self.progress.emit(40)

        # --- Render Mask as a Mesh Overlay ---
        if self.mask_volume is not None and np.any(self.mask_volume):
            mask_grid = pv.ImageData(
                dimensions=self.mask_volume.shape[::-1],
                origin=(0, 0, 0),
                spacing=(1, 1, 1)
            )
            mask_grid.point_data["mask"] = self.mask_volume.flatten(order="F")

            contours = mask_grid.contour([1], scalars="mask")
            if contours.n_points > 0:
                 self.plotter.add_mesh(contours, name="vessel_mask", color="red", opacity=0.4)
        self.progress.emit(60)

        # --- Render Flow Vectors as Glyphs ---
        if self.flow_data is not None:
            points, vectors = self.flow_data
            if points.any() and vectors.any():
                vector_poly = pv.PolyData(points)
                vector_poly['vectors'] = vectors
                glyphs = vector_poly.glyph(orient='vectors', factor=10.0, scale=False)
                self.plotter.add_mesh(glyphs, color='cyan')
        self.progress.emit(80)

        # --- Render 3D Path as a Tube ---
        if self.path_data is not None and len(self.path_data) > 1:
            # Convert (z, y, x) path to (x, y, z) for PyVista
            points_xyz = np.array([(p[2], p[1], p[0]) for p in self.path_data])
            spline = pv.Spline(points_xyz, 100)
            tube = spline.tube(radius=2)
            self.plotter.add_mesh(tube, color='yellow')

        self.plotter.camera_position = 'iso'
        self.plotter.reset_camera()
        self.progress.emit(100)
        self.finished.emit()


class YC_3DViewerDialog(QDialog):
    """A dialog to display a 3D rendering of the image volume using PyVista."""
    point_picked = pyqtSignal(tuple) # Signal to emit the picked (z, y, x) point

    def __init__(self, image_volume: np.ndarray, mask_volume: Optional[np.ndarray] = None,
                 flow_data: Optional[Tuple[np.ndarray, np.ndarray]] = None,
                 path_data: Optional[List[Tuple[int, int, int]]] = None, parent=None):
        super().__init__(parent)
        self.setWindowTitle("3D Volume Viewer - Hold 'P' to pick points")
        self.setGeometry(150, 150, 800, 600)
        self.set_stylesheet()

        self.image_volume = image_volume
        self.mask_volume = mask_volume
        self.flow_data = flow_data
        self.path_data = path_data

        self.layout = QVBoxLayout(self)
        self.plotter = QtInteractor(self)
        self.layout.addWidget(self.plotter.interactor)

        # Progress bar
        self.progress_bar = QProgressBar(self)
        self.progress_bar.setRange(0, 100)
        self.progress_bar.setValue(0)
        self.progress_bar.setTextVisible(True)
        self.progress_bar.setFormat("Loading Volume... %p%")
        self.layout.addWidget(self.progress_bar)

        self.render_volume()

    def set_stylesheet(self):
        self.setStyleSheet("""
            QDialog {
                background-color: #383838;
            }
            QProgressBar {
                border: 1px solid #555;
                border-radius: 5px;
                text-align: center;
                color: #E0E0E0;
            }
            QProgressBar::chunk {
                background-color: #00A0A0;
                width: 20px;
            }
        """)

    def render_volume(self):
        self.progress_bar.setValue(0)
        self.render_thread = VolumeRenderThread(self.plotter, self.image_volume, self.mask_volume, self.flow_data, self.path_data)
        self.render_thread.progress.connect(self.progress_bar.setValue)
        self.render_thread.finished.connect(self.on_render_finished)
        self.render_thread.start()

    def on_render_finished(self):
        self.progress_bar.setVisible(False)
        self.layout.removeWidget(self.progress_bar)
        self.progress_bar.deleteLater()
        # Enable point picking, using the 'vessel_mask' mesh as the target
        self.plotter.enable_point_picking(callback=self._handle_pick, use_mesh=True, show_point=True, color='yellow', point_size=10)

    def _handle_pick(self, *args):
        # We get the picked point coordinates directly from the callback argument
        if args and len(args) > 0 and isinstance(args[0], np.ndarray):
            picked_xyz = args[0]
            # Convert xyz to our (z, y, x) format
            picked_zyx = tuple(np.round(picked_xyz).astype(int)[::-1])

            # Ensure the point is within the mask bounds and on a vessel
            if self.mask_volume is not None and \
               all(0 <= val < dim for val, dim in zip(picked_zyx, self.mask_volume.shape)) and \
               self.mask_volume[picked_zyx] > 0:
                self.point_picked.emit(picked_zyx)

    def closeEvent(self, event):
        """Ensure the plotter is properly closed to free up resources."""
        self.plotter.close()
        super().closeEvent(event)
