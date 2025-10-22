from typing import Optional
from PyQt6.QtWidgets import QLabel, QSizePolicy
from PyQt6.QtGui import QPixmap, QPainter, QPen, QColor, QBrush
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QRect
from utils.helpers import AppState, DrawingMode

class YC_ImageLabel(QLabel):
    """
    A custom QLabel for displaying images with interactive capabilities.
    This label handles scaling, zooming, panning, and drawing ROIs.
    """
    point_clicked = pyqtSignal(QPoint)
    roi_drawn = pyqtSignal(QRect)
    expert_path_point_clicked = pyqtSignal(QPoint)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.main_window = parent
        self.current_pixmap: Optional[QPixmap] = None
        self.current_drawing_roi: Optional[QRect] = None
        self.is_drawing_roi: bool = False

        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.is_panning = False
        self.last_pan_pos = QPoint()

        self.engineering_mode = False
        self.current_polygon_points = []
        self.polygons = {} # Store polygons per frame index

        self.setMouseTracking(True)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        self.setMinimumSize(400, 400)
        self.setAlignment(Qt.AlignmentFlag.AlignCenter)

    def setPixmap(self, pixmap: QPixmap):
        self.current_pixmap = pixmap
        self.reset_zoom()
        self.update()

    def reset_zoom(self):
        self.zoom_factor = 1.0
        self.pan_offset = QPoint(0, 0)
        self.update()

    def zoom_in(self):
        self.zoom(1.25)

    def zoom_out(self):
        self.zoom(0.8)

    def zoom(self, factor: float):
        self.zoom_factor *= factor
        self.zoom_factor = max(0.1, min(self.zoom_factor, 20.0))
        self.update()

    def get_image_coords(self, event_pos: QPoint) -> Optional[QPoint]:
        if not self.current_pixmap:
            return None

        widget_center = self.rect().center()
        pixmap_size = self.current_pixmap.size() * self.zoom_factor
        top_left = widget_center - QPoint(int(pixmap_size.width() / 2), int(pixmap_size.height() / 2)) + self.pan_offset

        if not QRect(top_left, pixmap_size).contains(event_pos):
            return None

        relative_pos = event_pos - top_left
        img_x = relative_pos.x() / self.zoom_factor
        img_y = relative_pos.y() / self.zoom_factor

        return QPoint(int(img_x), int(img_y))

    def mousePressEvent(self, event):
        is_pan_mode = self.main_window.drawing_mode == DrawingMode.PAN
        if (event.button() == Qt.MouseButton.LeftButton and is_pan_mode) or \
           (event.button() == Qt.MouseButton.MiddleButton and self.current_pixmap):
            self.is_panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            image_coords = self.get_image_coords(event.pos())
            if not image_coords:
                return

            if self.engineering_mode and self.main_window.app_state == AppState.ANNOTATING_POLYGON:
                if self.main_window.engineering_mode_toggle.currentText() == "Draw Vessel Polygons":
                    self.handle_polygon_drawing(image_coords)
                else:
                    self.expert_path_point_clicked.emit(image_coords)
            elif self.main_window.app_state == AppState.MARKING_PATH and self.main_window.drawing_mode == DrawingMode.MARKING:
                self.point_clicked.emit(image_coords)
            elif self.main_window.drawing_mode == DrawingMode.NOISE_ROI:
                self.is_drawing_roi = True
                self.current_drawing_roi = QRect(image_coords, image_coords)
                self.update()

    def mouseMoveEvent(self, event):
        if self.is_panning and self.current_pixmap:
            delta = event.pos() - self.last_pan_pos
            self.pan_offset += delta
            self.last_pan_pos = event.pos()
            self.update()
        elif self.is_drawing_roi:
            end_pos = self.get_image_coords(event.pos())
            if end_pos and self.current_drawing_roi:
                self.current_drawing_roi.setBottomRight(end_pos)
                self.update()
        else:
            super().mouseMoveEvent(event)

    def mouseReleaseEvent(self, event):
        is_pan_mode = self.main_window.drawing_mode == DrawingMode.PAN
        if (event.button() == Qt.MouseButton.LeftButton and is_pan_mode) or \
           (event.button() == Qt.MouseButton.MiddleButton and self.is_panning):
            self.is_panning = False
            self.setCursor(Qt.CursorShape.OpenHandCursor if is_pan_mode else Qt.CursorShape.ArrowCursor)
        elif event.button() == Qt.MouseButton.LeftButton and self.is_drawing_roi:
            self.is_drawing_roi = False
            if self.current_drawing_roi and self.current_drawing_roi.width() > 5 and self.current_drawing_roi.height() > 5:
                self.roi_drawn.emit(self.current_drawing_roi.normalized())
            self.current_drawing_roi = None
            self.update()

    def wheelEvent(self, event):
        if event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if event.angleDelta().y() > 0:
                self.zoom_in()
            else:
                self.zoom_out()
            event.accept()
        else:
            super().wheelEvent(event)

    def keyPressEvent(self, event):
        if self.engineering_mode and event.key() == Qt.Key.Key_Z and event.modifiers() == Qt.KeyboardModifier.ControlModifier:
            if self.current_polygon_points:
                self.current_polygon_points.pop()
                self.update()
        else:
            super().keyPressEvent(event)

    def set_engineering_mode(self, enabled: bool):
        self.engineering_mode = enabled
        self.current_polygon_points = []
        if not enabled:
            self.polygons = {}
        self.update()

    def get_polygons(self):
        return self.polygons

    def handle_polygon_drawing(self, point: QPoint):
        # If the user clicks on the first point, close the polygon
        if self.current_polygon_points and (point - self.current_polygon_points[0]).manhattanLength() < 10:
            if len(self.current_polygon_points) > 2:
                frame_idx = self.main_window.current_frame_index
                if frame_idx not in self.polygons:
                    self.polygons[frame_idx] = []
                self.polygons[frame_idx].append([(p.x(), p.y()) for p in self.current_polygon_points])
                self.current_polygon_points = []
        else:
            self.current_polygon_points.append(point)

        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.current_pixmap:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.setRenderHint(QPainter.RenderHint.SmoothPixmapTransform)

        widget_center = self.rect().center()
        pixmap_size = self.current_pixmap.size() * self.zoom_factor
        target_rect = QRect(
            int(widget_center.x() - pixmap_size.width() / 2 + self.pan_offset.x()),
            int(widget_center.y() - pixmap_size.height() / 2 + self.pan_offset.y()),
            int(pixmap_size.width()),
            int(pixmap_size.height())
        )

        painter.drawPixmap(target_rect, self.current_pixmap, self.current_pixmap.rect())

        if self.is_drawing_roi and self.current_drawing_roi:
            if self.main_window.drawing_mode == DrawingMode.NOISE_ROI:
                pen = QPen(QColor(255, 0, 0, 255), 2, Qt.PenStyle.DashLine)
                brush = QBrush(QColor(255, 0, 0, 70))
            else:
                return

            painter.setPen(pen)
            painter.setBrush(brush)

            display_roi = QRect(
                int(target_rect.x() + self.current_drawing_roi.x() * self.zoom_factor),
                int(target_rect.y() + self.current_drawing_roi.y() * self.zoom_factor),
                int(self.current_drawing_roi.width() * self.zoom_factor),
                int(self.current_drawing_roi.height() * self.zoom_factor)
            )
            painter.drawRect(display_roi.normalized())

        if self.engineering_mode:
            painter.setPen(QPen(QColor(0, 255, 0, 200), 2))
            painter.setBrush(QBrush(QColor(0, 255, 0, 50)))

            # Draw completed polygons for the current frame
            frame_idx = self.main_window.current_frame_index
            if frame_idx in self.polygons:
                for poly_points in self.polygons[frame_idx]:
                    q_poly = [QPoint(p[0], p[1]) * self.zoom_factor + target_rect.topLeft() for p in poly_points]
                    painter.drawPolygon(q_poly)

            # Draw the current polygon being created
            if self.current_polygon_points:
                pen = QPen(QColor(255, 255, 0, 255), 2, Qt.PenStyle.DashLine)
                painter.setPen(pen)

                # Draw lines between points
                if len(self.current_polygon_points) > 1:
                    for i in range(len(self.current_polygon_points) - 1):
                        p1 = self.current_polygon_points[i] * self.zoom_factor + target_rect.topLeft()
                        p2 = self.current_polygon_points[i+1] * self.zoom_factor + target_rect.topLeft()
                        painter.drawLine(p1, p2)

                # Draw points
                for p in self.current_polygon_points:
                    center = p * self.zoom_factor + target_rect.topLeft()
                    painter.drawEllipse(center, 4, 4)

                # Draw a line to the current mouse position
                mouse_pos = self.mapFromGlobal(self.cursor().pos())
                if target_rect.contains(mouse_pos):
                     last_point = self.current_polygon_points[-1] * self.zoom_factor + target_rect.topLeft()
                     painter.drawLine(last_point, mouse_pos)