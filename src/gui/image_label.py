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
        if event.button() == Qt.MouseButton.MiddleButton and self.current_pixmap:
            self.is_panning = True
            self.last_pan_pos = event.pos()
            self.setCursor(Qt.CursorShape.ClosedHandCursor)
        elif event.button() == Qt.MouseButton.LeftButton:
            image_coords = self.get_image_coords(event.pos())
            if not image_coords:
                return

            if self.main_window.app_state == AppState.MARKING_PATH:
                self.point_clicked.emit(image_coords)
            elif self.main_window.drawing_mode == DrawingMode.NOISE_ROI:
                self.is_drawing_roi = True
                self.current_drawing_roi = QRect(image_coords, image_coords)
                self.update()

    def mouseMoveEvent(self, event):
        if self.is_panning:
            delta = event.pos() - self.last_pan_pos
            self.pan_offset += delta
            self.last_pan_pos = event.pos()
            self.update()
        elif self.is_drawing_roi:
            end_pos = self.get_image_coords(event.pos())
            if end_pos and self.current_drawing_roi:
                self.current_drawing_roi.setBottomRight(end_pos)
                self.update()

    def mouseReleaseEvent(self, event):
        if event.button() == Qt.MouseButton.MiddleButton and self.is_panning:
            self.is_panning = False
            self.setCursor(Qt.CursorShape.ArrowCursor)
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