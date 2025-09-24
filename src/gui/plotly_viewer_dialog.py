from PyQt6.QtWidgets import QDialog, QVBoxLayout, QDialogButtonBox
from PyQt6.QtWebEngineWidgets import QWebEngineView

class YC_PlotlyViewerDialog(QDialog):
    """A dialog for displaying an interactive Plotly graph using QWebEngineView."""
    def __init__(self, html_content: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("YC Interactive 3D View")
        self.setMinimumSize(900, 700)

        self.layout = QVBoxLayout(self)
        self.webview = QWebEngineView()
        self.layout.addWidget(self.webview)

        self.webview.setHtml(html_content)

        self.button_box = QDialogButtonBox(QDialogButtonBox.StandardButton.Close)
        self.layout.addWidget(self.button_box)
        self.button_box.rejected.connect(self.reject)