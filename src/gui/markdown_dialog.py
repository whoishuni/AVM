import os
from PyQt6.QtWidgets import QDialog, QVBoxLayout, QTextBrowser, QPushButton
from PyQt6.QtCore import Qt

class YC_MarkdownDialog(QDialog):
    """A dialog to display Markdown content, like a README file."""

    def __init__(self, file_path: str, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Introduction")
        self.setGeometry(200, 200, 800, 600)

        self.layout = QVBoxLayout(self)

        self.text_browser = QTextBrowser()
        self.text_browser.setOpenExternalLinks(True)
        self.layout.addWidget(self.text_browser)

        self.close_button = QPushButton("Close")
        self.close_button.clicked.connect(self.accept)
        self.layout.addWidget(self.close_button, alignment=Qt.AlignmentFlag.AlignRight)

        self.load_markdown(file_path)

    def load_markdown(self, file_path: str):
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.text_browser.setMarkdown(content)
        except FileNotFoundError:
            self.text_browser.setText(f"Error: Could not find the file at '{file_path}'.")
        except Exception as e:
            self.text_browser.setText(f"An error occurred while reading the file:\n{e}")