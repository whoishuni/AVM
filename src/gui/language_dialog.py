from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QPushButton
from typing import Optional

class YC_LanguageSelectionDialog(QDialog):
    """
    A dialog to allow the user to select the application language at startup.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Language / 選擇語言")
        self.setModal(True)
        self.set_stylesheet()

        self.layout = QVBoxLayout(self)
        self.layout.setSpacing(15)

        self.label = QLabel("Please select your preferred language:\n請選擇您的首選語言：")
        self.layout.addWidget(self.label)

        self.en_button = QPushButton("English")
        self.zh_button = QPushButton("繁體中文 (Traditional Chinese)")

        self.layout.addWidget(self.en_button)
        self.layout.addWidget(self.zh_button)

        self.en_button.clicked.connect(self.select_english)
        self.zh_button.clicked.connect(self.select_chinese)

        self.selected_language: Optional[str] = None

    def set_stylesheet(self):
        style = """
            QDialog {
                background-color: #2E2E2E;
                color: #E0E0E0;
                padding: 20px;
            }
            QLabel {
                color: #D0D0D0;
                font-size: 14px;
                margin-bottom: 10px;
            }
            QPushButton {
                background-color: #555555;
                color: #EEEEEE;
                border: 1px solid #666666;
                padding: 8px 16px;
                border-radius: 4px;
                font-size: 13px;
                min-height: 40px;
            }
            QPushButton:hover {
                background-color: #686868;
                border: 1px solid #777777;
            }
            QPushButton:pressed {
                background-color: #4A4A4A;
            }
        """
        self.setStyleSheet(style)

    def select_english(self):
        self.selected_language = "en"
        self.accept()

    def select_chinese(self):
        self.selected_language = "zh"
        self.accept()

    def get_selected_language(self) -> str:
        return self.selected_language or "en"