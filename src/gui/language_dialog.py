from PyQt6.QtWidgets import QDialog, QVBoxLayout, QLabel, QDialogButtonBox
from typing import Optional

class YC_LanguageSelectionDialog(QDialog):
    """
    A dialog to allow the user to select the application language at startup.
    """
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("Select Language / 選擇語言")
        self.setModal(True)

        self.layout = QVBoxLayout(self)

        self.label = QLabel("Please select your preferred language:\n請選擇您的首選語言：")
        self.layout.addWidget(self.label)

        self.button_box = QDialogButtonBox()
        self.en_button = self.button_box.addButton("English", QDialogButtonBox.ButtonRole.AcceptRole)
        self.zh_button = self.button_box.addButton("繁體中文 (Traditional Chinese)", QDialogButtonBox.ButtonRole.AcceptRole)
        self.layout.addWidget(self.button_box)

        self.en_button.clicked.connect(self.select_english)
        self.zh_button.clicked.connect(self.select_chinese)

        self.selected_language: Optional[str] = None

    def select_english(self):
        self.selected_language = "en"
        self.accept()

    def select_chinese(self):
        self.selected_language = "zh"
        self.accept()

    def get_selected_language(self) -> str:
        return self.selected_language or "en" # Default to English if somehow closed