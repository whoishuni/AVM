from PyInstaller.utils.hooks import collect_data_files

# Collect all necessary data files for QtWebEngine
datas = collect_data_files("PyQt6.QtWebEngineCore")