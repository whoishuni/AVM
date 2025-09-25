# hooks/hook-plotly.py
from PyInstaller.utils.hooks import collect_data_files

# Instruct PyInstaller to collect all data files for the plotly package.
# This includes the necessary JavaScript for offline rendering.
datas = collect_data_files('plotly')