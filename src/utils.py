import re
from enum import Enum, auto

# --- Global Helper Functions ---

def natural_sort_key(s: str) -> list:
    """Provides a key for natural sorting of filenames."""
    return [int(text) if text.isdigit() else text.lower() for text in re.split('([0-9]+)', s)]


# --- State Management Enums ---

class AppState(Enum):
    IDLE = auto()
    LOADED = auto()
    MARKING_PATH = auto()
    RANGE_CONFIRMED = auto()
    PROCESSING = auto()
    DONE = auto()


class DrawingMode(Enum):
    NOISE_ROI = auto()
