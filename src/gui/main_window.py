import sys
import os
import numpy as np
import cv2
from typing import List, Optional, Tuple, Dict, Any

# --- PyQt6 Imports ---
from PyQt6.QtWidgets import (
    QApplication, QWidget, QVBoxLayout, QHBoxLayout, QPushButton,
    QFileDialog, QLabel, QStatusBar, QMainWindow, QMessageBox,
    QSizePolicy, QProgressDialog, QSlider, QDialog, QGroupBox, QStyle,
    QCheckBox, QComboBox, QInputDialog, QLineEdit
)
from PyQt6.QtGui import QPixmap, QFont, QAction, QKeySequence, QIcon
from PyQt6.QtCore import Qt, QPoint, pyqtSignal, QRect

# --- Project-specific Imports ---
from gui.image_label import YC_ImageLabel
from gui.smoothing_dialog import YC_SmoothingPreviewDialog
from gui.step_viewer_dialog import YC_StepViewerDialog
from gui.help_dialog import YC_HelpDialog
from gui.parameter_dialog import YC_ParameterDialog
from gui.markdown_dialog import YC_MarkdownDialog
from core.image_processing import (
    create_enhanced_vessel_masks, create_maximum_intensity_projection,
    create_temporal_cost_map, build_vessel_identity_map, create_vessel_layers,
    create_combined_identity_map, generate_mask_steps
)
from core.pathfinding import find_path_astar
from utils.helpers import (
    load_images_from_folder, get_most_frequent_color, find_closest_pixel_on_mask,
    convert_np_to_pixmap, AppState, DrawingMode, create_yc_icon,
    load_central_annotations, save_central_annotations
)
from utils.threading import ProgressUpdater
import plotly.graph_objects as go
import subprocess
import webbrowser
import tempfile
import pathlib

class YC_VesselTracerApp(QMainWindow):
    """The main application window for the YC 2D vessel tracing tool."""
    DEFAULT_PARAMS = {
        "BG_REMOVAL_THRESHOLD_OFFSET": 15, "BG_REMOVAL_KERNEL_SIZE": 15,
        "MAX_NODE_SEARCH_RADIUS": 50, "MAX_GAP_BRIDGE_DISTANCE": 30,
        "FORBIDDEN_ZONE_RADIUS": 30, "TIME_COST_WEIGHT": 1.0,
        "PATHFINDING_OBSTACLE_COST": 1e9, "TURN_PENALTY_WEIGHT": 50.0,
        "DYNAMIC_COST_WEIGHT": 5.0, "STRAIGHT_PATH_THRESHOLD": 0.9,
        "CROSS_VESSEL_PENALTY": 1e6, "MAIN_VESSEL_WIDTH_TOLERANCE": 0.30,
        "SIDE_BRANCH_TURN_PENALTY_MULTIPLIER": 10.0
    }

    def __init__(self, language="en"):
        super().__init__()
        self.language = language
        self.translations = self.get_translations()
        # Check if running as a bundled executable
        self.is_packaged = getattr(sys, 'frozen', False) and hasattr(sys, '_MEIPASS')

        self.params = self.DEFAULT_PARAMS.copy()
        self.setWindowTitle(self.tr("app_title"))
        self.setGeometry(100, 100, 1280, 960)
        self.annotations = load_central_annotations()

        # Set window icon dynamically
        self.setWindowIcon(create_yc_icon())

        self.set_stylesheet()

        self.images: List[np.ndarray] = []
        self.global_background_color: int = 255
        self.vessel_masks: Optional[List[np.ndarray]] = None
        self.layered_vessel_mask: Optional[np.ndarray] = None
        self.vessel_identity_map: Optional[np.ndarray] = None
        self.noise_rois: List[QRect] = []
        self.drawing_mode: DrawingMode = DrawingMode.MARKING
        self.final_paths: Optional[List[List[Tuple[int, int]]]] = None
        self.alternative_paths: Optional[List[List[Tuple[int, int]]]] = None
        self.final_path_image: Optional[np.ndarray] = None
        self.final_path_base_image: Optional[np.ndarray] = None
        self.base_mask_projection: Optional[np.ndarray] = None
        self.temporal_cost_map: Optional[np.ndarray] = None
        self.current_frame_index: int = 0
        self.path_points_info: List[Dict[str, Any]] = []
        self.app_state: AppState = AppState.IDLE
        self.smoothing_level: int = 4
        self.visible_paths: List[bool] = []
        self.animation_data: Optional[Dict[str, Any]] = None
        self.expert_path_points: List[Dict[str, Any]] = []

        self.init_ui()
        self.create_actions()
        self.create_menus()
        self.connect_signals()
        self.retranslate_ui()
        self.update_ui_for_state()

    def get_translations(self):
        translations = {
            "en": {
                "app_title": "YC_VesselTracer", "select_folder": "Select Image Folder",
                "start_marking": "Start Marking Path", "confirm_points": "Confirm Points",
                "run_analysis": "Run Full Analysis", "analysis_complete": "Analysis Complete",
                "processing": "Processing...", "reset_all": "Reset All",
                "draw_noise": "Draw Noise Area", "adjust_smoothing": "Adjust Smoothing",
                "preview_mask": "Preview Mask", "show_3d_view": "Show 3D View",
                "view_steps": "View Steps", "file_menu": "&File", "edit_menu": "&Edit",
                "view_menu": "&View", "help_menu": "&Help", "tools_menu": "&Tools",
                "open_folder_action": "&Open Folder...", "reset_action": "&Reset",
                "exit_action": "E&xit", "parameters_action": "&Parameters...",
                "package_action": "Package Application", "update_action": "Check for Updates",
                "update_available_msg": "A new version is available. Please confirm with YC if an update is needed.",
                "zoom_in_action": "Zoom &In", "zoom_out_action": "Zoom &Out",
                "reset_zoom_action": "Reset &Zoom", "controls_action": "&Controls & Parameters...",
                "group_load": "Step 1: Load Images", "group_configure": "Step 2: Mark & Configure",
                "group_execute": "Step 3: Execute", "group_view_reset": "View & Reset",
                "frame_label": "Frame: {0}/{1}",
                "info_idle": "Click 'Select Image Folder' or use File > Open Folder to begin.",
                "info_loaded": "Images loaded. Click 'Start Marking Path' to begin selecting points.",
                "info_marking": "Marked {0} points. Use 'A'/'D' or slider to switch frames. Click to mark.",
                "info_confirmed": "Points confirmed. Smoothing: {0}. Draw noise areas or start analysis.",
                "info_processing": "Running analysis, please wait...",
                "info_done": "Path analysis is complete! Reset to start a new analysis.",
                "info_drawing_noise": "Drag the mouse on the image to draw a noise area to exclude.",
                "pan_mode": "Pan Tool", "reset_view": "Reset View", "replay_animation": "Replay Animation",
                "introduction": "Introduction",
                "param_bg_removal_offset_desc": "Offset for adaptive thresholding to remove background. Higher values remove more background but may clip vessels.",
                "param_bg_removal_kernel_desc": "Size of the kernel for background estimation. Must be an odd number. Larger values handle uneven lighting better.",
                "param_max_node_search_radius_desc": "The maximum distance (in pixels) to search for a vessel segment when a point is clicked.",
                "param_max_gap_bridge_distance_desc": "The maximum gap size (in pixels) the pathfinder will attempt to bridge between vessel segments.",
                "param_forbidden_zone_radius_desc": "Radius around a path start/end point where the pathfinder cannot re-enter, preventing loops.",
                "param_time_cost_weight_desc": "Weight multiplier for the cost of moving between frames (time). Higher values prefer shorter paths in time.",
                "param_pathfinding_obstacle_cost_desc": "The absolute cost of a pixel that is considered an obstacle. Should be a very large number.",
                "param_turn_penalty_weight_desc": "Penalty applied for turning. Higher values result in straighter paths.",
                "param_dynamic_cost_weight_desc": "Multiplier for the cost dynamically added to the path to discourage re-using the same pixels for alternative paths.",
                "param_straight_path_threshold_desc": "Cosine similarity threshold to consider a path segment 'straight'. Used for penalizing turns.",
                "param_cross_vessel_penalty_desc": "A large penalty applied when a path crosses into a different vessel, based on the identity map.",
                "param_main_vessel_width_tolerance_desc": "Tolerance (as a percentage) for how much a side branch's width can deviate from the main vessel's width.",
                "param_side_branch_turn_penalty_multiplier_desc": "Multiplier for the turn penalty specifically when the path is exploring a potential side branch.",
                "engineering_mode_action": "Engineering Mode",
                "info_annotating": "Engineering Mode: Drawing vessel polygons.",
                "status_annotating": "Engineering Mode"
            },
            "zh": {
                "app_title": "YC_血管尋路", "select_folder": "選擇圖片資料夾",
                "start_marking": "開始標記路徑", "confirm_points": "確認標記點",
                "run_analysis": "執行完整分析", "analysis_complete": "分析完成",
                "processing": "處理中...", "reset_all": "全部重置",
                "draw_noise": "繪製雜訊區域", "adjust_smoothing": "調整平滑度",
                "preview_mask": "預覽遮罩", "show_3d_view": "顯示3D視圖",
                "view_steps": "查看步驟", "pan_mode": "平移工具", "reset_view": "重置視圖",
                "replay_animation": "重播動畫", "introduction": "介紹",
                "file_menu": "檔案 (&F)", "edit_menu": "編輯 (&E)",
                "view_menu": "檢視 (&V)", "help_menu": "幫助 (&H)", "tools_menu": "工具 (&T)",
                "open_folder_action": "開啟資料夾 (&O)...", "reset_action": "重置 (&R)",
                "package_action": "一鍵打包", "update_action": "檢查更新",
                "update_available_msg": "發現系統已更新，請跟昱辰確認是否需要更新",
                "exit_action": "離開 (&X)", "parameters_action": "參數設定 (&P)...",
                "zoom_in_action": "放大 (&I)", "zoom_out_action": "縮小 (&O)",
                "reset_zoom_action": "重置縮放 (&Z)", "controls_action": "控制與參數說明 (&C)...",
                "group_load": "步驟一：載入圖片", "group_configure": "步驟二：標記與設定",
                "group_execute": "步驟三：執行", "group_view_reset": "檢視與重置",
                "frame_label": "幀: {0}/{1}",
                "info_idle": "點擊 '選擇圖片資料夾' 或使用 檔案 > 開啟資料夾 來開始。",
                "info_loaded": "圖片已載入。點擊 '開始標記路徑' 來選擇標記點。",
                "info_marking": "已標記 {0} 個點。使用 'A'/'D' 或滑桿來切換幀。點擊以進行標記。",
                "info_confirmed": "標記點已確認。平滑度: {0}。請繪製雜訊區域或開始分析。",
                "info_processing": "正在執行分析，請稍候...",
                "info_done": "路徑分析完成！點擊 '全部重置' 來開始新的分析。",
                "info_drawing_noise": "在影像上拖動滑鼠以繪製要排除的雜訊區域。",
                "param_bg_removal_offset_desc": "用於自適應閾值以去除背景的偏移量。值越高，去除的背景越多，但可能會裁切到血管。",
                "param_bg_removal_kernel_desc": "用於背景估計的核心大小。此數值必須為奇數。較大的值可以更好地處理不均勻的光照。",
                "param_max_node_search_radius_desc": "點擊標記點時，在血管遮罩上搜尋對應像素點的最大半徑（單位：像素）。",
                "param_max_gap_bridge_distance_desc": "路徑尋找演算法能夠連接的血管片段之間的最大間隙（單位：像素）。",
                "param_forbidden_zone_radius_desc": "環繞路徑起點/終點的區域半徑，禁止路徑重新進入此區域以防止產生迴圈。",
                "param_time_cost_weight_desc": "跨幀移動（時間維度）的成本權重。較高的值會傾向於選擇在時間上（幀數）更短的路徑。",
                "param_pathfinding_obstacle_cost_desc": "被視為障礙物的像素的絕對成本。應設為極大值以阻止路徑穿越。",
                "param_turn_penalty_weight_desc": "對路徑轉彎處施加的懲罰。值越高，產生的路徑越趨於直線。",
                "param_dynamic_cost_weight_desc": "為已走過的路徑動態增加的成本權重，用於在尋找替代路徑時避免重複。",
                "param_straight_path_threshold_desc": "用於判斷一段路徑是否為「直線」的餘弦相似度閾值，主要用於計算轉彎懲罰。",
                "param_cross_vessel_penalty_desc": "當路徑根據血管身份圖（Identity Map）跨越到不同血管時所施加的高額懲罰。",
                "param_main_vessel_width_tolerance_desc": "側枝血管寬度與主血管寬度的允許偏差容忍度（百分比）。",
                "param_side_branch_turn_penalty_multiplier_desc": "當路徑探索潛在的側枝時，對轉彎懲罰應用的特定乘數。"
            }
        }
        return translations

    def tr(self, key, *args):
        return self.translations.get(self.language, self.translations["en"]).get(key, key).format(*args)

    def retranslate_ui(self):
        self.setWindowTitle(self.tr("app_title"))
        self.group_load.setTitle(self.tr("group_load"))
        self.group_configure.setTitle(self.tr("group_configure"))
        self.group_execute.setTitle(self.tr("group_execute"))
        self.group_view_reset.setTitle(self.tr("group_view_reset"))
        self.btn_select_folder.setText(self.tr("select_folder"))
        self.btn_add_noise_roi.setText(self.tr("draw_noise"))
        self.btn_smoothing_preview.setText(self.tr("adjust_smoothing"))
        self.btn_show_path.setText(self.tr("preview_mask"))
        self.btn_show_3d_view.setText(self.tr("show_3d_view"))
        self.btn_step_view.setText(self.tr("view_steps"))
        self.btn_replay_animation.setText(self.tr("replay_animation"))
        self.btn_reset.setText(self.tr("reset_all"))
        self.btn_pan_mode.setText(self.tr("pan_mode"))
        self.btn_reset_view.setText(self.tr("reset_view"))
        self.open_action.setText(self.tr("open_folder_action"))
        self.reset_action.setText(self.tr("reset_action"))
        self.exit_action.setText(self.tr("exit_action"))
        self.settings_action.setText(self.tr("parameters_action"))
        self.zoom_in_action.setText(self.tr("zoom_in_action"))
        self.zoom_out_action.setText(self.tr("zoom_out_action"))
        self.reset_zoom_action.setText(self.tr("reset_zoom_action"))
        self.help_action.setText(self.tr("controls_action"))
        self.introduction_action.setText(self.tr("introduction"))
        self.file_menu.setTitle(self.tr("file_menu"))
        self.edit_menu.setTitle(self.tr("edit_menu"))
        self.view_menu.setTitle(self.tr("view_menu"))
        self.help_menu.setTitle(self.tr("help_menu"))
        self.tools_menu.setTitle(self.tr("tools_menu"))

        if self.is_packaged:
            self.packaging_action.setText(self.tr("update_action"))
        else:
            self.packaging_action.setText(self.tr("package_action"))

        self.update_ui_for_state()

    def show_introduction_dialog(self):
        # Always show the Chinese README as requested in the PR comment.
        readme_path = os.path.join(os.path.dirname(__file__), '..', '..', 'README.md')
        dialog = YC_MarkdownDialog(os.path.abspath(readme_path), self)
        dialog.exec()

    def set_stylesheet(self):
        style = """
            QMainWindow { background-color: #2E2E2E; }
            QGroupBox {
                background-color: #3C3C3C; border: 1px solid #555;
                border-radius: 5px; margin-top: 1ex;
                font-size: 14px; font-weight: bold; color: #E0E0E0;
            }
            QGroupBox::title {
                subcontrol-origin: margin; subcontrol-position: top center;
                padding: 0 3px; background-color: #3C3C3C;
            }
            QLabel, QStatusBar { color: #D0D0D0; font-size: 12px; }
            QPushButton {
                background-color: #555555; color: #EEEEEE;
                border: 1px solid #666666; padding: 8px 16px;
                border-radius: 4px; font-size: 13px;
            }
            QPushButton:hover { background-color: #686868; border: 1px solid #777777; }
            QPushButton:pressed { background-color: #4A4A4A; }
            QPushButton:disabled { background-color: #404040; color: #888888; border-color: #555555; }
            QSlider::groove:horizontal {
                border: 1px solid #4A4A4A; height: 8px; background: #404040;
                margin: 2px 0; border-radius: 4px;
            }
            QSlider::handle:horizontal {
                background: #00A0A0; border: 1px solid #00A0A0;
                width: 18px; margin: -5px 0; border-radius: 9px;
            }
            QSlider::handle:horizontal:disabled { background: #777; border-color: #777; }
            QDialog { background-color: #383838; }
            QMenuBar { background-color: #3C3C3C; color: #E0E0E0; }
            QMenuBar::item { background-color: transparent; padding: 4px 8px; }
            QMenuBar::item:selected { background-color: #555555; }
            QMenu { background-color: #3C3C3C; color: #E0E0E0; border: 1px solid #555; }
            QMenu::item:selected { background-color: #00A0A0; }
        """
        self.setStyleSheet(style)

    def init_ui(self):
        self.central_widget = QWidget()
        self.setCentralWidget(self.central_widget)
        # Main layout is now horizontal
        self.layout = QHBoxLayout(self.central_widget)

        # --- Left side: Image display and controls ---
        left_layout = QVBoxLayout()
        self.image_label = YC_ImageLabel(self)
        left_layout.addWidget(self.image_label, 1) # Set stretch factor to 1

        frame_nav_layout = QHBoxLayout()
        self.frame_slider = QSlider(Qt.Orientation.Horizontal)
        self.frame_slider.setEnabled(False)
        self.frame_info_label = QLabel("Frame: -- / --")
        frame_nav_layout.addWidget(self.frame_slider)
        frame_nav_layout.addWidget(self.frame_info_label)
        left_layout.addLayout(frame_nav_layout)

        self.info_label = QLabel()
        self.info_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        font = self.info_label.font()
        font.setPointSize(14)
        self.info_label.setFont(font)
        left_layout.addWidget(self.info_label)

        self.layout.addLayout(left_layout, 1) # Set stretch factor to 1

        # --- Right side: Control panels ---
        right_controls_layout = QVBoxLayout()
        right_controls_layout.setSpacing(15)

        # Group 1: Load
        self.group_load = QGroupBox()
        group1_layout = QVBoxLayout(self.group_load)
        self.btn_select_folder = QPushButton()
        group1_layout.addWidget(self.btn_select_folder)
        right_controls_layout.addWidget(self.group_load)

        # Group 2: Configure
        self.group_configure = QGroupBox()
        group2_layout = QVBoxLayout(self.group_configure)
        self.btn_add_noise_roi = QPushButton()
        self.btn_smoothing_preview = QPushButton()
        group2_layout.addWidget(self.btn_add_noise_roi)
        group2_layout.addWidget(self.btn_smoothing_preview)
        right_controls_layout.addWidget(self.group_configure)
        self.group_tools = self.group_configure

        # Group 3: Execute
        self.group_execute = QGroupBox()
        group3_layout = QVBoxLayout(self.group_execute)
        self.btn_main_action = QPushButton()
        group3_layout.addWidget(self.btn_main_action)
        right_controls_layout.addWidget(self.group_execute)

        # Group 4: View & Reset
        self.group_view_reset = QGroupBox()
        group4_layout = QVBoxLayout(self.group_view_reset)

        self.btn_pan_mode = QPushButton()
        self.btn_pan_mode.setCheckable(True)
        self.btn_reset_view = QPushButton()
        self.btn_show_path = QPushButton()
        self.btn_show_3d_view = QPushButton()
        self.btn_step_view = QPushButton()

        replay_layout = QHBoxLayout()
        self.path_replay_selector = QComboBox()
        self.btn_replay_animation = QPushButton()
        replay_layout.addWidget(self.path_replay_selector)
        replay_layout.addWidget(self.btn_replay_animation)

        self.path_selection_layout = QVBoxLayout()
        self.path_selection_layout.setSpacing(5)

        self.btn_reset = QPushButton()

        group4_layout.addWidget(self.btn_pan_mode)
        group4_layout.addWidget(self.btn_reset_view)
        group4_layout.addWidget(self.btn_show_path)
        group4_layout.addWidget(self.btn_show_3d_view)
        group4_layout.addWidget(self.btn_step_view)
        group4_layout.addLayout(replay_layout)
        group4_layout.addLayout(self.path_selection_layout)
        group4_layout.addWidget(self.btn_reset)
        right_controls_layout.addWidget(self.group_view_reset)

        right_controls_layout.addStretch(1) # Add stretch to push panels to the top

        # --- Engineering Mode Panel ---
        self.group_engineering = QGroupBox("Engineering Mode")
        engineering_layout = QVBoxLayout(self.group_engineering)
        self.engineering_mode_toggle = QComboBox()
        self.engineering_mode_toggle.addItems(["Draw Vessel Polygons", "Define Expert Paths"])
        self.btn_save_annotations = QPushButton("Save Annotations")
        engineering_layout.addWidget(self.engineering_mode_toggle)
        engineering_layout.addWidget(self.btn_save_annotations)
        right_controls_layout.addWidget(self.group_engineering)
        self.group_engineering.setVisible(False) # Initially hidden

        self.layout.addLayout(right_controls_layout)
        self.layout.setStretchFactor(left_layout, 3) # Image layout takes 3/4 of space
        self.layout.setStretchFactor(right_controls_layout, 1) # Controls layout takes 1/4 of space

        self.setStatusBar(QStatusBar(self))

    def create_actions(self):
        self.open_action = QAction(self)
        self.open_action.setShortcut(QKeySequence.StandardKey.Open)
        self.reset_action = QAction(self)
        self.reset_action.setShortcut("Ctrl+R")
        self.exit_action = QAction(self)
        self.exit_action.setShortcut(QKeySequence.StandardKey.Quit)
        self.settings_action = QAction(self)
        self.settings_action.setShortcut("Ctrl+P")
        self.zoom_in_action = QAction(self)
        self.zoom_in_action.setShortcut(QKeySequence.StandardKey.ZoomIn)
        self.zoom_out_action = QAction(self)
        self.zoom_out_action.setShortcut(QKeySequence.StandardKey.ZoomOut)
        self.reset_zoom_action = QAction(self)
        self.reset_zoom_action.setShortcut("Ctrl+0")
        self.help_action = QAction(self)
        self.help_action.setShortcut("F1")
        self.introduction_action = QAction(self)
        self.packaging_action = QAction(self)
        self.engineering_mode_action = QAction(self)


    def create_menus(self):
        menu_bar = self.menuBar()
        self.file_menu = menu_bar.addMenu("")
        self.file_menu.addAction(self.open_action)
        self.file_menu.addAction(self.reset_action)
        self.file_menu.addSeparator()
        self.file_menu.addAction(self.exit_action)
        self.edit_menu = menu_bar.addMenu("")
        self.edit_menu.addAction(self.settings_action)
        self.view_menu = menu_bar.addMenu("")
        self.view_menu.addAction(self.zoom_in_action)
        self.view_menu.addAction(self.zoom_out_action)
        self.view_menu.addAction(self.reset_zoom_action)
        self.help_menu = menu_bar.addMenu("")
        self.help_menu.addAction(self.introduction_action)
        self.help_menu.addAction(self.help_action)

        self.tools_menu = menu_bar.addMenu("")
        self.tools_menu.addAction(self.packaging_action)
        self.tools_menu.addSeparator()
        self.tools_menu.addAction(self.engineering_mode_action)

    def connect_signals(self):
        self.open_action.triggered.connect(self.select_folder)
        self.reset_action.triggered.connect(self.reset_system)
        self.exit_action.triggered.connect(self.close)
        self.settings_action.triggered.connect(self.open_parameter_settings)
        self.zoom_in_action.triggered.connect(self.image_label.zoom_in)
        self.zoom_out_action.triggered.connect(self.image_label.zoom_out)
        self.reset_zoom_action.triggered.connect(self.image_label.reset_zoom)
        self.help_action.triggered.connect(self.show_help_dialog)
        self.introduction_action.triggered.connect(self.show_introduction_dialog)

        self.btn_select_folder.clicked.connect(self.select_folder)
        self.btn_main_action.clicked.connect(self.handle_main_action)
        self.btn_reset.clicked.connect(self.reset_system)
        self.btn_add_noise_roi.clicked.connect(self.add_noise_roi_mode)
        self.btn_smoothing_preview.clicked.connect(self.open_smoothing_preview)
        self.btn_show_path.clicked.connect(self.show_segmented_path_preview)
        self.btn_show_3d_view.clicked.connect(self.show_3d_view)
        self.btn_step_view.clicked.connect(self.show_step_viewer)
        self.btn_replay_animation.clicked.connect(self.replay_path_animation)
        self.btn_pan_mode.clicked.connect(self.toggle_pan_mode)
        self.btn_reset_view.clicked.connect(self.image_label.reset_zoom)
        self.frame_slider.valueChanged.connect(self.slider_value_changed)
        self.image_label.point_clicked.connect(self.handle_point_selection)
        self.image_label.expert_path_point_clicked.connect(self.handle_expert_path_selection)
        self.image_label.roi_drawn.connect(self.handle_roi_drawn)
        self.packaging_action.triggered.connect(self.handle_packaging_action)
        self.engineering_mode_action.triggered.connect(self.enter_engineering_mode)
        self.btn_save_annotations.clicked.connect(self.save_annotations)

    def update_ui_for_state(self):
        is_interactive = self.app_state != AppState.PROCESSING
        # State configurations with translatable keys
        state_configs = {
            AppState.IDLE: {"main_action_key": "start_marking", "main_action_enabled": False, "info_key": "info_idle", "status_key": "Ready", "tools_visible": False, "slider_enabled": False, "select_folder_enabled": True},
            AppState.LOADED: {"main_action_key": "start_marking", "main_action_enabled": True, "info_key": "info_loaded", "status_key": "status_loaded", "tools_visible": False, "slider_enabled": True, "select_folder_enabled": True},
            AppState.MARKING_PATH: {"main_action_key": "confirm_points", "main_action_enabled": len(self.path_points_info) >= 2, "info_key": "info_marking", "status_key": "status_marking", "tools_visible": False, "slider_enabled": True, "select_folder_enabled": False},
            AppState.RANGE_CONFIRMED: {"main_action_key": "run_analysis", "main_action_enabled": True, "info_key": "info_confirmed", "status_key": "status_confirmed", "tools_visible": True, "slider_enabled": False, "select_folder_enabled": False},
            AppState.PROCESSING: {"main_action_key": "processing", "main_action_enabled": False, "info_key": "info_processing", "status_key": "status_processing", "tools_visible": False, "slider_enabled": False, "select_folder_enabled": False},
            AppState.DONE: {"main_action_key": "analysis_complete", "main_action_enabled": False, "info_key": "info_done", "status_key": "status_done", "tools_visible": True, "slider_enabled": False, "select_folder_enabled": False},
            AppState.ANNOTATING_POLYGON: {"main_action_key": "run_analysis", "main_action_enabled": True, "info_key": "info_annotating", "status_key": "status_annotating", "tools_visible": False, "slider_enabled": True, "select_folder_enabled": False}
        }
        config = state_configs.get(self.app_state, state_configs[AppState.IDLE])

        # Show/hide engineering panel based on state
        is_engineering_mode = self.app_state == AppState.ANNOTATING_POLYGON
        self.group_engineering.setVisible(is_engineering_mode)

        self.btn_main_action.setText(self.tr(config["main_action_key"]))
        self.btn_main_action.setEnabled(config["main_action_enabled"] and is_interactive)

        # Dynamic info text formatting
        info_text = self.tr(config["info_key"], len(self.path_points_info), self.smoothing_level)
        self.info_label.setText(info_text)

        self.statusBar().showMessage(self.tr(config.get("status_key", "Ready")))
        self.group_tools.setVisible(config["tools_visible"])
        self.btn_show_3d_view.setEnabled(self.app_state == AppState.DONE and is_interactive)
        self.btn_replay_animation.setEnabled(self.app_state == AppState.DONE and is_interactive)
        self.path_replay_selector.setEnabled(self.app_state == AppState.DONE and is_interactive)
        self.frame_slider.setEnabled(config["slider_enabled"])
        self.btn_select_folder.setEnabled(config["select_folder_enabled"] and is_interactive)
        self.btn_reset.setEnabled(is_interactive)
        self.open_action.setEnabled(config["select_folder_enabled"] and is_interactive)
        self.reset_action.setEnabled(is_interactive)
        self.settings_action.setEnabled(is_interactive)
        self.zoom_in_action.setEnabled(bool(self.images))
        self.zoom_out_action.setEnabled(bool(self.images))
        self.reset_zoom_action.setEnabled(bool(self.images))
        self.btn_pan_mode.setEnabled(bool(self.images) and is_interactive)
        self.btn_reset_view.setEnabled(bool(self.images) and is_interactive)
        if self.drawing_mode == DrawingMode.NOISE_ROI and self.app_state == AppState.RANGE_CONFIRMED:
            self.info_label.setText(self.tr("info_drawing_noise"))

    def keyPressEvent(self, event):
        if not self.images or self.app_state not in [AppState.LOADED, AppState.MARKING_PATH]:
            super().keyPressEvent(event)
            return
        current_idx = self.current_frame_index
        new_idx = -1
        if event.key() == Qt.Key.Key_D:
            new_idx = min(len(self.images) - 1, current_idx + 1)
        elif event.key() == Qt.Key.Key_A:
            new_idx = max(0, current_idx - 1)
        if new_idx != -1 and new_idx != current_idx:
            self.frame_slider.setValue(new_idx)
        else:
            super().keyPressEvent(event)

    def select_folder(self):
        path = QFileDialog.getExistingDirectory(self, self.tr("select_folder"))
        if path:
            self.reset_system()
            self.images = load_images_from_folder(path)
            if not self.images:
                QMessageBox.warning(self, "Error", "Could not load any images from the selected folder.")
                self.reset_system()
                return
            self.global_background_color = get_most_frequent_color(self.images[0])
            self.current_frame_index = 0
            self.frame_slider.setRange(0, len(self.images) - 1)
            self.frame_slider.setValue(0)
            self.update_frame_display(0)
            self.app_state = AppState.LOADED
            self.update_ui_for_state()

    def slider_value_changed(self, value: int):
        self.update_frame_display(value)

    def update_frame_display(self, frame_index: int):
        if not self.images or not (0 <= frame_index < len(self.images)):
            return
        self.current_frame_index = frame_index
        self.frame_info_label.setText(self.tr("frame_label", frame_index + 1, len(self.images)))
        base_img = self.images[frame_index].copy()
        display_img = self.get_overlayed_display_image(base_img, frame_index)
        pixmap = convert_np_to_pixmap(display_img)
        self.image_label.setPixmap(pixmap)

    def get_overlayed_display_image(self, base_image_gray: np.ndarray, frame_index: Optional[int]) -> np.ndarray:
        if frame_index is None: frame_index = self.current_frame_index
        display_img_bgr = cv2.cvtColor(base_image_gray, cv2.COLOR_GRAY2BGR)
        for i, p_info in enumerate(self.path_points_info):
            pt = p_info["point"]
            radius = 6
            color = (0, 255, 255)
            is_on_current_frame = p_info["frame"] == frame_index
            thickness = -1 if is_on_current_frame else 2
            if i == 0: color = (0, 0, 255)
            elif i == len(self.path_points_info) - 1 and self.app_state != AppState.MARKING_PATH: color = (255, 100, 0)
            cv2.circle(display_img_bgr, (pt.x(), pt.y()), radius + 1, (0, 0, 0), -1)
            cv2.circle(display_img_bgr, (pt.x(), pt.y()), radius, color, thickness)
        for r in self.noise_rois:
            cv2.rectangle(display_img_bgr, (r.x(), r.y()), (r.x() + r.width(), r.y() + r.height()), (0, 0, 255), 2)

        if self.app_state == AppState.ANNOTATING_POLYGON and self.engineering_mode_toggle.currentText() == "Define Expert Paths":
            for i, p_info in enumerate(self.expert_path_points):
                pt = QPoint(p_info["point"][0], p_info["point"][1])
                radius = 6
                color = (0, 255, 0) if i % 2 == 0 else (255, 0, 255)
                is_on_current_frame = p_info["frame"] == frame_index
                thickness = -1 if is_on_current_frame else 2
                cv2.circle(display_img_bgr, (pt.x(), pt.y()), radius + 1, (0, 0, 0), -1)
                cv2.circle(display_img_bgr, (pt.x(), pt.y()), radius, color, thickness)

        return display_img_bgr

    def handle_main_action(self):
        if self.app_state == AppState.LOADED:
            self.app_state = AppState.MARKING_PATH
        elif self.app_state == AppState.MARKING_PATH:
            if len(self.path_points_info) < 2:
                QMessageBox.warning(self, "Info", "Please mark at least a start and an end point.")
                return
            self.app_state = AppState.RANGE_CONFIRMED
            self.update_range_view()
        elif self.app_state == AppState.RANGE_CONFIRMED:
            self.start_analysis()
        self.update_ui_for_state()

    def handle_point_selection(self, point: QPoint):
        if self.app_state == AppState.MARKING_PATH:
            self.path_points_info.append({"point": point, "frame": self.current_frame_index})
            self.path_points_info.sort(key=lambda p: p['frame'])
            self.update_frame_display(self.current_frame_index)
            self.update_ui_for_state()

    def _get_frame_range(self, for_processing: bool = False) -> Optional[Tuple[int, int]]:
        if not self.path_points_info: return None
        all_frames = [p["frame"] for p in self.path_points_info]
        if not all_frames: return None
        end_f = max(all_frames)
        start_f = 0 if for_processing else min(all_frames)
        if start_f > end_f: return None
        return start_f, end_f

    def update_range_view(self):
        frame_range = self._get_frame_range(for_processing=False)
        if not frame_range: return
        start_f, end_f = frame_range
        range_pip = create_maximum_intensity_projection(self.images[start_f: end_f + 1])
        if range_pip is not None:
            img_with_overlays = self.get_overlayed_display_image(range_pip, -1)
            self.image_label.setPixmap(convert_np_to_pixmap(img_with_overlays))

    def add_noise_roi_mode(self):
        self.drawing_mode = DrawingMode.NOISE_ROI
        self.update_ui_for_state()

    def open_smoothing_preview(self):
        if self.app_state != AppState.RANGE_CONFIRMED: return
        frame_range = self._get_frame_range(for_processing=False)
        if not frame_range:
            QMessageBox.warning(self, "Error", "Please mark points first to define a preview range.")
            return
        start_f, end_f = frame_range
        pip_image = create_maximum_intensity_projection(self.images[start_f:end_f + 1])
        if pip_image is None:
            QMessageBox.warning(self, "Error", "Could not create a preview image.")
            return
        dialog = YC_SmoothingPreviewDialog(pip_image, self.smoothing_level, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            new_level = dialog.get_selected_level()
            if new_level != self.smoothing_level:
                self.smoothing_level = new_level
                self.vessel_masks = None
                self.statusBar().showMessage(f"Smoothing level set to: {self.smoothing_level}")
                self.update_ui_for_state()

    def handle_roi_drawn(self, roi: QRect):
        if self.drawing_mode == DrawingMode.NOISE_ROI and self.app_state == AppState.RANGE_CONFIRMED:
            self.noise_rois.append(roi)
            self.info_label.setText(f"Defined {len(self.noise_rois)} noise area(s).")
            self.drawing_mode = DrawingMode.MARKING
            self.update_ui_for_state()

    def toggle_pan_mode(self, checked: bool):
        if checked:
            self.drawing_mode = DrawingMode.PAN
            self.image_label.setCursor(Qt.CursorShape.OpenHandCursor)
        else:
            self.drawing_mode = DrawingMode.MARKING
            self.image_label.setCursor(Qt.CursorShape.ArrowCursor)

    def show_segmented_path_preview(self):
        if self.app_state != AppState.RANGE_CONFIRMED: return
        if self.base_mask_projection is not None:
            self.image_label.setPixmap(convert_np_to_pixmap(self.overlay_points_on_image(self.base_mask_projection)))
            self.statusBar().showMessage("Showing cached vessel mask.")
            return
        if self.prepare_and_generate_masks():
            self.image_label.setPixmap(convert_np_to_pixmap(self.overlay_points_on_image(self.base_mask_projection)))
            self.statusBar().showMessage("Vessel mask generated and displayed.")
        else:
            QMessageBox.warning(self, "Error", "Failed to generate vessel mask.")

    def prepare_and_generate_masks(self) -> bool:
        frame_range = self._get_frame_range(for_processing=True)
        if not frame_range: return False
        start_f, end_f = frame_range
        images_subset = self.images[start_f: end_f + 1]
        progress = QProgressDialog(self.tr("info_processing"), "Cancel", 0, len(images_subset), self)
        progress.setWindowModality(Qt.WindowModality.WindowModal)
        updater = ProgressUpdater(progress)
        masks = create_enhanced_vessel_masks(images_subset, self.noise_rois, self.global_background_color, self.params, self.smoothing_level, updater)
        updater.finish()
        if masks and updater.is_running:
            self.vessel_masks = masks
            self.base_mask_projection = np.max(np.stack(self.vessel_masks, axis=0), axis=0)
            self.temporal_cost_map = create_temporal_cost_map(self.vessel_masks, self.params["PATHFINDING_OBSTACLE_COST"])
            self.vessel_identity_map = build_vessel_identity_map(self.vessel_masks)
            return True
        else:
            self.vessel_masks = None
            return False

    def start_analysis(self):
        self.app_state = AppState.PROCESSING
        self.update_ui_for_state()
        QApplication.processEvents()
        if self.base_mask_projection is None:
            if not self.prepare_and_generate_masks():
                QMessageBox.warning(self, "Analysis Aborted", "Failed to generate vessel mask.")
                self.app_state = AppState.RANGE_CONFIRMED
                self.update_ui_for_state()
                return

        final_mask = self.base_mask_projection
        if np.sum(final_mask) == 0:
            QMessageBox.warning(self, "Analysis Aborted", "Generated vessel mask is empty.")
            self.app_state = AppState.RANGE_CONFIRMED
            self.update_ui_for_state()
            return

        mask_pixels = [find_closest_pixel_on_mask(p["point"], final_mask, self.params["MAX_NODE_SEARCH_RADIUS"]) for p in self.path_points_info]
        if not all(mask_pixels):
            QMessageBox.warning(self, "Pathfinding Failed", "Could not locate all marked points on the vessel mask. Try adjusting smoothing or re-marking points.")
            self.app_state = AppState.RANGE_CONFIRMED
            self.update_ui_for_state()
            return

        self.show_full_analysis_steps(final_mask, mask_pixels)

    def show_full_analysis_steps(self, final_mask, mask_pixels):
        self.statusBar().showMessage(self.tr("info_processing"))
        QApplication.processEvents()
        steps = []
        frame_range = self._get_frame_range(for_processing=False)
        if not frame_range: return
        start_f, end_f = frame_range
        base_original_pip = create_maximum_intensity_projection(self.images[start_f: end_f + 1])

        mask_steps_data = generate_mask_steps(base_original_pip, self.smoothing_level, self.params, self.global_background_color)
        for img, desc in mask_steps_data:
            steps.append((convert_np_to_pixmap(img), f"Mask Generation - {desc}"))

        self.layered_vessel_mask = create_vessel_layers(final_mask, base_original_pip)
        if self.layered_vessel_mask is not None:
            normalized_layers = cv2.normalize(self.layered_vessel_mask, None, 0, 255, cv2.NORM_MINMAX, dtype=cv2.CV_8U)
            layer_heatmap = cv2.applyColorMap(normalized_layers, cv2.COLORMAP_JET)
            layer_heatmap[self.layered_vessel_mask == 0] = [0, 0, 0]
            steps.append((convert_np_to_pixmap(layer_heatmap), "Vessel Layering (Z-depth)"))

        combined_identity_map = create_combined_identity_map(self.vessel_identity_map, self.layered_vessel_mask)

        width_map = cv2.distanceTransform(final_mask.astype(np.uint8), cv2.DIST_L2, 5)
        main_vessel_width = 2 * width_map[mask_pixels[0]] if width_map is not None else 0

        self.statusBar().showMessage(self.tr("info_processing"))
        QApplication.processEvents()

        cumulative_cost_map = self.temporal_cost_map.copy()
        all_found_paths = []
        folder_name = os.path.basename(self.image_folder_path) if self.image_folder_path else ""
        current_annotations = self.annotations.get(folder_name, {})

        for _ in range(3):
            full_path = []
            is_path_complete = True
            for i in range(len(mask_pixels) - 1):
                segment = find_path_astar(cumulative_cost_map, mask_pixels[i], mask_pixels[i+1], combined_identity_map, width_map, main_vessel_width, self.params, current_annotations)
                if segment:
                    full_path.extend(segment if i == 0 else segment[1:])
                else:
                    is_path_complete = False
                    break
            if is_path_complete and full_path:
                all_found_paths.append(full_path)
                for y, x in full_path:
                    cumulative_cost_map[y, x] += 1e7
            else:
                break

        if all_found_paths:
            self.final_paths = [all_found_paths[0]]
            self.alternative_paths = all_found_paths[1:]
        else:
            QMessageBox.warning(self, "Pathfinding Failed", "Could not find a continuous path between all points.")
            self.app_state = AppState.RANGE_CONFIRMED
            self.update_ui_for_state()
            return

        path_base_image = cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR)
        for path in self.final_paths:
            path_points = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(path_base_image, [path_points[:,:,::-1]], isClosed=False, color=(50, 255, 50), thickness=2)

        self.animation_data = {"type": "animation", "costmap": self.temporal_cost_map, "pixels": mask_pixels, "baseimage": cv2.cvtColor(final_mask, cv2.COLOR_GRAY2BGR), "identity_map": combined_identity_map, "width_map": width_map, "main_vessel_width": main_vessel_width}
        steps.append((convert_np_to_pixmap(path_base_image), "A* Search Result (Click Replay)", self.animation_data))

        self.generate_final_path_image(base_original_pip)
        steps.append((convert_np_to_pixmap(self.final_path_image), "Final Result"))

        dialog = YC_StepViewerDialog(steps, self)
        dialog.exec()

        self.app_state = AppState.DONE
        self.setup_path_visibility_controls()
        self.update_path_display()
        self.update_ui_for_state()

    def generate_final_path_image(self, base_original_pip):
        # This function now just prepares the base image. Drawing is handled by update_path_display.
        self.final_path_base_image = cv2.cvtColor(base_original_pip, cv2.COLOR_GRAY2BGR)
        if not self.final_paths:
            self.final_path_image = self.final_path_base_image
            return
        # The actual drawing is deferred to update_path_display
        self.update_path_display()

    def setup_path_visibility_controls(self):
        # Clear previous controls
        while self.path_selection_layout.count():
            child = self.path_selection_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()
        self.path_replay_selector.clear()

        all_paths = (self.final_paths or []) + (self.alternative_paths or [])
        self.visible_paths = [True] * len(all_paths)

        for i, path in enumerate(all_paths):
            path_name = f"Main Path" if i == 0 else f"Alternative {i}"

            # Add checkbox for visibility
            checkbox = QCheckBox(path_name)
            checkbox.setChecked(True)
            checkbox.stateChanged.connect(lambda state, index=i: self.toggle_path_visibility(index, state))
            self.path_selection_layout.addWidget(checkbox)

            # Add path to replay selector
            self.path_replay_selector.addItem(path_name)

    def toggle_path_visibility(self, index, state):
        self.visible_paths[index] = (state == Qt.CheckState.Checked.value)
        self.update_path_display()

    def update_path_display(self):
        if not hasattr(self, 'final_path_base_image') or self.final_path_base_image is None:
            return

        # Start with a fresh copy of the base image
        display_image = self.final_path_base_image.copy()

        all_paths = (self.final_paths or []) + (self.alternative_paths or [])
        path_colors = [(50, 255, 50), (255, 255, 0), (0, 100, 255)] # Green, Yellow, Blue

        for i, path in enumerate(all_paths):
            if i < len(self.visible_paths) and self.visible_paths[i]:
                path_points = np.array(path, dtype=np.int32).reshape(-1, 1, 2)
                color = path_colors[i % len(path_colors)]
                thickness = 3 if i == 0 else 2
                cv2.polylines(display_image, [path_points[:,:,::-1]], isClosed=False, color=color, thickness=thickness, lineType=cv2.LINE_AA)

        self.final_path_image = display_image
        self.overlay_points_on_image(self.final_path_image)
        self.image_label.setPixmap(convert_np_to_pixmap(self.final_path_image))

    def overlay_points_on_image(self, image):
        for i, p_info in enumerate(self.path_points_info):
            pt = p_info["point"]
            radius = 6
            color = (0, 255, 255)
            if i == 0: color = (0, 0, 255)
            elif i == len(self.path_points_info) - 1: color = (255, 100, 0)
            cv2.circle(image, (pt.x(), pt.y()), radius + 2, (0, 0, 0), -1)
            cv2.circle(image, (pt.x(), pt.y()), radius, color, -1)
        return image

    def show_3d_view(self):
        if self.app_state != AppState.DONE:
            QMessageBox.warning(self, "Not Ready", "Please run a full analysis first.")
            return
        self.statusBar().showMessage(self.tr("info_processing"))
        QApplication.processEvents()

        plot_traces = self.generate_3d_plot_data()
        if not plot_traces:
            QMessageBox.warning(self, "Error", "Could not generate data for the 3D plot.")
            self.statusBar().showMessage("Error generating 3D plot.", 5000)
            return

        fig = go.Figure(data=plot_traces)
        fig.update_layout(
            title_text='YC 3D Vessel Path',
            scene=dict(
                xaxis_title='X',
                yaxis_title='Y',
                zaxis_title='Frame (Time)',
                aspectratio=dict(x=1, y=1, z=0.5)
            ),
            margin=dict(l=0, r=0, b=0, t=40)
        )
        fig.update_scenes(yaxis_autorange="reversed")

        # Generate self-contained HTML
        html_content = fig.to_html(full_html=True, include_plotlyjs=True)
        file_url = ""
        try:
            # Save to a temporary file
            with tempfile.NamedTemporaryFile('w', delete=False, suffix='.html', encoding='utf-8') as f:
                f.write(html_content)
                # Get the file path as a URL
                file_url = pathlib.Path(f.name).as_uri()

            # Try to open in the default web browser
            opened = webbrowser.open(file_url)
            if not opened:
                raise webbrowser.Error("Browser could not be opened.")
            self.statusBar().showMessage("3D view opened in browser.", 5000)

        except Exception as e:
            # If it fails, show a message with the path
            error_msg = f"無法自動開啟瀏覽器。\n\n請手動開啟此檔案路徑:\n{file_url}\n\n錯誤: {e}"
            QMessageBox.information(self,
                                    self.tr("show_3d_view"),
                                    error_msg)
            self.statusBar().showMessage("無法自動開啟瀏覽器", 5000)

    def generate_3d_plot_data(self):
        if not self.vessel_masks: return []
        vessel_x, vessel_y, vessel_z = [], [], []
        for i, mask in enumerate(self.vessel_masks[::2]):
            points = np.argwhere(mask > 0)[::4]
            vessel_z.extend([i*2] * len(points))
            vessel_y.extend(points[:, 0])
            vessel_x.extend(points[:, 1])
        vessel_trace = go.Scatter3d(x=vessel_x, y=vessel_y, z=vessel_z, mode='markers', marker=dict(size=1, color='gray', opacity=0.3), name='Vessel Structure')
        traces = [vessel_trace]

        all_paths = (self.final_paths or []) + (self.alternative_paths or [])
        path_colors = ['lime', 'red', 'blue']
        for i, path in enumerate(all_paths):
            if not path: continue
            path_x, path_y, path_z = [], [], []
            for y, x in path:
                path_x.append(x)
                path_y.append(y)
                path_z.append(self.temporal_cost_map[y, x])
            path_name = f"Path {i+1}" if i > 0 else "Main Path"
            path_trace = go.Scatter3d(x=path_x, y=path_y, z=path_z, mode='lines', line=dict(color=path_colors[i % len(path_colors)], width=8), name=path_name)
            traces.append(path_trace)
        return traces

    def reset_system(self):
        self.images = []
        self.global_background_color = 255
        self.vessel_masks = None
        self.layered_vessel_mask = None
        self.vessel_identity_map = None
        self.noise_rois = []
        self.drawing_mode = DrawingMode.MARKING
        self.final_paths = None
        self.alternative_paths = None
        self.final_path_image = None
        self.final_path_base_image = None
        self.base_mask_projection = None
        self.temporal_cost_map = None
        self.path_points_info = []
        self.smoothing_level = 4
        self.visible_paths = []
        self.animation_data = None
        self.params = self.DEFAULT_PARAMS.copy()

        # Clear path selection checkboxes
        while self.path_selection_layout.count():
            child = self.path_selection_layout.takeAt(0)
            if child.widget():
                child.widget().deleteLater()

        self.image_label.setPixmap(QPixmap())
        self.frame_slider.setRange(0, 0)
        self.frame_info_label.setText("Frame: -- / --")
        self.app_state = AppState.IDLE
        self.update_ui_for_state()
        self.image_label.reset_zoom()

    def show_help_dialog(self):
        param_meta = {
            "BG_REMOVAL_THRESHOLD_OFFSET": (self.tr("param_bg_removal_offset_desc"), int, 0, 100),
            "BG_REMOVAL_KERNEL_SIZE": (self.tr("param_bg_removal_kernel_desc"), int, 1, 51),
            "MAX_NODE_SEARCH_RADIUS": (self.tr("param_max_node_search_radius_desc"), int, 5, 200),
            "MAX_GAP_BRIDGE_DISTANCE": (self.tr("param_max_gap_bridge_distance_desc"), int, 5, 100),
            "FORBIDDEN_ZONE_RADIUS": (self.tr("param_forbidden_zone_radius_desc"), int, 0, 100),
            "TIME_COST_WEIGHT": (self.tr("param_time_cost_weight_desc"), float, 0.0, 10.0),
            "PATHFINDING_OBSTACLE_COST": (self.tr("param_pathfinding_obstacle_cost_desc"), float, 1e6, 1e12),
            "TURN_PENALTY_WEIGHT": (self.tr("param_turn_penalty_weight_desc"), float, 0.0, 500.0),
            "DYNAMIC_COST_WEIGHT": (self.tr("param_dynamic_cost_weight_desc"), float, 0.0, 50.0),
            "STRAIGHT_PATH_THRESHOLD": (self.tr("param_straight_path_threshold_desc"), float, 0.0, 1.0),
            "CROSS_VESSEL_PENALTY": (self.tr("param_cross_vessel_penalty_desc"), float, 1e4, 1e9),
            "MAIN_VESSEL_WIDTH_TOLERANCE": (self.tr("param_main_vessel_width_tolerance_desc"), float, 0.0, 1.0),
            "SIDE_BRANCH_TURN_PENALTY_MULTIPLIER": (self.tr("param_side_branch_turn_penalty_multiplier_desc"), float, 1.0, 50.0),
        }
        dialog = YC_HelpDialog(param_meta, self)
        dialog.exec()

    def open_parameter_settings(self):
        dialog = YC_ParameterDialog(self.params, self)
        if dialog.exec() == QDialog.DialogCode.Accepted:
            self.params = dialog.get_parameters()
            self.vessel_masks = None
            self.statusBar().showMessage("Parameters updated.")

    def show_step_viewer(self):
        if self.app_state not in [AppState.RANGE_CONFIRMED, AppState.DONE]: return
        self.statusBar().showMessage(self.tr("info_processing"))
        QApplication.processEvents()

        frame_range = self._get_frame_range(for_processing=False)
        if not frame_range: return
        start_f, end_f = frame_range
        image_to_process = create_maximum_intensity_projection(self.images[start_f: end_f + 1])

        step_data = generate_mask_steps(image_to_process, self.smoothing_level, self.params, self.global_background_color)

        qt_steps = [(convert_np_to_pixmap(img), desc) for img, desc in step_data]
        dialog = YC_StepViewerDialog(qt_steps, self)
        dialog.exec()
        self.statusBar().showMessage("Ready")

    def replay_path_animation(self):
        if not self.final_paths and not self.alternative_paths:
            QMessageBox.warning(self, "No Paths Found", "No paths are available to animate.")
            return

        selected_index = self.path_replay_selector.currentIndex()
        all_paths = (self.final_paths or []) + (self.alternative_paths or [])
        if not (0 <= selected_index < len(all_paths)):
            return

        selected_path = all_paths[selected_index]
        base_image = self.final_path_base_image.copy()

        self.statusBar().showMessage(f"Animating: {self.path_replay_selector.currentText()}")

        for i in range(len(selected_path)):
            # Draw the path up to the current point
            temp_img = base_image.copy()
            path_segment = np.array(selected_path[:i+1], dtype=np.int32).reshape(-1, 1, 2)
            cv2.polylines(temp_img, [path_segment[:,:,::-1]], isClosed=False, color=(50, 255, 50), thickness=2, lineType=cv2.LINE_AA)

            # Overlay the marked points
            self.overlay_points_on_image(temp_img)

            self.image_label.setPixmap(convert_np_to_pixmap(temp_img))
            QApplication.processEvents()
            # A small delay to make the animation visible
            QApplication.instance().processEvents()
            # time.sleep(0.001) # Optional: for slower animation

        # Restore the full path view after animation
        self.update_path_display()
        self.statusBar().showMessage("Animation finished.", 3000)

    def handle_packaging_action(self):
        if self.is_packaged:
            self.check_for_updates()
        else:
            self.package_application()

    def package_application(self):
        msg_box = QMessageBox()
        msg_box.setWindowTitle(self.tr("package_action"))
        msg_box.setText("開始打包應用程式。\n此過程可能需要數分鐘，請稍候。\n完成後會跳出提示。")
        msg_box.setStandardButtons(QMessageBox.StandardButton.Ok)
        msg_box.show()
        QApplication.processEvents()

        try:
            # Command to run PyInstaller
            command = [
                sys.executable, "-m", "PyInstaller", "main.py",
                "--name", "YC_VesselTracer",
                "--windowed",
                "--paths", "src",
                "--collect-all", "skimage",
                "--collect-all", "plotly",
                "--hidden-import", "pytz",
                "--exclude-module", "PyQt5",
                "--noconfirm",
                "--additional-hooks-dir", "../../hooks"  # <-- Added the directory name here
            ]

            process = subprocess.Popen(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, creationflags=subprocess.CREATE_NO_WINDOW)
            stdout, stderr = process.communicate()

            if process.returncode == 0:
                QMessageBox.information(self, "打包成功", "應用程式已成功打包！\n請查看 'dist/YC_VesselTracer' 資料夾。")
            else:
                error_message = f"打包失敗！\n\n錯誤訊息:\n{stderr}"
                error_dialog = YC_MarkdownDialog(f"```\n{error_message}\n```", self)
                error_dialog.setWindowTitle("打包錯誤")
                error_dialog.exec()

        except Exception as e:
            QMessageBox.critical(self, "打包錯誤", f"執行打包時發生未知錯誤：\n{e}")

    def check_for_updates(self):
        try:
            # Fetch the latest info from the remote
            subprocess.check_output(["git", "fetch"], stderr=subprocess.STDOUT)

            # Get the commit hash of the local HEAD
            local_commit = subprocess.check_output(["git", "rev-parse", "HEAD"]).strip()

            # Get the commit hash of the remote main branch
            remote_commit = subprocess.check_output(["git", "rev-parse", "origin/main"]).strip()

            if local_commit == remote_commit:
                QMessageBox.information(self, self.tr("update_action"), "已是最新版本。")
            else:
                QMessageBox.information(self, self.tr("update_action"), self.tr("update_available_msg"))

        except subprocess.CalledProcessError as e:
            # This can happen if git is not installed, or this is not a git repository
            QMessageBox.warning(self, "更新錯誤", f"無法檢查更新。請確認您已安裝 Git，且此應用程式位於一個 Git 倉庫中。\n\n錯誤: {e.output.decode()}")
        except FileNotFoundError:
            QMessageBox.warning(self, "更新錯誤", "無法檢查更新。請確認您已安裝 Git 並將其加入系統路徑中。")
        except Exception as e:
            QMessageBox.critical(self, "更新錯誤", f"檢查更新時發生未知錯誤：\n{e}")

    def closeEvent(self, event):
        # Ask to save annotations if they have been modified
        # This is a placeholder for a more robust check
        if self.group_engineering.isVisible():
             reply = QMessageBox.question(self, 'Save Annotations',
                                          "Do you want to save changes to annotations?",
                                          QMessageBox.StandardButton.Save | QMessageBox.StandardButton.Discard | QMessageBox.StandardButton.Cancel,
                                          QMessageBox.StandardButton.Save)

             if reply == QMessageBox.StandardButton.Save:
                 self.save_annotations()
             elif reply == QMessageBox.StandardButton.Cancel:
                 event.ignore()
                 return

        self.reset_system()
        event.accept()

    def enter_engineering_mode(self):
        password, ok = QInputDialog.getText(self, "Engineering Mode", "Enter Password:", QLineEdit.EchoMode.Password)
        if ok and password == "nick910114":
            self.app_state = AppState.ANNOTATING_POLYGON
            self.update_ui_for_state()
            self.image_label.set_engineering_mode(True)
        elif ok:
            QMessageBox.warning(self, "Access Denied", "Incorrect password.")
            self.engineering_mode_action.setEnabled(False)

    def save_annotations(self):
        if not self.image_folder_path:
            QMessageBox.warning(self, "Error", "No image folder loaded.")
            return

        # Get annotations from the image label
        current_polygons = self.image_label.get_polygons()

        # Structure the data
        folder_name = os.path.basename(self.image_folder_path)
        if folder_name not in self.annotations:
            self.annotations[folder_name] = {"polygons": {}, "paths": []}

        self.annotations[folder_name]["polygons"] = current_polygons

        self.annotations[folder_name]["paths"] = self.expert_path_points

        save_central_annotations(self.annotations)
        QMessageBox.information(self, "Success", "Annotations saved to annotations.json")

    def handle_expert_path_selection(self, point: QPoint):
        self.expert_path_points.append({"point": (point.x(), point.y()), "frame": self.current_frame_index})
        if len(self.expert_path_points) % 2 == 0:
            QMessageBox.information(self, "Path Defined", f"Expert path {len(self.expert_path_points)//2} defined.")
        self.update_frame_display(self.current_frame_index)
