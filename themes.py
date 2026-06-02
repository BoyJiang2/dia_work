"""数字图像处理系统 — Professional Dark 主题."""
from __future__ import annotations

# ── 色彩常量 ──────────────────────────────────────────────
BG_DEEP = "#15181e"          # 最深背景
BG_PANEL = "#1e2128"         # 面板/卡片背景
BG_CARD = "#252830"          # 卡片
BG_HOVER = "#2c303a"         # hover 高亮
BG_INPUT = "#1a1d24"         # 输入控件背景
BG_CANVAS = "#0d1117"        # 图像画布
BORDER = "#333840"           # 边框
BORDER_FOCUS = "#4a9eff"     # 聚焦边框
TEXT_PRIMARY = "#e0e4ec"     # 主文字
TEXT_SECONDARY = "#8b949e"   # 次文字
TEXT_MUTED = "#5d6778"       # 更淡文字
ACCENT_BLUE = "#4a9eff"      # 强调蓝
ACCENT_GREEN = "#3fb950"     # 成功绿
ACCENT_ORANGE = "#d29922"    # 警示橙
ACCENT_RED = "#f85149"       # 错误红
ACCENT_TEAL = "#39d1b4"      # 青绿
ACCENT_PURPLE = "#a371f7"    # 紫色

FONT_STACK = '"Microsoft YaHei UI", "Segoe UI", "SF Pro Display", Arial, sans-serif'
FONT_MONO = '"Cascadia Code", "Fira Code", "JetBrains Mono", "Consolas", monospace'

# ── 全局 QSS ──────────────────────────────────────────────

GLOBAL_QSS = f"""
/* ====== 基础重置 ====== */
QWidget {{
    font-family: {FONT_STACK};
    font-size: 13px;
    color: {TEXT_PRIMARY};
    background: transparent;
}}

QMainWindow#rootWindow, QWidget#appShell {{
    background: {BG_DEEP};
}}

/* ====== 顶栏 ====== */
QFrame#topBar {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    background: {BG_PANEL};
    padding: 4px 0;
}}

QLabel#appTitle {{
    color: {TEXT_PRIMARY};
    font-size: 22px;
    font-weight: 700;
    letter-spacing: 0.5px;
}}

QLabel#appSubtitle {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
    font-weight: 400;
}}

QLabel#courseTag {{
    color: {ACCENT_TEAL};
    font-weight: 700;
    font-size: 12px;
    padding: 6px 12px;
    border: 1px solid rgba(57, 209, 180, 0.3);
    border-radius: 6px;
    background: rgba(57, 209, 180, 0.08);
}}

/* ====== 卡片 ====== */
QFrame#card {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    background: {BG_CARD};
}}

QLabel#cardTitle {{
    color: {TEXT_PRIMARY};
    font-size: 14px;
    font-weight: 700;
    padding-bottom: 2px;
    border-bottom: 1px solid rgba(255,255,255,0.06);
}}

QLabel#description, QLabel#pathLabel {{
    color: {TEXT_SECONDARY};
    font-size: 12px;
    line-height: 1.5;
    padding: 8px 10px;
    border-radius: 6px;
    background: {BG_INPUT};
}}

/* ====== 按钮 ====== */
QPushButton {{
    min-height: 34px;
    border: 1px solid transparent;
    border-radius: 7px;
    padding: 6px 14px;
    font-weight: 650;
    font-size: 13px;
}}

QPushButton#primaryButton {{
    color: #fff;
    background: {ACCENT_BLUE};
    border-color: {ACCENT_BLUE};
}}
QPushButton#primaryButton:hover {{
    background: #5cb6ff;
}}
QPushButton#primaryButton:pressed {{
    background: #3b8edb;
}}

QPushButton#accentButton {{
    color: #fff;
    min-height: 40px;
    font-size: 14px;
    font-weight: 700;
    background: {ACCENT_GREEN};
    border-color: {ACCENT_GREEN};
}}
QPushButton#accentButton:hover {{
    background: #4cc962;
}}
QPushButton#accentButton:pressed {{
    background: #34a348;
}}

QPushButton#secondaryButton {{
    color: {ACCENT_BLUE};
    background: rgba(74, 158, 255, 0.08);
    border-color: rgba(74, 158, 255, 0.25);
}}
QPushButton#secondaryButton:hover {{
    background: rgba(74, 158, 255, 0.15);
}}

QPushButton#ghostButton {{
    color: {TEXT_SECONDARY};
    background: {BG_HOVER};
    border-color: {BORDER};
}}
QPushButton#ghostButton:hover {{
    color: {TEXT_PRIMARY};
    background: #363b48;
}}

QPushButton#toolBtn {{
    color: {TEXT_SECONDARY};
    min-height: 24px;
    max-height: 28px;
    border: 1px solid {BORDER};
    border-radius: 6px;
    padding: 3px 10px;
    font-size: 11px;
    font-weight: 600;
    background: rgba(255,255,255,0.04);
}}
QPushButton#toolBtn:hover {{
    color: {TEXT_PRIMARY};
    background: rgba(255,255,255,0.1);
    border-color: #555c6b;
}}

/* ====== 输入控件 ====== */
QComboBox, QSpinBox, QDoubleSpinBox {{
    min-height: 32px;
    border: 1px solid {BORDER};
    border-radius: 7px;
    padding: 4px 10px;
    background: {BG_INPUT};
    color: {TEXT_PRIMARY};
    selection-background-color: {ACCENT_BLUE};
    selection-color: #fff;
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
    border-color: #555c6b;
}}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
    border-color: {ACCENT_BLUE};
}}

QComboBox QAbstractItemView {{
    background: {BG_CARD};
    border: 1px solid {BORDER};
    border-radius: 6px;
    selection-background-color: rgba(74, 158, 255, 0.2);
    selection-color: {ACCENT_BLUE};
    outline: none;
    padding: 4px;
}}
QComboBox::drop-down {{
    width: 24px;
    border: 0;
    subcontrol-position: right center;
    subcontrol-origin: padding;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 18px;
    border-left: 1px solid {BORDER};
    border-bottom: 1px solid {BORDER};
    border-top-right-radius: 6px;
    background: {BG_HOVER};
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 18px;
    border-left: 1px solid {BORDER};
    border-bottom-right-radius: 6px;
    background: {BG_HOVER};
}}

/* ====== 复选框 ====== */
QCheckBox {{
    spacing: 8px;
    color: {TEXT_SECONDARY};
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {BG_INPUT};
}}
QCheckBox::indicator:checked {{
    background: {ACCENT_BLUE};
    border-color: {ACCENT_BLUE};
}}
QCheckBox:hover {{
    color: {TEXT_PRIMARY};
}}

QCheckBox#compareCheck {{
    color: {TEXT_SECONDARY};
    spacing: 6px;
    font-size: 12px;
}}
QCheckBox#compareCheck::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {BG_INPUT};
}}
QCheckBox#compareCheck::indicator:checked {{
    background: {ACCENT_TEAL};
    border-color: {ACCENT_TEAL};
}}
QCheckBox#compareCheck:hover {{
    color: {TEXT_PRIMARY};
}}

/* ====== 滚动区域 ====== */
QScrollArea {{
    background: transparent;
    border: 0;
}}

QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: rgba(255,255,255,0.08);
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{
    background: rgba(255,255,255,0.15);
}}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0;
}}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{
    background: none;
}}

QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: rgba(255,255,255,0.08);
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{
    background: rgba(255,255,255,0.15);
}}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{
    width: 0;
}}

/* ====== 工作台 ====== */
QFrame#workspace {{
    border-radius: 10px;
    border: 1px solid {BORDER};
    background: {BG_PANEL};
}}

QLabel#stageTitle {{
    color: {TEXT_PRIMARY};
    font-size: 16px;
    font-weight: 700;
}}

QLabel#stageHint {{
    color: {TEXT_MUTED};
    font-size: 12px;
}}

/* ====== 图像面板 ====== */
QFrame#imagePanel {{
    border: 1px solid {BORDER};
    border-radius: 10px;
    background: {BG_DEEP};
}}

QLabel#imageTitle {{
    color: {TEXT_PRIMARY};
    font-size: 14px;
    font-weight: 700;
}}

QLabel#imageBadge {{
    color: {ACCENT_TEAL};
    font-size: 10px;
    font-weight: 800;
    padding: 2px 7px;
    border-radius: 5px;
    background: rgba(57, 209, 180, 0.1);
    border: 1px solid rgba(57, 209, 180, 0.2);
}}

QLabel#imageMeta {{
    color: {TEXT_MUTED};
    font-size: 11px;
}}

QLabel#imageCanvas {{
    color: {TEXT_MUTED};
    border: 1px dashed {BORDER};
    border-radius: 8px;
    background: {BG_CANVAS};
}}

/* ====== 指标卡片 ====== */
QFrame#metricChip {{
    border: 1px solid {BORDER};
    border-radius: 8px;
    background: {BG_DEEP};
    padding: 2px;
}}
QFrame#metricChip:hover {{
    border-color: #555c6b;
    background: #191c22;
}}

QLabel#metricName {{
    color: {TEXT_MUTED};
    font-size: 11px;
    font-weight: 500;
}}

QLabel#metricValue {{
    color: {TEXT_PRIMARY};
    font-size: 15px;
    font-weight: 750;
    font-family: {FONT_MONO};
}}

/* ====== 历史列表 ====== */
QListWidget#historyList {{
    min-height: 110px;
    border: 1px solid {BORDER};
    border-radius: 8px;
    background: {BG_DEEP};
    padding: 4px;
    outline: none;
}}
QListWidget#historyList::item {{
    padding: 8px 10px;
    border-radius: 5px;
    color: {TEXT_SECONDARY};
    font-size: 12px;
    margin: 1px 0;
}}
QListWidget#historyList::item:hover {{
    color: {TEXT_PRIMARY};
    background: {BG_HOVER};
}}
QListWidget#historyList::item:selected {{
    color: {TEXT_PRIMARY};
    background: rgba(74, 158, 255, 0.15);
    border-left: 2px solid {ACCENT_BLUE};
}}

/* ====== 分屏手柄 ====== */
QSplitter::handle {{
    background: transparent;
    width: 6px;
}}
QSplitter::handle:hover {{
    background: rgba(74, 158, 255, 0.3);
}}

/* ====== 状态栏 ====== */
QStatusBar#status {{
    color: {TEXT_MUTED};
    background: transparent;
    font-size: 12px;
    padding: 2px 8px;
}}

/* ====== 进度条 ====== */
QProgressBar {{
    border: 1px solid {BORDER};
    border-radius: 4px;
    background: {BG_INPUT};
    text-align: center;
    font-size: 10px;
    color: {TEXT_SECONDARY};
    max-height: 16px;
}}
QProgressBar::chunk {{
    background: {ACCENT_BLUE};
    border-radius: 3px;
}}

/* ====== 提示框 ====== */
QMessageBox {{
    background: {BG_CARD};
}}
QMessageBox QLabel {{
    color: {TEXT_PRIMARY};
    font-size: 13px;
}}
QMessageBox QPushButton {{
    min-width: 80px;
    min-height: 30px;
}}
"""
