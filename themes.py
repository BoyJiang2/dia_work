"""数字图像处理系统 — 三主题配色与 QSS 生成."""
from __future__ import annotations

from dataclasses import dataclass

FONT_STACK = '"Microsoft YaHei UI", "Segoe UI", "SF Pro Display", Arial, sans-serif'
FONT_MONO = '"Cascadia Code", "Fira Code", "JetBrains Mono", "Consolas", monospace'


@dataclass(frozen=True)
class Palette:
    """一组主题配色常量."""
    name: str
    description: str
    # 背景
    bg_deep: str
    bg_panel: str
    bg_card: str
    bg_hover: str
    bg_input: str
    bg_canvas: str
    # 边框
    border: str
    border_focus: str
    # 文字
    text_primary: str
    text_secondary: str
    text_muted: str
    # 强调色
    accent: str           # 主强调
    accent_hover: str
    accent_pressed: str
    green: str
    green_hover: str
    teal: str
    orange: str
    red: str
    # 按钮
    btn_primary_bg: str
    btn_primary_text: str
    btn_accent_bg: str
    btn_accent_text: str
    btn_secondary_text: str
    btn_secondary_bg: str
    btn_secondary_border: str
    btn_ghost_text: str
    btn_ghost_bg: str
    btn_ghost_border: str
    # 顶栏
    topbar_bg: str
    topbar_border: str
    # 工作台
    workspace_bg: str
    workspace_border: str
    # 特殊
    card_shadow_r: int = 0
    card_shadow_g: int = 0
    card_shadow_b: int = 0
    card_shadow_a: int = 60
    base_font_size: int = 14
    corner_radius: int = 10


# ═══════════════════════════════════════════════════════════════
#  方案 A：Professional Dark
# ═══════════════════════════════════════════════════════════════

DARK_PRO = Palette(
    name="Professional Dark",
    description="类 VS Code / Figma 深色专业工具风格",
    bg_deep="#15181e",
    bg_panel="#1e2128",
    bg_card="#252830",
    bg_hover="#2c303a",
    bg_input="#1a1d24",
    bg_canvas="#0d1117",
    border="#333840",
    border_focus="#4a9eff",
    text_primary="#e0e4ec",
    text_secondary="#8b949e",
    text_muted="#5d6778",
    accent="#4a9eff",
    accent_hover="#5cb6ff",
    accent_pressed="#3b8edb",
    green="#3fb950",
    green_hover="#4cc962",
    teal="#39d1b4",
    orange="#d29922",
    red="#f85149",
    btn_primary_bg="#4a9eff",
    btn_primary_text="#ffffff",
    btn_accent_bg="#3fb950",
    btn_accent_text="#ffffff",
    btn_secondary_text="#4a9eff",
    btn_secondary_bg="rgba(74,158,255,0.08)",
    btn_secondary_border="rgba(74,158,255,0.25)",
    btn_ghost_text="#8b949e",
    btn_ghost_bg="#2c303a",
    btn_ghost_border="#333840",
    topbar_bg="#1e2128",
    topbar_border="#333840",
    workspace_bg="#1e2128",
    workspace_border="#333840",
    card_shadow_r=0, card_shadow_g=0, card_shadow_b=0, card_shadow_a=80,
    base_font_size=15,
    corner_radius=10,
)


# ═══════════════════════════════════════════════════════════════
#  方案 B：Glassmorphism Light
# ═══════════════════════════════════════════════════════════════

GLASS_LIGHT = Palette(
    name="Glassmorphism Light",
    description="macOS / Win11 风格半透明毛玻璃，轻盈现代",
    bg_deep="#e8ecf1",
    bg_panel="#f0f2f5",
    bg_card="rgba(255,255,255,0.72)",
    bg_hover="rgba(255,255,255,0.92)",
    bg_input="#ffffff",
    bg_canvas="#fafbfc",
    border="rgba(0,0,0,0.08)",
    border_focus="#3b82f6",
    text_primary="#1e293b",
    text_secondary="#64748b",
    text_muted="#94a3b8",
    accent="#3b82f6",
    accent_hover="#609cf7",
    accent_pressed="#2563eb",
    green="#10b981",
    green_hover="#34d399",
    teal="#14b8a6",
    orange="#f59e0b",
    red="#ef4444",
    btn_primary_bg="#3b82f6",
    btn_primary_text="#ffffff",
    btn_accent_bg="#10b981",
    btn_accent_text="#ffffff",
    btn_secondary_text="#3b82f6",
    btn_secondary_bg="rgba(59,130,246,0.06)",
    btn_secondary_border="rgba(59,130,246,0.2)",
    btn_ghost_text="#64748b",
    btn_ghost_bg="rgba(0,0,0,0.03)",
    btn_ghost_border="rgba(0,0,0,0.08)",
    topbar_bg="rgba(255,255,255,0.78)",
    topbar_border="rgba(0,0,0,0.06)",
    workspace_bg="rgba(255,255,255,0.55)",
    workspace_border="rgba(0,0,0,0.06)",
    card_shadow_r=0, card_shadow_g=0, card_shadow_b=0, card_shadow_a=30,
    base_font_size=15,
    corner_radius=14,
)


# ═══════════════════════════════════════════════════════════════
#  方案 C：Dark Neon
# ═══════════════════════════════════════════════════════════════

DARK_NEON = Palette(
    name="Dark Neon",
    description="科技感终端风格，纯黑底色 + 青紫霓虹点缀",
    bg_deep="#08080d",
    bg_panel="#0e0e16",
    bg_card="#111118",
    bg_hover="#181822",
    bg_input="#0c0c14",
    bg_canvas="#040408",
    border="#1e1e2e",
    border_focus="#00e5ff",
    text_primary="#c8d6e5",
    text_secondary="#7a8ba0",
    text_muted="#4a5568",
    accent="#00e5ff",
    accent_hover="#3ef0ff",
    accent_pressed="#00b8d4",
    green="#00e676",
    green_hover="#5cff8a",
    teal="#1de9b6",
    orange="#ff9100",
    red="#ff1744",
    btn_primary_bg="rgba(0,229,255,0.15)",
    btn_primary_text="#00e5ff",
    btn_accent_bg="rgba(0,230,118,0.15)",
    btn_accent_text="#00e676",
    btn_secondary_text="#b388ff",
    btn_secondary_bg="rgba(179,136,255,0.08)",
    btn_secondary_border="rgba(179,136,255,0.25)",
    btn_ghost_text="#7a8ba0",
    btn_ghost_bg="rgba(255,255,255,0.03)",
    btn_ghost_border="#1e1e2e",
    topbar_bg="#0e0e16",
    topbar_border="#1e1e2e",
    workspace_bg="#0e0e16",
    workspace_border="#1e1e2e",
    card_shadow_r=0, card_shadow_g=229, card_shadow_b=255, card_shadow_a=12,
    base_font_size=15,
    corner_radius=10,
)


# ═══════════════════════════════════════════════════════════════
#  方案 D：Midnight Gold（用户设计稿）
# ═══════════════════════════════════════════════════════════════

MIDNIGHT_GOLD = Palette(
    name="Midnight Gold",
    description="深海蓝 + 香槟金 — 经典沉稳的设计稿风格",
    bg_deep="#fafaf8",
    bg_panel="#f4f3ef",
    bg_card="#ffffff",
    bg_hover="#f0efe9",
    bg_input="#fafaf8",
    bg_canvas="#fafaf8",
    border="#d8d5cc",
    border_focus="#3e8abb",
    text_primary="#1a1e28",
    text_secondary="#5a5d66",
    text_muted="#949698",
    accent="#3e8abb",
    accent_hover="#559cc9",
    accent_pressed="#2e6d94",
    green="#5a9e6f",
    green_hover="#6db083",
    teal="#3e8abb",
    orange="#c4883d",
    red="#c4554d",
    btn_primary_bg="#3e8abb",
    btn_primary_text="#ffffff",
    btn_accent_bg="#f8c15d",
    btn_accent_text="#2d2210",
    btn_secondary_text="#3e8abb",
    btn_secondary_bg="rgba(62,138,187,0.06)",
    btn_secondary_border="rgba(62,138,187,0.2)",
    btn_ghost_text="#5a5d66",
    btn_ghost_bg="rgba(0,0,0,0.03)",
    btn_ghost_border="#d8d5cc",
    topbar_bg="#0f1f32",
    topbar_border="#1a3148",
    workspace_bg="#f4f3ef",
    workspace_border="#d8d5cc",
    card_shadow_r=0, card_shadow_g=0, card_shadow_b=0, card_shadow_a=18,
    base_font_size=15,
    corner_radius=8,
)


# ═══════════════════════════════════════════════════════════════
#  QSS 生成器
# ═══════════════════════════════════════════════════════════════

def _icon_visible_in_theme(p: Palette) -> bool:
    """emojis look best on dark backgrounds, return False on light."""
    return p is not GLASS_LIGHT


def build_qss(p: Palette) -> str:
    fs = p.base_font_size
    fs_sm = max(11, fs - 3)
    fs_lg = fs + 3
    fs_xl = fs + 8
    cr = p.corner_radius
    cr_sm = max(4, cr - 4)

    # Midnight Gold 顶栏特殊处理：深色顶栏 + 金色点缀
    is_midnight = p is MIDNIGHT_GOLD
    if is_midnight:
        title_color = "#f8f4ea"
        subtitle_color = "#c4b998"
        tag_color = "#f8c15d"
        tag_bg = "rgba(248,193,93,0.12)"
        tag_border = "rgba(248,193,93,0.28)"
        btn_primary_bg = "#f8c15d"
        btn_primary_text = "#2d2210"
        btn_primary_hover = "#fcd580"
        btn_primary_pressed = "#d9a54a"
    else:
        title_color = p.text_primary
        subtitle_color = p.text_secondary
        tag_color = p.teal
        tag_bg = f"{p.teal}14"
        tag_border = f"{p.teal}33"
        btn_primary_bg = p.btn_primary_bg
        btn_primary_text = p.btn_primary_text
        btn_primary_hover = p.accent_hover
        btn_primary_pressed = p.accent_pressed

    return f"""
/* ═════ {p.name} ═════ */

QWidget {{
    font-family: {FONT_STACK};
    font-size: {fs}px;
    color: {p.text_primary};
    background: transparent;
}}

QMainWindow#rootWindow, QWidget#appShell {{
    background: {p.bg_deep};
}}

/* ── 顶栏 ── */
QFrame#topBar {{
    border: 1px solid {p.topbar_border};
    border-radius: {cr}px;
    background: {p.topbar_bg};
}}

QLabel#appTitle {{
    color: {title_color};
    font-size: {fs_xl}px;
    font-weight: 700;
    letter-spacing: 0.5px;
}}

QLabel#appSubtitle {{
    color: {subtitle_color};
    font-size: {fs_sm}px;
}}

QLabel#courseTag {{
    color: {tag_color};
    font-weight: 700;
    font-size: {fs_sm}px;
    padding: 6px 12px;
    border: 1px solid {tag_border};
    border-radius: {cr_sm}px;
    background: {tag_bg};
}}

/* ── 卡片 ── */
QFrame#card {{
    border: 1px solid {p.border};
    border-radius: {cr}px;
    background: {p.bg_card};
}}

QLabel#cardTitle {{
    color: {p.text_primary};
    font-size: {fs}px;
    font-weight: 700;
    padding-bottom: 4px;
    border-bottom: 1px solid {p.border};
}}

QLabel#description, QLabel#pathLabel {{
    color: {p.text_secondary};
    font-size: {fs_sm}px;
    line-height: 1.5;
    padding: 8px 10px;
    border-radius: {cr_sm}px;
    background: {p.bg_input};
}}

/* ── 按钮 ── */
QPushButton {{
    min-height: {fs + 22}px;
    border: 1px solid transparent;
    border-radius: {cr_sm}px;
    padding: 6px 16px;
    font-weight: 650;
    font-size: {fs}px;
}}

QPushButton#primaryButton {{
    color: {btn_primary_text};
    background: {btn_primary_bg};
    border-color: {btn_primary_bg};
}}
QPushButton#primaryButton:hover {{ background: {btn_primary_hover}; }}
QPushButton#primaryButton:pressed {{ background: {btn_primary_pressed}; }}

QPushButton#accentButton {{
    color: {p.btn_accent_text};
    min-height: {fs + 28}px;
    font-size: {fs_lg}px;
    font-weight: 700;
    background: {p.btn_accent_bg};
    border-color: {p.btn_accent_bg};
}}
QPushButton#accentButton:hover {{ background: {p.green_hover}; }}
QPushButton#accentButton:pressed {{ background: #2d9a40; }}

QPushButton#secondaryButton {{
    color: {p.btn_secondary_text};
    background: {p.btn_secondary_bg};
    border-color: {p.btn_secondary_border};
}}
QPushButton#secondaryButton:hover {{ background: {p.btn_secondary_border}; }}

QPushButton#ghostButton {{
    color: {p.btn_ghost_text};
    background: {p.btn_ghost_bg};
    border-color: {p.btn_ghost_border};
}}
QPushButton#ghostButton:hover {{
    color: {p.text_primary};
    background: {p.bg_hover};
}}

QPushButton#toolBtn {{
    color: {p.text_secondary};
    min-height: {fs + 10}px;
    max-height: {fs + 14}px;
    border: 1px solid {p.border};
    border-radius: {cr_sm}px;
    padding: 3px 10px;
    font-size: {fs_sm}px;
    font-weight: 600;
    background: {p.bg_input};
}}
QPushButton#toolBtn:hover {{
    color: {p.text_primary};
    background: {p.bg_hover};
    border-color: {p.text_muted};
}}

/* ── 下拉框 / 输入 ── */
QComboBox, QSpinBox, QDoubleSpinBox {{
    min-height: {fs + 20}px;
    border: 1px solid {p.border};
    border-radius: {cr_sm}px;
    padding: 4px 10px;
    background: {p.bg_input};
    color: {p.text_primary};
    font-size: {fs}px;
    selection-background-color: {p.accent};
    selection-color: #ffffff;
}}
QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{ border-color: {p.text_muted}; }}
QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{ border-color: {p.border_focus}; }}

QComboBox QAbstractItemView {{
    background: {p.bg_card};
    border: 1px solid {p.border};
    border-radius: {cr_sm}px;
    font-size: {fs}px;
    selection-background-color: {p.accent}33;
    selection-color: {p.accent};
    outline: none;
    padding: 4px;
}}
QComboBox::drop-down {{
    width: 24px;
    border: 0;
}}
QComboBox::down-arrow {{
    width: 10px;
    height: 10px;
}}

QSpinBox::up-button, QDoubleSpinBox::up-button {{
    subcontrol-origin: border;
    subcontrol-position: top right;
    width: 20px;
    border-left: 1px solid {p.border};
    border-bottom: 1px solid {p.border};
    border-top-right-radius: {cr_sm - 1}px;
    background: {p.bg_hover};
}}
QSpinBox::down-button, QDoubleSpinBox::down-button {{
    subcontrol-origin: border;
    subcontrol-position: bottom right;
    width: 20px;
    border-left: 1px solid {p.border};
    border-bottom-right-radius: {cr_sm - 1}px;
    background: {p.bg_hover};
}}

/* ── 复选框 ── */
QCheckBox {{
    spacing: 8px;
    color: {p.text_secondary};
    font-size: {fs}px;
}}
QCheckBox::indicator {{
    width: 16px;
    height: 16px;
    border: 1px solid {p.border};
    border-radius: 4px;
    background: {p.bg_input};
}}
QCheckBox::indicator:checked {{
    background: {p.accent};
    border-color: {p.accent};
}}
QCheckBox:hover {{ color: {p.text_primary}; }}

QCheckBox#compareCheck {{
    color: {p.text_secondary};
    spacing: 6px;
    font-size: {fs_sm}px;
}}
QCheckBox#compareCheck::indicator {{
    width: 15px;
    height: 15px;
    border: 1px solid {p.border};
    border-radius: 4px;
    background: {p.bg_input};
}}
QCheckBox#compareCheck::indicator:checked {{
    background: {p.teal};
    border-color: {p.teal};
}}
QCheckBox#compareCheck:hover {{ color: {p.text_primary}; }}

/* ── 滚动条 ── */
QScrollArea {{ background: transparent; border: 0; }}

QScrollBar:vertical {{
    background: transparent;
    width: 8px;
    margin: 0;
}}
QScrollBar::handle:vertical {{
    background: {p.text_muted}30;
    border-radius: 4px;
    min-height: 30px;
}}
QScrollBar::handle:vertical:hover {{ background: {p.text_muted}55; }}
QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{ height: 0; }}
QScrollBar::add-page:vertical, QScrollBar::sub-page:vertical {{ background: none; }}

QScrollBar:horizontal {{
    background: transparent;
    height: 8px;
    margin: 0;
}}
QScrollBar::handle:horizontal {{
    background: {p.text_muted}30;
    border-radius: 4px;
    min-width: 30px;
}}
QScrollBar::handle:horizontal:hover {{ background: {p.text_muted}55; }}
QScrollBar::add-line:horizontal, QScrollBar::sub-line:horizontal {{ width: 0; }}

/* ── 工作台 ── */
QFrame#workspace {{
    border-radius: {cr}px;
    border: 1px solid {p.workspace_border};
    background: {p.workspace_bg};
}}

QLabel#stageTitle {{
    color: {p.text_primary};
    font-size: {fs_lg}px;
    font-weight: 700;
}}
QLabel#stageHint {{
    color: {p.text_muted};
    font-size: {fs_sm}px;
}}

/* ── 图像面板 ── */
QFrame#imagePanel {{
    border: 1px solid {p.border};
    border-radius: {cr}px;
    background: {p.bg_deep};
}}

QLabel#imageTitle {{
    color: {p.text_primary};
    font-size: {fs}px;
    font-weight: 700;
}}
QLabel#imageBadge {{
    color: {p.teal};
    font-size: {fs_sm - 2}px;
    font-weight: 800;
    padding: 2px 7px;
    border-radius: 5px;
    background: {p.teal}1a;
    border: 1px solid {p.teal}33;
}}
QLabel#imageMeta {{
    color: {p.text_muted};
    font-size: {fs_sm - 1}px;
}}
QLabel#imageCanvas {{
    color: {p.text_muted};
    border: 1px dashed {p.border};
    border-radius: {cr_sm}px;
    background: {p.bg_canvas};
}}

/* ── 指标卡片 ── */
QFrame#metricChip {{
    border: 1px solid {p.border};
    border-radius: {cr_sm}px;
    background: {p.bg_deep};
    padding: 2px;
}}
QFrame#metricChip:hover {{
    border-color: {p.text_muted};
    background: {p.bg_panel};
}}
QLabel#metricName {{
    color: {p.text_muted};
    font-size: {fs_sm - 1}px;
}}
QLabel#metricValue {{
    color: {p.text_primary};
    font-size: {fs_lg}px;
    font-weight: 750;
    font-family: {FONT_MONO};
}}

/* ── 历史列表 ── */
QListWidget#historyList {{
    min-height: 110px;
    border: 1px solid {p.border};
    border-radius: {cr_sm}px;
    background: {p.bg_deep};
    padding: 4px;
    outline: none;
    font-size: {fs_sm}px;
}}
QListWidget#historyList::item {{
    padding: 8px 10px;
    border-radius: 5px;
    color: {p.text_secondary};
    font-size: {fs_sm}px;
    margin: 1px 0;
}}
QListWidget#historyList::item:hover {{
    color: {p.text_primary};
    background: {p.bg_hover};
}}
QListWidget#historyList::item:selected {{
    color: {p.text_primary};
    background: {p.accent}26;
    border-left: 2px solid {p.accent};
}}

/* ── 分屏手柄 ── */
QSplitter::handle {{
    background: transparent;
    width: 6px;
}}
QSplitter::handle:hover {{ background: {p.accent}40; }}

/* ── 状态栏 ── */
QStatusBar#status {{
    color: {p.text_muted};
    background: transparent;
    font-size: {fs_sm}px;
    padding: 2px 8px;
}}

/* ── 进度条 ── */
QProgressBar {{
    border: 1px solid {p.border};
    border-radius: 4px;
    background: {p.bg_input};
    text-align: center;
    font-size: 10px;
    color: {p.text_secondary};
    max-height: 16px;
}}
QProgressBar::chunk {{
    background: {p.accent};
    border-radius: 3px;
}}

/* ── 对话框 ── */
QMessageBox {{ background: {p.bg_card}; }}
QMessageBox QLabel {{ color: {p.text_primary}; font-size: {fs}px; }}
QMessageBox QPushButton {{ min-width: 80px; min-height: {fs + 18}px; }}

/* ── 主题选择器 ── */
QComboBox#themeSelector {{
    min-height: {fs + 14}px;
    max-width: 170px;
    border: 1px solid {p.border};
    border-radius: {cr_sm}px;
    padding: 2px 8px;
    background: {p.bg_input};
    color: {p.text_secondary};
    font-size: {fs_sm}px;
    font-weight: 600;
}}
QComboBox#themeSelector:hover {{ color: {p.text_primary}; border-color: {p.text_muted}; }}
QComboBox#themeSelector::drop-down {{ width: 20px; border: 0; }}
QComboBox#themeSelector QAbstractItemView {{
    background: {p.bg_card};
    border: 1px solid {p.border};
    border-radius: {cr_sm}px;
    font-size: {fs_sm}px;
    padding: 4px;
}}
"""


# ═══════════════════════════════════════════════════════════════
#  主题注册表
# ═══════════════════════════════════════════════════════════════

THEMES: dict[str, dict] = {}
for _p in (MIDNIGHT_GOLD, DARK_PRO, GLASS_LIGHT, DARK_NEON):
    THEMES[_p.name] = {
        "id": _p.name.lower().replace(" ", "_"),
        "name": _p.name,
        "description": _p.description,
        "palette": _p,
        "qss": build_qss(_p),
    }

THEME_IDS = list(THEMES.keys())
DEFAULT_THEME = MIDNIGHT_GOLD.name


def get_theme(name: str) -> dict:
    """按名称获取主题，fallback 到默认主题."""
    return THEMES.get(name, THEMES[DEFAULT_THEME])


def get_qss(name: str) -> str:
    return get_theme(name)["qss"]


def get_palette(name: str) -> Palette:
    return get_theme(name)["palette"]
