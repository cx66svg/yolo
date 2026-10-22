"""界面主题。

统一配色与控件样式：深色侧边导航 + 浅色内容区，单一强调色，
不使用渐变与大面积阴影，保证截图与打印观感一致。
"""
from __future__ import annotations

# ---------------------------------------------------------------- 语义色
BG = "#F5F7FB"                 # 页面底色
CARD = "#FFFFFF"               # 卡片/控件底色
BORDER = "#E5E9F2"             # 常规描边
BORDER_STRONG = "#D6DDEA"      # 输入类描边略深
TEXT = "#111827"             # 主文字
TEXT_MUTED = "#6B7280"       # 次要文字（说明、单位）
TEXT_FAINT = "#9CA3AF"       # 占位符、弱提示

ACCENT = "#2563EB"           # 强调色：主按钮、选中态、进度
ACCENT_HOVER = "#1D4ED8"
ACCENT_SOFT = "#EFF4FF"      # 强调色的浅底（选中行、聚焦背景）

SIDEBAR = "#0F172A"          # 侧边导航底色
SIDEBAR_ITEM = "#1E293B"     # 导航项悬停
SIDEBAR_TEXT = "#CBD5E1"
SIDEBAR_MUTED = "#7C8AA0"

SUCCESS = "#16A34A"
WARNING = "#D97706"
DANGER = "#DC2626"
DANGER_SOFT = "#FEF2F2"

FONT_FAMILY = "Microsoft YaHei"
MONO_FAMILY = "Consolas"

# 候选中文字体文件，按优先级尝试注册。
# 某些环境（离屏渲染、精简系统、打包后）字体库为空，仅靠 font-family
# 指定名字拿不到字形，中文会全部渲染成方框，因此需要显式注册。
_CJK_FONT_FILES = [
    r"C:\Windows\Fonts\msyh.ttc",
    r"C:\Windows\Fonts\msyhbd.ttc",
    r"C:\Windows\Fonts\simhei.ttf",
    r"C:\Windows\Fonts\simsun.ttc",
    r"C:\Windows\Fonts\Deng.ttf",
]


def ensure_fonts() -> list[str]:
    """注册系统中文字体，返回成功注册的字体族名。"""
    import os

    from PyQt6.QtGui import QFontDatabase

    registered: list[str] = []
    for path in _CJK_FONT_FILES:
        if not os.path.exists(path):
            continue
        font_id = QFontDatabase.addApplicationFont(path)
        if font_id >= 0:
            registered.extend(QFontDatabase.applicationFontFamilies(font_id))
    return registered


def resolve_family() -> str:
    """从已注册字体中挑一个可用的中文族名。"""
    from PyQt6.QtGui import QFontDatabase

    families = set(QFontDatabase.families())
    for name in (FONT_FAMILY, "Microsoft YaHei", "SimHei", "DengXian",
                 "SimSun", "Noto Sans CJK SC", "Source Han Sans SC"):
        if name in families:
            return name
    return FONT_FAMILY


def stylesheet() -> str:
    family = resolve_family()
    return f"""
    * {{
        font-family: "{family}", "SimHei", "DengXian", "SimSun", sans-serif;
        font-size: 13px;
        color: {TEXT};
    }}
    QWidget#Root {{ background: {BG}; }}
    QToolTip {{
        background: #1F2937; color: #F9FAFB; border: none;
        padding: 6px 8px; border-radius: 6px;
    }}

    /* ---------------- 侧边导航 ---------------- */
    QWidget#Sidebar {{ background: {SIDEBAR}; }}
    QLabel#Brand {{
        color: #FFFFFF; font-size: 18px; font-weight: bold;
        padding: 26px 20px 2px 20px;
    }}
    QLabel#BrandSub {{
        color: {SIDEBAR_MUTED}; font-size: 11px;
        padding: 0 20px 16px 20px;
    }}
    QFrame#BrandDivider {{
        background: #1E293B; border: none; max-height: 1px; margin: 0 16px 12px 16px;
    }}
    QPushButton#NavButton {{
        color: {SIDEBAR_TEXT}; background: transparent; border: none;
        border-radius: 8px; text-align: left; padding: 10px 14px;
        font-size: 13px;
    }}
    QPushButton#NavButton:hover {{ background: {SIDEBAR_ITEM}; color: #FFFFFF; }}
    QPushButton#NavButton:checked {{
        background: {ACCENT}; color: #FFFFFF; font-weight: bold;
    }}
    QLabel#SidebarFooter {{
        color: {SIDEBAR_MUTED}; font-size: 11px; padding: 14px 20px 18px 20px;
    }}

    /* ---------------- 卡片与标题 ---------------- */
    QFrame#Card {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
    }}
    QLabel#CardTitle {{
        font-size: 13px; font-weight: bold; color: {TEXT};
        padding-bottom: 2px;
    }}
    QLabel#Hint {{ color: {TEXT_MUTED}; font-size: 12px; }}
    QLabel#PageTitle {{ font-size: 21px; font-weight: bold; color: {TEXT}; }}
    QLabel#PageDesc {{ color: {TEXT_MUTED}; font-size: 12px; }}
    QFrame#Separator {{ background: {BORDER}; border: none; max-height: 1px; }}

    /* ---------------- 指标块 ---------------- */
    QFrame#StatCard {{
        background: {CARD};
        border: 1px solid {BORDER};
        border-radius: 12px;
    }}
    QLabel#StatValue {{ font-size: 23px; font-weight: bold; color: {TEXT}; }}
    QLabel#StatValue[placeholder="true"] {{
        color: {TEXT_FAINT}; font-size: 19px; font-weight: normal;
    }}
    QLabel#StatLabel {{ color: {TEXT_MUTED}; font-size: 11px; }}

    /* ---------------- 弹窗 ---------------- */
    /* 必须显式配色：系统深色主题会让它变成黑底深字，几乎看不清 */
    QMessageBox, QDialog {{ background: {CARD}; }}
    QMessageBox QLabel, QDialog QLabel {{ color: {TEXT}; font-size: 13px; }}
    QMessageBox QPushButton {{
        min-width: 82px; padding: 7px 16px; border-radius: 8px;
        background: {CARD}; color: {TEXT}; border: 1px solid {BORDER_STRONG};
    }}
    QMessageBox QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; }}
    QMessageBox QPushButton:default {{
        background: {ACCENT}; color: #FFFFFF; border-color: {ACCENT}; font-weight: bold;
    }}
    QMessageBox QPushButton:default:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; }}

    /* ---------------- 按钮 ---------------- */
    QPushButton {{
        background: {CARD}; border: 1px solid {BORDER_STRONG};
        border-radius: 8px; padding: 7px 14px; color: {TEXT};
    }}
    QPushButton:hover {{ border-color: {ACCENT}; color: {ACCENT}; background: {ACCENT_SOFT}; }}
    QPushButton:pressed {{ background: #E3EDFF; }}
    QPushButton:disabled {{
        color: {TEXT_FAINT}; border-color: {BORDER}; background: #FBFCFE;
    }}
    QPushButton#Primary {{
        background: {ACCENT}; color: #FFFFFF; border: 1px solid {ACCENT};
        font-weight: bold; padding: 8px 18px;
    }}
    QPushButton#Primary:hover {{ background: {ACCENT_HOVER}; border-color: {ACCENT_HOVER}; color: #FFFFFF; }}
    QPushButton#Primary:disabled {{ background: #AFC4EC; border-color: #AFC4EC; color: #F5F8FE; }}
    QPushButton#Danger {{ color: {DANGER}; border-color: #F3C6C6; }}
    QPushButton#Danger:hover {{ background: {DANGER_SOFT}; border-color: {DANGER}; color: {DANGER}; }}
    QPushButton#Danger:disabled {{ color: {TEXT_FAINT}; border-color: {BORDER}; background: #FBFCFE; }}

    /* ---------------- 输入类 ---------------- */
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox {{
        background: {CARD}; border: 1px solid {BORDER_STRONG};
        border-radius: 8px; padding: 6px 10px; min-height: 20px;
        selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT};
    }}
    QLineEdit:hover, QComboBox:hover, QSpinBox:hover, QDoubleSpinBox:hover {{
        border-color: #BFC9DC;
    }}
    QLineEdit:focus, QComboBox:focus, QSpinBox:focus, QDoubleSpinBox:focus {{
        border-color: {ACCENT}; background: #FFFFFF;
    }}
    QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled {{
        background: #F7F9FC; color: {TEXT_FAINT};
    }}
    QComboBox::drop-down {{ border: none; width: 20px; }}
    QComboBox QAbstractItemView {{
        background: {CARD}; color: {TEXT};
        border: 1px solid {BORDER_STRONG}; border-radius: 8px;
        padding: 4px;
        selection-background-color: {ACCENT_SOFT};
        selection-color: {TEXT}; outline: none;
    }}

    /* 微调按钮：默认箭头图在部分渲染后端会丢失，用 CSS 三角自绘 */
    QSpinBox::up-button, QDoubleSpinBox::up-button {{
        subcontrol-origin: border; subcontrol-position: top right;
        width: 16px; border-left: 1px solid {BORDER};
        border-top-right-radius: 8px;
    }}
    QSpinBox::down-button, QDoubleSpinBox::down-button {{
        subcontrol-origin: border; subcontrol-position: bottom right;
        width: 16px; border-left: 1px solid {BORDER};
        border-bottom-right-radius: 8px;
    }}
    QSpinBox::up-arrow, QDoubleSpinBox::up-arrow {{
        width: 0; height: 0;
        border-left: 4px solid transparent; border-right: 4px solid transparent;
        border-bottom: 5px solid {TEXT_MUTED};
    }}
    QSpinBox::down-arrow, QDoubleSpinBox::down-arrow {{
        width: 0; height: 0;
        border-left: 4px solid transparent; border-right: 4px solid transparent;
        border-top: 5px solid {TEXT_MUTED};
    }}

    QCheckBox {{ spacing: 8px; }}
    QCheckBox::indicator {{
        width: 16px; height: 16px; border-radius: 4px;
        border: 1px solid {BORDER_STRONG}; background: {CARD};
    }}
    QCheckBox::indicator:checked {{ background: {ACCENT}; border-color: {ACCENT}; }}

    QSlider::groove:horizontal {{
        height: 4px; background: #E8EDF6; border-radius: 2px;
    }}
    QSlider::sub-page:horizontal {{ background: {ACCENT}; border-radius: 2px; }}
    QSlider::handle:horizontal {{
        background: {CARD}; border: 2px solid {ACCENT};
        width: 12px; height: 12px; margin: -5px 0; border-radius: 8px;
    }}

    /* ---------------- 日志与列表 ---------------- */
    /* 浅底深字：深底配合系统主题与默认选中色会让文字糊成一片 */
    QPlainTextEdit, QTextEdit {{
        background: #FFFFFF; color: {TEXT};
        border: 1px solid {BORDER}; border-radius: 10px;
        font-family: "{MONO_FAMILY}", "Courier New", monospace;
        font-size: 12px; padding: 10px;
        selection-background-color: {ACCENT_SOFT};
        selection-color: {TEXT};
    }}
    QListWidget, QListView, QTreeView {{
        background: {CARD}; color: {TEXT};
        border: 1px solid {BORDER}; border-radius: 10px;
        outline: none; padding: 4px;
        selection-background-color: {ACCENT_SOFT};
        selection-color: {TEXT};
    }}
    QListWidget::item, QListView::item {{
        padding: 8px 10px; border-radius: 8px; margin: 1px 0;
    }}
    QListWidget::item:selected, QListView::item:selected {{
        background: {ACCENT_SOFT}; color: {TEXT}; font-weight: bold;
    }}
    QListWidget::item:hover, QListView::item:hover {{ background: #F4F7FD; }}

    QTableWidget {{
        background: {CARD}; border: 1px solid {BORDER};
        border-radius: 10px; gridline-color: #EEF2F8;
        selection-background-color: {ACCENT_SOFT}; selection-color: {TEXT};
        alternate-background-color: #FBFCFE;
    }}
    QTableWidget::item {{ padding: 4px 8px; }}
    QHeaderView::section {{
        background: #F8FAFD; border: none; border-bottom: 1px solid {BORDER};
        padding: 9px 8px; font-weight: bold; color: {TEXT_MUTED}; font-size: 12px;
    }}
    QTableCornerButton::section {{ background: #F8FAFD; border: none; }}

    /* ---------------- 进度与页签 ---------------- */
    QProgressBar {{
        background: #E8EDF6; border: none; border-radius: 5px;
        min-height: 10px; max-height: 10px;
        text-align: center; color: transparent;
    }}
    QProgressBar::chunk {{ background: {ACCENT}; border-radius: 5px; }}

    QTabWidget::pane {{
        border: 1px solid {BORDER}; border-radius: 10px; background: {CARD};
        top: -1px;
    }}
    QTabBar::tab {{
        background: transparent; padding: 8px 18px; border: none;
        color: {TEXT_MUTED}; margin-right: 4px;
    }}
    QTabBar::tab:hover {{ color: {TEXT}; }}
    QTabBar::tab:selected {{
        color: {ACCENT}; border-bottom: 2px solid {ACCENT}; font-weight: bold;
    }}

    /* ---------------- 滚动条 ---------------- */
    QScrollBar:vertical {{ background: transparent; width: 10px; margin: 2px; }}
    QScrollBar::handle:vertical {{
        background: #C8D2E0; border-radius: 5px; min-height: 32px;
    }}
    QScrollBar::handle:vertical:hover {{ background: #A9B6C9; }}
    QScrollBar::add-line, QScrollBar::sub-line {{ height: 0; width: 0; }}
    QScrollBar::add-page, QScrollBar::sub-page {{ background: transparent; }}
    QScrollBar:horizontal {{ background: transparent; height: 10px; margin: 2px; }}
    QScrollBar::handle:horizontal {{
        background: #C8D2E0; border-radius: 5px; min-width: 32px;
    }}
    QScrollBar::handle:horizontal:hover {{ background: #A9B6C9; }}
    """
