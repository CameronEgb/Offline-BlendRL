"""Vertical tab bar with themed SVG icons: a drop-in for the subset of QTabWidget the window uses."""
from pathlib import Path

from PyQt6.QtCore import QByteArray, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QHBoxLayout, QScrollArea, QSizePolicy, QStackedWidget,
                             QToolButton, QVBoxLayout, QWidget)

from .theme import theme_color

ICON_DIR = Path(__file__).resolve().parent / "icons"
ICON_SIZE = 20


def svg_icon(name, colors):
    """Render icons/<name>.svg once per state, replacing currentColor with that state's theme role color.

    colors maps (QIcon.Mode, QIcon.State) to a theme role such as "muted" or "accent".
    """
    source = (ICON_DIR / f"{name}.svg").read_text(encoding="utf-8")
    ratio = QApplication.instance().devicePixelRatio() if QApplication.instance() else 1.0
    icon = QIcon()
    for (mode, state), role in colors.items():
        renderer = QSvgRenderer(QByteArray(source.replace("currentColor", theme_color(role)).encode("utf-8")))
        pixmap = QPixmap(round(ICON_SIZE * ratio), round(ICON_SIZE * ratio))
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter, QRectF(0, 0, pixmap.width(), pixmap.height()))
        painter.end()
        pixmap.setDevicePixelRatio(ratio)
        icon.addPixmap(pixmap, mode, state)
    return icon


TAB_ICON_COLORS = {
    (QIcon.Mode.Normal, QIcon.State.Off): "muted",
    (QIcon.Mode.Active, QIcon.State.Off): "text",
    (QIcon.Mode.Normal, QIcon.State.On): "accent",
    (QIcon.Mode.Active, QIcon.State.On): "accent",
}


class SideTabs(QWidget):
    """Icon-over-label buttons stacked on the left, with the pages in a QStackedWidget beside them."""
    currentChanged = pyqtSignal(int)

    def __init__(self, parent=None):
        super().__init__(parent)
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        self.bar = QWidget()
        self.bar.setObjectName("sideTabs")
        self.bar.setAttribute(Qt.WidgetAttribute.WA_StyledBackground, True)
        self.bar.setSizePolicy(QSizePolicy.Policy.Fixed, QSizePolicy.Policy.Preferred)
        self.bar_layout = QVBoxLayout(self.bar)
        # Inset the buttons so the rounded selection sits inside the bar, clear of its 1 px right border.
        self.bar_layout.setContentsMargins(5, 6, 6, 6)
        self.bar_layout.setSpacing(3)
        self.bar_layout.addStretch()
        self.stack = QStackedWidget()
        layout.addWidget(self.bar)
        layout.addWidget(self.stack, 1)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.group.idClicked.connect(self.setCurrentIndex)
        self.buttons = []
        self.icon_names = []
        self.pages = []

    def addTab(self, widget, text, icon=None, short=None):
        """icon names an SVG in frontend/icons/; short is the label shown under it (text becomes the tooltip)."""
        # Each page scrolls when cramped. Otherwise the stack's minimum height is the tallest page's, and the
        # main window pushes the bottom dock (the console) off-screen to honor it on small or scaled displays.
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setFocusPolicy(Qt.FocusPolicy.NoFocus)  # keep Tab order and clicks on the page's own widgets
        scroll.setWidget(widget)
        self.pages.append(widget)
        index = self.stack.addWidget(scroll)
        button = QToolButton()
        button.setObjectName("sideTab")
        button.setCheckable(True)
        button.setAutoRaise(True)
        button.setToolButtonStyle(Qt.ToolButtonStyle.ToolButtonTextUnderIcon)
        button.setText(short or text)
        button.setToolTip(text)
        button.setAccessibleName(text)
        button.setIconSize(QSize(ICON_SIZE, ICON_SIZE))
        button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.group.addButton(button, index)
        self.bar_layout.insertWidget(self.bar_layout.count() - 1, button)
        self.buttons.append(button)
        self.icon_names.append(icon)
        if icon:
            button.setIcon(svg_icon(icon, TAB_ICON_COLORS))
        if index == 0:
            button.setChecked(True)
        return index

    def refresh_icons(self):
        """Re-render icons in the current theme's colors."""
        for button, name in zip(self.buttons, self.icon_names):
            if name:
                button.setIcon(svg_icon(name, TAB_ICON_COLORS))

    # ── QTabWidget-compatible API ────────────────────────────────────────────

    def count(self):
        return self.stack.count()

    def currentIndex(self):
        return self.stack.currentIndex()

    def currentWidget(self):
        return self.widget(self.currentIndex())

    def widget(self, index):
        return self.pages[index] if 0 <= index < len(self.pages) else None

    def indexOf(self, widget):
        return self.pages.index(widget) if widget in self.pages else -1

    def setCurrentIndex(self, index):
        if not 0 <= index < self.count():
            return
        self.buttons[index].setChecked(True)
        if index != self.stack.currentIndex():
            self.stack.setCurrentIndex(index)
            self.currentChanged.emit(index)

    def setCurrentWidget(self, widget):
        self.setCurrentIndex(self.indexOf(widget))
