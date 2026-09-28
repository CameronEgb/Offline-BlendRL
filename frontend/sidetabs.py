"""Vertical tab bar with themed SVG icons: a drop-in for the subset of QTabWidget the window uses."""
from pathlib import Path

from PyQt6.QtCore import QByteArray, QRectF, QSize, Qt, pyqtSignal
from PyQt6.QtGui import QIcon, QPainter, QPixmap
from PyQt6.QtSvg import QSvgRenderer
from PyQt6.QtWidgets import (QApplication, QButtonGroup, QHBoxLayout, QSizePolicy, QStackedWidget, QToolButton,
                             QVBoxLayout, QWidget)

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
    logoClicked = pyqtSignal()

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

        self.logo_button = QToolButton()
        self.logo_button.setObjectName("sideTabLogo")
        self.logo_button.setCheckable(True)
        self.logo_button.setAutoRaise(True)
        self.logo_button.setToolTip("Settings & About")
        self.logo_button.setAccessibleName("ThetaIDE Settings & About")
        self.logo_button.setIconSize(QSize(36, 36))
        self.logo_button.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Fixed)
        self.logo_button.setCursor(Qt.CursorShape.PointingHandCursor)
        self.logo_button.clicked.connect(self._on_logo_clicked)
        self.set_logo_icon()
        self.bar_layout.addWidget(self.logo_button)

        self.stack = QStackedWidget()
        layout.addWidget(self.bar)
        layout.addWidget(self.stack, 1)
        self.group = QButtonGroup(self)
        self.group.setExclusive(True)
        self.group.idClicked.connect(self.setCurrentIndex)
        self.buttons = []
        self.icon_names = []
        self.settings_widget = None
        self.settings_index = -1

    def set_logo_icon(self):
        source_path = ICON_DIR / "theta_logo.svg"
        if not source_path.exists():
            return
        ratio = QApplication.instance().devicePixelRatio() if QApplication.instance() else 1.0
        renderer = QSvgRenderer(str(source_path))
        pixmap = QPixmap(round(36 * ratio), round(36 * ratio))
        pixmap.fill(Qt.GlobalColor.transparent)
        painter = QPainter(pixmap)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        renderer.render(painter, QRectF(0, 0, pixmap.width(), pixmap.height()))
        painter.end()
        pixmap.setDevicePixelRatio(ratio)
        self.logo_button.setIcon(QIcon(pixmap))

    def addTab(self, widget, text, icon=None, short=None):
        """icon names an SVG in frontend/icons/; short is the label shown under it (text becomes the tooltip)."""
        index = self.stack.addWidget(widget)
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
        self.bar_layout.insertWidget(len(self.buttons), button)
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
        self.set_logo_icon()

    # ── QTabWidget-compatible API ────────────────────────────────────────────

    def count(self):
        return self.stack.count()

    def currentIndex(self):
        return self.stack.currentIndex()

    def currentWidget(self):
        return self.stack.currentWidget()

    def set_settings_widget(self, widget):
        self.settings_widget = widget
        self.settings_index = self.stack.addWidget(widget)
        return self.settings_index

    def _on_logo_clicked(self):
        self.logoClicked.emit()
        if self.settings_widget is not None and self.settings_index >= 0:
            self.setCurrentIndex(self.settings_index)

    def widget(self, index):
        return self.stack.widget(index)

    def indexOf(self, widget):
        return self.stack.indexOf(widget)

    def setCurrentIndex(self, index):
        if not 0 <= index < self.count():
            return
        if self.settings_index >= 0 and index == self.settings_index:
            self.group.setExclusive(False)
            for button in self.buttons:
                button.setChecked(False)
            self.group.setExclusive(True)
            self.logo_button.setChecked(True)
        elif 0 <= index < len(self.buttons):
            self.logo_button.setChecked(False)
            self.buttons[index].setChecked(True)
        if index != self.stack.currentIndex():
            self.stack.setCurrentIndex(index)
            self.currentChanged.emit(index)

    def setCurrentWidget(self, widget):
        self.setCurrentIndex(self.indexOf(widget))
