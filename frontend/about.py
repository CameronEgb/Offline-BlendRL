"""About dialog with a shaded, software-rendered ASCII theta sculpture."""
import math
import time

from PyQt6.QtCore import Qt, QTimer, QRectF
from PyQt6.QtGui import QColor, QFont, QPainter
from PyQt6.QtWidgets import QDialog, QHBoxLayout, QPushButton, QVBoxLayout, QWidget

from .widgets import label
from .theme import theme_color, current_theme


class AsciiTheta(QWidget):
    columns, rows = 76, 30
    ramp = ".,:;irsXA253hMHGS#9B&@"

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumSize(460, 280)
        self.setAccessibleName("Rotating three-dimensional ASCII theta logo")
        self.points = []
        # Elliptical ring, with a cylindrical crossbar to form a capital theta.
        for i in range(160):
            u = math.tau * i / 160
            for j in range(28):
                v = math.tau * j / 28
                cu, su, cv, sv = math.cos(u), math.sin(u), math.cos(v), math.sin(v)
                self.points.append(((1 + .18 * cv) * cu, 1.22 * (1 + .18 * cv) * su,
                                    .18 * sv, cv * cu, cv * su / 1.22, sv))
        for i in range(100):
            for j in range(28):
                v = math.tau * j / 28
                self.points.append((-1 + 2 * i / 99, .13 * math.cos(v), .13 * math.sin(v),
                                    0, math.cos(v), math.sin(v)))
        self.angle = .28
        self.paused = False
        self.last_time = time.monotonic()
        self.timer = QTimer(self)
        self.timer.setInterval(33)
        self.timer.timeout.connect(self.advance)

    def showEvent(self, event):
        super().showEvent(event)
        self.last_time = time.monotonic()
        if not self.paused:
            self.timer.start()

    def hideEvent(self, event):
        self.timer.stop()
        super().hideEvent(event)

    def set_paused(self, paused):
        self.paused = paused
        self.last_time = time.monotonic()
        if paused:
            self.timer.stop()
        elif self.isVisible():
            self.timer.start()

    def advance(self):
        now = time.monotonic()
        self.angle = (self.angle + min(now - self.last_time, .1) * .7) % math.tau
        self.last_time = now
        self.update()

    def frame(self):
        cells = {}
        ca, sa = math.cos(self.angle), math.sin(self.angle)
        tilt = .20 + .12 * math.sin(self.angle)
        ct, st = math.cos(tilt), math.sin(tilt)
        for x, y, z, nx, ny, nz in self.points:
            # Rotate about the vertical axis, then gently tilt toward the viewer.
            x, z = ca * x + sa * z, -sa * x + ca * z
            nx, nz = ca * nx + sa * nz, -sa * nx + ca * nz
            y, z = ct * y - st * z, st * y + ct * z
            ny, nz = ct * ny - st * nz, st * ny + ct * nz
            perspective = 4.8 / (4.8 - z)
            col = round(self.columns / 2 + x * perspective * 19)
            row = round(self.rows / 2 - y * perspective * 8.8)
            if not (0 <= col < self.columns and 0 <= row < self.rows):
                continue
            key = (col, row)
            if key in cells and cells[key][0] >= z:
                continue
            normal_length = math.sqrt(nx * nx + ny * ny + nz * nz)
            brightness = .20 + .80 * max(0, (-.35 * nx + .45 * ny + .82 * nz) / normal_length)
            index = min(len(self.ramp) - 1, int(brightness * (len(self.ramp) - 1)))
            cells[key] = (z, self.ramp[index], brightness)
        return cells

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.fillRect(self.rect(), QColor(theme_color("base")))
        painter.setRenderHint(QPainter.RenderHint.TextAntialiasing)
        cell = min(self.width() / self.columns, self.height() / (self.rows * 1.8))
        height = cell * 1.8
        left = (self.width() - cell * self.columns) / 2
        top = (self.height() - height * self.rows) / 2
        font = QFont("Consolas")
        font.setStyleHint(QFont.StyleHint.Monospace)
        font.setPixelSize(max(8, int(height)))
        painter.setFont(font)
        for (col, row), (_, char, brightness) in self.frame().items():
            color = "#ebdbb2" if brightness > .83 else "#fabd2f" if brightness > .48 else "#d79921" if brightness > .30 else "#665c54"
            painter.setPen(QColor(theme_color(color)))
            painter.drawText(QRectF(left + col * cell, top + row * height, cell, height),
                             Qt.AlignmentFlag.AlignCenter, char)


class AboutDialog(QDialog):
    def __init__(self, parent=None):
        super().__init__(parent)
        self.setWindowTitle("About ThetaIDE")
        self.resize(620, 630)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(24, 20, 24, 20)
        layout.setSpacing(12)
        title = label("ThetaIDE", "brand")
        title.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title)
        subtitle = label(f"PyQt6  /  {current_theme()['name']}  /  local research workspace", "muted")
        subtitle.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(subtitle)
        self.logo = AsciiTheta(self)
        layout.addWidget(self.logo, 1)
        for text in (
            "Working: configuration, synthetic live metrics, stop, saved run records, comparison, and linked notes.",
            "Not connected: actual training, checkpoints, plugin execution, datasets, IPython, or tmux.",
            "The demo curve depends on seed and step; other hyperparameters are recorded only.",
        ):
            description = label(text, "muted")
            description.setWordWrap(True)
            layout.addWidget(description)
        buttons = QHBoxLayout()
        pause = QPushButton("Pause animation")
        pause.setCheckable(True)
        pause.toggled.connect(self.logo.set_paused)
        pause.toggled.connect(lambda checked: pause.setText("Resume animation" if checked else "Pause animation"))
        buttons.addWidget(pause)
        buttons.addStretch()
        close = QPushButton("Close")
        close.clicked.connect(self.accept)
        close.setDefault(True)
        buttons.addWidget(close)
        layout.addLayout(buttons)
