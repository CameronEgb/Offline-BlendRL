import math
import re
from PyQt6.QtCore import Qt, QRectF, QSize
from PyQt6.QtGui import QColor, QPainter, QPainterPath, QPen, QFont, QSyntaxHighlighter, QTextCharFormat
from PyQt6.QtWidgets import QWidget, QLabel, QVBoxLayout, QFrame, QAbstractButton
from .theme import theme_color


def label(text, kind=None):
    result = QLabel(text)
    if kind:
        result.setObjectName(kind)
    return result


class MetricCard(QFrame):
    def __init__(self, title, subtitle):
        super().__init__()
        self.setObjectName("card")
        layout = QVBoxLayout(self)
        self.title = label(title, "muted")
        layout.addWidget(self.title)
        self.value = label("—", "value")
        layout.addWidget(self.value)
        self.subtitle = label(subtitle, "muted")
        layout.addWidget(self.subtitle)


class YamlHighlighter(QSyntaxHighlighter):
    def highlightBlock(self, text):
        for pattern, color in ((r'^\s*[\w_]+(?=:)', '#83a598'),
                               (r'\b\d+(?:\.\d+)?\b', '#d3869b'),
                               (r'"[^"\n]*"', '#b8bb26'), (r'#.*$', '#928374')):
            fmt = QTextCharFormat()
            fmt.setForeground(QColor(theme_color(color)))
            for match in re.finditer(pattern, text):
                self.setFormat(match.start(), len(match.group()), fmt)


class Chart(QWidget):
    """Small dependency-free plot; drawing only, no synthetic data hidden here.

    Axes fit the data with round tick values. `reference` (e.g. an environment's maximum
    reward) caps the y-axis and is drawn as a dashed line once the data approaches it.
    `band` names a per-point spread (e.g. "reward_std") drawn as a shaded ±1 band.
    """
    def __init__(self, metric="reward", title="Episode reward", band=None, zero_based=True):
        super().__init__()
        self.metric, self.title, self.band = metric, title, band
        self.zero_based = zero_based  # False fits the y-axis to the data (small changes stay visible)
        self.series = []
        self.xmax = None
        self.reference = None
        self.empty_text = "Launch training to see live metrics"
        self.corner = None
        self.setMinimumHeight(185)
        self.setMinimumWidth(240)

    def set_series(self, series, xmax=None, reference=None):
        """xmax fixes the x-axis extent (e.g. a run's training budget) instead of fitting the data."""
        self.xmax, self.reference = xmax, reference
        # Live runs report different metrics on different rows; skip points missing this chart's metric.
        self.series = [(name, [m for m in metrics if m.get(self.metric) is not None], color)
                       for name, metrics, color in series]
        self.update()

    def set_metric(self, metric, title):
        self.metric, self.title = metric, title

    def set_corner_widget(self, widget):
        """Place a small control (e.g. a metric selector) in the chart's top-right corner."""
        widget.setParent(self)
        self.corner = widget
        self.position_corner()

    def position_corner(self):
        if self.corner is not None:
            self.corner.adjustSize()
            self.corner.move(self.width() - self.corner.width() - 12, 6)

    def resizeEvent(self, event):
        self.position_corner()
        super().resizeEvent(event)

    def y_range(self, points):
        values = [m[self.metric] for m in points]
        if self.band:
            spreads = [(m[self.metric], m.get(self.band)) for m in points]
            values += [mean + s for mean, s in spreads if s is not None]
            values += [max(0.0, mean - s) for mean, s in spreads if s is not None]
        anchor = [0.0] if self.zero_based else []
        lo, hi = min(values + anchor), max(values + anchor)
        if hi == lo:
            lo, hi = lo - 0.5 * (abs(lo) or 1), hi + 0.5 * (abs(hi) or 1)
        pad = (hi - lo) * 0.05
        lo, hi = (lo if self.zero_based and lo == 0 else lo - pad), hi + pad
        step = nice_step((hi - lo) / 4)
        lo, hi = math.floor(lo / step) * step, math.ceil(hi / step) * step
        if self.reference is not None and hi >= self.reference * 0.9:
            hi = self.reference  # close to the ceiling: show the full scale up to the maximum
            step = nice_step((hi - lo) / 4)
        return lo, hi, step

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor(theme_color("panel")))
        painter.setFont(QFont("Segoe UI", 10))
        painter.setPen(QColor(theme_color("text")))
        painter.drawText(16, 24, self.title)
        area = QRectF(52, 45, self.width() - 74, self.height() - 80)
        points = [m for _, metrics, _ in self.series for m in metrics]
        xmax = self.xmax or max([m["step"] for m in points] or [10000])
        lo, hi, step = self.y_range(points) if points else (0.0, 1.0, 0.25)

        def y_of(value):
            return area.bottom() - area.height() * (value - lo) / max(1e-12, hi - lo)

        def x_of(steps):
            return area.left() + area.width() * steps / max(1, xmax)

        painter.setFont(QFont("Segoe UI", 8))
        value = lo
        while value <= hi + step / 2:
            y = y_of(value)
            painter.setPen(QColor(theme_color("raised")))
            painter.drawLine(int(area.left()), int(y), int(area.right()), int(y))
            painter.setPen(QColor(theme_color("muted")))
            painter.drawText(QRectF(0, y - 8, 46, 16), Qt.AlignmentFlag.AlignRight, format_tick(value, step))
            value += step
        for i in range(5):
            x = area.left() + area.width() * i / 4
            painter.drawText(QRectF(x - 25, area.bottom() + 8, 50, 18), Qt.AlignmentFlag.AlignCenter,
                             f"{xmax * i / 4000:g}k")
        if self.reference is not None and points:
            if hi >= self.reference:
                painter.setPen(QPen(QColor(theme_color("comment")), 1, Qt.PenStyle.DashLine))
                y = y_of(self.reference)
                painter.drawLine(int(area.left()), int(y), int(area.right()), int(y))
                painter.drawText(QRectF(area.right() - 90, y + 2, 88, 14), Qt.AlignmentFlag.AlignRight,
                                 f"max {self.reference:g}")
            else:
                painter.setPen(QColor(theme_color("muted")))
                painter.drawText(QRectF(area.right() - 160, 12, 160, 16), Qt.AlignmentFlag.AlignRight,
                                 f"max possible {self.reference:g} ↑")

        for _, metrics, color in self.series:
            if not metrics:
                continue
            if self.band and len(metrics) > 1 and all(m.get(self.band) is not None for m in metrics):
                band = QPainterPath()
                band.moveTo(x_of(metrics[0]["step"]), y_of(metrics[0][self.metric] + metrics[0][self.band]))
                for m in metrics[1:]:
                    band.lineTo(x_of(m["step"]), y_of(m[self.metric] + m[self.band]))
                for m in reversed(metrics):
                    band.lineTo(x_of(m["step"]), y_of(max(lo, m[self.metric] - m[self.band])))
                band.closeSubpath()
                fill = QColor(theme_color(color))
                fill.setAlpha(45)
                painter.fillPath(band, fill)
            path = QPainterPath()
            for i, m in enumerate(metrics):
                x, y = x_of(m["step"]), y_of(m[self.metric])
                if i == 0:
                    path.moveTo(x, y)
                else:
                    path.lineTo(x, y)
            painter.setPen(QPen(QColor(theme_color(color)), 2))
            painter.drawPath(path)
            if len(metrics) <= 12:  # sparse series such as evaluations: mark each point
                painter.setBrush(QColor(theme_color(color)))
                for m in metrics:
                    painter.drawEllipse(QRectF(x_of(m["step"]) - 2.5, y_of(m[self.metric]) - 2.5, 5, 5))
                painter.setBrush(Qt.BrushStyle.NoBrush)
        if not points:
            painter.setPen(QColor(theme_color("muted")))
            painter.drawText(area, Qt.AlignmentFlag.AlignCenter, self.empty_text)


def nice_step(raw):
    """Round a raw tick interval up to 1, 2, 2.5 or 5 times a power of ten."""
    if raw <= 0:
        return 1.0
    magnitude = 10 ** math.floor(math.log10(raw))
    return next(m * magnitude for m in (1, 2, 2.5, 5, 10) if m * magnitude >= raw * (1 - 1e-9))


def format_tick(value, step):
    """Tick label with just enough decimals for the tick interval (2.5-steps need one more)."""
    if abs(value) < step / 1e6:
        return "0"
    if abs(value) >= 1000 and step >= 100:
        return f"{value / 1000:g}k"
    exponent = math.floor(math.log10(step))
    mantissa = round(step / 10 ** exponent, 6)
    decimals = max(0, -exponent + (1 if mantissa == 2.5 else 0))
    return f"{value:.{decimals}f}"


class ToggleSlider(QAbstractButton):
    """Instant toggle slider switch with theme-aware pill track and knob."""

    def __init__(self, checked=True, parent=None):
        super().__init__(parent)
        self.setCheckable(True)
        self.setChecked(checked)
        self.setCursor(Qt.CursorShape.PointingHandCursor)
        self.setFixedSize(40, 22)

    def sizeHint(self):
        return QSize(40, 22)

    def paintEvent(self, event):
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        rect = QRectF(0.5, 0.5, self.width() - 1, self.height() - 1)
        radius = rect.height() / 2
        is_on = self.isChecked()

        track_color = QColor(theme_color("primary" if is_on else "raised"))
        border_color = QColor(theme_color("primary" if is_on else "border"))
        if not self.isEnabled():
            track_color.setAlpha(120)
            border_color.setAlpha(120)
        painter.setPen(QPen(border_color, 1))
        painter.setBrush(track_color)
        painter.drawRoundedRect(rect, radius, radius)

        thumb_diameter = rect.height() - 6
        thumb_y = 3.0
        thumb_x = (rect.width() - thumb_diameter - 3.0) if is_on else 3.0
        knob_color = QColor(theme_color("base" if is_on else "muted"))
        if not self.isEnabled():
            knob_color.setAlpha(150)
        painter.setPen(Qt.PenStyle.NoPen)
        painter.setBrush(knob_color)
        painter.drawEllipse(QRectF(thumb_x, thumb_y, thumb_diameter, thumb_diameter))
        painter.end()

