"""Transactional theme editing: preview freely, save explicitly, cancel to revert."""
import copy
import json
import re
from pathlib import Path

from PyQt6.QtCore import QTimer, Qt
from PyQt6.QtGui import QColor
from PyQt6.QtWidgets import (
    QDialog, QVBoxLayout, QHBoxLayout, QFormLayout, QComboBox, QLineEdit,
    QPushButton, QColorDialog, QFileDialog, QMessageBox, QScrollArea, QWidget,
)
from .theme import BUILTINS, ROLES, validate_theme, write_json
from .widgets import label


def contrast_ratio(a, b):
    def luminance(color):
        rgb = [int(color[i:i + 2], 16) / 255 for i in (1, 3, 5)]
        rgb = [v / 12.92 if v <= .04045 else ((v + .055) / 1.055) ** 2.4 for v in rgb]
        return sum(v * weight for v, weight in zip(rgb, (.2126, .7152, .0722)))
    x, y = sorted((luminance(a), luminance(b)))
    return (y + .05) / (x + .05)


class ThemeBuilder(QDialog):
    def __init__(self, manager, parent=None):
        super().__init__(parent)
        self.manager = manager
        self.original = copy.deepcopy(manager.active)
        self.loading = False
        self.setWindowTitle("Themes & theme builder")
        self.resize(640, 750)
        self.preview_timer = QTimer(self)
        self.preview_timer.setSingleShot(True)
        self.preview_timer.setInterval(150)
        self.preview_timer.timeout.connect(self.preview)
        layout = QVBoxLayout(self)
        layout.setContentsMargins(22, 20, 22, 20)
        layout.setSpacing(12)
        layout.addWidget(label("Make the workspace yours.", "heading"))
        hint = label("Choose a preset or build your own palette. Valid edits preview throughout the app. Cancel restores your previous theme.", "muted")
        hint.setWordWrap(True)
        layout.addWidget(hint)
        presets = QHBoxLayout()
        presets.addWidget(label("Start from", "muted"))
        self.presets = QComboBox()
        self.presets.addItems(list(manager.themes()))
        self.presets.setCurrentText(manager.active["name"])
        presets.addWidget(self.presets, 1)
        load_button = QPushButton("Reset to preset")
        load_button.clicked.connect(self.load_preset)
        presets.addWidget(load_button)
        layout.addLayout(presets)
        self.name = QLineEdit()
        self.name.setMaxLength(60)
        name_row = QFormLayout()
        name_row.addRow("Theme name", self.name)
        layout.addLayout(name_row)
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        fields = QWidget()
        form = QFormLayout(fields)
        form.setVerticalSpacing(9)
        self.fields, self.swatches = {}, {}
        for role, (title, _) in ROLES.items():
            row = QHBoxLayout()
            edit = QLineEdit()
            edit.setMaxLength(7)
            edit.setAccessibleName(title + " hex color")
            edit.setPlaceholderText("#RRGGBB")
            swatch = label("", "swatch")
            choose = QPushButton("Choose…")
            choose.setAccessibleName("Choose " + title.lower())
            choose.clicked.connect(lambda _, key=role: self.choose_color(key))
            edit.textChanged.connect(lambda _, key=role: self.color_edited(key))
            row.addWidget(swatch)
            row.addWidget(edit, 1)
            row.addWidget(choose)
            form.addRow(title, row)
            self.fields[role], self.swatches[role] = edit, swatch
        scroll.setWidget(fields)
        layout.addWidget(scroll, 1)
        self.feedback = label("", "muted")
        self.feedback.setWordWrap(True)
        layout.addWidget(self.feedback)
        buttons = QHBoxLayout()
        for title, callback in (("Import JSON…", self.import_file), ("Export JSON…", self.export_file)):
            button = QPushButton(title)
            button.clicked.connect(callback)
            buttons.addWidget(button)
        buttons.addStretch()
        cancel = QPushButton("Cancel")
        cancel.clicked.connect(self.reject)
        buttons.addWidget(cancel)
        self.save_button = QPushButton("Save && apply")
        self.save_button.setObjectName("primary")
        self.save_button.clicked.connect(self.save)
        buttons.addWidget(self.save_button)
        layout.addLayout(buttons)
        self.presets.currentTextChanged.connect(self.load_preset)
        self.name.textChanged.connect(self.schedule_preview)
        self.load_theme(self.original)

    def load_theme(self, theme):
        self.loading = True
        self.name.setText(theme["name"])
        for role, color in theme["colors"].items():
            self.fields[role].setText(color)
        self.loading = False
        self.preview()

    def load_preset(self, *_):
        self.load_theme(self.manager.themes()[self.presets.currentText()])

    def draft(self):
        return validate_theme({"name": self.name.text(), "colors": {key: edit.text() for key, edit in self.fields.items()}})

    def color_edited(self, role):
        if self.loading:
            return
        if self.name.text() in BUILTINS:
            self.name.setText(self.name.text() + " custom")
        self.schedule_preview()

    def schedule_preview(self, *_):
        if not self.loading:
            self.preview_timer.start()

    def preview(self):
        self.preview_timer.stop()
        try:
            theme = self.draft()
        except ValueError as exc:
            self.feedback.setText(str(exc) + " Preview keeps the last valid palette.")
            self.save_button.setEnabled(False)
            return
        self.save_button.setEnabled(True)
        for role, color in theme["colors"].items():
            self.swatches[role].setStyleSheet(f"background: {color};")
        colors = theme["colors"]
        contrast = contrast_ratio(colors["text"], colors["panel"])
        warning = " · Low text contrast; consider adjusting these colors." if contrast < 4.5 else ""
        existing = theme["name"] in self.manager.custom
        self.feedback.setText(f"Live preview · Text/panel contrast {contrast:.1f}:1{warning}" +
                              ("\nSaving updates the existing custom theme with this name." if existing else ""))
        self.manager.apply(theme)

    def choose_color(self, role):
        initial = self.fields[role].text()
        if not re.fullmatch(r"#[0-9a-fA-F]{6}", initial):
            initial = self.manager.active["colors"][role]
        color = QColorDialog.getColor(QColor(initial), self, ROLES[role][0],
                                      QColorDialog.ColorDialogOption.DontUseNativeDialog)
        if color.isValid():
            self.fields[role].setText(color.name())

    def save(self):
        self.preview_timer.stop()
        try:
            self.manager.commit(self.draft())
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Theme not saved", str(exc))
            return
        self.accept()

    def reject(self):
        self.preview_timer.stop()
        self.manager.apply(self.original)
        super().reject()

    def import_path(self, path):
        data = json.loads(Path(path).read_text(encoding="utf-8"))
        if not isinstance(data, dict) or data.get("version") != 1:
            raise ValueError("Expected a ThetaIDE theme JSON file with version 1.")
        theme = validate_theme(data.get("theme"))
        if theme["name"].casefold() in {name.casefold() for name in BUILTINS}:
            theme["name"] += " imported"
        self.load_theme(theme)

    def import_file(self):
        path, _ = QFileDialog.getOpenFileName(self, "Import theme", "", "ThetaIDE theme (*.json)")
        if path:
            try:
                self.import_path(path)
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Could not import theme", str(exc))

    def export_path(self, path):
        write_json(path, {"version": 1, "theme": self.draft()})

    def export_file(self):
        path, _ = QFileDialog.getSaveFileName(self, "Export theme", "thetaide-theme.json", "ThetaIDE theme (*.json)")
        if path:
            try:
                self.export_path(path)
            except (OSError, ValueError) as exc:
                QMessageBox.warning(self, "Could not export theme", str(exc))
