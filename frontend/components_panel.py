"""Dedicated Components panel for managing modular configurations (agent, env, model, paradigms, site, hydra, etc.)."""
from pathlib import Path

import yaml
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QFrame,
    QHBoxLayout,
    QMessageBox,
    QPlainTextEdit,
    QPushButton,
    QSplitter,
    QToolButton,
    QVBoxLayout,
    QWidget,
)

from .config_tree import ConfigTreeWidget
from .config_viewer import ConfigViewer
from .widgets import YamlHighlighter, label


class ComponentsPanel(QWidget):
    """Panel in Theta-IDE dedicated to modular configuration components outside experiments:
    agent profiles, environment definitions, model architectures, paradigms, site profiles,
    and global configs.
    """
    component_saved = pyqtSignal(str)  # rel_path

    def __init__(self, log_fn=None, parent=None):
        super().__init__(parent)
        self.log_fn = log_fn or (lambda msg: None)
        self.active_path = None
        self.active_rel_path = None
        self._init_ui()
        self.init_default_component()

    def _init_ui(self):
        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(18, 14, 18, 14)
        root_layout.setSpacing(10)

        # ── Actions bar ────────────────────────────────────────────────────────
        actions_bar = QHBoxLayout()
        actions_bar.setSpacing(8)

        self.btn_new = QPushButton("+  New component…")
        self.btn_new.setToolTip("Create a new modular component config (agent, env, model, etc.)")
        self.btn_new.clicked.connect(self.new_component)
        actions_bar.addWidget(self.btn_new)

        self.btn_duplicate = QPushButton("📑  Duplicate…")
        self.btn_duplicate.setToolTip("Duplicate the currently selected component configuration")
        self.btn_duplicate.clicked.connect(self.duplicate_component)
        actions_bar.addWidget(self.btn_duplicate)

        self.btn_save = QPushButton("💾  Save YAML")
        self.btn_save.setToolTip("Save changes to disk (Ctrl+S)")
        self.btn_save.clicked.connect(self.save_current)
        actions_bar.addWidget(self.btn_save)

        self.btn_reload = QPushButton("↺  Reload")
        self.btn_reload.setToolTip("Reload component files and tree from disk")
        self.btn_reload.clicked.connect(self.reload_components)
        actions_bar.addWidget(self.btn_reload)

        actions_bar.addStretch()

        self.btn_toggle_raw = QPushButton("{ }  View Raw YAML")
        self.btn_toggle_raw.setCheckable(True)
        self.btn_toggle_raw.setChecked(False)
        self.btn_toggle_raw.setToolTip("Toggle side-by-side view of the raw YAML file content")
        self.btn_toggle_raw.clicked.connect(lambda: self.toggle_raw_preview())
        actions_bar.addWidget(self.btn_toggle_raw)

        self.btn_hub = QPushButton("🌐 Component Hub…")
        self.btn_hub.setToolTip("Browse and install new RL methods, models, and environments from the Community Hub")
        self.btn_hub.clicked.connect(self._open_hub)
        actions_bar.addWidget(self.btn_hub)

        root_layout.addLayout(actions_bar)

        # ── Horizontal Splitter ───────────────────────────────────────────────
        self.splitter = QSplitter(Qt.Orientation.Horizontal)

        # 1. Components Tree
        self.components_tree = ConfigTreeWidget(mode="components")
        self.components_tree.setMinimumWidth(220)
        self.components_tree.file_selected.connect(self.on_component_selected)
        self.splitter.addWidget(self.components_tree)

        # 2. Boxed Config Viewer
        self.viewer = ConfigViewer()
        self.viewer.config_changed.connect(self._on_viewer_changed)
        self.viewer.save_requested.connect(self._on_viewer_saved)
        self.splitter.addWidget(self.viewer)

        # 3. Raw YAML Preview Panel (hidden by default)
        raw_panel = QWidget()
        raw_layout = QVBoxLayout(raw_panel)
        raw_layout.setContentsMargins(0, 0, 0, 0)
        raw_layout.setSpacing(6)

        raw_header = QHBoxLayout()
        raw_header.setSpacing(8)
        raw_header.addWidget(label("RAW YAML CONFIG", "eyebrow"))
        raw_header.addStretch()

        btn_close_raw = QToolButton()
        btn_close_raw.setText("✕")
        btn_close_raw.setToolTip("Close raw YAML preview")
        btn_close_raw.clicked.connect(lambda: self.toggle_raw_preview(False))
        raw_header.addWidget(btn_close_raw)
        raw_layout.addLayout(raw_header)

        self.raw_status = label("Source file view", "muted")
        raw_layout.addWidget(self.raw_status)

        self.raw_edit = QPlainTextEdit()
        self.raw_edit.setReadOnly(True)
        self.raw_highlighter = YamlHighlighter(self.raw_edit.document())
        raw_layout.addWidget(self.raw_edit, 1)

        self.raw_panel = raw_panel
        self.raw_panel.hide()
        self.splitter.addWidget(raw_panel)

        self.splitter.setSizes([260, 1000, 0])
        root_layout.addWidget(self.splitter, 1)

    def init_default_component(self):
        """Select a default component config if available."""
        preferred_defaults = [
            "agent/ppo.yaml",
            "env/cartpole.yaml",
            "model/dueling_resnet.yaml",
            "config.yaml",
        ]
        for rel in preferred_defaults:
            target = self.components_tree.root_dir / rel
            if target.exists() and target.is_file():
                if self.components_tree.select_file(rel):
                    return

    def on_component_selected(self, file_path, rel_path):
        """Handle component file selection from the tree."""
        self.active_path = Path(file_path)
        self.active_rel_path = rel_path
        self.viewer.load_file(self.active_path, rel_path)
        self._update_raw_yaml_view()

    def _update_raw_yaml_view(self):
        """Refresh raw YAML editor content."""
        if not self.active_path or not self.active_path.exists():
            self.raw_edit.setPlainText("")
            self.raw_status.setText("No component loaded")
            return
        try:
            content = self.active_path.read_text(encoding="utf-8")
            self.raw_edit.setPlainText(content)
            self.raw_status.setText(f"{self.active_rel_path} • {len(content.splitlines())} lines")
        except Exception as exc:
            self.raw_edit.setPlainText(f"# Error reading file:\n# {exc}")
            self.raw_status.setText(f"Error: {exc}")

    def _on_viewer_changed(self):
        """When viewer data changes, update the raw view if visible."""
        if self.raw_panel.isVisible():
            try:
                yaml_str = yaml.safe_dump(self.viewer.raw_data, sort_keys=False)
                self.raw_edit.setPlainText(yaml_str)
                self.raw_status.setText(f"{self.active_rel_path} • modified (unsaved)")
            except Exception:
                pass

    def _on_viewer_saved(self):
        """Fired when viewer saves to disk."""
        self._update_raw_yaml_view()
        if self.active_rel_path:
            self.log_fn(f"Saved component configuration: {self.active_rel_path}")
            self.component_saved.emit(self.active_rel_path)

    def save_current(self):
        """Save changes to the currently loaded component config."""
        success = self.viewer.save_to_disk()
        if success:
            self._update_raw_yaml_view()
            if self.active_rel_path:
                self.log_fn(f"Saved component configuration: {self.active_rel_path}")
                self.component_saved.emit(self.active_rel_path)

    def new_component(self):
        """Delegate to tree's new component creation dialog."""
        self.components_tree.prompt_new_component()

    def duplicate_component(self):
        """Delegate to tree's duplicate dialog."""
        self.components_tree.prompt_duplicate()

    def reload_components(self, target_rel_path: str | None = None, ensure_expanded: str | None = None):
        """Reload the tree and active file from disk without collapsing folders."""
        curr = target_rel_path or self.active_rel_path
        if not ensure_expanded and curr:
            p = Path(curr).parent
            if str(p) and str(p) != ".":
                ensure_expanded = str(p).replace("\\", "/")

        self.components_tree.populate(ensure_expanded=ensure_expanded)
        if curr and (self.components_tree.root_dir / curr).exists():
            self.components_tree.select_file(curr)
        else:
            # If the current active file no longer exists (e.g. was uninstalled),
            # try to select a sibling file in the same folder first so we don't jump away.
            selected_alt = False
            if curr:
                parent_dir = self.components_tree.root_dir / Path(curr).parent
                if parent_dir.exists() and parent_dir.is_dir():
                    siblings = sorted([p for p in parent_dir.iterdir() if p.is_file() and p.suffix in (".yaml", ".yml")])
                    if siblings:
                        sibling_rel = str(siblings[0].relative_to(self.components_tree.root_dir)).replace("\\", "/")
                        if self.components_tree.select_file(sibling_rel):
                            selected_alt = True
            if not selected_alt:
                self.init_default_component()
        self.log_fn("Reloaded component files from disk.")

    def toggle_raw_preview(self, checked=None):
        """Show or hide the raw YAML preview panel."""
        if checked is None:
            checked = self.btn_toggle_raw.isChecked()
        else:
            self.btn_toggle_raw.setChecked(checked)
        self.raw_panel.setVisible(checked)
        if checked:
            self.splitter.setSizes([240, 600, 420])
            self._update_raw_yaml_view()
        else:
            self.splitter.setSizes([240, 1000, 0])

    def _open_hub(self):
        """Open the Community Hub filtered to RL methods and models."""
        w = self.window()
        if hasattr(w, "open_hub"):
            w.open_hub(initial_kind="method")

