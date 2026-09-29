"""Tree widget mirroring in/config/ directory for Theta-IDE."""
from pathlib import Path
import yaml
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QTreeWidget, QTreeWidgetItem,
    QLineEdit, QPushButton, QLabel, QInputDialog, QMessageBox, QMenu, QToolButton
)
from .widgets import label


def find_config_root():
    """Locate in/config directory from cwd or parent directories."""
    candidates = [
        Path.cwd() / "in" / "config",
        Path.cwd() / "in" / "configs",
        Path(__file__).resolve().parent.parent / "in" / "config",
        Path(__file__).resolve().parent.parent / "in" / "configs",
    ]
    for p in candidates:
        if p.exists() and p.is_dir():
            return p.resolve()
    return (Path(__file__).resolve().parent.parent / "in" / "config").resolve()


FOLDER_ICONS = {
    "experiment": "🧪",
    "agent": "🤖",
    "env": "🌐",
    "model": "🧠",
    "paradigms": "⚙️",
    "site": "🖥️",
    "hydra": "⚡",
}


class ConfigTreeWidget(QWidget):
    """File tree that mirrors in/config/ and allows selecting, duplicating,

    and creating experiments in groups.
    """
    file_selected = pyqtSignal(object, str)  # (Path, rel_path)
    duplicate_requested = pyqtSignal(str)     # (rel_path)
    new_in_group_requested = pyqtSignal()

    def __init__(self, root_dir=None, parent=None):
        super().__init__(parent)
        self.root_dir = Path(root_dir) if root_dir else find_config_root()
        self.current_rel_path = None
        self._init_ui()
        self.populate()

    def _init_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(6)

        # Header with title and actions
        header = QHBoxLayout()
        header.setSpacing(4)
        title = label("CONFIG REPOSITORY", "eyebrow")
        header.addWidget(title)
        header.addStretch()

        self.btn_refresh = QToolButton()
        self.btn_refresh.setText("↺")
        self.btn_refresh.setToolTip("Reload configuration files from disk")
        self.btn_refresh.clicked.connect(self.populate)
        header.addWidget(self.btn_refresh)

        self.btn_new_group = QToolButton()
        self.btn_new_group.setText("+ New")
        self.btn_new_group.setToolTip("Create a new default experiment in a group")
        self.btn_new_group.clicked.connect(self.prompt_new_in_group)
        header.addWidget(self.btn_new_group)

        self.btn_duplicate = QToolButton()
        self.btn_duplicate.setText("📑 Copy")
        self.btn_duplicate.setToolTip("Duplicate currently selected experiment")
        self.btn_duplicate.clicked.connect(self.prompt_duplicate)
        header.addWidget(self.btn_duplicate)

        layout.addLayout(header)

        # Search filter
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter configs…")
        self.search.textChanged.connect(self.filter_tree)
        layout.addWidget(self.search)

        # Tree widget
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.itemClicked.connect(self._on_item_clicked)
        self.tree.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.tree.customContextMenuRequested.connect(self._show_context_menu)
        layout.addWidget(self.tree, 1)

        # Status footer
        self.footer = label(f"●  in/config/ ({self.root_dir.name})", "muted")
        layout.addWidget(self.footer)

    def populate(self):
        """Recursively scan root_dir and populate the tree."""
        self.tree.clear()
        if not self.root_dir.exists():
            item = QTreeWidgetItem(self.tree, ["in/config (not found)"])
            item.setToolTip(0, f"Directory not found: {self.root_dir}")
            return

        # Top-level directory ordering
        preferred_order = ["experiment", "agent", "env", "model", "paradigms", "site", "hydra"]
        existing_dirs = {p.name: p for p in self.root_dir.iterdir() if p.is_dir() and not p.name.startswith(".")}
        top_files = sorted([p for p in self.root_dir.iterdir() if p.is_file() and p.suffix in (".yaml", ".yml")])

        # 1. Preferred top-level directories
        for cat in preferred_order:
            if cat in existing_dirs:
                self._add_dir_node(self.tree, existing_dirs[cat], expand=(cat == "experiment"))

        # 2. Any other directories not in preferred_order
        for name, dir_path in sorted(existing_dirs.items()):
            if name not in preferred_order:
                self._add_dir_node(self.tree, dir_path, expand=False)

        # 3. Top-level files
        for f in top_files:
            self._add_file_node(self.tree, f)

        # Re-apply current selection if possible
        if self.current_rel_path:
            self.select_file(self.current_rel_path)

    def _add_dir_node(self, parent_widget, dir_path: Path, expand=False):
        name = dir_path.name
        icon = FOLDER_ICONS.get(name, "📁")
        node = QTreeWidgetItem(parent_widget, [f"{icon}  {name}"])
        rel_path = str(dir_path.relative_to(self.root_dir))
        node.setData(0, Qt.ItemDataRole.UserRole, {
            "type": "dir",
            "path": str(dir_path),
            "rel_path": rel_path,
        })
        node.setToolTip(0, rel_path)

        # Subdirectories first, then files
        subdirs = sorted([p for p in dir_path.iterdir() if p.is_dir() and not p.name.startswith(".")])
        files = sorted([p for p in dir_path.iterdir() if p.is_file() and p.suffix in (".yaml", ".yml")])

        for sub in subdirs:
            # Under experiment, expand group folders like cartpole, mimic
            sub_expand = (name == "experiment")
            self._add_dir_node(node, sub, expand=sub_expand)

        for f in files:
            self._add_file_node(node, f)

        if expand:
            node.setExpanded(True)

        return node

    def _add_file_node(self, parent_node, file_path: Path):
        name = file_path.name
        rel_path = str(file_path.relative_to(self.root_dir))
        is_exp = rel_path.startswith("experiment/") and not name.startswith("_")

        node = QTreeWidgetItem(parent_node, [f"📄  {name}"])
        node.setData(0, Qt.ItemDataRole.UserRole, {
            "type": "file",
            "path": str(file_path),
            "rel_path": rel_path,
            "is_experiment": is_exp,
        })
        node.setToolTip(0, rel_path)
        return node

    def _on_item_clicked(self, item, column):
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return
        if data["type"] == "file":
            self.current_rel_path = data["rel_path"]
            self.file_selected.emit(Path(data["path"]), data["rel_path"])

    def filter_tree(self, text):
        query = text.strip().lower()

        def match_and_filter(item):
            data = item.data(0, Qt.ItemDataRole.UserRole)
            item_text = item.text(0).lower()
            rel_path = (data.get("rel_path") or "").lower() if data else ""
            self_match = (query in item_text) or (query in rel_path)

            child_matched = False
            for i in range(item.childCount()):
                if match_and_filter(item.child(i)):
                    child_matched = True

            visible = self_match or child_matched or (not query)
            item.setHidden(not visible)
            if query and (self_match or child_matched):
                item.setExpanded(True)
            return visible

        for i in range(self.tree.topLevelItemCount()):
            match_and_filter(self.tree.topLevelItem(i))

    def select_file(self, target_rel_path: str):
        """Find and select an item by its relative path."""
        target_norm = str(Path(target_rel_path)).replace("\\", "/")

        def find_item(parent):
            count = parent.topLevelItemCount() if isinstance(parent, QTreeWidget) else parent.childCount()
            for i in range(count):
                item = parent.topLevelItem(i) if isinstance(parent, QTreeWidget) else parent.child(i)
                data = item.data(0, Qt.ItemDataRole.UserRole)
                if data and data.get("type") == "file":
                    item_rel = str(Path(data.get("rel_path", ""))).replace("\\", "/")
                    if item_rel == target_norm or item_rel.endswith(target_norm):
                        return item
                found = find_item(item)
                if found:
                    return found
            return None

        found = find_item(self.tree)
        if found:
            self.tree.setCurrentItem(found)
            # Ensure all parent items are expanded
            curr = found.parent()
            while curr:
                curr.setExpanded(True)
                curr = curr.parent()
            data = found.data(0, Qt.ItemDataRole.UserRole)
            self.current_rel_path = data["rel_path"]
            self.file_selected.emit(Path(data["path"]), data["rel_path"])
            return True
        return False

    def get_available_groups(self):
        """Return sorted list of experiment groups in in/config/experiment/."""
        exp_dir = self.root_dir / "experiment"
        if not exp_dir.exists():
            return []
        groups = [p.name for p in exp_dir.iterdir() if p.is_dir() and not p.name.startswith(".")]
        return sorted(groups)

    def prompt_new_in_group(self):
        """Prompt to create a new default experiment in an existing or new group."""
        groups = self.get_available_groups()
        if not groups:
            groups = ["cartpole", "mimic", "thetaide"]

        group, ok = QInputDialog.getItem(
            self, "New Experiment in Group", "Select or type experiment group:",
            groups, 0, True
        )
        if not ok or not group.strip():
            return

        group = group.strip()
        name, ok2 = QInputDialog.getText(
            self, "New Experiment Name", f"Experiment name in group '{group}':",
            QLineEdit.EchoMode.Normal, "new_experiment"
        )
        if not ok2 or not name.strip():
            return

        name = name.strip()
        if not name.endswith(".yaml"):
            name_file = f"{name}.yaml"
            exp_id = name
        else:
            name_file = name
            exp_id = name[:-5]

        target_dir = self.root_dir / "experiment" / group
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / name_file

        if target_file.exists():
            QMessageBox.warning(self, "File Exists", f"Experiment file already exists:\n{target_file}")
            return

        # Default experiment template
        content = (
            "# @package _global_\n"
            "defaults:\n"
            f"- {group}/_base\n\n"
            f"experiment_id: {exp_id}\n"
            "seed: 42\n"
            "total_timesteps: 10000\n"
            "intervals_count: 4\n"
            "eval_episodes: 100\n\n"
            "# Custom overrides for this experiment\n"
            "tensorboard: true\n"
        )
        try:
            target_file.write_text(content, encoding="utf-8")
            self.populate()
            rel_path = f"experiment/{group}/{name_file}"
            self.select_file(rel_path)
        except OSError as exc:
            QMessageBox.critical(self, "Error Creating Experiment", f"Could not create file:\n{exc}")

    def prompt_duplicate(self):
        """Prompt to duplicate the currently selected experiment."""
        if not self.current_rel_path:
            QMessageBox.information(self, "No File Selected", "Please select an experiment YAML file first.")
            return

        source_file = self.root_dir / self.current_rel_path
        if not source_file.exists() or not source_file.is_file():
            QMessageBox.warning(self, "Invalid Selection", "Selected item is not a valid file.")
            return

        stem = source_file.stem
        parent_dir = source_file.parent
        new_name, ok = QInputDialog.getText(
            self, "Duplicate Experiment", f"Duplicate '{stem}' as:",
            QLineEdit.EchoMode.Normal, f"{stem}_copy"
        )
        if not ok or not new_name.strip():
            return

        new_name = new_name.strip()
        new_filename = f"{new_name}.yaml" if not new_name.endswith(".yaml") else new_name
        target_file = parent_dir / new_filename

        if target_file.exists():
            QMessageBox.warning(self, "File Exists", f"Target file already exists:\n{target_file}")
            return

        try:
            raw_text = source_file.read_text(encoding="utf-8")
            # If experiment_id is declared, update it
            try:
                parsed = yaml.safe_load(raw_text)
                if isinstance(parsed, dict) and "experiment_id" in parsed:
                    parsed["experiment_id"] = new_name.replace(".yaml", "")
                    new_text = yaml.safe_dump(parsed, sort_keys=False)
                else:
                    new_text = raw_text
            except Exception:
                new_text = raw_text

            target_file.write_text(new_text, encoding="utf-8")
            self.populate()
            new_rel = str(target_file.relative_to(self.root_dir))
            self.select_file(new_rel)
        except OSError as exc:
            QMessageBox.critical(self, "Duplicate Error", f"Could not duplicate file:\n{exc}")

    def _show_context_menu(self, position):
        item = self.tree.itemAt(position)
        if not item:
            return
        data = item.data(0, Qt.ItemDataRole.UserRole)
        if not data:
            return

        menu = QMenu(self)
        if data["type"] == "file":
            action_select = menu.addAction("Load in Config Viewer")
            action_select.triggered.connect(lambda: self._on_item_clicked(item, 0))
            action_dup = menu.addAction("Duplicate Experiment…")
            action_dup.triggered.connect(self.prompt_duplicate)
        elif data["type"] == "dir":
            if data["rel_path"].startswith("experiment"):
                action_new = menu.addAction("New Experiment in this group…")
                group_name = Path(data["rel_path"]).name
                action_new.triggered.connect(lambda: self._create_in_specific_group(group_name))

        menu.exec(self.tree.viewport().mapToGlobal(position))

    def _create_in_specific_group(self, group_name):
        name, ok = QInputDialog.getText(
            self, "New Experiment", f"New experiment name in '{group_name}':",
            QLineEdit.EchoMode.Normal, "new_experiment"
        )
        if not ok or not name.strip():
            return
        name = name.strip()
        filename = f"{name}.yaml" if not name.endswith(".yaml") else name
        exp_id = name.replace(".yaml", "")

        target_dir = self.root_dir / "experiment" / group_name
        target_dir.mkdir(parents=True, exist_ok=True)
        target_file = target_dir / filename
        if target_file.exists():
            QMessageBox.warning(self, "File Exists", f"Experiment file already exists:\n{target_file}")
            return

        content = (
            "# @package _global_\n"
            "defaults:\n"
            f"- {group_name}/_base\n\n"
            f"experiment_id: {exp_id}\n"
            "seed: 42\n"
            "total_timesteps: 10000\n"
            "intervals_count: 4\n"
            "eval_episodes: 100\n"
            "tensorboard: true\n"
        )
        try:
            target_file.write_text(content, encoding="utf-8")
            self.populate()
            self.select_file(f"experiment/{group_name}/{filename}")
        except OSError as exc:
            QMessageBox.critical(self, "Error", str(exc))
