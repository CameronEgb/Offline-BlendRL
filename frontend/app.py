"""Spyder-inspired research workspace. Configs are composed and trained by the NeSyRL backend."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import os
from pathlib import Path
import sys

from PyQt6.QtCore import Qt, QTimer, QRegularExpression
from PyQt6.QtGui import QAction, QFont, QRegularExpressionValidator
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QDockWidget, QTabWidget, QPlainTextEdit, QLineEdit, QComboBox, QSpinBox, QCheckBox,
    QDoubleSpinBox, QPushButton, QTreeWidget, QTreeWidgetItem, QToolBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QProgressBar, QScrollArea, QFileDialog, QMessageBox, QDialog,
)
from .api import DEFAULT_URL, Backend
from .model import (BASE_EXPERIMENT, FINAL_STATUSES, LIVE_STATUSES, Config, Store, available_metrics, example_runs,
                    latest, metric_points, new_run, sample)
from .theme import STYLE, ThemeManager, theme_color
from .theme_builder import ThemeBuilder
from .widgets import Chart, MetricCard, YamlHighlighter, label
from .about import AboutDialog
from .plots import PlotViewer
from .queue_panel import QueuePanel
from .sidetabs import SideTabs
from .tensorboard import TensorBoardPanel


# Metrics the monitor's second chart can show: key -> (card title, chart title, subtitle, value format)
SECOND_METRICS = {
    "entropy": ("POLICY ENTROPY", "Policy entropy", "how random actions are · ln 2 ≈ 0.693 is uniform", ".3f"),
    "approx_kl": ("APPROX. KL", "Approximate KL per update", "size of each policy update", ".5f"),
    "loss": ("TRAINING LOSS", "Total loss", "policy + 0.5 × value − 0.01 × entropy", ".4f"),
    "policy_loss": ("POLICY LOSS", "Policy loss", "PPO clipped surrogate objective", ".4f"),
    "value_loss": ("VALUE LOSS", "Value loss", "value-function error · grows as episodes lengthen", ".3f"),
}
# Highest possible evaluation reward per environment, drawn as the reward chart's ceiling.
ENV_MAX_REWARD = {"cartpole": 500}


class Window(QMainWindow):
    def __init__(self, data_dir, api_url=DEFAULT_URL):
        super().__init__()
        self.setWindowTitle("ThetaIDE — Research workspace")
        self.resize(1480, 940)
        self.setMinimumSize(1080, 720)
        self.setDockNestingEnabled(True)
        self.store = Store(data_dir)
        self.theme_manager = ThemeManager(Path(data_dir) / ".appearance.json", self)
        saved, errors = self.store.load()
        self.runs = saved or example_runs()
        self.active = None
        self.selected = None
        self.timer = QTimer(self)
        self.timer.setInterval(350)
        self.timer.timeout.connect(self.tick)
        self.backend = Backend(api_url, self)
        self.schema = None
        self.schema_pending = False
        self.compose_serial = 0
        self.compose_timer = QTimer(self)
        self.compose_timer.setSingleShot(True)
        self.compose_timer.setInterval(250)
        self.compose_timer.timeout.connect(self.request_compose)
        self.compose_valid = False
        self.poll_timer = QTimer(self)
        self.poll_timer.setInterval(1000)
        self.poll_timer.timeout.connect(self.poll_job)
        self.poll_busy = False
        self.queue_timer = QTimer(self)
        self.queue_timer.setInterval(2000)
        self.queue_timer.timeout.connect(self.poll_queue)
        self.queue_busy = False
        self.issued_ids = set()
        self.queue_running = False
        self.make_toolbar()
        self.make_center()
        self.make_explorer()
        self.make_inspector()
        self.make_console()
        self.make_menus()
        self.theme_status = label("", "muted")
        self.statusBar().addWidget(self.theme_status)
        self.theme_manager.changed.connect(self.theme_changed)
        self.theme_changed()
        self.state_label = label("TRAINING   •   idle  ", "muted")
        self.statusBar().addPermanentWidget(self.state_label)
        self.backend_label = label("BACKEND   ○   connecting…  ", "muted")
        self.statusBar().addPermanentWidget(self.backend_label)
        self.refresh_runs()
        self.update_config()
        if self.runs:
            self.select_run(self.runs[0])
        self.log("ThetaIDE ready. Launch trains on this machine through the backend; "
                 "Run → Start simulated demo needs no backend.")
        self.request_schema(apply_defaults=True)
        self.reconnect_live_run()
        for error in errors:
            self.log(error)
        if self.theme_manager.error:
            self.log(self.theme_manager.error)
        self.resizeDocks([self.explorer_dock, self.inspector_dock], [230, 340], Qt.Orientation.Horizontal)
        self.resizeDocks([self.console_dock], [170], Qt.Orientation.Vertical)
        self.default_layout = self.saveState()

    def button(self, text, callback, primary=False):
        button = QPushButton(text)
        if primary:
            button.setObjectName("primary")
        button.clicked.connect(callback)
        return button

    def make_toolbar(self):
        toolbar = QToolBar("Workspace")
        toolbar.setMovable(False)
        self.addToolBar(toolbar)
        toolbar.addWidget(label("θ  ThetaIDE", "brand"))
        toolbar.addWidget(label("RESEARCH WORKSPACE", "muted"))
        toolbar.addSeparator()
        toolbar.addWidget(self.button("+  New experiment", self.new_experiment))
        self.start_button = self.button("▶  Launch training", self.launch_training, True)
        self.start_button.setToolTip("Train the builder's config on this machine through the backend (F5)")
        self.start_button.setEnabled(False)
        toolbar.addWidget(self.start_button)
        self.queue_button = self.button("＋  Add to queue", self.add_to_queue)
        self.queue_button.setToolTip("Queue the builder's config; queued jobs train one at a time, in order "
                                     "(Ctrl+Shift+Q)")
        self.queue_button.setEnabled(False)
        toolbar.addWidget(self.queue_button)
        self.stop_button = self.button("■  Stop", self.stop_run)
        self.stop_button.setEnabled(False)
        toolbar.addWidget(self.stop_button)
        spacer = QWidget()
        spacer.setObjectName("toolbarSpacer")
        from PyQt6.QtWidgets import QSizePolicy
        spacer.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Preferred)
        toolbar.addWidget(spacer)
        toolbar.addWidget(label("FRONTEND PREVIEW", "badge"))

    def make_center(self):
        self.tabs = SideTabs()
        self.setCentralWidget(self.tabs)
        monitor = QWidget()
        layout = QVBoxLayout(monitor)
        layout.setContentsMargins(20, 18, 20, 14)
        layout.setSpacing(14)
        layout.addWidget(label("EXPERIMENT / OVERVIEW", "eyebrow"))
        self.run_title = label("Your next experiment", "heading")
        layout.addWidget(self.run_title)
        self.run_caption = label("Configure an experiment, then launch training.", "muted")
        layout.addWidget(self.run_caption)
        row = QHBoxLayout()
        self.reward_card = MetricCard("EPISODE REWARD", "synthetic evaluation / mean")
        self.loss_card = MetricCard("POLICY ENTROPY", "")
        self.steps_card = MetricCard("TIMESTEPS", "configured training budget")
        for card in (self.reward_card, self.loss_card, self.steps_card):
            row.addWidget(card)
        layout.addLayout(row)
        self.progress = QProgressBar()
        self.progress.setTextVisible(False)
        self.progress.setFixedHeight(5)
        layout.addWidget(self.progress)
        self.reward_chart = Chart("reward", "Episode reward  ·  mean ± 1 std over evaluation episodes",
                                  band="reward_std")
        self.second_metric = "entropy"
        self.loss_chart = Chart(self.second_metric, SECOND_METRICS[self.second_metric][1], zero_based=False)
        self.metric_select = QComboBox()
        self.metric_select.setToolTip("Metric shown in this chart and the second card")
        self.metric_select.currentIndexChanged.connect(self.second_metric_changed)
        self.loss_chart.set_corner_widget(self.metric_select)
        layout.addWidget(self.reward_chart, 3)
        layout.addWidget(self.loss_chart, 2)
        self.monitor_note = label("", "muted")
        self.monitor_note.setWordWrap(True)
        monitor_footer = QHBoxLayout()
        monitor_footer.addWidget(self.monitor_note, 1)
        monitor_footer.addWidget(self.button("Open TensorBoard  →", self.show_tensorboard))
        layout.addLayout(monitor_footer)
        self.tabs.addTab(monitor, "Training monitor", "monitor", "Monitor")

        results = QWidget()
        results_layout = QVBoxLayout(results)
        results_layout.setContentsMargins(18, 18, 18, 18)
        results_layout.addWidget(label("Experiment history", "heading"))
        results_layout.addWidget(label("Select one run to inspect, or two to compare (Ctrl + click).", "muted"))
        self.search = QLineEdit()
        self.search.setPlaceholderText("Filter by experiment name, seed, or status…")
        self.search.textChanged.connect(self.refresh_runs)
        results_layout.addWidget(self.search)
        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["Experiment", "Seed", "Status", "Reward", "Source"])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.ResizeToContents)
        self.table.verticalHeader().hide()
        self.table.verticalHeader().setDefaultSectionSize(40)
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.ExtendedSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.itemSelectionChanged.connect(self.table_selected)
        self.table.itemDoubleClicked.connect(lambda _: self.tabs.setCurrentIndex(0))
        results_layout.addWidget(self.table, 1)
        actions = QHBoxLayout()
        actions.addWidget(self.button("Load saved config", self.load_selected_config))
        actions.addWidget(self.button("Compare two runs", self.compare))
        actions.addWidget(self.button("View plot", self.view_selected_plot))
        actions.addStretch()
        results_layout.addLayout(actions)
        self.tabs.addTab(results, "Results browser", "results", "Results")

        preview = QWidget()
        preview_layout = QVBoxLayout(preview)
        self.preview_status = label("Resolved config • waiting for backend", "muted")
        preview_layout.addWidget(self.preview_status)
        self.preview_errors = label("", "configError")
        self.preview_errors.setWordWrap(True)
        self.preview_errors.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        self.preview_errors.hide()
        preview_layout.addWidget(self.preview_errors)
        self.preview = QPlainTextEdit()
        self.preview.setReadOnly(True)
        self.highlighter = YamlHighlighter(self.preview.document())
        preview_layout.addWidget(self.preview)
        preview_layout.addWidget(self.button("Export recipe YAML…", self.export_config))
        self.tabs.addTab(preview, "config.yaml", "config", "Config")
        self.plot_viewer = PlotViewer()
        self.tabs.addTab(self.plot_viewer, "Plot viewer", "plots", "Plots")
        self.tensorboard_panel = TensorBoardPanel(self.backend, self.log)
        self.tabs.addTab(self.tensorboard_panel, "TensorBoard", "tensorboard", "TensorBoard")
        self.queue_panel = QueuePanel(self.move_queued, self.remove_queued, self.open_job, self.set_queue_running)
        self.tabs.addTab(self.queue_panel, "Job queue", "queue", "Queue")
        self.tabs.currentChanged.connect(self.tab_changed)

    def dock(self, title, name, widget, area):
        dock = QDockWidget(title, self)
        dock.setObjectName(name)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def make_explorer(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(10, 12, 10, 8)
        layout.addWidget(label("THETA / LOCAL", "eyebrow"))
        self.tree = QTreeWidget()
        self.tree.setHeaderHidden(True)
        self.tree.setIndentation(14)
        self.tree.itemClicked.connect(self.tree_selected)
        layout.addWidget(self.tree)
        layout.addWidget(label("●  Run records stay local", "muted"))
        self.explorer_dock = self.dock("Workspace", "explorer", panel, Qt.DockWidgetArea.LeftDockWidgetArea)

    def make_inspector(self):
        self.inspector_tabs = QTabWidget()
        form_widget = QWidget()
        layout = QVBoxLayout(form_widget)
        layout.setContentsMargins(15, 16, 15, 14)
        layout.setSpacing(12)
        layout.addWidget(label("BUILD AN EXPERIMENT", "eyebrow"))
        layout.addWidget(label("Start with a question.", "heading"))
        description = label("A small, reproducible CartPole experiment.\nTune the parameters and watch it train.", "muted")
        description.setWordWrap(True)
        layout.addWidget(description)
        form = QFormLayout()
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setVerticalSpacing(10)
        self.name = QLineEdit(Config().name)
        self.name.setMaxLength(100)
        self.name.setValidator(QRegularExpressionValidator(QRegularExpression(r"[A-Za-z0-9_\-]+"), self.name))
        form.addRow("Experiment name", self.name)
        self.fixed_fields = {}
        for title, value in (("Task", "Reinforcement learning"), ("Environment", "CartPole-v1"),
                             ("Method", "PPO · neural policy"), ("Training mode", "Online")):
            combo = QComboBox()
            combo.addItem(value)
            combo.setToolTip("This proof of concept supports the CartPole / PPO workflow.")
            form.addRow(title, combo)
            self.fixed_fields[title] = combo
        self.seed = QSpinBox()
        self.seed.setRange(0, 2147483647)
        self.seed.setValue(42)
        form.addRow("Random seed", self.seed)
        self.steps = QSpinBox()
        self.steps.setRange(1000, 10000000)
        self.steps.setSingleStep(1000)
        self.steps.setValue(10000)
        form.addRow("Total timesteps", self.steps)
        self.lr = QDoubleSpinBox()
        self.lr.setDecimals(6)
        self.lr.setRange(0.000001, 1)
        self.lr.setSingleStep(0.0001)
        self.lr.setValue(0.0003)
        form.addRow("Learning rate", self.lr)
        self.batch = QComboBox()
        self.batch.addItems(["32", "64", "128", "256"])
        self.batch.setCurrentText("64")
        form.addRow("Batch size", self.batch)
        self.gamma = QDoubleSpinBox()
        self.gamma.setDecimals(3)
        self.gamma.setRange(0, 1)
        self.gamma.setSingleStep(0.01)
        self.gamma.setValue(0.99)
        form.addRow("Discount factor · γ", self.gamma)
        self.tensorboard = QCheckBox("Log to TensorBoard")
        self.tensorboard.setChecked(True)
        form.addRow("Logging", self.tensorboard)
        layout.addLayout(form)
        self.fields = {"experiment_id": self.name, "seed": self.seed, "total_timesteps": self.steps,
                       "methods.ppo.lr": self.lr, "methods.ppo.batch_size": self.batch, "methods.ppo.gamma": self.gamma,
                       "tensorboard": self.tensorboard}
        self.builder_status = label("Checking config with backend…", "muted")
        self.builder_status.setWordWrap(True)
        layout.addWidget(self.builder_status)
        layout.addWidget(label("Trains locally  /  ≈ 30 s for 20k steps on CPU", "badge"))
        layout.addWidget(self.button("Preview config  →", lambda: self.tabs.setCurrentIndex(2)))
        layout.addStretch()
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setWidget(form_widget)
        self.inspector_tabs.addTab(scroll, "Experiment builder")

        notes = QWidget()
        notes_layout = QVBoxLayout(notes)
        self.note_target = label("No run selected", "muted")
        self.note_target.setWordWrap(True)
        notes_layout.addWidget(self.note_target)
        self.notes = QPlainTextEdit()
        self.notes.setPlaceholderText("## Hypothesis\nWhat do you expect to learn?\n\n## Observations\n\n## Next iteration")
        notes_layout.addWidget(self.notes)
        self.note_status = label("Notes are linked to the selected run.", "muted")
        notes_layout.addWidget(self.note_status)
        notes_layout.addWidget(self.button("Save notes", self.save_notes))
        self.inspector_tabs.addTab(notes, "Notes")
        self.inspector_dock = self.dock("Experiment", "inspector", self.inspector_tabs,
                                        Qt.DockWidgetArea.RightDockWidgetArea)
        self.inspector_dock.setMinimumWidth(300)
        self.name.textChanged.connect(self.update_config)
        self.batch.currentTextChanged.connect(self.update_config)
        for field in (self.seed, self.steps, self.lr, self.gamma):
            field.valueChanged.connect(self.update_config)
        self.tensorboard.toggled.connect(self.update_config)
        self.notes.textChanged.connect(self.notes_changed)

    def make_console(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(0, 0, 0, 0)
        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(1000)
        layout.addWidget(self.console)
        self.command = QLineEdit()
        self.command.setPlaceholderText("Console › help, status, config, clear (not a system shell)")
        self.command.returnPressed.connect(self.console_command)
        layout.addWidget(self.command)
        self.console_dock = self.dock("Console", "console", panel, Qt.DockWidgetArea.BottomDockWidgetArea)

    def make_menus(self):
        file_menu = self.menuBar().addMenu("File")
        for title, shortcut, callback in (("New experiment", "Ctrl+N", self.new_experiment),
                                          ("Export draft YAML…", "Ctrl+Shift+S", self.export_config),
                                          ("Save notes", "Ctrl+S", self.save_notes),
                                          ("Quit", "Ctrl+Q", self.close)):
            action = QAction(title, self)
            action.setShortcut(shortcut)
            action.triggered.connect(callback)
            file_menu.addAction(action)
        run_menu = self.menuBar().addMenu("Run")
        for title, shortcut, callback in (("Launch training", "F5", self.launch_training),
                                          ("Stop", "Shift+F5", self.stop_run),
                                          ("Add to queue", "Ctrl+Shift+Q", self.add_to_queue),
                                          ("Start or pause queue", "Ctrl+Shift+R",
                                           lambda: self.set_queue_running(not self.queue_running)),
                                          ("Start simulated demo", "Ctrl+F5", self.start_demo)):
            action = QAction(title, self)
            action.setShortcut(shortcut)
            action.triggered.connect(callback)
            run_menu.addAction(action)
        view_menu = self.menuBar().addMenu("View")
        self.themes_menu = view_menu.addMenu("Themes")
        self.themes_menu.aboutToShow.connect(self.populate_themes_menu)
        view_menu.addAction("Theme builder…", self.show_theme_builder)
        view_menu.addAction("Plot viewer", lambda: self.tabs.setCurrentWidget(self.plot_viewer))
        view_menu.addAction("TensorBoard", self.show_tensorboard)
        for dock in (self.explorer_dock, self.inspector_dock, self.console_dock):
            view_menu.addAction(dock.toggleViewAction())
        view_menu.addAction("Restore default layout", lambda: self.restoreState(self.default_layout))
        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("About this prototype", self.show_about)

    def show_tensorboard(self):
        self.tabs.setCurrentWidget(self.tensorboard_panel)

    def tab_changed(self, index):
        if self.tabs.widget(index) is self.tensorboard_panel:
            self.tensorboard_panel.activate()
        if self.tabs.widget(index) is self.queue_panel:
            self.poll_queue()
            self.queue_timer.start()

    def show_about(self):
        dialog = AboutDialog(self)
        dialog.exec()
        dialog.deleteLater()

    def theme_changed(self):
        self.theme_status.setText(f"  ●  Local workspace    /    {self.theme_manager.active['name']}")
        self.highlighter.rehighlight()
        self.tabs.refresh_icons()

    def populate_themes_menu(self):
        self.themes_menu.clear()
        for name, theme in self.theme_manager.themes().items():
            action = self.themes_menu.addAction(name)
            action.setCheckable(True)
            action.setChecked(name == self.theme_manager.active["name"])
            action.triggered.connect(lambda _, palette=theme: self.select_theme(palette))
        self.themes_menu.addSeparator()
        self.themes_menu.addAction("Theme builder…", self.show_theme_builder)

    def select_theme(self, theme):
        try:
            self.theme_manager.commit(theme)
        except (OSError, ValueError) as exc:
            QMessageBox.warning(self, "Could not save theme", str(exc))

    def show_theme_builder(self):
        dialog = ThemeBuilder(self.theme_manager, self)
        dialog.exec()
        dialog.deleteLater()

    def config(self):
        return Config(name=self.name.text().strip() or "untitled_experiment", seed=self.seed.value(),
                      total_timesteps=self.steps.value(), lr=self.lr.value(),
                      batch_size=int(self.batch.currentText()), gamma=self.gamma.value(),
                      tensorboard=self.tensorboard.isChecked())

    def update_config(self, *_):
        self.compose_timer.start()

    def request_compose(self):
        self.compose_serial += 1
        serial, config = self.compose_serial, self.config()
        payload = {"experiment": BASE_EXPERIMENT, "overrides": config.overrides()}
        self.backend.post("/api/config/compose", payload,
                          lambda data, error: self.compose_finished(serial, config, data, error))

    def compose_finished(self, serial, config, data, error):
        if serial != self.compose_serial:
            return  # superseded by a newer edit
        self.compose_valid = not error and data["valid"]
        self.update_launch_state()
        if error:
            self.set_backend_state(False, error)
            self.preview_status.setText("Backend offline • local recipe draft, not validated")
            self.preview_errors.hide()
            self.builder_status.setObjectName("muted")
            self.builder_status.setText("Backend offline: this config has not been validated.")
            self.builder_status.setStyle(self.builder_status.style())
            self.preview.setPlainText(config.recipe_yaml())
            return
        self.set_backend_state(True)
        if self.schema is None and not self.schema_pending:
            self.request_schema(apply_defaults=False)
        header = ["# Resolved by the backend exactly as the command line would:",
                  "#   " + " ".join(data["argv"])]
        header += [f"# Notice: {notice}" for notice in data["notices"]]
        sections = ["\n".join(header)]
        if data.get("methods"):
            sections.append("# Methods — the settings each method actually trains with\nmethods:\n"
                            + "".join(f"  {line}\n" for line in data["methods_yaml"].splitlines()).rstrip())
        sections.append("# Full composed config — agent.* holds defaults that the method settings above override\n"
                        + data["config_yaml"])
        self.preview.setPlainText("\n\n".join(sections))
        if data["valid"]:
            self.preview_status.setText("Resolved config • ✓ valid for the training pipeline")
            self.preview_errors.hide()
            self.builder_status.setObjectName("configOk")
            rollouts = [plan["rollout"] for plan in data.get("methods", {}).values() if plan.get("rollout")]
            detail = ""
            if rollouts and rollouts[0]["timesteps"] != config.total_timesteps:
                r = rollouts[0]
                detail = f" · trains {r['timesteps']:,} steps ({r['rollouts']} PPO rollouts × {r['size']})"
            self.builder_status.setText("✓ Config validated by backend" + detail)
        else:
            messages = [f"[{e['stage']}] {e['message']}" for e in data["errors"]]
            self.preview_status.setText(f"Resolved config • ✗ {len(messages)} error(s)")
            self.preview_errors.setText("\n".join(messages))
            self.preview_errors.show()
            self.builder_status.setObjectName("configError")
            self.builder_status.setText("✗ " + messages[0].splitlines()[0])
        self.builder_status.setStyle(self.builder_status.style())

    def set_backend_state(self, connected, error=None):
        if connected:
            self.backend_label.setText(f"BACKEND   ●   {self.backend.base_url}  ")
            self.backend_label.setToolTip("Configs are composed and validated by the NeSyRL API.")
        else:
            self.backend_label.setText("BACKEND   ○   offline  ")
            self.backend_label.setToolTip(f"{self.backend.base_url}: {error}\nStart it from the repository root with:\n"
                                          "uvicorn src.app.api.app:app --host 127.0.0.1 --port 8000")

    def request_schema(self, apply_defaults):
        self.schema_pending = True
        self.backend.get("/api/config/schema", lambda data, error: self.schema_loaded(data, error, apply_defaults))

    def schema_loaded(self, data, error, apply_defaults):
        self.schema_pending = False
        if error:
            self.set_backend_state(False, error)
            self.log(f"Backend not reachable at {self.backend.base_url} ({error}). "
                     "Start it with: uvicorn src.app.api.app:app --host 127.0.0.1 --port 8000")
            return
        self.schema = data
        env = data["environment"]
        method = data["method"]
        for title, text in (("Environment", env.get("env_id") or env["name"]),
                            ("Method", f"{method['agent'].upper()} · {method['model']}"),
                            ("Training mode", f"Online · {data['paradigm']}")):
            self.fixed_fields[title].setItemText(0, text)
        for field in data["fields"]:
            widget = self.fields.get(field["key"])
            if widget is None:
                continue
            widget.setToolTip(f"{field['help']}\nOverride: {field['key']}")
            if "choices" in field:
                current = widget.currentText()
                widget.blockSignals(True)
                widget.clear()
                widget.addItems([str(choice) for choice in field["choices"]])
                widget.setCurrentText(current)
                widget.blockSignals(False)
            elif "min" in field:
                widget.setRange(field["min"], field["max"])
                if "step" in field:
                    widget.setSingleStep(field["step"])
            if apply_defaults and field["key"] != "experiment_id" and field.get("default") is not None:
                if "choices" in field:
                    widget.setCurrentText(str(field["default"]))
                elif field["type"] == "bool":
                    widget.setChecked(bool(field["default"]))
                else:
                    widget.setValue(field["default"])
        self.log(f"Experiment builder loaded from backend: {env['name']} / {method['agent']} ({data['base_experiment']}).")
        self.update_config()

    def set_config(self, config):
        self.name.setText(config.name)
        self.seed.setValue(config.seed)
        self.steps.setValue(config.total_timesteps)
        self.lr.setValue(config.lr)
        self.batch.setCurrentText(str(config.batch_size))
        self.gamma.setValue(config.gamma)
        self.tensorboard.setChecked(config.tensorboard)
        self.update_config()

    def new_experiment(self):
        self.set_config(Config(name="cartpole_ppo_experiment"))
        self.inspector_tabs.setCurrentIndex(0)
        self.inspector_dock.show()
        self.name.setFocus()
        self.name.selectAll()

    def persist(self, run):
        try:
            self.store.save(run)
            return True
        except OSError as exc:
            self.log(f"SAVE FAILED: {exc}")
            self.statusBar().showMessage(f"Could not save run: {exc}", 12000)
            return False

    def update_launch_state(self):
        idle = self.active is None
        ready = idle and self.compose_valid
        self.start_button.setEnabled(ready)
        self.queue_button.setEnabled(self.compose_valid)
        self.stop_button.setEnabled(not idle)
        if not idle:
            tip = "A run is active. Stop it before launching another."
        elif not self.compose_valid:
            tip = "Launch needs the backend running and a config it has validated."
        else:
            tip = "Train the builder's config on this machine through the backend (F5)"
        self.start_button.setToolTip(tip)
        run = self.active
        if run is None:
            self.state_label.setText("TRAINING   •   idle  ")
        elif run["simulated"]:
            self.state_label.setText(f"DEMO   •   {run['status']}  ")
        else:
            self.state_label.setText(f"TRAINING   •   {run['status']}   •   {run['backend']['experiment_id']}  ")

    def activate(self, run):
        self.active = run
        if run not in self.runs:
            self.runs.insert(0, run)
        self.select_run(run)
        self.refresh_runs()
        self.tabs.setCurrentIndex(0)
        self.update_launch_state()

    # ── Live training through the backend ────────────────────────────────────

    def launch_training(self):
        if self.active:
            self.statusBar().showMessage("A run is already active. Stop it before launching another.", 4000)
            return
        if not self.compose_valid:
            self.statusBar().showMessage("Start the backend and fix config errors before launching.", 5000)
            return
        config = self.config()
        # Each launch gets its own experiment_id: the pipeline purges results of a reused ID.
        experiment_id = self.unique_experiment_id(config.name)
        overrides = replace(config, name=experiment_id).overrides()
        self.start_button.setEnabled(False)
        self.log(f"Launching {experiment_id}:\n  python run_pipeline.py {BASE_EXPERIMENT} {' '.join(overrides)}")
        self.backend.post("/api/experiments/launch", {"experiment": BASE_EXPERIMENT, "overrides": overrides},
                          lambda data, error: self.launch_finished(config, data, error))

    def launch_finished(self, config, job, error):
        if error:
            self.log(f"Launch failed: {error}")
            self.statusBar().showMessage("Launch failed; see the console.", 8000)
            self.update_launch_state()
            return
        run = new_run(config, simulated=False)
        run.update(status="starting", log_since=0, metrics_since=0, backend={
            key: job.get(key) for key in ("job_id", "experiment", "group", "experiment_id", "agents", "total_timesteps",
                                          "effective_timesteps")})
        if not self.persist(run):
            return
        self.activate(run)
        self.log(f"Job {job['job_id']} started. Metrics: results/logs/{job['group']}/{job['experiment_id']}/")
        self.poll_timer.start()
        self.poll_job()

    def reconnect_live_run(self):
        """Resume monitoring a run that was still training when the app last closed."""
        run = next((r for r in self.runs if not r["simulated"] and r["status"] in LIVE_STATUSES), None)
        if run:
            self.log(f"Reconnecting to {run['backend']['experiment_id']} (job {run['backend']['job_id']})…")
            self.activate(run)
            self.poll_timer.start()
        if any(r["status"] == "queued" for r in self.runs):
            self.queue_timer.start()

    def poll_job(self):
        run = self.active
        if self.poll_busy or not run or run["simulated"]:
            return
        self.poll_busy = True
        job_id = run["backend"]["job_id"]
        self.backend.get(f"/api/experiments/{job_id}/status?since={run['log_since']}",
                         lambda data, error: self.status_received(run, data, error))

    def status_received(self, run, data, error):
        if run is not self.active:
            self.poll_busy = False
            return
        if error:
            self.poll_busy = False
            if "404" in error:  # the backend restarted and no longer knows this job
                self.log("The backend no longer tracks this job (it was restarted). Metrics on disk are kept.")
                self.finish("interrupted")
            else:
                self.set_backend_state(False, error)
            return
        self.set_backend_state(True)
        for line in data["log"]:
            self.log(f"│ {line}")
            if "Auto-Generating" in line:
                run["phase"] = "plotting"
        run["log_since"] = data["log_total"]
        status = {"pending": "starting"}.get(data["status"], data["status"])
        if run["status"] != status:
            run["status"] = status
            self.update_launch_state()
            self.refresh_runs()
        # Fetch metrics after the status, so a finished job's final rows are always collected.
        self.backend.get(f"/api/experiments/{run['backend']['job_id']}/metrics?since={run['metrics_since']}",
                         lambda metrics, err: self.metrics_received(run, metrics, err, data))

    def metrics_received(self, run, data, error, status):
        self.poll_busy = False
        if run is not self.active:
            return
        if not error:
            agent = data["agents"].get(run["backend"]["agents"][0], {}) if run["backend"]["agents"] else {}
            if agent.get("reset"):
                run["metrics"] = []
            run["metrics"].extend(metric_points(agent.get("rows", [])))
            run["metrics_since"] = agent.get("total", run["metrics_since"])
            if self.selected is run:
                self.render_run()
        if status["status"] in FINAL_STATUSES:
            detail = f" (exit code {status['returncode']})" if status.get("returncode") not in (None, 0) else ""
            self.finish(status["status"], detail)
        elif len(run["metrics"]) and run["status"] == "running":
            self.persist(run)

    # ── Job queue ────────────────────────────────────────────────────────────

    def unique_experiment_id(self, name):
        """<name>_<date>-<time>, with a counter if several jobs are created within the same second."""
        base = f"{name}_{datetime.now():%Y%m%d-%H%M%S}"
        experiment_id, n = base, 2
        while experiment_id in self.issued_ids:
            experiment_id, n = f"{base}-{n}", n + 1
        self.issued_ids.add(experiment_id)
        return experiment_id

    def add_to_queue(self):
        if not self.compose_valid:
            self.statusBar().showMessage("Start the backend and fix config errors before queueing.", 5000)
            return
        config = self.config()
        experiment_id = self.unique_experiment_id(config.name)
        overrides = replace(config, name=experiment_id).overrides()
        self.backend.post("/api/experiments/launch",
                          {"experiment": BASE_EXPERIMENT, "overrides": overrides, "queue": True},
                          lambda data, error: self.queued(config, data, error))

    def queued(self, config, job, error):
        if error:
            self.log(f"Could not add to queue: {error}")
            self.statusBar().showMessage("Could not add to queue; see the console.", 8000)
            return
        run = new_run(config, simulated=False)
        run.update(status="queued", log_since=0, metrics_since=0, queue_position=job.get("position"), backend={
            key: job.get(key) for key in ("job_id", "experiment", "group", "experiment_id", "agents", "total_timesteps",
                                          "effective_timesteps")})
        if not self.persist(run):
            return
        self.runs.insert(0, run)
        self.refresh_runs()
        self.queue_running = bool(job.get("queue_running"))
        place = f"position {job['position'] + 1}"
        hint = ("it runs after the jobs ahead of it" if self.queue_running
                else "the queue is paused; press Start queue in the Queue tab to begin")
        self.log(f"Queued {job['experiment_id']} ({place}): {hint}.")
        self.statusBar().showMessage(f"Added {job['experiment_id']} to the queue ({place}); {hint}.", 8000)
        self.queue_timer.start()
        self.poll_queue()

    def poll_queue(self):
        if self.queue_busy:
            return
        self.queue_busy = True
        self.backend.get("/api/queue", self.queue_received)

    def queue_received(self, data, error):
        self.queue_busy = False
        if error:
            self.queue_panel.set_offline(error)
            return
        if self.queue_running != data["running"]:
            self.queue_running = data["running"]
            self.render_run()
        self.queue_panel.set_data(data)
        where = {job["job_id"]: ("active", job) for job in data["active"]}
        where.update({job["job_id"]: ("queued", job) for job in data["queued"]})
        where.update({job["job_id"]: ("finished", job) for job in data["finished"]})
        changed = False
        for run in [r for r in self.runs if r["status"] == "queued"]:
            kind, job = where.get(run["backend"]["job_id"], (None, None))
            if kind == "queued":
                changed |= run.get("queue_position") != job["position"]
                run["queue_position"] = job["position"]
            elif kind == "active" and self.active is None:
                run["status"] = "starting"
                self.log(f"Queue: {run['backend']['experiment_id']} started.")
                self.persist(run)
                self.activate(run)
                self.poll_timer.start()
                self.poll_job()
            elif kind == "finished":  # removed from the queue, or finished while another run was followed
                run["status"] = job["status"]
                self.persist(run)
                changed = True
            elif kind is None:  # not in the listing (it keeps only recent finished jobs): ask about this job
                job_id = run["backend"]["job_id"]
                self.backend.get(f"/api/experiments/{job_id}/status",
                                 lambda status, err, run=run: self.queued_job_status(run, status, err))
        if changed:
            self.refresh_runs()
            self.render_run()
        # Keep watching while jobs wait or the queue runs, so its self-pause after draining is seen too
        waiting = any(r["status"] == "queued" for r in self.runs)
        if not waiting and not self.queue_running and self.tabs.currentWidget() is not self.queue_panel:
            self.queue_timer.stop()

    def queued_job_status(self, run, status, error):
        if run["status"] != "queued":
            return
        if error and "404" in error:
            run["status"] = "interrupted"  # the backend restarted and lost its queue
            self.log(f"The backend no longer has queued job {run['backend']['experiment_id']} (it was restarted).")
        elif not error and status["status"] in FINAL_STATUSES:
            run["status"] = status["status"]
        else:
            return
        self.persist(run)
        self.refresh_runs()
        self.render_run()

    def set_queue_running(self, running):
        action = "start" if running else "pause"
        self.backend.post(f"/api/queue/{action}", {}, lambda data, error: self.queue_toggled(action, data, error))

    def queue_toggled(self, action, data, error):
        if error:
            self.log(f"Could not {action} the queue: {error}")
            return
        if action == "start" and not data["running"]:
            self.statusBar().showMessage("The queue is empty; add experiments first.", 5000)
        else:
            self.log("Queue started: jobs will train one after another." if data["running"] else
                     "Queue paused: no new job will start; one already training keeps running.")
        self.queue_timer.start()
        self.poll_queue()

    def move_queued(self, job_id, position):
        self.backend.post(f"/api/queue/{job_id}/move", {"position": position},
                          lambda data, error: self.log(f"Move failed: {error}") if error else self.poll_queue())

    def remove_queued(self, job_id):
        self.backend.post(f"/api/experiments/{job_id}/cancel", {},
                          lambda data, error: self.log(f"Remove failed: {error}") if error else self.poll_queue())

    def open_job(self, job_id):
        run = next((r for r in self.runs if not r["simulated"] and r["backend"]["job_id"] == job_id), None)
        if run is None:
            self.statusBar().showMessage("That job was queued from another client; it has no record here.", 5000)
            return
        self.select_run(run)
        self.tabs.setCurrentIndex(0)

    # ── Simulated demo (no backend needed) ───────────────────────────────────

    def start_demo(self):
        if self.active:
            self.statusBar().showMessage("A run is already active. Stop it before starting another.", 4000)
            return
        run = new_run(self.config())
        if not self.persist(run):
            return
        self.activate(run)
        self.log(f"[demo] Started {run['config']['name']} · seed {run['config']['seed']} · {run['id']}")
        self.timer.start()

    def tick(self):
        if not self.active or not self.active["simulated"]:
            return
        run = self.active
        metric = sample(Config(**run["config"]), len(run["metrics"]) + 1)
        run["metrics"].append(metric)
        self.plot_viewer.update_runs(self.runs)
        if self.selected is run:
            self.render_run()
        if len(run["metrics"]) % 10 == 0:
            self.log(f"[demo] step={metric['step']:>6}  reward={metric['reward']:.2f}  loss={metric['loss']:.4f}")
            self.persist(run)
        if len(run["metrics"]) >= 60:
            self.finish("completed")

    # ── Shared lifecycle ─────────────────────────────────────────────────────

    def stop_run(self):
        run = self.active
        if not run:
            return
        if run["simulated"]:
            self.finish("stopped")
            return
        self.stop_button.setEnabled(False)
        self.log(f"Stopping {run['backend']['experiment_id']}: terminating the pipeline and its training processes…")
        self.backend.post(f"/api/experiments/{run['backend']['job_id']}/cancel", {},
                          lambda data, error: self.log(f"Stop failed: {error}") if error else None)
        # The next status poll reports "cancelled" and finishes the run.

    def finish(self, status, detail=""):
        self.timer.stop()
        self.poll_timer.stop()
        run = self.active
        run["status"] = status
        self.persist(run)
        prefix = "[demo] " if run["simulated"] else ""
        self.log(f"{prefix}{status.capitalize()}{detail} · {run['id']} · config and metrics saved locally")
        if status == "failed":
            self.statusBar().showMessage("Training failed; the console shows the pipeline output.", 10000)
        self.active = None
        self.update_launch_state()
        self.refresh_runs()
        self.render_run()
        if any(r["status"] == "queued" for r in self.runs):
            self.poll_queue()  # follow the next queued job as soon as it starts

    def select_run(self, run):
        self.selected = run
        self.notes.blockSignals(True)
        self.notes.setPlainText(run["notes"])
        self.notes.blockSignals(False)
        self.note_target.setText(f"Linked to {run['config']['name']}\nRun {run['id']}")
        self.note_status.setText("Notes save automatically as you type.")
        self.render_run()
        self.backfill_metrics(run)

    def sync_metric_select(self, run):
        """Offer the second-chart metrics this run has; keep the user's choice when available."""
        available = [key for key in SECOND_METRICS if key in available_metrics(run)] or ["loss"]
        chosen = self.second_metric if self.second_metric in available else available[0]
        self.metric_select.blockSignals(True)
        self.metric_select.clear()
        for key in available:
            self.metric_select.addItem(SECOND_METRICS[key][1], key)
        self.metric_select.setCurrentIndex(available.index(chosen))
        self.metric_select.blockSignals(False)
        self.loss_chart.position_corner()
        return chosen

    def second_metric_changed(self, index):
        key = self.metric_select.itemData(index)
        if key:
            self.second_metric = key
            self.render_run()

    def backfill_metrics(self, run):
        """Fetch the full metrics of a finished trained run recorded before all metrics were kept."""
        if run["simulated"] or run["status"] not in FINAL_STATUSES or run.get("metrics_backfilled"):
            return
        if not run["metrics"] or any("entropy" in m for m in run["metrics"]):
            return
        backend = run["backend"]
        agent = (backend.get("agents") or ["ppo"])[0]
        path = f"/api/runs/{backend['group']}/{backend['experiment_id']}/{agent}/metrics"
        self.backend.get(path, lambda data, error: self.backfill_received(run, data, error))

    def backfill_received(self, run, data, error):
        run["metrics_backfilled"] = True  # try once; the results folder may have been deleted
        if error:
            return
        rows = []
        for row in data["metrics"]:
            numeric = {}
            for key, value in row.items():
                try:
                    numeric[key] = float(value)
                except (TypeError, ValueError):
                    pass
            rows.append(numeric)
        points = metric_points(rows)
        if points:
            run["metrics"] = points
            self.persist(run)
            if self.selected is run:
                self.render_run()

    def render_run(self):
        run = self.selected
        if not run:
            return
        config = run["config"]
        metrics = run["metrics"]
        live = not run["simulated"]
        title = run["backend"]["experiment_id"] if live else config["name"]
        source = f"LIVE · job {run['backend']['job_id'][:8]}" if live else "SIMULATED"
        self.run_title.setText(title)
        self.run_caption.setText(f"CartPole-v1  /  PPO  /  seed {config['seed']}   •   {run['status'].upper()}   •   {source}")
        reward = latest(run, "reward")
        second = self.sync_metric_select(run)
        card_title, chart_title, subtitle, fmt = SECOND_METRICS[second]
        second_value = latest(run, second)
        step = max((m["step"] for m in metrics), default=0)
        requested = config["total_timesteps"]
        # PPO trains whole rollouts, so the real budget can exceed the requested one (10,000 -> 10,240).
        # Records from before the backend reported it fall back to the steps actually run.
        budget = (run["backend"].get("effective_timesteps") or max(step, requested)) if live else requested
        self.reward_card.value.setText("—" if reward is None else f"{reward:.1f}")
        self.loss_card.title.setText(card_title)
        self.loss_card.value.setText("—" if second_value is None else format(second_value, fmt))
        self.loss_card.subtitle.setText(subtitle if live else "synthetic training loss")
        self.steps_card.value.setText(f"{step:,}")
        self.progress.setValue(min(100, round(100 * step / max(1, budget))))
        if live:
            self.reward_card.subtitle.setText("evaluation / mean episode reward")
            rounding = f"\n{requested:,} requested, rounded up to whole PPO rollouts" if budget != requested else ""
            self.steps_card.subtitle.setText(f"of {budget:,} environment steps{rounding}")
            where = f"results/logs/{run['backend']['group']}/{run['backend']['experiment_id']}/"
            if run["status"] == "queued":
                position = run.get("queue_position")
                ahead = f"position {position + 1}" if position is not None else "waiting"
                note = (f"In the job queue ({ahead}). Queued experiments train one at a time; "
                        "this view follows it when it starts.")
                if not self.queue_running:
                    note += " The queue is paused: press Start queue in the Queue tab."
            elif run["status"] == "starting" or (run["status"] == "running" and not metrics):
                note = "Starting the pipeline — the first evaluation usually appears after about 15 seconds."
            elif run["status"] == "running" and run.get("phase") == "plotting":
                note = f"Training finished — the pipeline is generating plots in results/plots/{run['backend']['group']}/…"
            elif run["status"] in FINAL_STATUSES:
                note = f"Final metrics from {where}"
                if run["status"] == "completed":
                    note += f" · plots in results/plots/{run['backend']['group']}/{run['backend']['experiment_id']}/"
            else:
                note = f"Live metrics from {where} · updated every second while training."
            self.monitor_note.setText(f"●  {note}")
        else:
            self.reward_card.subtitle.setText("synthetic evaluation / mean")
            self.steps_card.subtitle.setText("configured training budget")
            self.monitor_note.setText("●  Selected run     •     Synthetic metrics / frontend demonstration")
        xmax = budget if live else None
        has_spread = any(m.get("reward_std") is not None for m in metrics)
        self.reward_chart.set_metric("reward", "Episode reward  ·  mean ± 1 std over evaluation episodes"
                                     if has_spread else "Episode reward")
        self.reward_chart.set_series([(title, metrics, "#b8bb26")], xmax, ENV_MAX_REWARD.get(config["env"]))
        self.loss_chart.set_metric(second, chart_title)
        self.loss_chart.set_series([(title, metrics, "#83a598")], xmax)

    def refresh_runs(self, *_):
        self.plot_viewer.update_runs(self.runs)
        query = self.search.text().lower()
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        self.tree.clear()
        project = QTreeWidgetItem(self.tree, ["▾  theta-workspace"])
        draft = QTreeWidgetItem(project, ["◇  config.yaml"])
        draft.setData(0, Qt.ItemDataRole.UserRole, "draft")
        folder = QTreeWidgetItem(project, [f"▾  Experiments ({len(self.runs)})"])
        for run in self.runs:
            config = run["config"]
            name = config["name"] if run["simulated"] else run["backend"]["experiment_id"]
            item = QTreeWidgetItem(folder, [name])
            item.setData(0, Qt.ItemDataRole.UserRole, run["id"])
            item.setToolTip(0, f"{run['status']} · seed {config['seed']} · "
                               + ("synthetic data" if run["simulated"] else "trained by the backend"))
            if query not in f"{name} {config['seed']} {run['status']}".lower():
                continue
            row = self.table.rowCount()
            self.table.insertRow(row)
            last_reward = latest(run, "reward")
            reward = "—" if last_reward is None else f"{last_reward:.1f}"
            source = "Simulated" if run["simulated"] else "Trained"
            for col, value in enumerate((name, str(config["seed"]), run["status"], reward, source)):
                cell = QTableWidgetItem(value)
                cell.setData(Qt.ItemDataRole.UserRole, run["id"])
                self.table.setItem(row, col, cell)
        catalog = QTreeWidgetItem(project, ["◇  Backend reference"])
        for title in ("PPO / dnn", "CartPole-v1", "Hydra config groups"):
            QTreeWidgetItem(catalog, [title]).setToolTip(0, "Reference only; backend is not loaded")
        self.tree.expandAll()
        self.table.blockSignals(False)

    def by_id(self, run_id):
        return next((run for run in self.runs if run["id"] == run_id), None)

    def view_selected_plot(self):
        if self.selected:
            self.plot_viewer.show_run(self.selected["id"])
        self.tabs.setCurrentWidget(self.plot_viewer)

    def tree_selected(self, item, _):
        run_id = item.data(0, Qt.ItemDataRole.UserRole)
        if run_id == "draft":
            self.tabs.setCurrentIndex(2)
        elif run := self.by_id(run_id):
            self.select_run(run)
            self.tabs.setCurrentIndex(0)

    def table_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if len(rows) == 1:
            run = self.by_id(self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole))
            self.select_run(run)

    def notes_changed(self):
        if self.selected:
            self.selected["notes"] = self.notes.toPlainText()
            ok = self.persist(self.selected)
            self.note_status.setText("Saved locally · linked to this run" if ok else "Save failed · see console")

    def save_notes(self):
        self.notes_changed()

    def load_selected_config(self):
        rows = self.table.selectionModel().selectedRows()
        if len(rows) != 1:
            self.statusBar().showMessage("Select exactly one run to load its saved config.", 5000)
            return
        run = self.by_id(self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole))
        self.set_config(Config(**run["config"]))
        self.inspector_tabs.setCurrentIndex(0)
        self.inspector_dock.show()
        self.tabs.setCurrentIndex(2)
        self.log(f"Loaded exact configuration from {run['id']}. Launch to create a new run.")

    def compare(self):
        rows = self.table.selectionModel().selectedRows()
        if len(rows) != 2:
            self.statusBar().showMessage("Select exactly two runs with Ctrl + click to compare.", 5000)
            return
        runs = [self.by_id(self.table.item(index.row(), 0).data(Qt.ItemDataRole.UserRole)) for index in rows]
        dialog = QDialog(self)
        dialog.setWindowTitle("Compare experiments")
        dialog.resize(820, 660)
        layout = QVBoxLayout(dialog)
        layout.addWidget(label("Two runs. One question.", "heading"))
        chart = Chart("reward", "Episode reward", band="reward_std")
        chart.set_series([(run["config"]["name"], run["metrics"], color)
                          for run, color in zip(runs, ("#b8bb26", "#83a598"))],
                         reference=ENV_MAX_REWARD.get(runs[0]["config"]["env"]))
        layout.addWidget(chart, 1)
        for run, color in zip(runs, ("#b8bb26", "#83a598")):
            legend = label(f"●  {run['config']['name']} · seed {run['config']['seed']} · {run['id']}")
            legend.setStyleSheet(f"color: {theme_color(color)}")
            layout.addWidget(legend)
        diff = QTableWidget(len(asdict(Config())), 3)
        diff.setHorizontalHeaderLabels(["Parameter", "Run A", "Run B"])
        diff.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        diff.verticalHeader().hide()
        diff.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        configs = [asdict(Config(**run["config"])) for run in runs]  # older records lack newer fields
        for row, key in enumerate(asdict(Config())):
            for col, value in enumerate((key, configs[0][key], configs[1][key])):
                cell = QTableWidgetItem(str(value))
                if configs[0][key] != configs[1][key]:
                    from PyQt6.QtGui import QColor
                    cell.setForeground(QColor(theme_color("accent")))
                diff.setItem(row, col, cell)
        layout.addWidget(diff, 1)
        synthetic = [run["config"]["name"] for run in runs if run["simulated"]]
        note = f" Synthetic curves: {', '.join(synthetic)}." if synthetic else ""
        layout.addWidget(label("Changed parameters are highlighted." + note, "muted"))
        dialog.exec()

    def export_config(self):
        config = self.config()
        path, _ = QFileDialog.getSaveFileName(self, "Export experiment recipe", f"{config.name}.yaml", "YAML (*.yaml)")
        if path:
            try:
                Path(path).write_text(config.recipe_yaml(), encoding="utf-8")
                self.log(f"Exported recipe to {path}. Place it in in/config/experiment/thetaide/ and run:\n"
                         f"  python run_pipeline.py thetaide/{Path(path).stem}")
            except OSError as exc:
                QMessageBox.warning(self, "Export failed", str(exc))

    def log(self, message):
        self.console.appendPlainText(message)

    def console_command(self):
        command = self.command.text().strip()
        self.command.clear()
        self.log(f"theta › {command}")
        if command == "clear":
            self.console.clear()
        elif command == "help":
            self.log("help    Show commands\nstatus  Show the active run\nconfig  Show the equivalent run_pipeline command\nclear   Clear output\nThis console does not execute Python or shell commands.")
        elif command == "status":
            run = self.active
            if run is None:
                state = "idle"
            elif run["simulated"]:
                state = f"simulated demo {run['status']}"
            else:
                state = f"{run['status']} · {run['backend']['experiment_id']} · job {run['backend']['job_id']}"
            self.log(f"Training: {state} · {len(self.runs)} records")
        elif command == "config":
            self.log(self.config().command())
        elif command:
            self.log("Unknown demo command. Type help for available commands.")

    def closeEvent(self, event):
        if self.active and self.active["simulated"]:
            self.stop_run()
        elif self.active:
            self.persist(self.active)  # training keeps running in the backend; reopening reconnects
        self.save_notes()
        event.accept()


def main():
    parser = argparse.ArgumentParser(description="ThetaIDE PyQt frontend proof of concept")
    parser.add_argument("--data-dir", type=Path, default=Path(__file__).resolve().parent.parent / ".thetaide" / "runs")
    parser.add_argument("--api-url", default=os.environ.get("THETAIDE_API_URL", DEFAULT_URL),
                        help="NeSyRL backend API (default: %(default)s)")
    args = parser.parse_args()
    app = QApplication(sys.argv[:1])
    app.setStyle("Fusion")
    app.setFont(QFont("Segoe UI", 10))
    app.setStyleSheet(STYLE)
    try:
        window = Window(args.data_dir, args.api_url)
    except OSError as exc:
        QMessageBox.critical(None, "Workspace unavailable", f"Could not open local run storage:\n{exc}")
        return 1
    window.show()
    return app.exec()
