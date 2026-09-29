"""Spyder-inspired research workspace. Configs are composed and trained by the NeSyRL backend."""
import argparse
from dataclasses import asdict, replace
from datetime import datetime
import json
import os
from pathlib import Path
import sys

from PyQt6.QtCore import Qt, QTimer, QRegularExpression, QUrl
from PyQt6.QtGui import QAction, QFont, QRegularExpressionValidator, QDesktopServices
from PyQt6.QtWidgets import (
    QApplication, QMainWindow, QWidget, QVBoxLayout, QHBoxLayout, QFormLayout,
    QDockWidget, QTabWidget, QPlainTextEdit, QLineEdit, QComboBox, QSpinBox, QCheckBox,
    QDoubleSpinBox, QPushButton, QTreeWidget, QTreeWidgetItem, QToolBar,
    QTableWidget, QTableWidgetItem, QHeaderView, QAbstractItemView,
    QProgressBar, QScrollArea, QFileDialog, QMessageBox, QDialog, QSplitter,
    QFrame, QToolButton,
)
from .api import DEFAULT_URL, Backend
from .model import (BASE_EXPERIMENT, FINAL_STATUSES, LIVE_STATUSES, Config, Store, available_metrics, example_runs,
                    latest, metric_points, new_run, sample)
from .theme import STYLE, ThemeManager, theme_color
from .theme_builder import ThemeBuilder
from .widgets import Chart, MetricCard, ToggleSlider, YamlHighlighter, label
from .about import AboutDialog, AsciiTheta
from .config_tree import ConfigTreeWidget
from .config_viewer import ConfigViewer
from .plots import PlotViewer
from .queue_panel import QueuePanel
from .sidetabs import SideTabs
from .tensorboard import TensorBoardPanel
from .terminal import TerminalPanel


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
        self.layout_file = Path(data_dir) / ".layout.json"
        self.pane_sliders = {}
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
        self.active_config_path = None
        self.active_config_rel_path = None
        self.current_experiment = None
        self.fixed_fields = {}
        self.fields = {}
        self.make_center()
        self.init_default_experiment()
        self.tabs.tabOrderChanged.connect(lambda _: self.save_layout())
        self.load_layout()
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
        self.default_layout = self.saveState()

    def button(self, text, callback, primary=False):
        button = QPushButton(text)
        if primary:
            button.setObjectName("primary")
        button.clicked.connect(callback)
        return button

    def make_center(self):
        self.tabs = SideTabs()
        self.tabs.logoClicked.connect(self.show_about)
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
        self.monitor_panel = monitor
        self.tabs.addTab(self.monitor_panel, "Training monitor", "monitor", "Monitor", tab_id="monitor")

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
        self.table.itemDoubleClicked.connect(lambda _: self.tabs.setCurrentWidget(self.monitor_panel))
        results_layout.addWidget(self.table, 1)
        actions = QHBoxLayout()
        actions.addWidget(self.button("Load saved config", self.load_selected_config))
        actions.addWidget(self.button("Compare two runs", self.compare))
        actions.addWidget(self.button("View plot", self.view_selected_plot))
        actions.addStretch()
        results_layout.addLayout(actions)
        self.results_panel = results
        self.tabs.addTab(self.results_panel, "Results browser", "results", "Results", tab_id="results")

        config_panel = QWidget()
        config_layout = QVBoxLayout(config_panel)
        config_layout.setContentsMargins(18, 14, 18, 14)
        config_layout.setSpacing(10)

        actions_bar = QHBoxLayout()
        actions_bar.setSpacing(8)
        actions_bar.addWidget(self.button("+  New in group…", self.new_experiment))
        actions_bar.addWidget(self.button("📑  Duplicate…", self.duplicate_experiment))
        self.start_button = self.button("▶  Launch training", self.launch_training, True)
        self.start_button.setToolTip("Train the loaded experiment config through the backend (F5)")
        self.start_button.setEnabled(False)
        actions_bar.addWidget(self.start_button)
        self.queue_button = self.button("＋  Add to queue", self.add_to_queue)
        self.queue_button.setToolTip("Queue the loaded config; queued jobs train one at a time, in order (Ctrl+Shift+Q)")
        self.queue_button.setEnabled(False)
        actions_bar.addWidget(self.queue_button)
        self.stop_button = self.button("■  Stop", self.stop_run)
        self.stop_button.setEnabled(False)
        actions_bar.addWidget(self.stop_button)
        actions_bar.addWidget(self.button("💾  Save YAML", self.save_current_config))
        actions_bar.addWidget(self.button("Export recipe YAML…", self.export_config))
        actions_bar.addStretch()

        self.btn_toggle_yaml = QPushButton("{ }  View Hydra YAML")
        self.btn_toggle_yaml.setCheckable(True)
        self.btn_toggle_yaml.setChecked(False)
        self.btn_toggle_yaml.setToolTip("Toggle preview of the resolved Hydra YAML configuration")
        self.btn_toggle_yaml.clicked.connect(lambda: self.toggle_yaml_preview())
        actions_bar.addWidget(self.btn_toggle_yaml)

        config_layout.addLayout(actions_bar)

        splitter = QSplitter(Qt.Orientation.Horizontal)
        self.config_splitter = splitter

        # 1. Config Tree (mirrors in/config/)
        self.config_tree = ConfigTreeWidget()
        self.config_tree.setMinimumWidth(220)
        self.tree = self.config_tree.tree  # backwards compatibility alias
        self.config_tree.file_selected.connect(self.on_config_file_selected)
        splitter.addWidget(self.config_tree)

        # 2. Boxed Config Viewer (takes up the main area of the screen)
        self.config_viewer = ConfigViewer()
        self.config_viewer.config_changed.connect(self.update_config)
        self.config_viewer.save_requested.connect(self.on_config_saved)
        splitter.addWidget(self.config_viewer)

        # 3. Preview Panel (Hydra YAML - hidden by default!)
        preview_panel = QWidget()
        preview_layout = QVBoxLayout(preview_panel)
        preview_layout.setContentsMargins(0, 0, 0, 0)
        preview_layout.setSpacing(6)

        prev_header = QHBoxLayout()
        prev_header.setSpacing(8)
        prev_header.addWidget(label("RESOLVED HYDRA YAML", "eyebrow"))
        prev_header.addStretch()
        btn_close_prev = QToolButton()
        btn_close_prev.setText("✕")
        btn_close_prev.setToolTip("Close preview and expand config viewer")
        btn_close_prev.clicked.connect(lambda: self.toggle_yaml_preview(False))
        prev_header.addWidget(btn_close_prev)
        preview_layout.addLayout(prev_header)

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
        preview_layout.addWidget(self.preview, 1)

        self.builder_status = label("", "muted")
        self.builder_status.setWordWrap(True)
        preview_layout.addWidget(self.builder_status)

        self.preview_panel = preview_panel
        self.preview_panel.hide()  # Hidden by default!
        splitter.addWidget(preview_panel)

        splitter.setSizes([260, 1000, 0])
        config_layout.addWidget(splitter, 1)
        self.config_panel = config_panel
        self.tabs.addTab(self.config_panel, "Experiment", "config", "Experiment", tab_id="config")
        self.plot_viewer = PlotViewer()
        self.tabs.addTab(self.plot_viewer, "Plot viewer", "plots", "Plots", tab_id="plots")
        self.tensorboard_panel = TensorBoardPanel(self.backend, self.log)
        self.tabs.addTab(self.tensorboard_panel, "TensorBoard", "tensorboard", "TensorBoard", tab_id="tensorboard")
        self.queue_panel = QueuePanel(self.move_queued, self.remove_queued, self.open_job, self.set_queue_running)
        self.tabs.addTab(self.queue_panel, "Job queue", "queue", "Queue", tab_id="queue")
        self.terminal_panel = TerminalPanel(cwd=str(Path.cwd()), parent=self)
        self.tabs.addTab(self.terminal_panel, "Terminal", "terminal", "Terminal", tab_id="terminal")
        self.console_panel = self.make_console()
        self.tabs.addTab(self.console_panel, "Console", "console", "Console", tab_id="console")
        self.settings_panel = self.make_settings()
        self.tabs.set_settings_widget(self.settings_panel)
        self.tabs.currentChanged.connect(self.tab_changed)

    def dock(self, title, name, widget, area):
        dock = QDockWidget(title, self)
        dock.setObjectName(name)
        dock.setWidget(widget)
        self.addDockWidget(area, dock)
        return dock

    def init_default_experiment(self):
        default_exp = "experiment/cartpole/quick_test.yaml"
        if not (self.config_tree.root_dir / default_exp).exists():
            default_exp = "experiment/thetaide/cartpole_ppo_reference.yaml"
        self.config_tree.select_file(default_exp)

    def toggle_yaml_preview(self, checked=None):
        if checked is None:
            checked = self.btn_toggle_yaml.isChecked()
        else:
            self.btn_toggle_yaml.setChecked(checked)
        self.preview_panel.setVisible(checked)
        if checked:
            self.config_splitter.setSizes([240, 600, 420])
            self.request_compose()
        else:
            self.config_splitter.setSizes([240, 1000, 0])

    def on_config_file_selected(self, file_path, rel_path):
        self.active_config_path = file_path
        self.active_config_rel_path = rel_path
        self.config_viewer.load_file(file_path, rel_path)

        norm_rel = str(Path(rel_path)).replace("\\", "/")
        if norm_rel.startswith("experiment/") and not Path(rel_path).name.startswith("_"):
            exp_name = str(Path(norm_rel).relative_to("experiment").with_suffix(""))
            self.current_experiment = exp_name
            self.start_button.setEnabled(self.compose_valid)
            self.queue_button.setEnabled(self.compose_valid)
            self.start_button.setToolTip(f"Train {exp_name} through the backend (F5)")
        else:
            self.current_experiment = None
            self.start_button.setEnabled(False)
            self.queue_button.setEnabled(False)
            self.start_button.setToolTip("Select an experiment from experiment/ to launch training")

        self.request_compose()

    def duplicate_experiment(self):
        self.config_tree.prompt_duplicate()

    def save_current_config(self):
        if hasattr(self, "config_viewer"):
            ok = self.config_viewer.save_to_disk()
            if ok:
                self.statusBar().showMessage(f"Saved {self.config_viewer.current_rel_path}", 4000)

    def on_config_saved(self):
        self.statusBar().showMessage(f"Saved {self.config_viewer.current_rel_path}", 4000)
        self.request_compose()

    def make_console(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(18, 16, 18, 14)
        layout.setSpacing(10)

        header = QHBoxLayout()
        header.setSpacing(8)
        header.addWidget(label("CONSOLE / SYSTEM OUTPUT", "eyebrow"))
        header.addStretch()
        for cmd_name, desc in (
            ("status", "Show active training and run summary"),
            ("config", "Show pipeline config command"),
            ("help", "List available console commands"),
        ):
            btn = QPushButton(f"› {cmd_name}")
            btn.setToolTip(desc)
            btn.clicked.connect(lambda _, c=cmd_name: self.run_quick_command(c))
            header.addWidget(btn)
        header.addWidget(self.button("Clear output", self.console_clear))
        layout.addLayout(header)

        self.console = QPlainTextEdit()
        self.console.setReadOnly(True)
        self.console.setMaximumBlockCount(3000)
        layout.addWidget(self.console, 1)

        self.command = QLineEdit()
        self.command.setPlaceholderText("Console › help, status, config, clear (see Terminal tab for system shell)")
        self.command.returnPressed.connect(self.console_command)
        layout.addWidget(self.command)
        return panel

    def console_clear(self):
        if hasattr(self, "console"):
            self.console.clear()

    def make_settings(self):
        panel = QWidget()
        layout = QVBoxLayout(panel)
        layout.setContentsMargins(20, 18, 20, 18)
        layout.setSpacing(14)

        layout.addWidget(label("WORKSPACE & PREFERENCES", "eyebrow"))
        layout.addWidget(label("Settings & About", "heading"))
        layout.addWidget(label("Configure visual themes, backend connectivity, workspace storage, and quick commands.", "muted"))

        splitter = QSplitter(Qt.Orientation.Horizontal)
        splitter.setChildrenCollapsible(False)

        # Left Column: Settings Cards
        left_container = QWidget()
        left_layout = QVBoxLayout(left_container)
        left_layout.setContentsMargins(0, 0, 10, 0)
        left_layout.setSpacing(14)

        # Card 1: Appearance & Theme
        theme_card = QFrame()
        theme_card.setObjectName("card")
        tc_layout = QVBoxLayout(theme_card)
        tc_layout.setContentsMargins(14, 12, 14, 12)
        tc_layout.setSpacing(10)
        tc_layout.addWidget(label("APPEARANCE", "eyebrow"))
        tc_layout.addWidget(label("Theme & Palette", "heading"))
        tc_layout.addWidget(label("Select a color palette or customize individual UI roles.", "muted"))
        theme_row = QHBoxLayout()
        theme_row.addWidget(label("Active theme:", "muted"))
        self.settings_theme_select = QComboBox()
        for name in self.theme_manager.themes():
            self.settings_theme_select.addItem(name)
        self.settings_theme_select.setCurrentText(self.theme_manager.active["name"])
        self.settings_theme_select.currentTextChanged.connect(self.settings_theme_selected)
        theme_row.addWidget(self.settings_theme_select, 1)
        btn_builder = QPushButton("Customize palette…")
        btn_builder.clicked.connect(self.show_theme_builder)
        theme_row.addWidget(btn_builder)
        tc_layout.addLayout(theme_row)
        left_layout.addWidget(theme_card)

        # Card 1.5: Sidebar Navigation & Panes
        sidebar_card = QFrame()
        sidebar_card.setObjectName("card")
        sb_layout = QVBoxLayout(sidebar_card)
        sb_layout.setContentsMargins(14, 12, 14, 12)
        sb_layout.setSpacing(10)
        sb_layout.addWidget(label("SIDEBAR & NAVIGATION", "eyebrow"))
        sb_layout.addWidget(label("Panels & Visibility", "heading"))
        sb_layout.addWidget(label("Toggle which panels appear in the sidebar. Drag icons on the left activity bar to reorder.", "muted"))

        panes_grid = QVBoxLayout()
        panes_grid.setSpacing(8)

        pane_metadata = [
            ("monitor", "Training monitor", "Overview charts & live training curves"),
            ("results", "Results browser", "Experiment runs, comparison, & metrics"),
            ("config", "Experiment builder", "Hydra configurations & hyperparameter tuner"),
            ("plots", "Plot viewer", "Saved figure plots & multi-seed comparisons"),
            ("tensorboard", "TensorBoard", "Interactive TensorBoard event visualizer"),
            ("queue", "Job queue", "Local sequential run scheduler & manager"),
            ("terminal", "Terminal", "Embedded terminal shell"),
            ("console", "Console", "Live Theta IDE system log & command line"),
        ]

        self.pane_sliders = {}
        for pid, name, desc in pane_metadata:
            row = QHBoxLayout()
            info_layout = QVBoxLayout()
            info_layout.setSpacing(1)
            title_lbl = label(name)
            title_lbl.setStyleSheet("font-weight: 600;")
            desc_lbl = label(desc, "muted")
            info_layout.addWidget(title_lbl)
            info_layout.addWidget(desc_lbl)
            row.addLayout(info_layout, 1)

            slider = ToggleSlider(checked=self.tabs.is_tab_visible(pid))
            slider.setToolTip(f"Show or hide {name} in the sidebar")
            slider.setAccessibleName(f"Toggle {name} visibility in sidebar")
            slider.toggled.connect(lambda chk, p=pid: self.on_pane_slider_toggled(p, chk))
            self.pane_sliders[pid] = slider
            row.addWidget(slider)
            panes_grid.addLayout(row)

        sb_layout.addLayout(panes_grid)

        sb_btn_row = QHBoxLayout()
        btn_reset_sidebar = QPushButton("Restore default sidebar")
        btn_reset_sidebar.setToolTip("Show all panels and restore original sidebar order")
        btn_reset_sidebar.clicked.connect(self.reset_sidebar_layout)
        sb_btn_row.addWidget(btn_reset_sidebar)
        sb_btn_row.addStretch()
        sb_layout.addLayout(sb_btn_row)

        left_layout.addWidget(sidebar_card)

        # Card 2: Backend API Connection
        backend_card = QFrame()
        backend_card.setObjectName("card")
        bc_layout = QVBoxLayout(backend_card)
        bc_layout.setContentsMargins(14, 12, 14, 12)
        bc_layout.setSpacing(10)
        bc_layout.addWidget(label("BACKEND SERVICES", "eyebrow"))
        bc_layout.addWidget(label("NeSyRL API & Training Engine", "heading"))
        bc_layout.addWidget(label("Connects to the FastAPI backend managing training runs and pipelines.", "muted"))
        url_row = QHBoxLayout()
        url_row.addWidget(label("API URL:", "muted"))
        self.settings_backend_url = QLineEdit(self.backend.base_url)
        self.settings_backend_url.setReadOnly(True)
        url_row.addWidget(self.settings_backend_url, 1)
        btn_test = QPushButton("Test connection")
        btn_test.clicked.connect(self.test_backend_connection)
        url_row.addWidget(btn_test)
        btn_docs = QPushButton("Swagger docs ↗")
        btn_docs.clicked.connect(self.open_swagger_docs)
        url_row.addWidget(btn_docs)
        bc_layout.addLayout(url_row)
        self.settings_backend_status = label("Status: Checking connection…", "muted")
        bc_layout.addWidget(self.settings_backend_status)
        left_layout.addWidget(backend_card)

        # Card 3: Storage & Workspace
        storage_card = QFrame()
        storage_card.setObjectName("card")
        sc_layout = QVBoxLayout(storage_card)
        sc_layout.setContentsMargins(14, 12, 14, 12)
        sc_layout.setSpacing(10)
        sc_layout.addWidget(label("LOCAL STORAGE", "eyebrow"))
        sc_layout.addWidget(label("Workspace & Cache", "heading"))
        sc_layout.addWidget(label(f"Data root:  {self.store.root}", "muted"))
        self.settings_runs_count_label = label(f"Total run records:  {len(self.runs)} runs", "muted")
        sc_layout.addWidget(self.settings_runs_count_label)
        btn_row = QHBoxLayout()
        btn_reset_layout = QPushButton("Reset UI layout")
        btn_reset_layout.setToolTip("Restore default pane sizes and layout")
        btn_reset_layout.clicked.connect(lambda: (self.restoreState(self.default_layout), self.statusBar().showMessage("Restored default UI layout.", 4000)))
        btn_row.addWidget(btn_reset_layout)
        btn_row.addStretch()
        sc_layout.addLayout(btn_row)
        left_layout.addWidget(storage_card)

        # Card 4: Quick Console Commands (Boilerplate commands)
        cmd_card = QFrame()
        cmd_card.setObjectName("card")
        cc_layout = QVBoxLayout(cmd_card)
        cc_layout.setContentsMargins(14, 12, 14, 12)
        cc_layout.setSpacing(10)
        cc_layout.addWidget(label("QUICK ACTIONS", "eyebrow"))
        cc_layout.addWidget(label("Console Commands", "heading"))
        cc_layout.addWidget(label("Execute boilerplate commands in the Theta console:", "muted"))
        cmd_row = QHBoxLayout()
        for cmd_name, desc in (
            ("status", "Show active training and run summary"),
            ("config", "Show pipeline config command"),
            ("help", "List available console commands"),
            ("clear", "Clear console output buffer"),
        ):
            btn = QPushButton(f"› {cmd_name}")
            btn.setToolTip(desc)
            btn.clicked.connect(lambda _, c=cmd_name: self.run_quick_command(c))
            cmd_row.addWidget(btn)
        cc_layout.addLayout(cmd_row)
        left_layout.addWidget(cmd_card)
        left_layout.addStretch()

        left_scroll = QScrollArea()
        left_scroll.setWidgetResizable(True)
        left_scroll.setWidget(left_container)
        left_scroll.setFrameShape(QFrame.Shape.NoFrame)

        # Right Column: About ThetaIDE & ASCII Sculpture
        right_card = QFrame()
        right_card.setObjectName("card")
        rc_layout = QVBoxLayout(right_card)
        rc_layout.setContentsMargins(16, 14, 16, 14)
        rc_layout.setSpacing(10)
        rc_layout.addWidget(label("ABOUT THETA-IDE", "eyebrow"))
        title_row = QHBoxLayout()
        title_row.addWidget(label("ThetaIDE", "heading"))
        badge = label("v0.1.0-alpha", "badge")
        title_row.addWidget(badge)
        title_row.addStretch()
        rc_layout.addLayout(title_row)
        desc = label("Neuro-symbolic reinforcement learning research studio.\n"
                     "Jointly configure, inspect, and evaluate hybrid logic-neural agents "
                     "with live monitoring and experiment comparison.", "muted")
        desc.setWordWrap(True)
        rc_layout.addWidget(desc)

        self.settings_ascii = AsciiTheta(right_card)
        rc_layout.addWidget(self.settings_ascii, 1)

        anim_row = QHBoxLayout()
        self.anim_toggle_btn = QPushButton("Pause animation")
        self.anim_toggle_btn.setCheckable(True)
        self.anim_toggle_btn.toggled.connect(self.toggle_ascii_animation)
        anim_row.addWidget(self.anim_toggle_btn)
        anim_row.addStretch()
        anim_row.addWidget(label("3D Software-Rendered ASCII Sculpture", "muted"))
        rc_layout.addLayout(anim_row)

        splitter.addWidget(left_scroll)
        splitter.addWidget(right_card)
        splitter.setSizes([520, 500])

        layout.addWidget(splitter, 1)
        return panel

    def settings_theme_selected(self, theme_name):
        if theme_name and theme_name != self.theme_manager.active["name"]:
            theme = self.theme_manager.themes().get(theme_name)
            if theme:
                self.select_theme(theme)

    def on_pane_slider_toggled(self, pane_id, checked):
        visible_count = sum(1 for s in self.pane_sliders.values() if s.isChecked())
        if not checked and visible_count == 0:
            slider = self.pane_sliders.get(pane_id)
            if slider:
                slider.blockSignals(True)
                slider.setChecked(True)
                slider.blockSignals(False)
            self.statusBar().showMessage("At least one panel must remain visible in the sidebar.", 3000)
            return

        self.tabs.set_tab_visible(pane_id, checked)
        self.update_pane_sliders_state()
        self.save_layout()

    def update_pane_sliders_state(self):
        visible_sliders = [s for s in self.pane_sliders.values() if s.isChecked()]
        if len(visible_sliders) == 1:
            visible_sliders[0].setEnabled(False)
            visible_sliders[0].setToolTip("At least one panel must remain visible in the sidebar")
        else:
            for s in self.pane_sliders.values():
                s.setEnabled(True)
                s.setToolTip("Show or hide this panel in the sidebar")

    def reset_sidebar_layout(self):
        default_order = ["monitor", "results", "config", "plots", "tensorboard", "queue", "terminal", "console"]
        self.tabs.apply_tab_order(default_order)
        for pid, slider in self.pane_sliders.items():
            slider.blockSignals(True)
            slider.setChecked(True)
            slider.setEnabled(True)
            slider.blockSignals(False)
            self.tabs.set_tab_visible(pid, True)
        self.save_layout()
        self.statusBar().showMessage("Restored default sidebar panels and order.", 4000)

    def save_layout(self):
        if not hasattr(self, "layout_file"):
            return
        data = {
            "tab_order": list(self.tabs.tab_order),
            "visible_tabs": {pid: self.tabs.is_tab_visible(pid) for pid in self.tabs.tab_order}
        }
        try:
            self.layout_file.write_text(json.dumps(data, indent=2) + "\n", encoding="utf-8")
        except OSError as exc:
            self.log(f"Failed to save sidebar layout: {exc}")

    def load_layout(self):
        if not hasattr(self, "layout_file") or not self.layout_file.exists():
            return
        try:
            data = json.loads(self.layout_file.read_text(encoding="utf-8"))
            if isinstance(data, dict):
                order = data.get("tab_order")
                if isinstance(order, list):
                    self.tabs.apply_tab_order(order)
                vis = data.get("visible_tabs")
                if isinstance(vis, dict):
                    for pid, val in vis.items():
                        self.tabs.set_tab_visible(pid, bool(val))
                        if hasattr(self, "pane_sliders") and pid in self.pane_sliders:
                            slider = self.pane_sliders[pid]
                            slider.blockSignals(True)
                            slider.setChecked(bool(val))
                            slider.blockSignals(False)
                    self.update_pane_sliders_state()
        except (OSError, ValueError, TypeError) as exc:
            self.log(f"Could not load sidebar layout preferences: {exc}")

    def test_backend_connection(self):
        if hasattr(self, "settings_backend_status"):
            self.settings_backend_status.setText("Status: Testing connection…")
            self.settings_backend_status.setStyleSheet("")
            self.backend.get("/api/health", self.backend_health_callback)

    def backend_health_callback(self, data, error):
        if not hasattr(self, "settings_backend_status"):
            return
        if error:
            self.settings_backend_status.setText(f"Status: Disconnected ({error})")
            self.settings_backend_status.setStyleSheet("color: #fb4934;")
        else:
            status_text = data.get("status", "ok") if isinstance(data, dict) else "ok"
            self.settings_backend_status.setText(f"Status: Connected (Server status: {status_text})")
            self.settings_backend_status.setStyleSheet(f"color: {theme_color('primary')};")

    def open_swagger_docs(self):
        docs_url = f"{self.backend.base_url.rstrip('/')}/docs"
        QDesktopServices.openUrl(QUrl(docs_url))

    def run_quick_command(self, command):
        self.tabs.setCurrentWidget(self.console_panel)
        if hasattr(self, "command"):
            self.command.setText(command)
        self.console_command(command)

    def toggle_ascii_animation(self, paused):
        if hasattr(self, "settings_ascii"):
            self.settings_ascii.set_paused(paused)
        if hasattr(self, "anim_toggle_btn"):
            self.anim_toggle_btn.setText("Resume animation" if paused else "Pause animation")

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
        view_menu.addAction("Experiment config", lambda: self.tabs.setCurrentWidget(self.config_panel))
        view_menu.addAction("Plot viewer", lambda: self.tabs.setCurrentWidget(self.plot_viewer))
        view_menu.addAction("TensorBoard", self.show_tensorboard)
        view_menu.addAction("Terminal", lambda: self.tabs.setCurrentWidget(self.terminal_panel))
        view_menu.addAction("Console", lambda: self.tabs.setCurrentWidget(self.console_panel))
        view_menu.addAction("Settings & About", lambda: self.tabs.setCurrentWidget(self.settings_panel))
        view_menu.addAction("Restore default layout", lambda: self.restoreState(self.default_layout))
        help_menu = self.menuBar().addMenu("Help")
        help_menu.addAction("Settings & About", self.show_about)

    def show_tensorboard(self):
        self.tabs.setCurrentWidget(self.tensorboard_panel)

    def tab_changed(self, index):
        if self.tabs.widget(index) is self.tensorboard_panel:
            self.tensorboard_panel.activate()
        if self.tabs.widget(index) is self.queue_panel:
            self.poll_queue()
            self.queue_timer.start()
        if hasattr(self, "terminal_panel") and self.tabs.widget(index) is self.terminal_panel:
            self.terminal_panel.terminal.focus_terminal()
            self.terminal_panel.terminal.fit_terminal()
        if hasattr(self, "console_panel") and self.tabs.widget(index) is self.console_panel:
            self.command.setFocus()
        if hasattr(self, "settings_panel") and self.tabs.widget(index) is self.settings_panel:
            self.test_backend_connection()

    def show_about(self):
        if hasattr(self, "settings_panel"):
            self.tabs.setCurrentWidget(self.settings_panel)
        else:
            dialog = AboutDialog(self)
            dialog.exec()
            dialog.deleteLater()

    def theme_changed(self):
        self.theme_status.setText(f"  ●  Local workspace    /    {self.theme_manager.active['name']}")
        self.highlighter.rehighlight()
        self.tabs.refresh_icons()
        if hasattr(self, "settings_theme_select"):
            self.settings_theme_select.blockSignals(True)
            self.settings_theme_select.setCurrentText(self.theme_manager.active["name"])
            self.settings_theme_select.blockSignals(False)
        if hasattr(self, "settings_ascii"):
            self.settings_ascii.update()
        if hasattr(self, "settings_backend_status"):
            self.test_backend_connection()
        if hasattr(self, "terminal_panel"):
            self.terminal_panel.apply_theme(self.theme_manager.active)

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
        if hasattr(self, "config_viewer") and getattr(self.config_viewer, "raw_data", None):
            d = self.config_viewer.raw_data
            exp_id = d.get("experiment_id") or (self.config_viewer.current_path.stem if getattr(self.config_viewer, "current_path", None) else "cartpole_ppo_baseline")
            seed = int(d.get("seed") if d.get("seed") is not None else 42)
            steps = int(d.get("total_timesteps") or 10000)
            tb = bool(d.get("tensorboard", True))
            methods = d.get("methods", {})
            ppo_spec = methods.get("ppo", {}) if isinstance(methods, dict) else {}
            lr = float(ppo_spec.get("lr") or 0.0003)
            batch = int(ppo_spec.get("batch_size") or 64)
            gamma = float(ppo_spec.get("gamma") or 0.99)
            return Config(name=str(exp_id), seed=seed, total_timesteps=steps, lr=lr, batch_size=batch, gamma=gamma, tensorboard=tb)
        return Config()

    def update_config(self, *_):
        self.compose_timer.start()

    def request_compose(self):
        self.compose_serial += 1
        serial, config = self.compose_serial, self.config()
        exp_target = getattr(self, "current_experiment", None) or BASE_EXPERIMENT
        overrides = self.config_viewer.get_overrides() if hasattr(self, "config_viewer") else config.overrides()
        payload = {"experiment": exp_target, "overrides": overrides}
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
            if title in self.fixed_fields:
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
        if hasattr(self, "config_viewer"):
            d = self.config_viewer.raw_data
            d["experiment_id"] = config.name
            d["seed"] = config.seed
            d["total_timesteps"] = config.total_timesteps
            d["tensorboard"] = config.tensorboard
            methods = d.setdefault("methods", {})
            if isinstance(methods, dict):
                ppo_spec = methods.setdefault("ppo", {})
                if isinstance(ppo_spec, dict):
                    ppo_spec["lr"] = config.lr
                    ppo_spec["batch_size"] = config.batch_size
                    ppo_spec["gamma"] = config.gamma
            self.config_viewer._render_boxes()
        self.update_config()

    def new_experiment(self):
        if hasattr(self, "config_tree"):
            self.config_tree.prompt_new_in_group()
        else:
            self.set_config(Config(name="cartpole_ppo_experiment"))
            self.tabs.setCurrentWidget(self.config_panel)

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
        self.tabs.setCurrentWidget(self.monitor_panel)
        self.update_launch_state()

    # ── Live training through the backend ────────────────────────────────────

    def launch_training(self):
        if self.active:
            self.statusBar().showMessage("A run is already active. Stop it before launching another.", 4000)
            return
        if not self.compose_valid:
            self.statusBar().showMessage("Start the backend and fix config errors before launching.", 5000)
            return
        if hasattr(self, "config_viewer") and self.config_viewer.is_dirty:
            self.config_viewer.save_to_disk()

        config = self.config()
        exp_target = getattr(self, "current_experiment", None) or BASE_EXPERIMENT
        experiment_id = self.unique_experiment_id(config.name)
        overrides = self.config_viewer.get_overrides() if hasattr(self, "config_viewer") else config.overrides()
        overrides = [o for o in overrides if not o.startswith("++experiment_id=")] + [f"++experiment_id='{experiment_id}'"]
        self.start_button.setEnabled(False)
        self.log(f"Launching {experiment_id}:\n  python run_pipeline.py {exp_target} {' '.join(overrides)}")
        self.backend.post("/api/experiments/launch", {"experiment": exp_target, "overrides": overrides},
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
        if hasattr(self, "config_viewer") and self.config_viewer.is_dirty:
            self.config_viewer.save_to_disk()

        config = self.config()
        exp_target = getattr(self, "current_experiment", None) or BASE_EXPERIMENT
        experiment_id = self.unique_experiment_id(config.name)
        overrides = self.config_viewer.get_overrides() if hasattr(self, "config_viewer") else config.overrides()
        overrides = [o for o in overrides if not o.startswith("++experiment_id=")] + [f"++experiment_id='{experiment_id}'"]
        self.queue_button.setEnabled(False)
        self.log(f"Queuing {experiment_id}:\n  python run_pipeline.py {exp_target} {' '.join(overrides)}")
        self.backend.post("/api/experiments/launch",
                          {"experiment": exp_target, "overrides": overrides, "queue": True},
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
        self.tabs.setCurrentWidget(self.monitor_panel)

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
        if hasattr(self, "notes"):
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
        if hasattr(self, "settings_runs_count_label"):
            self.settings_runs_count_label.setText(f"Total run records:  {len(self.runs)} runs")
        query = self.search.text().lower()
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for run in self.runs:
            config = run["config"]
            name = config["name"] if run["simulated"] else run["backend"]["experiment_id"]
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
        self.table.blockSignals(False)

    def by_id(self, run_id):
        return next((run for run in self.runs if run["id"] == run_id), None)

    def view_selected_plot(self):
        if self.selected:
            self.plot_viewer.show_run(self.selected["id"])
        self.tabs.setCurrentWidget(self.plot_viewer)

    def tree_selected(self, item, _):
        pass

    def table_selected(self):
        rows = self.table.selectionModel().selectedRows()
        if len(rows) == 1:
            run = self.by_id(self.table.item(rows[0].row(), 0).data(Qt.ItemDataRole.UserRole))
            self.select_run(run)

    def notes_changed(self):
        if hasattr(self, "notes") and self.selected:
            self.selected["notes"] = self.notes.toPlainText()
            ok = self.persist(self.selected)
            if hasattr(self, "note_status"):
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
        self.tabs.setCurrentWidget(self.config_panel)
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
                if hasattr(self, "config_viewer") and getattr(self.config_viewer, "raw_data", None):
                    text = yaml.safe_dump(self.config_viewer.raw_data, sort_keys=False)
                else:
                    text = config.recipe_yaml()
                Path(path).write_text(text, encoding="utf-8")
                self.log(f"Exported configuration to {path}.")
            except OSError as exc:
                QMessageBox.warning(self, "Export failed", str(exc))

    def log(self, message):
        if hasattr(self, "console"):
            self.console.appendPlainText(message)

    def console_command(self, cmd_override=None):
        if cmd_override is not None:
            command = cmd_override.strip()
        elif hasattr(self, "command"):
            command = self.command.text().strip()
            self.command.clear()
        else:
            command = ""
        self.log(f"theta › {command}")
        if command == "clear":
            self.console.clear()
        elif command == "help":
            self.log("help    Show commands\nstatus  Show the active run\nconfig  Show the equivalent run_pipeline command\nclear   Clear output\n(Use the Terminal tab for an interactive shell with tmux support.)")
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
            self.log(f"Unknown command '{command}'. Type help or switch to the Terminal tab.")

    def closeEvent(self, event):
        if self.active and self.active["simulated"]:
            self.stop_run()
        elif self.active:
            self.persist(self.active)  # training keeps running in the backend; reopening reconnects
        if hasattr(self, "terminal_panel"):
            self.terminal_panel.terminal.close()
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
