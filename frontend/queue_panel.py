"""Job queue tab: the backend's running, queued and recently finished training jobs."""
import time

from PyQt6.QtCore import Qt
from PyQt6.QtWidgets import (QAbstractItemView, QHBoxLayout, QHeaderView, QPushButton, QTableWidget,
                             QTableWidgetItem, QVBoxLayout, QWidget)

from .widgets import label

FINISHED_MARKS = {"completed": "✓", "failed": "✗", "error": "✗", "cancelled": "■"}


def clock(seconds):
    seconds = int(max(0, seconds))
    return f"{seconds // 60}:{seconds % 60:02d}"


class QueuePanel(QWidget):
    """Shows GET /api/queue. Actions call back into the window, which talks to the backend."""

    def __init__(self, on_move, on_remove, on_open, on_toggle, parent=None):
        super().__init__(parent)
        self.on_move, self.on_remove, self.on_open, self.on_toggle = on_move, on_remove, on_open, on_toggle
        self.rows = []  # (kind, job) per table row: kind is "active", "queued" or "finished"
        self.queued_count = 0
        self.running = False

        layout = QVBoxLayout(self)
        layout.setContentsMargins(18, 18, 18, 18)
        layout.addWidget(label("Job queue", "heading"))
        about = label("Line up experiments with “Add to queue” in the toolbar, then press Start queue. "
                      "They train one at a time, in order, on this machine, and keep running if you close ThetaIDE.",
                      "muted")
        about.setWordWrap(True)
        layout.addWidget(about)
        state_row = QHBoxLayout()
        self.run_button = self.button("▶  Start queue", lambda: self.on_toggle(not self.running))
        state_row.addWidget(self.run_button)
        self.state = label("", "muted")
        self.state.setWordWrap(True)
        state_row.addWidget(self.state, 1)
        layout.addLayout(state_row)

        self.table = QTableWidget(0, 5)
        self.table.setHorizontalHeaderLabels(["#", "Experiment", "Steps", "Status", "Time"])
        header = self.table.horizontalHeader()
        header.setSectionResizeMode(QHeaderView.ResizeMode.ResizeToContents)
        header.setSectionResizeMode(1, QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().hide()
        self.table.setSelectionBehavior(QAbstractItemView.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QAbstractItemView.SelectionMode.SingleSelection)
        self.table.setEditTriggers(QAbstractItemView.EditTrigger.NoEditTriggers)
        self.table.setAlternatingRowColors(True)
        self.table.setShowGrid(False)
        self.table.itemSelectionChanged.connect(self.update_buttons)
        self.table.itemDoubleClicked.connect(lambda _: self.open_selected())
        layout.addWidget(self.table, 1)

        actions = QHBoxLayout()
        self.up_button = self.button("▲  Move up", lambda: self.move_selected(-1))
        self.down_button = self.button("▼  Move down", lambda: self.move_selected(1))
        self.remove_button = self.button("✕  Remove from queue", self.remove_selected)
        self.open_button = self.button("Open in monitor", self.open_selected)
        for button in (self.up_button, self.down_button, self.remove_button, self.open_button):
            actions.addWidget(button)
        actions.addStretch()
        self.summary = label("", "muted")
        actions.addWidget(self.summary)
        layout.addLayout(actions)
        self.update_buttons()

    def button(self, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        return button

    def selected(self):
        rows = self.table.selectionModel().selectedRows()
        return self.rows[rows[0].row()] if rows else (None, None)

    def set_offline(self, error):
        self.summary.setText("Backend offline — the queue is kept by the API server")
        self.summary.setToolTip(error)
        self.run_button.setEnabled(False)

    def set_data(self, data):
        _, keep = self.selected()
        keep_id = keep and keep["job_id"]
        self.rows = ([("active", j) for j in data["active"]] + [("queued", j) for j in data["queued"]]
                     + [("finished", j) for j in data["finished"]])
        self.queued_count = len(data["queued"])
        now = time.time()
        self.table.blockSignals(True)
        self.table.setRowCount(0)
        for kind, job in self.rows:
            if kind == "active":
                marker = "▶"
                timing = f"running {clock(now - job.get('started', job['created']))}"
            elif kind == "queued":
                marker = str(job["position"] + 1)
                timing = "queued " + time.strftime("%H:%M", time.localtime(job["created"]))
            else:
                marker = FINISHED_MARKS.get(job["status"], "·")
                start = job.get("started")
                timing = (f"took {clock(job['finished'] - start)}" if start and job.get("finished")
                          else "never started")
            steps = job.get("effective_timesteps") or job.get("total_timesteps")
            row = self.table.rowCount()
            self.table.insertRow(row)
            for col, text in enumerate((marker, job["experiment_id"], f"{steps:,}" if steps else "—",
                                        job["status"], timing)):
                cell = QTableWidgetItem(text)
                if col == 0:
                    cell.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
                self.table.setItem(row, col, cell)
            if job["job_id"] == keep_id:
                self.table.selectRow(row)
        self.table.blockSignals(False)
        self.running = data["running"]
        active = len(data["active"])
        self.summary.setText(f"{active} running  ·  {self.queued_count} queued")
        self.summary.setToolTip("")
        if self.running:
            self.state.setText("Queue running: jobs start one after another. "
                               "It pauses itself when the last one finishes.")
        elif self.queued_count:
            self.state.setText(f"Queue paused: {self.queued_count} job{'s' * (self.queued_count != 1)} waiting. "
                               "Press Start queue to begin."
                               + ("  The job already training will finish." if active else ""))
        else:
            self.state.setText("Queue empty. Add experiments with “Add to queue”.")
        self.run_button.setText("⏸  Pause queue" if self.running else "▶  Start queue")
        self.run_button.setObjectName("" if self.running else "primary")
        self.run_button.setStyle(self.run_button.style())
        self.run_button.setEnabled(self.running or self.queued_count > 0)
        self.update_buttons()

    def update_buttons(self):
        kind, job = self.selected()
        queued = kind == "queued"
        self.up_button.setEnabled(queued and job["position"] > 0)
        self.down_button.setEnabled(queued and job["position"] < self.queued_count - 1)
        self.remove_button.setEnabled(queued)
        self.open_button.setEnabled(job is not None)

    def move_selected(self, delta):
        kind, job = self.selected()
        if kind == "queued":
            self.on_move(job["job_id"], job["position"] + delta)

    def remove_selected(self):
        kind, job = self.selected()
        if kind == "queued":
            self.on_remove(job["job_id"])

    def open_selected(self):
        _, job = self.selected()
        if job is not None:
            self.on_open(job["job_id"])
