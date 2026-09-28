"""TensorBoard tab: the backend runs the TensorBoard server and this panel embeds it."""
from PyQt6.QtCore import Qt, QTimer, QUrl
from PyQt6.QtGui import QColor, QDesktopServices
from PyQt6.QtWidgets import QHBoxLayout, QPushButton, QStackedWidget, QVBoxLayout, QWidget

from .theme import theme_color
from .widgets import label

try:  # PyQt6-WebEngine is optional; without it TensorBoard opens in the system browser.
    from PyQt6.QtWebEngineWidgets import QWebEngineView
except ImportError:
    QWebEngineView = None


class TensorBoardPanel(QWidget):
    def __init__(self, backend, log, parent=None):
        super().__init__(parent)
        self.backend, self.log = backend, log
        self.url = None
        self.loaded_url = None
        self.busy = False
        self.timer = QTimer(self)
        self.timer.setInterval(1000)
        self.timer.timeout.connect(self.refresh)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)
        bar = QHBoxLayout()
        bar.setContentsMargins(14, 8, 14, 8)
        self.status = label("TensorBoard", "muted")
        bar.addWidget(self.status, 1)
        self.start_button = self.button("Start", lambda: self.refresh(start=True))
        self.stop_button = self.button("Stop", self.stop)
        self.reload_button = self.button("Reload", self.reload)
        self.browser_button = self.button("Open in browser ↗", self.open_in_browser)
        for button in (self.start_button, self.stop_button, self.reload_button, self.browser_button):
            bar.addWidget(button)
        layout.addLayout(bar)

        self.stack = QStackedWidget()
        self.placeholder = label("", "muted")
        self.placeholder.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.placeholder.setWordWrap(True)
        self.stack.addWidget(self.placeholder)
        self.view = None
        if QWebEngineView is not None:
            self.view = QWebEngineView()
            self.view.page().setBackgroundColor(QColor(theme_color("base")))
            self.stack.addWidget(self.view)
        layout.addWidget(self.stack, 1)
        self.show_message("Open this tab to start TensorBoard.", running=False)

    def button(self, text, callback):
        button = QPushButton(text)
        button.clicked.connect(callback)
        return button

    def activate(self):
        """Called when the tab is shown: start TensorBoard on first use, or show the running server."""
        if self.loaded_url is None:
            self.refresh(start=True)

    def show_message(self, text, running):
        self.placeholder.setText(text)
        self.stack.setCurrentWidget(self.placeholder)
        self.start_button.setVisible(not running)
        self.stop_button.setVisible(running)
        self.reload_button.setEnabled(False)
        self.browser_button.setEnabled(running and self.url is not None)

    def refresh(self, start=False):
        if self.busy:
            return
        self.busy = True
        self.backend.get("/api/tensorboard", lambda data, error: self.status_received(data, error, start))

    def status_received(self, data, error, start):
        self.busy = False
        if error:
            self.timer.stop()
            self.status.setText("TensorBoard  •  backend offline")
            self.show_message("The backend is not reachable. Start the API to use TensorBoard:\n"
                              "uvicorn src.app.api.app:app --host 127.0.0.1 --port 8000", running=False)
            return
        if not data["available"]:
            self.timer.stop()
            self.show_message("TensorBoard is not installed in the backend environment.\n"
                              "Install it there with: pip install tensorboard", running=False)
            return
        if not data["running"]:
            self.url = self.loaded_url = None
            if start:
                self.status.setText("TensorBoard  •  starting…")
                self.show_message("Starting TensorBoard…", running=False)
                self.backend.post("/api/tensorboard/start", {}, self.start_received)
                return
            self.timer.stop()
            self.status.setText("TensorBoard  •  stopped")
            detail = f"\n\nIt exited with code {data['exit_code']}:\n{data.get('error') or ''}" if "exit_code" in data else ""
            self.show_message("TensorBoard is stopped." + detail, running=False)
            return
        self.url = data["url"]
        if not data["ready"]:
            self.status.setText("TensorBoard  •  starting…")
            self.show_message("Starting TensorBoard…", running=True)
            self.timer.start()
            return
        self.timer.stop()
        self.status.setText(f"TensorBoard  •  {self.url}  •  logs from {data['logdir']}/ (runs with “Log to TensorBoard”)")
        self.stop_button.setVisible(True)
        self.start_button.setVisible(False)
        self.browser_button.setEnabled(True)
        if self.view is None:
            self.show_message("PyQt6-WebEngine is not installed, so TensorBoard cannot be embedded here.\n"
                              "Use “Open in browser”, or install it: pip install PyQt6-WebEngine", running=True)
            self.browser_button.setEnabled(True)
            return
        self.reload_button.setEnabled(True)
        self.stack.setCurrentWidget(self.view)
        if self.loaded_url != self.url:
            self.view.load(QUrl(self.url))
            self.loaded_url = self.url
            self.log(f"TensorBoard running at {self.url}")

    def start_received(self, data, error):
        if error:
            self.status.setText("TensorBoard  •  failed to start")
            self.show_message(f"Could not start TensorBoard: {error}", running=False)
            return
        self.timer.start()  # poll until the server answers

    def stop(self):
        self.timer.stop()
        self.backend.post("/api/tensorboard/stop", {}, lambda data, error: self.status_received(data, error, False))
        self.loaded_url = None
        if self.view is not None:
            self.view.setUrl(QUrl("about:blank"))

    def reload(self):
        if self.view is not None and self.loaded_url:
            self.view.reload()

    def open_in_browser(self):
        if self.url:
            QDesktopServices.openUrl(QUrl(self.url))
