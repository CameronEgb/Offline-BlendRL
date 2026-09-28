"""PyQt6 QWebEngineView wrapper for xterm.js terminal emulator with PTY integration."""
import json
from pathlib import Path
from typing import Optional

from PyQt6.QtCore import Qt, QUrl, pyqtSignal, pyqtSlot, QObject, QTimer
from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QPushButton, QLabel,
    QStackedWidget, QPlainTextEdit, QFrame, QSizePolicy
)
from PyQt6.QtWebEngineWidgets import QWebEngineView
from PyQt6.QtWebEngineCore import QWebEngineSettings
from PyQt6.QtWebChannel import QWebChannel

from .pty_session import PtySession


HTML_PATH = Path(__file__).resolve().parent / "terminal.html"

# Authentic terminal palettes for built-in themes
NATIVE_TERMINAL_PALETTES = {
    "catppuccin": {
        "background": "#1e1e2e",
        "foreground": "#cdd6f4",
        "cursor": "#f5e0dc",
        "cursorAccent": "#1e1e2e",
        "selectionBackground": "rgba(88, 91, 112, 0.45)",
        "selectionForeground": "#cdd6f4",
        "black": "#45475a",
        "red": "#f38ba8",
        "green": "#a6e3a1",
        "yellow": "#f9e2af",
        "blue": "#89b4fa",
        "magenta": "#f5c2e7",
        "cyan": "#94e2d5",
        "white": "#bac2de",
        "brightBlack": "#585b70",
        "brightRed": "#f38ba8",
        "brightGreen": "#a6e3a1",
        "brightYellow": "#f9e2af",
        "brightBlue": "#89b4fa",
        "brightMagenta": "#cba6f7",
        "brightCyan": "#94e2d5",
        "brightWhite": "#a6adc8",
    },
    "catppuccin latte": {
        "background": "#eff1f5",
        "foreground": "#4c4f69",
        "cursor": "#dc8a78",
        "cursorAccent": "#eff1f5",
        "selectionBackground": "rgba(172, 176, 190, 0.45)",
        "selectionForeground": "#4c4f69",
        "black": "#5c5f77",
        "red": "#d20f39",
        "green": "#40a02b",
        "yellow": "#df8e1d",
        "blue": "#1e66f5",
        "magenta": "#ea76cb",
        "cyan": "#179299",
        "white": "#acb0be",
        "brightBlack": "#6c6f85",
        "brightRed": "#d20f39",
        "brightGreen": "#40a02b",
        "brightYellow": "#df8e1d",
        "brightBlue": "#1e66f5",
        "brightMagenta": "#8839ef",
        "brightCyan": "#179299",
        "brightWhite": "#bcc0cc",
    },
    "dracula": {
        "background": "#282a36",
        "foreground": "#f8f8f2",
        "cursor": "#f8f8f2",
        "cursorAccent": "#282a36",
        "selectionBackground": "rgba(68, 71, 90, 0.5)",
        "black": "#21222c",
        "red": "#ff5555",
        "green": "#50fa7b",
        "yellow": "#f1fa8c",
        "blue": "#bd93f9",
        "magenta": "#ff79c6",
        "cyan": "#8be9fd",
        "white": "#f8f8f2",
        "brightBlack": "#6272a4",
        "brightRed": "#ff6e6e",
        "brightGreen": "#69ff94",
        "brightYellow": "#ffffa5",
        "brightBlue": "#d6acff",
        "brightMagenta": "#ff92df",
        "brightCyan": "#a4ffff",
        "brightWhite": "#ffffff",
    },
    "nord": {
        "background": "#2e3440",
        "foreground": "#d8dee9",
        "cursor": "#d8dee9",
        "cursorAccent": "#2e3440",
        "selectionBackground": "rgba(76, 86, 106, 0.5)",
        "black": "#3b4252",
        "red": "#bf616a",
        "green": "#a3be8c",
        "yellow": "#ebcb8b",
        "blue": "#81a1c1",
        "magenta": "#b48ead",
        "cyan": "#88c0d0",
        "white": "#e5e9f0",
        "brightBlack": "#4c566a",
        "brightRed": "#bf616a",
        "brightGreen": "#a3be8c",
        "brightYellow": "#ebcb8b",
        "brightBlue": "#81a1c1",
        "brightMagenta": "#b48ead",
        "brightCyan": "#8fbcbb",
        "brightWhite": "#eceff4",
    },
}


class TerminalBridge(QObject):
    """Bridge object exposed to JavaScript via QWebChannel."""

    data_received = pyqtSignal(str)
    theme_received = pyqtSignal(str)
    font_size_received = pyqtSignal(int)
    clear_requested = pyqtSignal()

    def __init__(self, terminal_widget: "TerminalWidget"):
        super().__init__()
        self.terminal_widget = terminal_widget

    @pyqtSlot(str)
    def send_input(self, data: str):
        """Called from xterm.js on user keyboard input."""
        self.terminal_widget.pty.write(data)

    @pyqtSlot(int, int)
    def resize_pty(self, cols: int, rows: int):
        """Called from xterm.js when container or window resizes."""
        self.terminal_widget.pty.resize(cols, rows)

    @pyqtSlot()
    def terminal_ready(self):
        """Called when xterm.js is initialized and mounted in DOM."""
        self.terminal_widget._on_terminal_ready()


class TerminalWidget(QWidget):
    """Interactive xterm.js terminal widget backed by an OS pseudo-terminal."""

    session_started = pyqtSignal()
    session_exited = pyqtSignal(int)

    def __init__(self, cwd: Optional[str] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.cwd = cwd or str(Path.cwd())
        self.pty = PtySession(cwd=self.cwd, parent=self)
        self.bridge = TerminalBridge(self)
        self._is_ready = False
        self._pending_theme: Optional[dict] = None
        self._current_cols = 80
        self._current_rows = 24

        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(0)

        self.web_view = QWebEngineView(self)
        self.web_view.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)
        
        # Configure WebEngine settings
        settings = self.web_view.settings()
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessFileUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.LocalContentCanAccessRemoteUrls, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.JavascriptCanAccessClipboard, True)
        settings.setAttribute(QWebEngineSettings.WebAttribute.ScrollAnimatorEnabled, False)

        # Attach WebChannel
        self.channel = QWebChannel(self.web_view.page())
        self.channel.registerObject("bridge", self.bridge)
        self.web_view.page().setWebChannel(self.channel)

        layout.addWidget(self.web_view)

        # Connect PTY signals
        self.pty.data_ready.connect(self._on_pty_data)
        self.pty.process_exited.connect(self._on_pty_exit)

        # Load terminal HTML
        self.web_view.load(QUrl.fromLocalFile(str(HTML_PATH)))

    def _on_terminal_ready(self):
        """xterm.js has loaded, mounted, and reported ready."""
        self._is_ready = True
        if not self.pty.is_alive():
            self.pty.start(cols=self._current_cols, rows=self._current_rows)
            self.session_started.emit()

        if self._pending_theme:
            self.apply_theme(self._pending_theme)
            self._pending_theme = None

        self.focus_terminal()

    def _on_pty_data(self, text: str):
        """Send raw data from PTY process to xterm.js."""
        self.bridge.data_received.emit(text)

    def _on_pty_exit(self, exit_code: int):
        """Child shell exited; notify xterm and signal."""
        msg = f"\r\n\x1b[90m[Process completed with exit code {exit_code} — Press '+ New shell' or Enter to restart]\x1b[0m\r\n"
        self.bridge.data_received.emit(msg)
        self.session_exited.emit(exit_code)

    def send_command(self, cmd: str):
        """Write a command string followed by Enter to the shell."""
        if not self.pty.is_alive():
            self.restart_session()
        self.pty.write(f"{cmd}\n")
        self.focus_terminal()

    def restart_session(self):
        """Terminate current process and start a fresh shell session."""
        self.pty.close()
        self.bridge.clear_requested.emit()
        self.pty.start(cols=self._current_cols, rows=self._current_rows)
        self.fit_terminal()
        self.focus_terminal()

    def clear(self):
        """Clear xterm terminal viewport."""
        self.bridge.clear_requested.emit()

    def fit_terminal(self):
        """Trigger fitAddon.fit() in xterm.js."""
        if self._is_ready:
            self.web_view.page().runJavaScript("if (typeof fitTerminal === 'function') fitTerminal();")

    def focus_terminal(self):
        """Focus keyboard events on the terminal."""
        self.web_view.setFocus()
        if self._is_ready:
            self.web_view.page().runJavaScript("if (typeof term !== 'undefined') term.focus();")

    def resizeEvent(self, event):
        super().resizeEvent(event)
        QTimer.singleShot(50, self.fit_terminal)

    def apply_theme(self, theme_dict: dict):
        """Map Theta-IDE palette to xterm.js theme object and update web view."""
        if not self._is_ready:
            self._pending_theme = theme_dict
            return

        name_key = (theme_dict.get("name") or "").strip().casefold()
        if name_key in NATIVE_TERMINAL_PALETTES:
            xterm_theme = NATIVE_TERMINAL_PALETTES[name_key]
        elif "catppuccin" in name_key and "latte" in name_key:
            xterm_theme = NATIVE_TERMINAL_PALETTES["catppuccin latte"]
        elif "catppuccin" in name_key:
            xterm_theme = NATIVE_TERMINAL_PALETTES["catppuccin"]
        elif "dracula" in name_key:
            xterm_theme = NATIVE_TERMINAL_PALETTES["dracula"]
        elif "nord" in name_key:
            xterm_theme = NATIVE_TERMINAL_PALETTES["nord"]
        else:
            colors = theme_dict.get("colors", {})
            if not colors:
                return

            base = colors.get("base", "#1d2021")
            text = colors.get("text", "#ebdbb2")
            accent = colors.get("accent", "#fabd2f")
            primary = colors.get("primary", "#b8bb26")
            secondary = colors.get("secondary", "#83a598")
            number = colors.get("number", "#d3869b")
            comment = colors.get("comment", "#928374")
            border = colors.get("border", "#504945")

            xterm_theme = {
                "background": base,
                "foreground": text,
                "cursor": accent,
                "cursorAccent": base,
                "selectionBackground": border,
                "black": base,
                "red": "#ea6962",
                "green": primary,
                "yellow": accent,
                "blue": secondary,
                "magenta": number,
                "cyan": "#8ec07c",
                "white": text,
                "brightBlack": comment,
                "brightRed": "#fb4934",
                "brightGreen": primary,
                "brightYellow": accent,
                "brightBlue": secondary,
                "brightMagenta": number,
                "brightCyan": "#8ec07c",
                "brightWhite": "#fbf1c7",
            }
        self.bridge.theme_received.emit(json.dumps(xterm_theme))

    def close(self):
        self.pty.close()
        super().close()


class TerminalPanel(QWidget):
    """Pure, edge-to-edge interactive terminal pane without top bars or headers."""

    def __init__(self, cwd: Optional[str] = None, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.cwd = cwd or str(Path.cwd())

        root_layout = QVBoxLayout(self)
        root_layout.setContentsMargins(0, 0, 0, 0)
        root_layout.setSpacing(0)

        # Full-bleed, edge-to-edge interactive terminal
        self.terminal = TerminalWidget(cwd=self.cwd, parent=self)
        root_layout.addWidget(self.terminal, 1)

    def apply_theme(self, theme_dict: dict):
        """Apply active theme palette to terminal."""
        self.terminal.apply_theme(theme_dict)

    def closeEvent(self, event):
        self.terminal.close()
        super().closeEvent(event)
