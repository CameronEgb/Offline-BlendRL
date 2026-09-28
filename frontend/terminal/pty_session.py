"""Pseudo-terminal (PTY) session management for interactive shells."""
import codecs
import errno
import fcntl
import os
from pathlib import Path
import pty
import struct
import subprocess
import termios
from typing import Optional

from PyQt6.QtCore import QObject, QSocketNotifier, pyqtSignal


class PtySession(QObject):
    """Manages an interactive shell process bound to a pseudo-terminal."""

    data_ready = pyqtSignal(str)
    process_exited = pyqtSignal(int)

    def __init__(self, cwd: Optional[str] = None, command: Optional[list[str]] = None, parent: Optional[QObject] = None):
        super().__init__(parent)
        self.cwd = str(cwd or Path.cwd())
        self.command = command
        self.master_fd: Optional[int] = None
        self.process: Optional[subprocess.Popen] = None
        self.notifier: Optional[QSocketNotifier] = None
        self.decoder = codecs.getincrementaldecoder("utf-8")(errors="replace")
        self.initial_cols = 80
        self.initial_rows = 24

    def start(self, cols: int = 80, rows: int = 24) -> bool:
        """Start the child shell process inside a new PTY."""
        self.initial_cols = max(10, cols)
        self.initial_rows = max(4, rows)

        try:
            self.master_fd, slave_fd = pty.openpty()
        except OSError:
            return False

        # Set initial PTY dimensions before process launch
        try:
            winsize = struct.pack("HHHH", self.initial_rows, self.initial_cols, 0, 0)
            fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, winsize)
        except OSError:
            pass

        shell = self.command or [os.environ.get("SHELL", "/bin/zsh"), "-l"]

        env = os.environ.copy()
        env["TERM"] = "xterm-256color"
        env["COLORTERM"] = "truecolor"
        env["LANG"] = "en_US.UTF-8"
        env["LC_ALL"] = "en_US.UTF-8"

        try:
            self.process = subprocess.Popen(
                shell,
                stdin=slave_fd,
                stdout=slave_fd,
                stderr=slave_fd,
                cwd=self.cwd,
                env=env,
                preexec_fn=os.setsid,
                close_fds=True,
            )
        except Exception:
            os.close(slave_fd)
            if self.master_fd is not None:
                os.close(self.master_fd)
                self.master_fd = None
            return False

        # Close slave in parent process now that child inherited it
        os.close(slave_fd)

        # Set master_fd non-blocking
        flags = fcntl.fcntl(self.master_fd, fcntl.F_GETFL)
        fcntl.fcntl(self.master_fd, fcntl.F_SETFL, flags | os.O_NONBLOCK)

        # Hook Qt event loop notifier for ready-to-read events
        self.notifier = QSocketNotifier(self.master_fd, QSocketNotifier.Type.Read, self)
        self.notifier.activated.connect(self._on_readable)
        return True

    def _on_readable(self):
        """Read pending bytes from master PTY and emit decoded text."""
        if self.master_fd is None:
            return

        try:
            while True:
                chunk = os.read(self.master_fd, 8192)
                if not chunk:
                    self._handle_exit()
                    return
                text = self.decoder.decode(chunk)
                if text:
                    self.data_ready.emit(text)
        except BlockingIOError:
            # All available bytes read for this cycle
            return
        except OSError as exc:
            if exc.errno in (errno.EIO, errno.EBADF):
                self._handle_exit()
            return

    def _handle_exit(self):
        """Process reached EOF or closed terminal."""
        if self.notifier:
            self.notifier.setEnabled(False)
            self.notifier = None

        exit_code = 0
        if self.process:
            try:
                exit_code = self.process.poll()
                if exit_code is None:
                    exit_code = self.process.wait(timeout=0.2)
            except Exception:
                exit_code = 0

        self.process_exited.emit(exit_code or 0)

    def write(self, data: str):
        """Write string to the master PTY."""
        if self.master_fd is None:
            return
        try:
            encoded = data.encode("utf-8", errors="replace")
            os.write(self.master_fd, encoded)
        except OSError:
            pass

    def resize(self, cols: int, rows: int):
        """Resize terminal window via ioctl TIOCSWINSZ."""
        if self.master_fd is None:
            return
        cols = max(10, cols)
        rows = max(4, rows)
        try:
            winsize = struct.pack("HHHH", rows, cols, 0, 0)
            fcntl.ioctl(self.master_fd, termios.TIOCSWINSZ, winsize)
        except OSError:
            pass

    def close(self):
        """Cleanly terminate child process and close file descriptors."""
        if self.notifier:
            self.notifier.setEnabled(False)
            self.notifier = None

        if self.process and self.process.poll() is None:
            try:
                self.process.terminate()
                self.process.wait(timeout=0.3)
            except Exception:
                try:
                    self.process.kill()
                except Exception:
                    pass

        if self.master_fd is not None:
            try:
                os.close(self.master_fd)
            except OSError:
                pass
            self.master_fd = None

    def is_alive(self) -> bool:
        return self.process is not None and self.process.poll() is None
