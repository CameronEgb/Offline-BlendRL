"""Plugin discovery, lifecycle management, and persistence for ThetaIDE."""
from __future__ import annotations
import importlib
import importlib.util
import json
from pathlib import Path
import sys
from typing import TYPE_CHECKING, Dict, List, Optional
from PyQt6.QtCore import QObject, pyqtSignal

from .base import Plugin, PluginManifest
from .context import PluginContext

if TYPE_CHECKING:
    from ..app import Window
    from ..settings import SettingsManager


class PluginManager(QObject):
    """Manages discovery, activation, deactivation, and settings for plugins."""

    pluginStateChanged = pyqtSignal(str, bool)  # (plugin_id, enabled)

    def __init__(self, window: Window, plugin_dirs: Optional[List[Path]] = None,
                 settings_manager: Optional["SettingsManager"] = None):
        super().__init__()
        self.window = window
        self.plugin_dirs = plugin_dirs or [Path(__file__).parent]
        self._settings = settings_manager
        store = getattr(window, "store", None)
        root_dir = getattr(store, "root", getattr(store, "data_dir", Path("."))) if store else Path(".")
        # Legacy file path — used only for one-time migration reads
        self.state_file = Path(root_dir) / ".plugins.json"

        self.manifests: Dict[str, PluginManifest] = {}
        self.instances: Dict[str, Plugin] = {}
        self.contexts: Dict[str, PluginContext] = {}
        self.enabled_states: Dict[str, bool] = {}

    def discover(self) -> None:
        """Scan configured plugin directories for plugin.json manifests."""
        self._load_states()
        self.manifests.clear()
        for pdir in self.plugin_dirs:
            if not pdir.exists():
                continue
            for manifest_path in pdir.glob("*/plugin.json"):
                try:
                    data = json.loads(manifest_path.read_text(encoding="utf-8"))
                    if data.get("kind", "plugin") != "plugin":
                        continue
                    pid = data["id"]
                    manifest = PluginManifest(
                        id=pid,
                        name=data.get("name", pid),
                        version=data.get("version", "0.1.0"),
                        description=data.get("description", ""),
                        author=data.get("author", "ThetaIDE Team"),
                        default_enabled=data.get("default_enabled", False),
                        icon=data.get("icon"),
                        entry_point=data.get("entry_point", "Plugin"),
                        plugin_dir=manifest_path.parent,
                        extra=data,
                    )
                    self.manifests[pid] = manifest
                except Exception as exc:
                    self.window.log(f"Failed to load plugin manifest at {manifest_path}: {exc}")

        # Plugins discovered for the first time default to their manifest setting
        for pid, manifest in self.manifests.items():
            if pid not in self.enabled_states:
                self.enabled_states[pid] = bool(manifest.default_enabled)

    def _load_states(self) -> None:
        """Load which plugins are enabled from settings.toml (preferred) or
        the legacy .plugins.json (one-time migration fallback)."""
        if self._settings is not None:
            enabled_list = self._settings.plugins_enabled
            self.enabled_states = {pid: True for pid in enabled_list}
            # One-time migration: pull any extra states from .plugins.json
            self._merge_legacy_states()
        else:
            self._load_legacy_states()

    def _load_legacy_states(self) -> None:
        """Read enabled states from the legacy .plugins.json."""
        saved: dict = {}
        if self.state_file.exists():
            try:
                saved = json.loads(self.state_file.read_text(encoding="utf-8"))
            except Exception as exc:
                self.window.log(f"Failed to read plugin state file: {exc}")
        saved_enabled = saved.get("enabled", saved) if isinstance(saved, dict) else {}
        self.enabled_states = {
            k: bool(v) for k, v in saved_enabled.items()
            if k not in ("uninstalled", "enabled")
        }

    def _merge_legacy_states(self) -> None:
        """Pull any extra enabled/disabled states from .plugins.json not yet in settings."""
        if not self.state_file.exists():
            return
        try:
            saved = json.loads(self.state_file.read_text(encoding="utf-8"))
        except Exception:
            return
        saved_enabled = saved.get("enabled", saved) if isinstance(saved, dict) else {}
        for k, v in saved_enabled.items():
            if k not in ("uninstalled", "enabled") and k not in self.enabled_states:
                self.enabled_states[k] = bool(v)

    def save_states(self) -> None:
        """Persist which plugins are enabled to settings.toml (or legacy .plugins.json)."""
        enabled_list = sorted(pid for pid, on in self.enabled_states.items() if on)
        if self._settings is not None:
            try:
                self._settings.set("plugins", "enabled", enabled_list)
            except Exception as exc:
                self.window.log(f"Failed to save plugin states to settings.toml: {exc}")
        else:
            try:
                self.state_file.parent.mkdir(parents=True, exist_ok=True)
                payload: dict = {"enabled": dict(self.enabled_states)}
                self.state_file.write_text(json.dumps(payload, indent=2) + "\n", encoding="utf-8")
            except OSError as exc:
                self.window.log(f"Failed to save plugin states: {exc}")

    def mark_uninstalled(self, plugin_id: str) -> None:
        """Uninstall a plugin: deactivate it and delete its directory from disk.

        After deletion the plugin won't be discovered on the next startup.
        No blocklist needed — if it's not on disk, it doesn't exist.
        """
        aliases = {
            plugin_id,
            plugin_id.replace("-", "_"),
            plugin_id.replace("_", "-"),
        }
        for suffix in ("-notes", "_notes", "-sim", "_sim", "-plugin", "_plugin"):
            if plugin_id.endswith(suffix):
                aliases.add(plugin_id[:-len(suffix)])
            else:
                aliases.add(f"{plugin_id}{suffix}")

        for alias in aliases:
            self.disable_plugin(alias)
            self.manifests.pop(alias, None)
            self.enabled_states.pop(alias, None)

        store = getattr(self.window, "store", None)
        root_dir = getattr(store, "root", getattr(store, "data_dir", None)) if store else None
        if root_dir:
            import shutil
            data_plugins = Path(root_dir) / "plugins"
            for alias in aliases:
                target = data_plugins / alias
                if target.exists():
                    shutil.rmtree(target, ignore_errors=True)
        self.save_states()



    def initialize_plugins(self) -> None:
        """Activate plugins that are currently enabled in the saved state."""
        for pid, is_enabled in list(self.enabled_states.items()):
            if is_enabled:
                self.enable_plugin(pid)

    def is_plugin_enabled(self, plugin_id: str) -> bool:
        """Check if a plugin is currently enabled."""
        return self.enabled_states.get(plugin_id, False)

    def enable_plugin(self, plugin_id: str) -> bool:
        """Dynamically activate and mount a plugin."""
        if plugin_id not in self.manifests:
            self.window.log(f"Cannot enable unknown plugin: {plugin_id}")
            return False

        if plugin_id in self.instances:
            return True  # Already active

        manifest = self.manifests[plugin_id]
        plugin_dir = manifest.plugin_dir

        try:
            # Load the plugin module
            try:
                module = importlib.import_module(f"frontend.plugins.{plugin_id}")
            except (ImportError, ModuleNotFoundError):
                init_file = plugin_dir / "__init__.py" if plugin_dir else None
                if init_file and init_file.exists():
                    mod_name = f"theta_plugin_{plugin_id}"
                    spec = importlib.util.spec_from_file_location(
                        mod_name, init_file, submodule_search_locations=[str(plugin_dir)]
                    )
                    if spec is None or spec.loader is None:
                        raise ImportError(f"Cannot load spec from {init_file}")
                    module = importlib.util.module_from_spec(spec)
                    module.__path__ = [str(plugin_dir)]
                    sys.modules[mod_name] = module
                    sys.modules[f"frontend.plugins.{plugin_id}"] = module
                    spec.loader.exec_module(module)
                else:
                    raise

            plugin_cls = getattr(module, manifest.entry_point)
            instance: Plugin = plugin_cls(manifest)

            root_dir = getattr(self.window.store, "root", getattr(self.window.store, "data_dir", "."))
            storage_file = (
                Path(root_dir) / "plugins" / f"{plugin_id}.json"
            )
            context = PluginContext(plugin_id, self.window, storage_file)

            instance.activate(context)

            self.instances[plugin_id] = instance
            self.contexts[plugin_id] = context
            self.enabled_states[plugin_id] = True
            self.save_states()
            self.pluginStateChanged.emit(plugin_id, True)
            self.window.log(f"Plugin activated: {manifest.name} ({plugin_id})")
            return True

        except Exception as exc:
            self.window.log(f"Failed to activate plugin {plugin_id}: {exc}")
            import traceback
            traceback.print_exc()
            return False

    def disable_plugin(self, plugin_id: str) -> bool:
        """Dynamically deactivate and unmount a plugin."""
        if plugin_id not in self.manifests:
            return False

        instance = self.instances.pop(plugin_id, None)
        context = self.contexts.pop(plugin_id, None)

        if instance:
            try:
                instance.deactivate()
            except Exception as exc:
                self.window.log(f"Error during deactivation of plugin {plugin_id}: {exc}")

        if context:
            try:
                context.cleanup()
            except Exception as exc:
                self.window.log(f"Error cleaning up context for {plugin_id}: {exc}")

        self.enabled_states[plugin_id] = False
        self.save_states()
        self.pluginStateChanged.emit(plugin_id, False)
        self.window.log(f"Plugin deactivated: {self.manifests[plugin_id].name} ({plugin_id})")
        return True

    def notify_experiment_changed(
        self, exp_name: Optional[str], file_path: Optional[str]
    ) -> None:
        """Dispatch experiment changes to all active plugin contexts."""
        for context in self.contexts.values():
            context.notify_experiment_changed(exp_name, file_path)
