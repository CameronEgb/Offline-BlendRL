"""Theta Hub client for fetching community registries, searching, and managing installations."""
from __future__ import annotations
import json
from pathlib import Path
import threading
import urllib.error
import urllib.request
from typing import Dict, List, Optional
from PyQt6.QtCore import QObject, pyqtSignal

from .installer import HubInstaller
from .models import AuthorInfo, HubComponent, ReleaseInfo

DEFAULT_REGISTRY_URL = "https://raw.githubusercontent.com/CameronEgb/theta-hub/main/dist/index.json"

# Built-in fallback index for offline use or initial development
SAMPLE_COMPONENTS = [
    {
        "id": "obsidian-notes",
        "name": "Obsidian Notes",
        "kind": "plugin",
        "version": "0.1.0",
        "description": "Embedded Obsidian vault browser and research markdown notebook linked to NeSyRL experiment runs.",
        "author": {"name": "Cameron Egbert", "github": "CameronEgb"},
        "repository": "https://github.com/CameronEgb/Theta-IDE",
        "tags": ["notes", "obsidian", "markdown", "research"],
        "target_path": "plugins/obsidian",
        "releases": {
            "0.1.0": {
                "tag": "v0.1.0",
                "url": "https://raw.githubusercontent.com/CameronEgb/theta-hub/main/packages/obsidian-0.1.0.zip",
                "sha256": "48c21bb5df26d2158637856320d3da12943e1e4cccb31105d7b4801775a44167",
                "published_at": "2026-09-29T00:00:00Z"
            }
        }
    },
    {
        "id": "wandb-monitor",
        "name": "Weights & Biases Live Sync",
        "kind": "plugin",
        "version": "0.2.1",
        "description": "Streams real-time training losses, rewards, and evaluation metrics directly to W&B dashboards.",
        "author": {"name": "NeSyRL Community", "github": "nesyrl"},
        "repository": "https://github.com/nesyrl/theta-plugin-wandb",
        "tags": ["wandb", "telemetry", "visualization", "monitoring"],
        "target_path": "plugins/wandb",
        "releases": {
            "0.2.1": {
                "tag": "v0.2.1",
                "url": "https://raw.githubusercontent.com/CameronEgb/theta-hub/main/packages/wandb-0.2.1.zip",
                "sha256": "18ef286e692c9757221c472bff6a8747cb7548159d3be55c4e1bb66b1346dc83",
                "published_at": "2026-09-25T12:00:00Z"
            }
        }
    },
    {
        "id": "cql-continuous",
        "name": "Conservative Q-Learning (Continuous)",
        "kind": "method",
        "version": "1.0.0",
        "description": "Offline continuous-action CQL algorithm with automated Lagrange multiplier tuning for out-of-distribution state penalties.",
        "author": {"name": "NeSyRL Lab", "github": "nesyrl"},
        "repository": "https://github.com/nesyrl/theta-method-cql-cont",
        "tags": ["offline-rl", "cql", "continuous", "actor-critic"],
        "target_path": "src/usr/methods/cql_continuous",
        "releases": {
            "1.0.0": {
                "tag": "v1.0.0",
                "url": "https://raw.githubusercontent.com/CameronEgb/theta-hub/main/packages/cql_continuous-1.0.0.zip",
                "sha256": "cc8da4c62655cdb94ece4d63f84888e12eb17ff0a7723617a72b43e9382d6564",
                "published_at": "2026-09-20T10:00:00Z"
            }
        }
    },
    {
        "id": "neumann-fast",
        "name": "Neumann Fast Reasoner",
        "kind": "model",
        "version": "0.3.0",
        "description": "High-throughput vector-matrix NeSy symbolic logic forward reasoner for hybrid RL policies.",
        "author": {"name": "LogicRL Contributors", "github": "logicrl"},
        "repository": "https://github.com/logicrl/neumann-fast",
        "tags": ["logic", "symbolic", "neumann", "hybrid"],
        "target_path": "src/usr/models/neumann_fast",
        "releases": {
            "0.3.0": {
                "tag": "v0.3.0",
                "url": "https://raw.githubusercontent.com/CameronEgb/theta-hub/main/packages/neumann_fast-0.3.0.zip",
                "sha256": "179d06547babbc51baae187fe6b4bb72492c181e8a15b9d032208e9dec7a2599",
                "published_at": "2026-09-18T08:00:00Z"
            }
        }
    },
    {
        "id": "mujoco-ant-maze",
        "name": "MuJoCo AntMaze Navigation",
        "kind": "env",
        "version": "0.1.5",
        "description": "Vectorized D4RL AntMaze benchmark environment wrappers with sparse and shaped reward profiles.",
        "author": {"name": "Gym Contributors", "github": "farama"},
        "repository": "https://github.com/farama/antmaze-wrappers",
        "tags": ["mujoco", "antmaze", "d4rl", "offline-rl"],
        "target_path": "in/envs/antmaze",
        "releases": {
            "0.1.5": {
                "tag": "v0.1.5",
                "url": "https://raw.githubusercontent.com/CameronEgb/theta-hub/main/packages/antmaze-0.1.5.zip",
                "sha256": "a2f5575a23d7b4a604efe0065975cd34d25d2e33b039995b4cef2501c1b16caf",
                "published_at": "2026-09-15T14:00:00Z"
            }
        }
    }
]


class HubClient(QObject):
    """Client for browsing, searching, and installing community hub items."""

    indexLoaded = pyqtSignal(list)
    fetchFailed = pyqtSignal(str)
    installProgress = pyqtSignal(str, int)  # (component_id, percentage)
    installFinished = pyqtSignal(str, bool, str)  # (component_id, success, message)
    uninstallFinished = pyqtSignal(str, bool, str)  # (component_id, success, message)
    componentChanged = pyqtSignal(str, str)  # (component_id, action: "install" | "uninstall")

    def __init__(
        self,
        workspace_dir: Path,
        data_dir: Path,
        registry_url: str = DEFAULT_REGISTRY_URL,
        on_change_callback=None,
    ):
        super().__init__()
        self.workspace_dir = Path(workspace_dir)
        self.data_dir = Path(data_dir)
        self.registry_url = registry_url

        self.cache_dir = self.data_dir / "cache" / "hub"
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.cache_file = self.cache_dir / "index.json"
        self.meta_file = self.cache_dir / "meta.json"

        self.installer = HubInstaller(
            workspace_dir=self.workspace_dir,
            data_dir=self.data_dir,
            on_change_callback=self._handle_installer_change,
        )
        if on_change_callback:
            self.componentChanged.connect(on_change_callback)

        self.components: List[HubComponent] = []

    def _handle_installer_change(self, component_id: str, action: str):
        self.componentChanged.emit(component_id, action)

    def refresh_installed_status(self) -> None:
        """Update each loaded component with its local installation state."""
        for comp in self.components:
            is_installed, ver = self.installer.check_installed(comp)
            comp.is_installed = is_installed
            comp.installed_version = ver

    def fetch_index_sync(self, force: bool = False) -> List[HubComponent]:
        """Fetch index synchronously using HTTP conditional caching (ETag)."""
        headers = {"User-Agent": "ThetaIDE-HubClient/1.0"}
        etag = None

        if not force and self.meta_file.exists():
            try:
                meta = json.loads(self.meta_file.read_text(encoding="utf-8"))
                etag = meta.get("etag")
                if etag:
                    headers["If-None-Match"] = etag
            except Exception:
                pass

        data = None

        if self.registry_url.startswith("file://"):
            local_file = Path(self.registry_url[7:])
            if local_file.exists():
                data = json.loads(local_file.read_text(encoding="utf-8"))
        else:
            candidates = [self.registry_url]
            for alt in [
                "https://raw.githubusercontent.com/CameronEgb/theta-hub/main/dist/index.json",
                "https://cameronegb.github.io/theta-hub/index.json",
                "https://cdn.jsdelivr.net/gh/CameronEgb/theta-hub@main/dist/index.json",
            ]:
                if alt not in candidates:
                    candidates.append(alt)

            for target_url in candidates:
                try:
                    req = urllib.request.Request(target_url, headers=headers)
                    with urllib.request.urlopen(req, timeout=8) as response:
                        raw_body = response.read().decode("utf-8")
                        data = json.loads(raw_body)

                        # Update local cache and etag
                        self.cache_file.write_text(raw_body, encoding="utf-8")
                        new_etag = response.headers.get("ETag")
                        if new_etag:
                            self.meta_file.write_text(
                                json.dumps({"etag": new_etag}, indent=2) + "\n",
                                encoding="utf-8",
                            )
                        break
                except urllib.error.HTTPError as http_err:
                    if http_err.code == 304 and self.cache_file.exists():
                        # Cache is fresh
                        data = json.loads(self.cache_file.read_text(encoding="utf-8"))
                        break
                except Exception:
                    continue

        # Fallback to cached copy if network request failed
        if data is None and self.cache_file.exists():
            try:
                data = json.loads(self.cache_file.read_text(encoding="utf-8"))
            except Exception:
                data = None

        # Ultimate fallback: Sample community feed
        if data is None:
            data = {"components": SAMPLE_COMPONENTS}

        raw_list = data.get("components", data if isinstance(data, list) else [])
        components = [HubComponent.from_dict(item) for item in raw_list]
        self.components = components
        self.refresh_installed_status()
        return self.components

    def fetch_index_async(self, force: bool = False) -> None:
        """Fetch registry asynchronously in a background thread."""
        def worker():
            try:
                comps = self.fetch_index_sync(force=force)
                self.indexLoaded.emit(comps)
            except Exception as exc:
                self.fetchFailed.emit(str(exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    @property
    def registry(self) -> Dict[str, HubComponent]:
        """Dictionary lookup of components by ID."""
        return {comp.id: comp for comp in self.components}

    @registry.setter
    def registry(self, mapping: Dict[str, HubComponent]) -> None:
        self.components = list(mapping.values())

    def get_component(self, component_id: str) -> Optional[HubComponent]:
        """Find a component by its unique ID, target path, or alias."""
        cid = component_id.lower()
        cid_norm = cid.replace("-", "_")
        for comp in self.components:
            comp_id = comp.id.lower()
            if comp_id == cid or comp_id.replace("-", "_") == cid_norm:
                return comp
            if comp.target_path:
                tname = Path(comp.target_path).name.lower()
                if tname == cid or tname.replace("-", "_") == cid_norm:
                    return comp
            if comp.id.startswith(cid) or cid.startswith(comp.id):
                return comp
        return None

    def search(
        self,
        query: str = "",
        kind: Optional[str] = None,
        tag: Optional[str] = None,
    ) -> List[HubComponent]:
        """Filter components by search query, component kind, or tag."""
        q = query.strip().lower()
        results = []

        for comp in self.components:
            # Kind filter
            if kind and kind.lower() != "all" and comp.kind.lower() != kind.lower():
                continue

            # Tag filter
            if tag and tag.lower() not in [t.lower() for t in comp.tags]:
                continue

            # Query filter (matches name, description, tags, author, or id)
            if q:
                match_id = q in comp.id.lower()
                match_name = q in comp.name.lower()
                match_desc = q in comp.description.lower()
                match_author = q in comp.author.name.lower()
                match_tags = any(q in t.lower() for t in comp.tags)
                if not (match_id or match_name or match_desc or match_author or match_tags):
                    continue

            results.append(comp)

        return results

    def install_async(self, component: HubComponent, version: Optional[str] = None) -> None:
        """Install component in a background thread with progress notifications."""
        def progress_cb(downloaded: int, total: int):
            if total > 0:
                pct = int((downloaded / total) * 100)
                self.installProgress.emit(component.id, pct)

        def worker():
            try:
                self.installer.install(
                    component=component,
                    version=version,
                    progress_callback=progress_cb,
                )
                self.refresh_installed_status()
                self.installFinished.emit(component.id, True, "Installation complete.")
            except Exception as exc:
                self.installFinished.emit(component.id, False, str(exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()

    def uninstall_async(self, component: HubComponent) -> None:
        """Uninstall component in a background thread."""
        def worker():
            try:
                self.installer.uninstall(component)
                self.refresh_installed_status()
                self.uninstallFinished.emit(component.id, True, "Uninstalled.")
            except Exception as exc:
                self.uninstallFinished.emit(component.id, False, str(exc))

        thread = threading.Thread(target=worker, daemon=True)
        thread.start()
