"""Component installer, archive extractor, and integrity verification."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import shutil
import tempfile
import urllib.request
from typing import Callable, Optional, Tuple
from zipfile import ZipFile

from .models import HubComponent, ReleaseInfo


class HubInstaller:
    """Manages the download, verification, extraction, and removal of Hub components."""

    def __init__(self, workspace_dir: Path, data_dir: Path, on_change_callback: Optional[Callable[[str, str], None]] = None):
        self.workspace_dir = Path(workspace_dir)
        self.data_dir = Path(data_dir)
        self.on_change_callback = on_change_callback  # (component_id, action: "install" | "uninstall")

    def resolve_target_dir(self, component: HubComponent) -> Path:
        """Determine the filesystem destination for a component based on its kind."""
        if component.target_path:
            # If target_path is relative, determine root
            if component.kind == "plugin":
                return self.data_dir / component.target_path
            return self.workspace_dir / component.target_path

        # Standard conventions
        if component.kind == "plugin":
            return self.data_dir / "plugins" / component.id
        elif component.kind == "method":
            return self.workspace_dir / "src" / "usr" / "methods" / component.id
        elif component.kind == "model":
            return self.workspace_dir / "src" / "usr" / "models" / component.id
        elif component.kind == "env":
            return self.workspace_dir / "in" / "envs" / component.id
        elif component.kind == "experiment":
            return self.workspace_dir / "in" / "config" / "experiment" / component.id
        else:
            return self.workspace_dir / "components" / component.id

    def check_installed(self, component: HubComponent) -> Tuple[bool, Optional[str]]:
        """Check if a component is installed and determine its version."""
        target_dir = self.resolve_target_dir(component)
        if not target_dir.exists() or not target_dir.is_dir():
            return False, None

        # 1. Check .theta_component.json manifest
        meta_file = target_dir / ".theta_component.json"
        if meta_file.exists():
            try:
                data = json.loads(meta_file.read_text(encoding="utf-8"))
                return True, data.get("version")
            except Exception:
                pass

        # 2. Check plugin.json manifest
        plugin_file = target_dir / "plugin.json"
        if plugin_file.exists():
            try:
                data = json.loads(plugin_file.read_text(encoding="utf-8"))
                return True, data.get("version")
            except Exception:
                pass

        return True, "unknown"

    def download_and_verify(
        self,
        url: str,
        expected_sha256: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> Path:
        """Download an archive and verify its cryptographic SHA-256 checksum."""
        tmp_fd, tmp_path_str = tempfile.mkstemp(prefix="theta_pkg_", suffix=".zip")
        tmp_path = Path(tmp_path_str)

        hasher = hashlib.sha256()
        total_downloaded = 0

        try:
            if url.startswith("file://"):
                local_source = Path(url[7:])
                content = local_source.read_bytes()
                hasher.update(content)
                tmp_path.write_bytes(content)
                total_downloaded = len(content)
                if progress_callback:
                    progress_callback(total_downloaded, total_downloaded)
            else:
                req = urllib.request.Request(
                    url,
                    headers={"User-Agent": "ThetaIDE-HubClient/1.0"}
                )
                with urllib.request.urlopen(req, timeout=30) as response, open(tmp_path, "wb") as f:
                    content_length = response.headers.get("Content-Length")
                    total_size = int(content_length) if content_length else -1

                    chunk_size = 64 * 1024
                    while True:
                        chunk = response.read(chunk_size)
                        if not chunk:
                            break
                        f.write(chunk)
                        hasher.update(chunk)
                        total_downloaded += len(chunk)
                        if progress_callback:
                            progress_callback(total_downloaded, total_size)

            computed_sha = hasher.hexdigest().lower()
            if expected_sha256:
                expected_clean = expected_sha256.strip().lower()
                if computed_sha != expected_clean:
                    raise ValueError(
                        f"Checksum verification failed!\n"
                        f"Expected SHA-256: {expected_clean}\n"
                        f"Computed SHA-256: {computed_sha}"
                    )

            return tmp_path

        except Exception:
            if tmp_path.exists():
                tmp_path.unlink()
            raise

    def install(
        self,
        component: HubComponent,
        version: Optional[str] = None,
        progress_callback: Optional[Callable[[int, int], None]] = None,
    ) -> bool:
        """Download, verify, and unpack a component release into its destination directory."""
        ver = version or component.version
        if ver not in component.releases:
            raise ValueError(f"Version {ver} not found in releases for {component.id}")

        release = component.releases[ver]
        archive_path = self.download_and_verify(
            url=release.url,
            expected_sha256=release.sha256,
            progress_callback=progress_callback,
        )

        target_dir = self.resolve_target_dir(component)
        staging_dir = Path(tempfile.mkdtemp(prefix="theta_staging_"))

        try:
            with ZipFile(archive_path, "r") as zf:
                zf.extractall(staging_dir)

            # Handle case where zip contains a single enclosing root directory
            extracted_items = list(staging_dir.iterdir())
            if len(extracted_items) == 1 and extracted_items[0].is_dir():
                source_dir = extracted_items[0]
            else:
                source_dir = staging_dir

            # Write component tracking metadata
            meta = {
                "id": component.id,
                "name": component.name,
                "kind": component.kind,
                "version": ver,
                "installed_from": release.url,
                "sha256": release.sha256,
            }
            (source_dir / ".theta_component.json").write_text(
                json.dumps(meta, indent=2) + "\n", encoding="utf-8"
            )

            # Atomic swap into target_dir
            if target_dir.exists():
                backup_dir = target_dir.with_name(f"{target_dir.name}.backup")
                if backup_dir.exists():
                    shutil.rmtree(backup_dir, ignore_errors=True)
                target_dir.rename(backup_dir)
                try:
                    shutil.move(str(source_dir), str(target_dir))
                    shutil.rmtree(backup_dir, ignore_errors=True)
                except Exception:
                    # Rollback
                    if target_dir.exists():
                        shutil.rmtree(target_dir, ignore_errors=True)
                    backup_dir.rename(target_dir)
                    raise
            else:
                target_dir.parent.mkdir(parents=True, exist_ok=True)
                shutil.move(str(source_dir), str(target_dir))

            # Update component object state
            component.is_installed = True
            component.installed_version = ver

            if self.on_change_callback:
                self.on_change_callback(component.id, "install")
                plugin_file = target_dir / "plugin.json"
                if plugin_file.exists():
                    try:
                        pdata = json.loads(plugin_file.read_text(encoding="utf-8"))
                        pid = pdata.get("id")
                        if pid and pid != component.id:
                            self.on_change_callback(pid, "install")
                    except Exception:
                        pass
                if component.target_path:
                    tname = Path(component.target_path).name
                    if tname != component.id:
                        self.on_change_callback(tname, "install")

            return True

        finally:
            if archive_path.exists():
                archive_path.unlink()
            if staging_dir.exists():
                shutil.rmtree(staging_dir, ignore_errors=True)

    def uninstall(self, component: HubComponent) -> bool:
        """Remove an installed component from disk."""
        target_dir = self.resolve_target_dir(component)
        if not target_dir.exists():
            component.is_installed = False
            component.installed_version = None
            if self.on_change_callback:
                self.on_change_callback(component.id, "uninstall")
                if component.target_path:
                    tname = Path(component.target_path).name
                    if tname != component.id:
                        self.on_change_callback(tname, "uninstall")
            return True

        try:
            shutil.rmtree(target_dir)
            component.is_installed = False
            component.installed_version = None

            if self.on_change_callback:
                self.on_change_callback(component.id, "uninstall")
                if component.target_path:
                    tname = Path(component.target_path).name
                    if tname != component.id:
                        self.on_change_callback(tname, "uninstall")

            return True
        except Exception as exc:
            raise OSError(f"Failed to remove {target_dir}: {exc}")
