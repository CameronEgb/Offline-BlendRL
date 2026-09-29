"""Unit and integration tests for ThetaIDE Plugin System."""
import json
import os
import shutil
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.app import Window
    from frontend.plugins import Plugin, PluginContext, PluginManager, PluginManifest
    from frontend.plugins.obsidian import ObsidianPlugin
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestPluginSystem(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="theta_test_plugins_")
        self.window = Window(data_dir=self.temp_dir)

    def tearDown(self):
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_obsidian_plugin_discovered(self):
        """Obsidian plugin is discovered and parsed with correct metadata."""
        manager = self.window.plugin_manager
        self.assertIn("obsidian", manager.manifests)
        manifest = manager.manifests["obsidian"]
        self.assertEqual(manifest.id, "obsidian")
        self.assertEqual(manifest.name, "Obsidian Notes")
        self.assertFalse(manifest.default_enabled)

    def test_obsidian_disabled_on_initialization(self):
        """Plugin must be disabled by default on clean initialization."""
        manager = self.window.plugin_manager
        # Check manager state
        self.assertFalse(manager.is_plugin_enabled("obsidian"))
        self.assertNotIn("obsidian", manager.instances)

        # Check UI sidebar tabs: obsidian must NOT be in tabs
        self.assertNotIn("obsidian", self.window.tabs.tabs)
        self.assertNotIn("obsidian", self.window.tabs.tab_order)

        # Check toggle slider in settings
        if "obsidian" in self.window.plugin_sliders:
            self.assertFalse(self.window.plugin_sliders["obsidian"].isChecked())

    def test_enable_and_disable_plugin_lifecycle(self):
        """Enabling dynamically mounts the sidebar tab; disabling unmounts it cleanly."""
        manager = self.window.plugin_manager

        # 1. Enable plugin
        success = manager.enable_plugin("obsidian")
        self.assertTrue(success)
        self.assertTrue(manager.is_plugin_enabled("obsidian"))
        self.assertIn("obsidian", manager.instances)
        self.assertIsInstance(manager.instances["obsidian"], ObsidianPlugin)

        # Tab should now be present in SideTabs
        self.assertIn("obsidian", self.window.tabs.tabs)
        self.assertIn("obsidian", self.window.tabs.tab_order)

        # 2. Disable plugin
        success = manager.disable_plugin("obsidian")
        self.assertTrue(success)
        self.assertFalse(manager.is_plugin_enabled("obsidian"))
        self.assertNotIn("obsidian", manager.instances)

        # Tab should be cleanly removed from SideTabs
        self.assertNotIn("obsidian", self.window.tabs.tabs)
        self.assertNotIn("obsidian", self.window.tabs.tab_order)

    def test_persistence_of_plugin_state(self):
        """State is persisted to .plugins.json and restored on reload."""
        manager = self.window.plugin_manager
        state_file = Path(self.temp_dir) / ".plugins.json"

        # Initially saved as disabled
        self.assertFalse(manager.is_plugin_enabled("obsidian"))

        # Enable and verify disk write
        manager.enable_plugin("obsidian")
        self.assertTrue(state_file.exists())
        saved_data = json.loads(state_file.read_text(encoding="utf-8"))
        self.assertTrue(saved_data.get("obsidian"))

        # Simulate new Window instance loading same data directory
        new_window = Window(data_dir=self.temp_dir)
        self.assertTrue(new_window.plugin_manager.is_plugin_enabled("obsidian"))
        self.assertIn("obsidian", new_window.tabs.tabs)

        # Cleanup new_window
        new_window.plugin_manager.disable_plugin("obsidian")

    def test_settings_toggle_slider_interaction(self):
        """Toggling the slider in the Settings card enables/disables the plugin."""
        self.assertIn("obsidian", self.window.plugin_sliders)
        slider = self.window.plugin_sliders["obsidian"]
        self.assertFalse(slider.isChecked())

        # Toggle to True
        slider.click()
        self.assertTrue(slider.isChecked())
        self.assertTrue(self.window.plugin_manager.is_plugin_enabled("obsidian"))
        self.assertIn("obsidian", self.window.tabs.tabs)

        # Toggle back to False
        slider.click()
        self.assertFalse(slider.isChecked())
        self.assertFalse(self.window.plugin_manager.is_plugin_enabled("obsidian"))
        self.assertNotIn("obsidian", self.window.tabs.tabs)

    def test_uninstalled_plugin_lifecycle_and_disappearance(self):
        """Uninstalling a plugin marks it uninstalled and removes it from the Settings menu."""
        manager = self.window.plugin_manager
        self.assertIn("obsidian", manager.manifests)
        self.assertIn("obsidian", self.window.plugin_sliders)

        # Mark uninstalled
        manager.mark_uninstalled("obsidian")
        self.assertNotIn("obsidian", manager.manifests)
        self.assertIn("obsidian", manager.uninstalled_ids)

        # UI refresh removes it from settings menu
        self.window.refresh_plugins_ui()
        self.assertNotIn("obsidian", self.window.plugin_sliders)

        # Rescanning discovery does not resurrect it
        manager.discover()
        self.assertNotIn("obsidian", manager.manifests)

        # Re-marking as installed restores it
        manager.unmark_uninstalled("obsidian")
        manager.discover()
        self.assertIn("obsidian", manager.manifests)
        self.window.refresh_plugins_ui()
        self.assertIn("obsidian", self.window.plugin_sliders)

        # Re-test via Hub component changed with alias 'obsidian-notes'
        manager.mark_uninstalled("obsidian")
        self.window.refresh_plugins_ui()
        self.assertNotIn("obsidian", self.window.plugin_sliders)

        self.window._on_hub_component_changed("obsidian-notes", "install")
        self.assertIn("obsidian", manager.manifests)
        self.assertIn("obsidian", self.window.plugin_sliders)

    def test_plugins_menu_only_shows_plugins_not_other_components(self):
        """Settings plugin menu strictly ignores components whose kind != 'plugin'."""
        data_plugins = Path(self.temp_dir) / "plugins"
        data_plugins.mkdir(parents=True, exist_ok=True)

        # Create a non-plugin component in plugin search path
        method_dir = data_plugins / "cql_algo"
        method_dir.mkdir(parents=True, exist_ok=True)
        (method_dir / "plugin.json").write_text(json.dumps({
            "id": "cql_algo",
            "name": "CQL Algorithm",
            "kind": "method",
            "version": "1.0.0"
        }), encoding="utf-8")

        self.window.plugin_manager.discover()
        self.assertNotIn("cql_algo", self.window.plugin_manager.manifests)

        self.window.refresh_plugins_ui()
        self.assertNotIn("cql_algo", self.window.plugin_sliders)

    def test_hub_dialog_sort_uninstalled_first(self):
        """HubDialog can sort components such that uninstalled items appear first."""
        from frontend.hub.models import HubComponent
        from frontend.hub.dialog import HubDialog

        client = self.window.hub_client
        c1 = HubComponent(id="comp_inst", name="Alpha Installed", kind="plugin", version="1.0.0", is_installed=True)
        c2 = HubComponent(id="comp_uninst", name="Beta Uninstalled", kind="plugin", version="1.0.0", is_installed=False)
        client.components = [c1, c2]
        client.registry = {"comp_inst": c1, "comp_uninst": c2}

        dialog = HubDialog(client, parent=self.window)
        dialog._populate_cards([c1, c2])

        # Sort by uninstalled first
        idx = dialog.sort_combo.findData("uninstalled_first")
        self.assertGreaterEqual(idx, 0)
        dialog.sort_combo.setCurrentIndex(idx)

        # Search / filter run
        dialog._apply_filters()
        # Card for comp_uninst should be positioned before comp_inst in card layout
        idx_uninst = dialog.card_layout.indexOf(dialog.cards["comp_uninst"])
        idx_inst = dialog.card_layout.indexOf(dialog.cards["comp_inst"])
        self.assertLess(idx_uninst, idx_inst)
        dialog.close()

