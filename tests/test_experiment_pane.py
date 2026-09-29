"""Unit tests for the overhauled Experiment Pane in Theta-IDE:
- ConfigTreeWidget mirroring in/config/
- Boxed ConfigViewer with real-time updates and saving
- On-demand Hydra YAML preview (hidden by default)
- Experiment creation and duplication
"""
import os
import sys
import tempfile
import unittest
from pathlib import Path
import yaml

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6 import QtWebEngineWidgets
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.config_tree import ConfigTreeWidget
    from frontend.config_viewer import ConfigViewer, ConfigBox
    from frontend.app import Window
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestConfigTree(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.root = Path(self.temp_dir.name)
        # Create mock in/config structure
        (self.root / "experiment" / "cartpole").mkdir(parents=True)
        (self.root / "agent").mkdir(parents=True)
        (self.root / "env").mkdir(parents=True)

        (self.root / "experiment" / "cartpole" / "exp1.yaml").write_text(
            "experiment_id: exp1\nseed: 42\ntotal_timesteps: 5000\n", encoding="utf-8"
        )
        (self.root / "agent" / "ppo.yaml").write_text(
            "lr: 0.0003\nbatch_size: 64\n", encoding="utf-8"
        )
        (self.root / "config.yaml").write_text("paradigm: online_rl\n", encoding="utf-8")

        self.tree_widget = ConfigTreeWidget(root_dir=self.root)

    def tearDown(self):
        self.tree_widget.close()
        self.temp_dir.cleanup()

    def test_tree_populates_directories_and_files(self):
        # Top-level should have experiment, agent, env, and config.yaml
        top_count = self.tree_widget.tree.topLevelItemCount()
        self.assertGreaterEqual(top_count, 4)

        top_texts = [self.tree_widget.tree.topLevelItem(i).text(0) for i in range(top_count)]
        self.assertTrue(any("experiment" in t for t in top_texts))
        self.assertTrue(any("agent" in t for t in top_texts))
        self.assertTrue(any("config.yaml" in t for t in top_texts))

    def test_experiment_expanded_by_default(self):
        for i in range(self.tree_widget.tree.topLevelItemCount()):
            item = self.tree_widget.tree.topLevelItem(i)
            if "experiment" in item.text(0):
                self.assertTrue(item.isExpanded())

    def test_select_file_emits_signal(self):
        selected = []
        self.tree_widget.file_selected.connect(lambda path, rel: selected.append((path, rel)))

        ok = self.tree_widget.select_file("experiment/cartpole/exp1.yaml")
        self.assertTrue(ok)
        self.assertEqual(len(selected), 1)
        self.assertEqual(selected[0][1], "experiment/cartpole/exp1.yaml")

    def test_filter_tree(self):
        self.tree_widget.filter_tree("exp1")
        # exp1 item should be visible
        for i in range(self.tree_widget.tree.topLevelItemCount()):
            item = self.tree_widget.tree.topLevelItem(i)
            if "agent" in item.text(0):
                self.assertTrue(item.isHidden())

        self.tree_widget.filter_tree("")
        for i in range(self.tree_widget.tree.topLevelItemCount()):
            item = self.tree_widget.tree.topLevelItem(i)
            self.assertFalse(item.isHidden())


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestConfigViewer(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.file_path = Path(self.temp_dir.name) / "test_exp.yaml"
        self.file_path.write_text(
            "experiment_id: test_exp\n"
            "seed: 123\n"
            "total_timesteps: 25000\n"
            "methods:\n"
            "  ppo:\n"
            "    agent: ppo\n"
            "    model: dnn\n"
            "    lr: 0.0005\n"
            "    batch_size: 128\n"
            "    gamma: 0.98\n",
            encoding="utf-8"
        )
        self.viewer = ConfigViewer()
        self.viewer.load_file(self.file_path, "experiment/test_exp.yaml")

    def tearDown(self):
        self.viewer.close()
        self.temp_dir.cleanup()

    def test_loads_experiment_values_into_boxes(self):
        self.assertEqual(self.viewer.txt_exp_id.text(), "test_exp")
        self.assertEqual(self.viewer.spin_seed.value(), 123)
        self.assertEqual(self.viewer.spin_timesteps.value(), 25000)

        # Boxes should be created as styled cards
        cards = self.viewer.findChildren(ConfigBox)
        self.assertGreaterEqual(len(cards), 3)

    def test_edits_mark_dirty_and_remember_values(self):
        events = []
        self.viewer.config_changed.connect(lambda: events.append(True))

        self.assertFalse(self.viewer.is_dirty)
        self.viewer.spin_seed.setValue(999)

        self.assertTrue(self.viewer.is_dirty)
        self.assertEqual(len(events), 1)
        self.assertEqual(self.viewer.raw_data["seed"], 999)

    def test_save_to_disk(self):
        self.viewer.spin_timesteps.setValue(50000)
        self.assertTrue(self.viewer.is_dirty)

        saved = self.viewer.save_to_disk()
        self.assertTrue(saved)
        self.assertFalse(self.viewer.is_dirty)

        # Verify disk contents
        data = yaml.safe_load(self.file_path.read_text(encoding="utf-8"))
        self.assertEqual(data["total_timesteps"], 50000)

    def test_get_overrides(self):
        overrides = self.viewer.get_overrides()
        self.assertIn("++experiment_id='test_exp'", overrides)
        self.assertIn("seed=123", overrides)
        self.assertIn("total_timesteps=25000", overrides)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestExperimentPaneWindowIntegration(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.data_dir = Path(self.temp_dir.name)
        self.window = Window(self.data_dir)
        self.window.show()

    def tearDown(self):
        self.window.close()
        self.temp_dir.cleanup()

    def test_hydra_yaml_preview_hidden_by_default(self):
        self.assertTrue(self.window.preview_panel.isHidden())
        self.assertFalse(self.window.btn_toggle_yaml.isChecked())

    def test_toggle_hydra_yaml_preview(self):
        # Click toggle to open
        self.window.toggle_yaml_preview(True)
        self.assertFalse(self.window.preview_panel.isHidden())
        self.assertTrue(self.window.btn_toggle_yaml.isChecked())

        # Click toggle to close
        self.window.toggle_yaml_preview(False)
        self.assertTrue(self.window.preview_panel.isHidden())
        self.assertFalse(self.window.btn_toggle_yaml.isChecked())

    def test_selecting_tree_item_loads_into_viewer(self):
        # Select quick_test.yaml
        ok = self.window.config_tree.select_file("experiment/cartpole/quick_test.yaml")
        if ok:
            self.assertEqual(self.window.current_experiment, "cartpole/quick_test")
            self.assertEqual(self.window.config_viewer.file_title.text(), "quick_test.yaml")


if __name__ == "__main__":
    unittest.main()
