"""Tests for paradigm constraints driving the ConfigViewer form.

Covers the wiring between frontend/config_model.py and the experiment pane:
selecting a paradigm should restrict the environment and agent choices and
disable the fields that paradigm forbids, so an invalid experiment cannot be
constructed in the UI at all.
"""

import os
import sys
import tempfile
import unittest
from pathlib import Path

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtWidgets import QApplication, QComboBox

    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.config_model import ConfigTree
    from frontend.config_viewer import ConfigViewer, config_tree

    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


ONLINE_EXPERIMENT = (
    "paradigm: online_rl\n"
    "experiment_id: cp_demo\n"
    "seed: 42\n"
    "total_timesteps: 5000\n"
    "eval_episodes: 10\n"
    "methods:\n"
    "  ppo:\n"
    "    agent: ppo\n"
    "    model: dnn\n"
)

OFFLINE_EXPERIMENT = (
    "paradigm: offline_rl\n"
    "experiment_id: mimic_demo\n"
    "seed: 7\n"
    "total_timesteps: 5000\n"
    "methods:\n"
    "  cql:\n"
    "    agent: cql\n"
    "    model: dnn\n"
)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class ConfigViewerConstraintTest(unittest.TestCase):
    """Base fixture: writes an experiment file and loads it into a viewer."""

    def load(self, text):
        self.temp_dir = tempfile.TemporaryDirectory()
        self.addCleanup(self.temp_dir.cleanup)
        path = Path(self.temp_dir.name) / "exp.yaml"
        path.write_text(text, encoding="utf-8")
        viewer = ConfigViewer()
        self.addCleanup(viewer.close)
        viewer.load_file(path, "experiment/exp.yaml")
        return viewer

    def agent_widget(self, viewer):
        for row in viewer.findChildren(QComboBox):
            if row.toolTip().startswith(("Agents permitted", "'")):
                return row
        return None


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestConfigTreeAvailable(unittest.TestCase):
    def test_config_tree_is_discoverable(self):
        self.assertIsNotNone(config_tree(), "frontend must find in/config from a checkout")

    def test_config_tree_is_cached(self):
        self.assertIs(config_tree(), config_tree())


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestParadigmSelector(ConfigViewerConstraintTest):
    def test_paradigm_combo_is_created(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertIn("paradigm", viewer.field_widgets)
        self.assertIsInstance(viewer.combo_paradigm, QComboBox)

    def test_paradigm_combo_reflects_file(self):
        viewer = self.load(OFFLINE_EXPERIMENT)
        self.assertEqual(viewer.combo_paradigm.currentText(), "offline_rl")

    def test_paradigm_combo_lists_every_paradigm(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        listed = {viewer.combo_paradigm.itemText(i) for i in range(viewer.combo_paradigm.count())}
        self.assertEqual(listed, set(ConfigTree.discover().paradigms))

    def test_changing_paradigm_updates_raw_data(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        viewer.combo_paradigm.setCurrentText("offline_rl")
        self.assertEqual(viewer.raw_data["paradigm"], "offline_rl")

    def test_changing_paradigm_marks_dirty(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertFalse(viewer.is_dirty)
        viewer.combo_paradigm.setCurrentText("offline_rl")
        self.assertTrue(viewer.is_dirty)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestEnvironmentFiltering(ConfigViewerConstraintTest):
    def test_online_offers_only_simulator_environments(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        offered = {viewer.combo_env.itemText(i) for i in range(viewer.combo_env.count())}
        offered.discard(ConfigViewer.INHERIT)
        tree = ConfigTree.discover()
        self.assertTrue(offered)
        self.assertTrue(all(not tree.environments[name].offline_only for name in offered))

    def test_offline_offers_only_static_dataset_environments(self):
        viewer = self.load(OFFLINE_EXPERIMENT)
        offered = {viewer.combo_env.itemText(i) for i in range(viewer.combo_env.count())}
        offered.discard(ConfigViewer.INHERIT)
        tree = ConfigTree.discover()
        self.assertTrue(offered)
        self.assertTrue(all(tree.environments[name].offline_only for name in offered))

    def test_cartpole_not_offered_for_offline(self):
        viewer = self.load(OFFLINE_EXPERIMENT)
        offered = {viewer.combo_env.itemText(i) for i in range(viewer.combo_env.count())}
        self.assertNotIn("cartpole", offered)

    def test_env_defaults_to_inherit_when_file_omits_it(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertEqual(viewer.combo_env.currentText(), ConfigViewer.INHERIT)

    def test_inherit_is_not_written_into_config(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertNotIn("env", viewer.raw_data)

    def test_selecting_env_records_it(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        viewer.combo_env.setCurrentText("cartpole")
        self.assertEqual(viewer.raw_data["env"], "cartpole")

    def test_switching_paradigm_drops_incompatible_env(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        viewer.combo_env.setCurrentText("cartpole")
        self.assertEqual(viewer.raw_data["env"], "cartpole")
        viewer.combo_paradigm.setCurrentText("offline_rl")
        self.assertNotIn("env", viewer.raw_data)


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestAgentFiltering(ConfigViewerConstraintTest):
    def test_agent_is_a_dropdown_not_free_text(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertIsNotNone(self.agent_widget(viewer))

    def test_online_agent_choices_exclude_offline_algorithms(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        combo = self.agent_widget(viewer)
        offered = {combo.itemText(i) for i in range(combo.count())}
        self.assertIn("ppo", offered)
        self.assertFalse(offered & {"cql", "iql", "cew"})

    def test_offline_agent_choices_exclude_ppo(self):
        viewer = self.load(OFFLINE_EXPERIMENT)
        combo = self.agent_widget(viewer)
        offered = {combo.itemText(i) for i in range(combo.count())}
        self.assertIn("cql", offered)
        self.assertNotIn("ppo", offered)

    def test_off_list_agent_is_preserved_not_silently_rewritten(self):
        """An existing file with a mismatched agent must not be quietly changed."""
        viewer = self.load(ONLINE_EXPERIMENT.replace("agent: ppo", "agent: cql"))
        combo = self.agent_widget(viewer)
        self.assertEqual(combo.currentText(), "cql")
        self.assertEqual(viewer.raw_data["methods"]["ppo"]["agent"], "cql")

    def test_off_list_agent_explains_itself(self):
        viewer = self.load(ONLINE_EXPERIMENT.replace("agent: ppo", "agent: cql"))
        self.assertIn("not permitted", self.agent_widget(viewer).toolTip())


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestForbiddenFields(ConfigViewerConstraintTest):
    def test_eval_episodes_enabled_for_online(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertTrue(viewer.spin_eval_ep.isEnabled())

    def test_eval_episodes_disabled_for_offline(self):
        viewer = self.load(OFFLINE_EXPERIMENT)
        self.assertFalse(viewer.spin_eval_ep.isEnabled())

    def test_disabled_field_explains_why(self):
        viewer = self.load(OFFLINE_EXPERIMENT)
        self.assertIn("eval_episodes", viewer.spin_eval_ep.toolTip())

    def test_switching_paradigm_disables_the_field(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertTrue(viewer.spin_eval_ep.isEnabled())
        viewer.combo_paradigm.setCurrentText("offline_rl")
        self.assertFalse(viewer.spin_eval_ep.isEnabled())


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestOverrides(ConfigViewerConstraintTest):
    def test_paradigm_is_emitted(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertIn("paradigm=online_rl", viewer.get_overrides())

    def test_selected_env_is_emitted(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        viewer.combo_env.setCurrentText("cartpole")
        self.assertIn("env=cartpole", viewer.get_overrides())

    def test_inherited_env_is_not_emitted(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        self.assertFalse(any(o.startswith("env=") for o in viewer.get_overrides()))

    def test_existing_override_behaviour_is_unchanged(self):
        viewer = self.load(ONLINE_EXPERIMENT)
        overrides = viewer.get_overrides()
        self.assertIn("++experiment_id='cp_demo'", overrides)
        self.assertIn("seed=42", overrides)
        self.assertIn("total_timesteps=5000", overrides)


if __name__ == "__main__":
    unittest.main()
