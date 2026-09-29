"""Unit and integration tests for DOOM Simulator plugin, raycasting engine, and RL bot."""
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import unittest

os.environ["QT_QPA_PLATFORM"] = "offscreen"

try:
    from PyQt6.QtCore import Qt
    from PyQt6.QtWidgets import QApplication
    QApplication.setAttribute(Qt.ApplicationAttribute.AA_ShareOpenGLContexts, True)
    app = QApplication.instance() or QApplication(sys.argv[:1])
    from frontend.app import Window
    from frontend.plugins.context import PluginContext
    from frontend.plugins.doom import DoomPlugin
    from frontend.plugins.doom.engine import (
        DoomEngine, Entity, SCREEN_H, SCREEN_W,
        WALL_TECH, WALL_RED_DOOR, WALL_BLUE_DOOR, WALL_EXIT
    )
    from frontend.plugins.doom.panel import DoomPanel
    HAS_PYQT6 = True
except ImportError:
    HAS_PYQT6 = False


@unittest.skipIf(not HAS_PYQT6, "PyQt6 not installed in current environment")
class TestDoomPlugin(unittest.TestCase):
    def setUp(self):
        self.temp_dir = tempfile.mkdtemp(prefix="theta_test_doom_")
        self.window = Window(data_dir=self.temp_dir)
        self.storage_file = Path(self.temp_dir) / "plugins" / "doom.json"
        self.context = PluginContext("doom", self.window, self.storage_file)

    def tearDown(self):
        if hasattr(self, "panel") and self.panel:
            self.panel.cleanup()
        shutil.rmtree(self.temp_dir, ignore_errors=True)

    def test_doom_plugin_discovered_in_manifests(self):
        """DOOM plugin manifest is discovered with correct metadata."""
        manager = self.window.plugin_manager
        self.assertIn("doom", manager.manifests)
        manifest = manager.manifests["doom"]
        self.assertEqual(manifest.id, "doom")
        self.assertEqual(manifest.name, "DOOM Simulator")
        self.assertFalse(manifest.default_enabled)
        self.assertEqual(manifest.icon, "doom")

    def test_doom_plugin_activation_and_deactivation_lifecycle(self):
        """Activating mounts the DOOM tab; deactivating unmounts cleanly."""
        manager = self.window.plugin_manager
        self.assertNotIn("doom", self.window.tabs.tabs)

        # Enable plugin
        success = manager.enable_plugin("doom")
        self.assertTrue(success)
        self.assertTrue(manager.is_plugin_enabled("doom"))
        self.assertIn("doom", self.window.tabs.tabs)
        self.assertIn("doom", self.window.tabs.tab_order)

        plugin_instance = manager.instances["doom"]
        self.assertIsInstance(plugin_instance, DoomPlugin)
        self.assertIsNotNone(plugin_instance.panel)

        # Disable plugin
        success = manager.disable_plugin("doom")
        self.assertTrue(success)
        self.assertFalse(manager.is_plugin_enabled("doom"))
        self.assertNotIn("doom", self.window.tabs.tabs)
        self.assertNotIn("doom", self.window.tabs.tab_order)

    def test_raycasting_engine_dda_and_zbuffer(self):
        """Engine casts rays, generates distance buffer, and detects wall heights."""
        engine = DoomEngine("E1M1: Hangar")
        z_buffer, wall_hits = engine.render_raycast()

        self.assertEqual(len(z_buffer), SCREEN_W)
        self.assertEqual(len(wall_hits), SCREEN_W)

        for col, hit in enumerate(wall_hits):
            self.assertEqual(hit["col"], col)
            self.assertGreater(hit["dist"], 0.0)
            self.assertGreater(hit["height"], 0)
            self.assertGreaterEqual(hit["tile"], 1)

    def test_player_movement_and_wall_collision(self):
        """Player moves forward in open space but cannot pass through walls."""
        engine = DoomEngine("E1M1: Hangar")
        initial_x = engine.player_x
        initial_y = engine.player_y

        # Move forward into open corridor
        engine.move_player(forward=1.0, strafe=0.0, dt=0.1)
        self.assertNotEqual(engine.player_x, initial_x)

        # Attempt to walk into outer perimeter wall (x=0)
        engine.player_x = 1.05
        engine.player_y = 1.5
        engine.player_angle = 3.14159  # Facing West toward wall at x=0
        engine.move_player(forward=1.0, strafe=0.0, dt=0.5)
        # Should be stopped by wall collision
        self.assertGreater(engine.player_x, 1.0)

    def test_weapon_firing_hitscan_and_demon_damage(self):
        """Firing equipped weapon consumes ammo and damages enemies in line-of-sight."""
        engine = DoomEngine("E1M1: Hangar")
        initial_shells = engine.ammo["shells"]

        # Face directly toward imp1 at (6.5, 1.5) from player at (5.5, 1.5) in open corridor
        engine.player_x = 5.5
        engine.player_y = 1.5
        engine.player_angle = 0.0  # Facing East along y=1.5
        engine.select_weapon("shotgun")

        fired = engine.fire_weapon()
        self.assertTrue(fired)
        self.assertEqual(engine.ammo["shells"], initial_shells - 1)

        # Imp should take damage
        imp = next(e for e in engine.entities if e.id == "imp1")
        self.assertLess(imp.health, imp.max_health)
        self.assertGreater(engine.episode_reward, 0.0)

    def test_doom_cheats_iddqd_and_idkfa(self):
        """Cheats IDDQD and IDKFA grant god mode, all weapons, and keys."""
        engine = DoomEngine("E1M1: Hangar")

        # Test IDDQD (God Mode)
        self.assertFalse(engine.god_mode)
        msg = engine.cheat_iddqd()
        self.assertIn("GOD MODE", msg)
        self.assertTrue(engine.god_mode)
        self.assertEqual(engine.face_state, "god")

        # In god mode, damage has no effect
        engine.take_player_damage(50)
        self.assertEqual(engine.health, 100)

        # Test IDKFA (Very Happy Ammo)
        msg2 = engine.cheat_idkfa()
        self.assertIn("IDKFA", msg2)
        self.assertTrue(engine.owned_weapons["bfg"])
        self.assertTrue(engine.owned_weapons["chaingun"])
        self.assertIn("blue", engine.keys_held)
        self.assertIn("red", engine.keys_held)
        self.assertEqual(engine.armor, 100)

    def test_autonomous_ai_bot_simulation_step(self):
        """AI Bot steps decision policy, navigates, and emits RL trajectory transitions."""
        engine = DoomEngine("E1M1: Hangar")
        initial_trajectory_len = len(engine.trajectory)

        res = engine.step_ai_bot(dt=0.05)
        self.assertIn("action", res)
        self.assertIn("reward", res)
        self.assertEqual(len(engine.trajectory), initial_trajectory_len + 1)

        # Verify state vector dimensions
        obs = engine.get_state_vector()
        self.assertEqual(len(obs), 8)

    def test_can_it_run_doom_benchmark_execution(self):
        """Performance benchmark executes and verifies 'CAN IT RUN DOOM? YES!'."""
        panel = DoomPanel(self.context)
        self.panel = panel

        bench_results = panel.run_benchmark_test()
        self.assertIn("CAN IT RUN DOOM? YES!", bench_results["verdict"])
        self.assertGreater(bench_results["fps"], 30.0)
        self.assertGreater(bench_results["total_rays"], 1000)
        self.assertIn("Theta-Certified", bench_results["rating"])

    def test_export_rl_dataset(self):
        """Exporting transitions saves a valid JSON dataset."""
        panel = DoomPanel(self.context)
        self.panel = panel

        # Populate trajectory
        for _ in range(5):
            panel.engine.step_ai_bot(dt=0.05)

        self.assertGreater(len(panel.engine.trajectory), 0)

        # Simulate dataset export
        save_dir = self.context.get_data_dir() / "datasets" / "doom"
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / "doom_transitions_test.json"

        data = {
            "environment": "doom_raycaster_sim",
            "level": panel.engine.level_name,
            "total_transitions": len(panel.engine.trajectory),
            "transitions": [
                {
                    "obs": t[0],
                    "action": t[1],
                    "reward": t[2],
                    "next_obs": t[3],
                    "done": t[4],
                }
                for t in panel.engine.trajectory
            ],
        }
        file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
        self.assertTrue(file_path.exists())

        loaded = json.loads(file_path.read_text(encoding="utf-8"))
        self.assertEqual(loaded["environment"], "doom_raycaster_sim")
        self.assertEqual(len(loaded["transitions"]), len(panel.engine.trajectory))
