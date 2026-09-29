"""DOOM Simulator and RL Agent Benchmark Plugin for ThetaIDE."""
from __future__ import annotations
from typing import Optional

from frontend.plugins.base import Plugin, PluginManifest
from frontend.plugins.context import PluginContext
from frontend.plugins.doom.panel import DoomPanel


class DoomPlugin(Plugin):
    """Real-time 3D DOOM raycaster simulator and NeSyRL RL agent benchmark plugin."""

    def __init__(self, manifest: PluginManifest):
        super().__init__(manifest)
        self.context: Optional[PluginContext] = None
        self.panel: Optional[DoomPanel] = None

    def activate(self, context: PluginContext) -> None:
        self.context = context
        self.panel = DoomPanel(context=context)

        # Register tab into the primary IDE sidebar
        context.add_sidebar_tab(
            tab_id="doom",
            widget=self.panel,
            title="DOOM Simulator",
            icon_name="doom",
            short_label="DOOM",
        )
        context.log("DOOM Simulator plugin activated: Can it run DOOM? YES!")

    def deactivate(self) -> None:
        if self.context and self.panel:
            self.context.remove_sidebar_tab("doom")
            self.panel.cleanup()
            self.panel = None
            self.context.log("DOOM Simulator plugin deactivated.")
            self.context = None
