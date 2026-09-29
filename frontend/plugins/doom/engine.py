"""DOOM Raycasting Engine, Game Simulator, and NeSyRL Autonomous Bot.

Provides a Wolf3D/DOOM-style pseudo-3D first-person raycasting simulation engine
running at high framerate with zero external dependencies, featuring classic maps,
demons, pickups, weapons, HUD status bar with animated Doomguy face, cheats,
and autonomous RL bot simulation.
"""
from __future__ import annotations
import math
import random
import time
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

# ── Screen & Raycast Constants ───────────────────────────────────────────────
SCREEN_W = 160
SCREEN_H = 100
FOV = math.pi / 3  # 60 degrees field of view
HALF_FOV = FOV / 2.0


# ── Wall Types ───────────────────────────────────────────────────────────────
WALL_EMPTY = 0
WALL_TECH = 1        # Standard steel/tech wall
WALL_BRONZE = 2      # Bronze rust armor plate
WALL_HAZARD = 3      # Yellow/black hazard stripes
WALL_COMPUTER = 4    # Computer terminals with blinkers
WALL_SLIME = 5       # Toxic green brick
WALL_RED_DOOR = 6    # Locked Red Door
WALL_BLUE_DOOR = 7   # Locked Blue Door
WALL_EXIT = 8        # Exit Switch Gateway
WALL_DOOR_OPEN = 9   # Temporarily opened door


# ── Level Maps (16x16 Grids) ────────────────────────────────────────────────
MAP_E1M1: List[List[int]] = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 2, 0, 1, 0, 4, 4, 4, 4, 0, 1, 0, 8, 0, 1],
    [1, 0, 0, 0, 6, 0, 4, 0, 0, 4, 0, 7, 0, 0, 0, 1],
    [1, 1, 6, 1, 1, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 3, 3, 3, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 1],
    [1, 0, 1, 1, 0, 0, 0, 0, 0, 0, 0, 5, 5, 0, 0, 1],
    [1, 0, 0, 0, 0, 4, 0, 0, 0, 4, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 4, 0, 2, 0, 4, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 3, 0, 0, 0, 1, 1, 1, 0, 0, 0, 0, 3, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

MAP_E1M2: List[List[int]] = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 4, 0, 1, 0, 5, 0, 1, 0, 4, 4, 4, 4, 0, 1],
    [1, 0, 4, 0, 6, 0, 5, 0, 7, 0, 4, 0, 0, 4, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 0, 1, 1, 1, 1, 0, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 3, 3, 0, 0, 1, 0, 2, 2, 0, 1, 0, 8, 0, 1],
    [1, 0, 3, 3, 0, 0, 0, 0, 2, 2, 0, 0, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 1, 1, 0, 1],
    [1, 1, 1, 0, 1, 1, 1, 1, 1, 0, 1, 1, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 3, 0, 1],
    [1, 0, 5, 0, 1, 0, 4, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 0, 5, 0, 0, 0, 4, 0, 0, 0, 0, 6, 0, 0, 0, 1],
    [1, 0, 0, 0, 1, 0, 0, 0, 1, 0, 0, 1, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

MAP_E1M8: List[List[int]] = [
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
    [1, 0, 0, 0, 0, 0, 1, 8, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
    [1, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 0, 0, 0, 0, 1],
    [1, 0, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 0, 0, 1],
    [1, 0, 0, 5, 5, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 4, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 4, 0, 1],
    [1, 0, 4, 0, 0, 0, 3, 0, 3, 0, 0, 0, 0, 4, 0, 1],
    [1, 0, 0, 0, 5, 5, 0, 0, 0, 5, 5, 0, 0, 0, 0, 1],
    [1, 0, 0, 5, 5, 5, 5, 0, 5, 5, 5, 5, 0, 0, 0, 1],
    [1, 0, 0, 5, 5, 0, 0, 0, 0, 0, 5, 5, 0, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 1],
    [1, 0, 2, 0, 0, 0, 0, 0, 0, 0, 0, 0, 2, 0, 0, 1],
    [1, 0, 0, 0, 0, 0, 1, 6, 1, 0, 0, 0, 0, 0, 0, 1],
    [1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1, 1],
]

LEVELS = {
    "E1M1: Hangar": MAP_E1M1,
    "E1M2: Nuclear Plant": MAP_E1M2,
    "E1M8: Phobos Anomaly": MAP_E1M8,
}


# ── Entity Data Class ────────────────────────────────────────────────────────
@dataclass
class Entity:
    """Represents an enemy demon, pickup item, or hazard barrel."""
    id: str
    kind: str  # "imp", "pinky", "cacodemon", "baron", "medikit", "stimpack", "armor", "shotgun_ammo", "cells", "blue_key", "red_key", "barrel"
    x: float
    y: float
    health: int
    max_health: int
    is_enemy: bool = False
    is_pickup: bool = False
    is_barrel: bool = False
    state: str = "idle"  # "idle", "chase", "attack", "hurt", "dead"
    attack_cooldown: float = 0.0
    pain_timer: float = 0.0
    speed: float = 1.6
    damage: int = 10
    score_val: int = 100


# ── Weapon Specifications ────────────────────────────────────────────────────
WEAPONS: Dict[str, Dict[str, Any]] = {
    "fist": {
        "name": "Fists / Brass Knuckles",
        "slot": 1,
        "ammo_type": None,
        "ammo_cost": 0,
        "damage": 25,
        "range": 1.4,
        "cooldown": 0.35,
        "sound": "punch",
    },
    "pistol": {
        "name": "Pistol",
        "slot": 2,
        "ammo_type": "bullets",
        "ammo_cost": 1,
        "damage": 18,
        "range": 14.0,
        "cooldown": 0.3,
        "sound": "pistol_shot",
    },
    "shotgun": {
        "name": "Super Shotgun",
        "slot": 3,
        "ammo_type": "shells",
        "ammo_cost": 1,
        "damage": 85,
        "range": 10.0,
        "cooldown": 0.75,
        "sound": "shotgun_blast",
    },
    "chaingun": {
        "name": "Chaingun",
        "slot": 4,
        "ammo_type": "bullets",
        "ammo_cost": 1,
        "damage": 16,
        "range": 14.0,
        "cooldown": 0.12,
        "sound": "chaingun_fire",
    },
    "bfg": {
        "name": "BFG 9000",
        "slot": 5,
        "ammo_type": "cells",
        "ammo_cost": 40,
        "damage": 450,
        "range": 16.0,
        "cooldown": 1.2,
        "sound": "bfg_fire",
    },
}


# ── Core Engine ─────────────────────────────────────────────────────────────
class DoomEngine:
    """Simulates DOOM world physics, DDA raycasting, demon AI, and RL transitions."""

    def __init__(self, level_name: str = "E1M1: Hangar"):
        self.level_name = level_name
        self.map_grid = [row[:] for row in LEVELS.get(level_name, MAP_E1M1)]
        self.map_h = len(self.map_grid)
        self.map_w = len(self.map_grid[0])

        # Player state
        self.player_x: float = 2.5
        self.player_y: float = 2.5
        self.player_angle: float = 0.0  # Radians
        self.health: int = 100
        self.armor: int = 0
        self.ammo: Dict[str, int] = {"bullets": 50, "shells": 8, "cells": 0}
        self.owned_weapons: Dict[str, bool] = {
            "fist": True,
            "pistol": True,
            "shotgun": True,
            "chaingun": False,
            "bfg": False,
        }
        self.current_weapon: str = "shotgun"
        self.weapon_cooldown: float = 0.0
        self.weapon_anim: float = 0.0  # 0.0 = resting, >0.0 = firing recoil
        self.keys_held: set[str] = set()

        # Score & stats
        self.frags: int = 0
        self.score: int = 0
        self.level_completed: bool = False
        self.is_dead: bool = False

        # Visual FX & face expressions
        self.screen_flash: Optional[Tuple[int, int, int, float]] = None  # (R, G, B, alpha)
        self.face_state: str = "normal"  # "normal", "look_left", "look_right", "grin", "hurt", "god", "dead"
        self.face_timer: float = 0.0
        self.god_mode: bool = False  # IDDQD
        self.no_clip: bool = False   # IDCLIP

        # Entities
        self.entities: List[Entity] = []
        self._spawn_entities_for_level()

        # Telemetry / RL stats
        self.total_steps: int = 0
        self.episode_reward: float = 0.0
        self.trajectory: List[Tuple[Any, Any, float, Any, bool]] = []
        self.last_action_desc: str = "IDLE"

        # Precomputed ray angles
        self.ray_angles = [
            math.atan2((i - SCREEN_W / 2.0) / (SCREEN_W / 2.0) * math.tan(HALF_FOV), 1.0)
            for i in range(SCREEN_W)
        ]

    def _spawn_entities_for_level(self) -> None:
        """Populate level with demons, pickups, and barrels."""
        self.entities.clear()
        if self.level_name == "E1M1: Hangar":
            self.player_x, self.player_y, self.player_angle = 1.5, 1.5, 0.0
            self.entities = [
                # Pickups
                Entity("m1", "medikit", 2.5, 1.5, 1, 1, is_pickup=True),
                Entity("arm1", "armor", 2.0, 2.0, 1, 1, is_pickup=True),
                Entity("sh1", "shotgun_ammo", 7.5, 1.5, 1, 1, is_pickup=True),
                Entity("bk1", "blue_key", 10.5, 1.5, 1, 1, is_pickup=True),
                Entity("rk1", "red_key", 1.5, 6.5, 1, 1, is_pickup=True),
                Entity("cells1", "cells", 13.5, 13.5, 1, 1, is_pickup=True),
                # Hazards
                Entity("b1", "barrel", 6.5, 6.5, 20, 20, is_barrel=True),
                Entity("b2", "barrel", 8.5, 6.5, 20, 20, is_barrel=True),
                # Demons
                Entity("imp1", "imp", 6.5, 1.5, 40, 40, is_enemy=True, speed=1.4, damage=10, score_val=100),
                Entity("imp2", "imp", 8.5, 1.5, 40, 40, is_enemy=True, speed=1.4, damage=10, score_val=100),
                Entity("pink1", "pinky", 9.5, 6.5, 70, 70, is_enemy=True, speed=2.1, damage=15, score_val=150),
                Entity("caco1", "cacodemon", 13.5, 6.5, 110, 110, is_enemy=True, speed=1.2, damage=22, score_val=250),
            ]
        elif self.level_name == "E1M2: Nuclear Plant":
            self.player_x, self.player_y, self.player_angle = 1.5, 1.5, 0.0
            self.entities = [
                Entity("m1", "medikit", 1.5, 4.5, 1, 1, is_pickup=True),
                Entity("sh1", "shotgun_ammo", 5.5, 1.5, 1, 1, is_pickup=True),
                Entity("bk1", "blue_key", 14.5, 1.5, 1, 1, is_pickup=True),
                Entity("rk1", "red_key", 1.5, 14.5, 1, 1, is_pickup=True),
                Entity("imp1", "imp", 4.5, 3.5, 40, 40, is_enemy=True, speed=1.5, damage=10, score_val=100),
                Entity("imp2", "imp", 7.5, 3.5, 40, 40, is_enemy=True, speed=1.5, damage=10, score_val=100),
                Entity("pink1", "pinky", 8.5, 8.5, 70, 70, is_enemy=True, speed=2.0, damage=15, score_val=150),
                Entity("caco1", "cacodemon", 13.5, 13.5, 110, 110, is_enemy=True, speed=1.3, damage=25, score_val=250),
            ]
        else:  # E1M8: Phobos Anomaly
            self.player_x, self.player_y, self.player_angle = 7.5, 14.0, -math.pi / 2.0
            self.owned_weapons["bfg"] = True
            self.owned_weapons["chaingun"] = True
            self.ammo["cells"] = 120
            self.ammo["shells"] = 30
            self.ammo["bullets"] = 200
            self.current_weapon = "bfg"
            self.entities = [
                Entity("baron1", "baron", 5.5, 6.5, 300, 300, is_enemy=True, speed=1.6, damage=35, score_val=500),
                Entity("baron2", "baron", 9.5, 6.5, 300, 300, is_enemy=True, speed=1.6, damage=35, score_val=500),
                Entity("caco1", "cacodemon", 7.5, 4.5, 110, 110, is_enemy=True, speed=1.4, damage=22, score_val=250),
                Entity("m1", "medikit", 1.5, 1.5, 1, 1, is_pickup=True),
                Entity("m2", "medikit", 13.5, 1.5, 1, 1, is_pickup=True),
                Entity("arm1", "armor", 7.5, 7.5, 1, 1, is_pickup=True),
                Entity("cells1", "cells", 3.5, 12.5, 1, 1, is_pickup=True),
                Entity("cells2", "cells", 11.5, 12.5, 1, 1, is_pickup=True),
            ]

    # ── Cheats ───────────────────────────────────────────────────────────────
    def cheat_iddqd(self) -> str:
        """Toggle God Mode."""
        self.god_mode = not self.god_mode
        if self.god_mode:
            self.health = 100
            self.face_state = "god"
            self.face_timer = 99999.0
            return "DEGREELESSNESS MODE (GOD MODE) ON"
        else:
            self.face_state = "normal"
            self.face_timer = 0.0
            return "GOD MODE OFF"

    def cheat_idkfa(self) -> str:
        """Very Happy Ammo (All weapons, max ammo, all keys)."""
        for w in self.owned_weapons:
            self.owned_weapons[w] = True
        self.ammo["bullets"] = 200
        self.ammo["shells"] = 50
        self.ammo["cells"] = 300
        self.keys_held = {"blue", "red", "yellow"}
        self.armor = 100
        self.face_state = "grin"
        self.face_timer = 2.0
        return "VERY HAPPY AMMO & KEYS ENABLED (IDKFA)"

    def cheat_idclip(self) -> str:
        """Toggle No-Clip."""
        self.no_clip = not self.no_clip
        return f"NO-CLIP {'ON' if self.no_clip else 'OFF'}"

    # ── Player Movement & Actions ────────────────────────────────────────────
    def move_player(self, forward: float, strafe: float, dt: float) -> None:
        """Translate player along forward and strafe axes."""
        if self.is_dead:
            return
        speed = 3.8 * dt
        dx = (math.cos(self.player_angle) * forward - math.sin(self.player_angle) * strafe) * speed
        dy = (math.sin(self.player_angle) * forward + math.cos(self.player_angle) * strafe) * speed

        new_x = self.player_x + dx
        new_y = self.player_y + dy

        if self.no_clip or not self._is_wall_blocking(new_x, self.player_y):
            self.player_x = new_x
        if self.no_clip or not self._is_wall_blocking(self.player_x, new_y):
            self.player_y = new_y

        self._check_pickups()

    def turn_player(self, delta_radians: float) -> None:
        """Rotate player viewpoint."""
        if self.is_dead:
            return
        self.player_angle = (self.player_angle + delta_radians) % (2 * math.pi)

    def select_weapon(self, weapon_id: str) -> bool:
        """Switch to owned weapon."""
        if weapon_id in self.owned_weapons and self.owned_weapons[weapon_id]:
            self.current_weapon = weapon_id
            return True
        return False

    def fire_weapon(self) -> bool:
        """Fire equipped weapon."""
        if self.is_dead or self.weapon_cooldown > 0.0:
            return False

        spec = WEAPONS.get(self.current_weapon, WEAPONS["pistol"])
        ammo_type = spec["ammo_type"]
        ammo_cost = spec["ammo_cost"]

        if ammo_type and not self.god_mode:
            if self.ammo.get(ammo_type, 0) < ammo_cost:
                return False
            self.ammo[ammo_type] -= ammo_cost

        self.weapon_cooldown = spec["cooldown"]
        self.weapon_anim = 0.25

        # Visual flash & grin
        if self.current_weapon == "bfg":
            self.screen_flash = (0, 255, 60, 0.4)
            self.face_state = "grin"
            self.face_timer = 1.0
        elif self.current_weapon == "shotgun":
            self.screen_flash = (255, 180, 50, 0.25)
            self.face_state = "grin"
            self.face_timer = 0.6
        else:
            self.screen_flash = (255, 230, 100, 0.15)

        # Hitscan detection
        num_pellets = 7 if self.current_weapon == "shotgun" else (1 if self.current_weapon != "bfg" else 20)
        spread = 0.12 if self.current_weapon == "shotgun" else (0.4 if self.current_weapon == "bfg" else 0.02)
        base_dmg = spec["damage"] / num_pellets if self.current_weapon == "shotgun" else spec["damage"]

        reward_earned = 0.0
        for _ in range(num_pellets):
            angle_jitter = (random.random() - 0.5) * spread
            hit_entity, dist = self._raycast_entity(self.player_angle + angle_jitter, max_dist=spec["range"])
            if hit_entity:
                dmg = int(base_dmg * random.uniform(0.85, 1.15))
                reward_earned += self._damage_entity(hit_entity, dmg)

        self.episode_reward += reward_earned
        return True

    def interact_use(self) -> str:
        """Open adjacent door or activate switch."""
        if self.is_dead:
            return "NONE"

        # Check tile directly in front of player
        front_x = int(self.player_x + math.cos(self.player_angle) * 1.1)
        front_y = int(self.player_y + math.sin(self.player_angle) * 1.1)

        if not (0 <= front_x < self.map_w and 0 <= front_y < self.map_h):
            return "OUT_OF_BOUNDS"

        tile = self.map_grid[front_y][front_x]
        if tile == WALL_RED_DOOR:
            if "red" in self.keys_held or self.god_mode:
                self.map_grid[front_y][front_x] = WALL_DOOR_OPEN
                self.screen_flash = (255, 50, 50, 0.3)
                self.episode_reward += 50.0
                return "UNLOCKED_RED_DOOR"
            return "NEED_RED_KEY"
        elif tile == WALL_BLUE_DOOR:
            if "blue" in self.keys_held or self.god_mode:
                self.map_grid[front_y][front_x] = WALL_DOOR_OPEN
                self.screen_flash = (50, 100, 255, 0.3)
                self.episode_reward += 50.0
                return "UNLOCKED_BLUE_DOOR"
            return "NEED_BLUE_KEY"
        elif tile == WALL_EXIT:
            self.level_completed = True
            self.score += 1000
            self.episode_reward += 200.0
            return "LEVEL_COMPLETED"

        return "NOTHING_TO_USE"

    # ── Entity Combat & Pickups ──────────────────────────────────────────────
    def _damage_entity(self, entity: Entity, damage: int) -> float:
        """Apply damage to demon or barrel."""
        if entity.state == "dead":
            return 0.0

        entity.health -= damage
        entity.pain_timer = 0.25
        entity.state = "hurt"

        reward = 5.0
        if entity.health <= 0:
            entity.health = 0
            entity.state = "dead"
            if entity.is_enemy:
                self.frags += 1
                self.score += entity.score_val
                reward += float(entity.score_val)
                # Drop ammo
                if entity.kind in ("imp", "pinky"):
                    self.entities.append(Entity(f"drop_{entity.id}", "shotgun_ammo", entity.x, entity.y, 1, 1, is_pickup=True))
                elif entity.kind in ("cacodemon", "baron"):
                    self.entities.append(Entity(f"drop_{entity.id}", "cells", entity.x, entity.y, 1, 1, is_pickup=True))
            elif entity.is_barrel:
                # Barrel explosion dealing area damage
                reward += self._explode_barrel(entity.x, entity.y)

        return reward

    def _explode_barrel(self, bx: float, by: float) -> float:
        """Explode barrel and damage adjacent enemies and player."""
        self.screen_flash = (255, 120, 20, 0.4)
        reward = 0.0
        radius = 2.2
        for e in self.entities:
            if e.state != "dead" and (e.is_enemy or e.is_barrel):
                dist = math.hypot(e.x - bx, e.y - by)
                if dist <= radius:
                    dmg = int(80 * (1.0 - dist / radius))
                    reward += self._damage_entity(e, dmg)

        # Damage player if close
        p_dist = math.hypot(self.player_x - bx, self.player_y - by)
        if p_dist <= radius and not self.god_mode:
            p_dmg = int(50 * (1.0 - p_dist / radius))
            self.take_player_damage(p_dmg)

        return reward

    def take_player_damage(self, amount: int) -> None:
        """Inflict damage on player."""
        if self.god_mode or self.is_dead:
            return

        if self.armor > 0:
            armor_absorbed = int(amount * 0.4)
            self.armor = max(0, self.armor - armor_absorbed)
            amount -= armor_absorbed

        self.health -= amount
        self.screen_flash = (220, 20, 20, 0.45)
        self.face_state = "hurt"
        self.face_timer = 0.8
        self.episode_reward -= float(amount)

        if self.health <= 0:
            self.health = 0
            self.is_dead = True
            self.face_state = "dead"
            self.episode_reward -= 100.0

    def _check_pickups(self) -> None:
        """Collect items player touches."""
        for e in self.entities:
            if not e.is_pickup or e.state == "dead":
                continue
            if math.hypot(e.x - self.player_x, e.y - self.player_y) < 0.7:
                e.state = "dead"
                self.screen_flash = (255, 230, 80, 0.3)
                self.score += 50
                self.episode_reward += 20.0
                if e.kind == "medikit":
                    self.health = min(100, self.health + 25)
                elif e.kind == "stimpack":
                    self.health = min(100, self.health + 10)
                elif e.kind == "armor":
                    self.armor = min(100, self.armor + 50)
                elif e.kind == "shotgun_ammo":
                    self.ammo["shells"] = min(50, self.ammo["shells"] + 8)
                    self.owned_weapons["shotgun"] = True
                elif e.kind == "cells":
                    self.ammo["cells"] = min(300, self.ammo["cells"] + 40)
                    self.owned_weapons["bfg"] = True
                elif e.kind == "blue_key":
                    self.keys_held.add("blue")
                elif e.kind == "red_key":
                    self.keys_held.add("red")

    def _is_wall_blocking(self, x: float, y: float) -> bool:
        """Check if grid cell at x, y contains solid geometry."""
        gx = int(x)
        gy = int(y)
        if not (0 <= gx < self.map_w and 0 <= gy < self.map_h):
            return True
        tile = self.map_grid[gy][gx]
        return tile not in (WALL_EMPTY, WALL_DOOR_OPEN)

    def _raycast_entity(self, ray_angle: float, max_dist: float) -> Tuple[Optional[Entity], float]:
        """Check hitscan intersection with living entities."""
        best_e = None
        best_dist = max_dist

        cos_a = math.cos(ray_angle)
        sin_a = math.sin(ray_angle)

        for e in self.entities:
            if e.state == "dead":
                continue
            dx = e.x - self.player_x
            dy = e.y - self.player_y
            dist = math.hypot(dx, dy)
            if dist > best_dist:
                continue

            # Dot product with ray direction
            proj = dx * cos_a + dy * sin_a
            if proj <= 0.2:
                continue
            perp_dist = math.sqrt(max(0.0, dist * dist - proj * proj))
            if perp_dist < 0.45:
                # Confirm wall does not occlude entity
                if not self._is_line_of_sight_blocked(self.player_x, self.player_y, e.x, e.y):
                    best_dist = dist
                    best_e = e

        return best_e, best_dist

    def _is_line_of_sight_blocked(self, x0: float, y0: float, x1: float, y1: float) -> bool:
        """Raymarch to check if wall obstructs view between two coordinates."""
        dx = x1 - x0
        dy = y1 - y0
        dist = math.hypot(dx, dy)
        if dist < 0.001:
            return False
        steps = int(dist * 8)
        for s in range(1, steps):
            t = s / steps
            if self._is_wall_blocking(x0 + dx * t, y0 + dy * t):
                return True
        return False

    # ── Simulation Update (Tick) ─────────────────────────────────────────────
    def update(self, dt: float) -> None:
        """Update demon state machines, weapon cooldowns, and visual timers."""
        self.total_steps += 1
        self.episode_reward -= 0.02 * dt  # slight time penalty

        # Weapon cooldown & anim
        if self.weapon_cooldown > 0.0:
            self.weapon_cooldown = max(0.0, self.weapon_cooldown - dt)
        if self.weapon_anim > 0.0:
            self.weapon_anim = max(0.0, self.weapon_anim - dt * 2.0)

        # Screen flash decay
        if self.screen_flash:
            r, g, b, alpha = self.screen_flash
            alpha -= dt * 2.5
            self.screen_flash = (r, g, b, max(0.0, alpha)) if alpha > 0.05 else None

        # Face animation state
        if self.face_timer > 0.0:
            self.face_timer -= dt
            if self.face_timer <= 0.0 and not self.god_mode and not self.is_dead:
                self.face_state = "normal"

        # Update demons
        for e in self.entities:
            if not e.is_enemy or e.state == "dead":
                continue

            if e.pain_timer > 0.0:
                e.pain_timer -= dt
                continue

            dist = math.hypot(self.player_x - e.x, self.player_y - e.y)

            # Alert / Chase behavior
            if dist < 11.0 and not self._is_line_of_sight_blocked(e.x, e.y, self.player_x, self.player_y):
                e.state = "chase"
                # Move toward player
                angle_to_p = math.atan2(self.player_y - e.y, self.player_x - e.x)
                step = e.speed * dt
                nx = e.x + math.cos(angle_to_p) * step
                ny = e.y + math.sin(angle_to_p) * step
                if not self._is_wall_blocking(nx, ny):
                    e.x = nx
                    e.y = ny

                # Attack when in range
                if dist < 1.3:
                    e.attack_cooldown -= dt
                    if e.attack_cooldown <= 0.0:
                        e.attack_cooldown = 1.0
                        self.take_player_damage(e.damage)
                elif e.kind in ("imp", "cacodemon", "baron") and dist < 8.0:
                    e.attack_cooldown -= dt
                    if e.attack_cooldown <= 0.0:
                        e.attack_cooldown = 2.2
                        # Demon ranged attack
                        if random.random() < 0.65:
                            self.take_player_damage(int(e.damage * 0.8))

    # ── Autonomous RL Bot / Heuristic Policy ─────────────────────────────────
    def step_ai_bot(self, dt: float) -> Dict[str, Any]:
        """Execute autonomous Neurosymbolic RL agent decision cycle."""
        if self.is_dead or self.level_completed:
            return {"action": "TERMINAL", "reward": self.episode_reward}

        action_desc = "EXPLORE"
        forward = 0.0
        turn = 0.0
        strafe = 0.0
        do_fire = False
        do_use = False

        # 1. Perception: Check nearest visible enemy
        nearest_enemy = None
        min_enemy_dist = 999.0
        for e in self.entities:
            if e.is_enemy and e.state != "dead":
                dist = math.hypot(e.x - self.player_x, e.y - self.player_y)
                if dist < min_enemy_dist and not self._is_line_of_sight_blocked(self.player_x, self.player_y, e.x, e.y):
                    min_enemy_dist = dist
                    nearest_enemy = e

        # 2. Perception: Check nearest pickup if hurt
        nearest_pickup = None
        if self.health < 60:
            for e in self.entities:
                if e.is_pickup and e.state != "dead" and e.kind in ("medikit", "stimpack"):
                    p_dist = math.hypot(e.x - self.player_x, e.y - self.player_y)
                    if not self._is_line_of_sight_blocked(self.player_x, self.player_y, e.x, e.y):
                        nearest_pickup = e
                        break

        # 3. Decision Logic
        if nearest_pickup and min_enemy_dist > 3.0:
            # Navigate to health
            action_desc = "RETRIEVE_HEALTH"
            target_angle = math.atan2(nearest_pickup.y - self.player_y, nearest_pickup.x - self.player_x)
            diff = (target_angle - self.player_angle + math.pi) % (2 * math.pi) - math.pi
            turn = 1.0 if diff > 0.05 else (-1.0 if diff < -0.05 else 0.0)
            forward = 1.0
        elif nearest_enemy and min_enemy_dist < 10.0:
            # Target and fire at demon
            target_angle = math.atan2(nearest_enemy.y - self.player_y, nearest_enemy.x - self.player_x)
            diff = (target_angle - self.player_angle + math.pi) % (2 * math.pi) - math.pi

            if abs(diff) < 0.18:
                action_desc = "ENGAGE_FIRE"
                do_fire = True
                forward = 0.4 if min_enemy_dist > 3.0 else -0.5
                strafe = 0.6 if random.random() < 0.5 else -0.6
            else:
                action_desc = "TARGET_LOCK"
                turn = 1.0 if diff > 0 else -1.0
                forward = 0.2
        else:
            # Exploration & wall avoidance using depth scan
            action_desc = "PATROL_MAZE"
            front_tile_x = int(self.player_x + math.cos(self.player_angle) * 1.0)
            front_tile_y = int(self.player_y + math.sin(self.player_angle) * 1.0)
            if 0 <= front_tile_x < self.map_w and 0 <= front_tile_y < self.map_h:
                tile = self.map_grid[front_tile_y][front_tile_x]
                if tile in (WALL_RED_DOOR, WALL_BLUE_DOOR, WALL_EXIT):
                    do_use = True
                    action_desc = "ACTIVATE_DOOR"

            # Check forward clearance
            clearance = self._get_forward_clearance()
            if clearance < 1.3:
                # Turn away from obstacle
                turn = 1.2 if random.random() < 0.5 else -1.2
                forward = 0.0
            else:
                forward = 1.0
                turn = (random.random() - 0.5) * 0.4

        # Apply actions
        self.last_action_desc = action_desc
        self.turn_player(turn * 2.8 * dt)
        self.move_player(forward, strafe, dt)
        if do_fire:
            self.fire_weapon()
        if do_use:
            self.interact_use()

        self.update(dt)

        # Record trajectory transition tuple
        obs = self.get_state_vector()
        action_val = [forward, strafe, turn, 1.0 if do_fire else 0.0, 1.0 if do_use else 0.0]
        self.trajectory.append((obs, action_val, self.episode_reward, self.get_state_vector(), self.is_dead or self.level_completed))

        return {
            "action": action_desc,
            "forward": forward,
            "turn": turn,
            "strafe": strafe,
            "fire": do_fire,
            "reward": self.episode_reward,
        }

    def _get_forward_clearance(self) -> float:
        """Measure distance to obstacle directly in front."""
        dist = 0.2
        while dist < 8.0:
            cx = int(self.player_x + math.cos(self.player_angle) * dist)
            cy = int(self.player_y + math.sin(self.player_angle) * dist)
            if self._is_wall_blocking(cx, cy):
                return dist
            dist += 0.2
        return 8.0

    def get_state_vector(self) -> List[float]:
        """Normalized 8-dimensional feature vector for RL observation."""
        return [
            self.player_x / max(1, self.map_w),
            self.player_y / max(1, self.map_h),
            self.player_angle / (2 * math.pi),
            self.health / 100.0,
            self.armor / 100.0,
            min(1.0, self.ammo.get("shells", 0) / 50.0),
            float(self.frags),
            1.0 if self.level_completed else 0.0,
        ]

    # ── Benchmark Mode: Can It Run DOOM? ─────────────────────────────────────
    def run_benchmark(self, num_frames: int = 150) -> Dict[str, Any]:
        """Perform high-stress unthrottled raycasting benchmark."""
        start_time = time.perf_counter()
        rays_cast_total = num_frames * SCREEN_W

        for _ in range(num_frames):
            # Step camera
            self.turn_player(0.04)
            # Full DDA Raycast pass
            z_buffer, wall_hits = self.render_raycast()
            # Sprite sorting pass
            self.get_sorted_sprites(z_buffer)

        duration = max(0.0001, time.perf_counter() - start_time)
        fps = num_frames / duration
        avg_ms = (duration / num_frames) * 1000.0

        verdict = "CAN IT RUN DOOM? YES!"
        status = "BLAZING FAST (Theta-Certified)" if fps > 120 else "SMOOTH (Theta-Certified)"

        return {
            "verdict": verdict,
            "fps": round(fps, 1),
            "avg_ms": round(avg_ms, 2),
            "total_frames": num_frames,
            "total_rays": rays_cast_total,
            "duration_sec": round(duration, 3),
            "rating": status,
        }

    # ── DDA Raycasting ───────────────────────────────────────────────────────
    def render_raycast(self) -> Tuple[List[float], List[Dict[str, Any]]]:
        """Cast rays across viewport using Digital Differential Analysis."""
        z_buffer = [999.0] * SCREEN_W
        wall_hits = []

        px = self.player_x
        py = self.player_y
        pa = self.player_angle

        for col, rel_angle in enumerate(self.ray_angles):
            ray_a = pa + rel_angle
            sin_a = math.sin(ray_a)
            cos_a = math.cos(ray_a)

            # Prevent division by zero
            cos_a = 0.00001 if cos_a == 0 else cos_a
            sin_a = 0.00001 if sin_a == 0 else sin_a

            map_x = int(px)
            map_y = int(py)

            delta_dist_x = abs(1.0 / cos_a)
            delta_dist_y = abs(1.0 / sin_a)

            if cos_a < 0:
                step_x = -1
                side_dist_x = (px - map_x) * delta_dist_x
            else:
                step_x = 1
                side_dist_x = (map_x + 1.0 - px) * delta_dist_x

            if sin_a < 0:
                step_y = -1
                side_dist_y = (py - map_y) * delta_dist_y
            else:
                step_y = 1
                side_dist_y = (map_y + 1.0 - py) * delta_dist_y

            hit = False
            side = 0
            tile = 0
            dist = 0.0

            while not hit and dist < 16.0:
                if side_dist_x < side_dist_y:
                    side_dist_x += delta_dist_x
                    map_x += step_x
                    side = 0
                else:
                    side_dist_y += delta_dist_y
                    map_y += step_y
                    side = 1

                if 0 <= map_x < self.map_w and 0 <= map_y < self.map_h:
                    tile = self.map_grid[map_y][map_x]
                    if tile > 0 and tile != WALL_DOOR_OPEN:
                        hit = True
                else:
                    hit = True
                    tile = WALL_TECH

            # Perpendicular distance to remove fisheye
            if side == 0:
                perp_wall_dist = (map_x - px + (1 - step_x) / 2.0) / cos_a
            else:
                perp_wall_dist = (map_y - py + (1 - step_y) / 2.0) / sin_a

            perp_wall_dist = max(0.1, perp_wall_dist)
            z_buffer[col] = perp_wall_dist

            wall_hits.append({
                "col": col,
                "dist": perp_wall_dist,
                "tile": tile,
                "side": side,
                "height": int(SCREEN_H / perp_wall_dist),
            })

        return z_buffer, wall_hits

    def get_sorted_sprites(self, z_buffer: List[float]) -> List[Dict[str, Any]]:
        """Project and sort sprites by depth."""
        sprites = []
        for e in self.entities:
            dx = e.x - self.player_x
            dy = e.y - self.player_y
            dist = math.hypot(dx, dy)
            if dist < 0.2 or dist > 14.0:
                continue

            # Angle relative to player angle
            sprite_angle = math.atan2(dy, dx) - self.player_angle
            sprite_angle = (sprite_angle + math.pi) % (2 * math.pi) - math.pi

            if abs(sprite_angle) > HALF_FOV + 0.35:
                continue

            screen_x = int((SCREEN_W / 2.0) + math.tan(sprite_angle) * (SCREEN_W / 2.0) / math.tan(HALF_FOV))
            sprites.append({
                "entity": e,
                "dist": dist,
                "screen_x": screen_x,
                "scale": int(SCREEN_H / dist),
            })

        sprites.sort(key=lambda s: s["dist"], reverse=True)
        return sprites
