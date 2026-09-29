"""DOOM Panel: Interactive 3D Raycasting Viewport, HUD, and NeSyRL Telemetry."""
from __future__ import annotations
import json
import math
from pathlib import Path
import time
from typing import TYPE_CHECKING, Any, Dict, List, Optional, Set

from PyQt6.QtCore import QPoint, QRect, QRectF, QSize, Qt, QTimer, QUrl
from PyQt6.QtGui import (
    QBrush, QColor, QFont, QIcon, QKeyEvent, QPainter, QPaintEvent,
    QPen, QPixmap, QPolygon
)
from PyQt6.QtWidgets import (
    QComboBox, QFileDialog, QFrame, QGridLayout, QGroupBox,
    QHBoxLayout, QLabel, QMenu, QMessageBox, QPushButton,
    QScrollArea, QSizePolicy, QSplitter, QStackedWidget, QVBoxLayout, QWidget
)

try:
    from PyQt6.QtWebEngineWidgets import QWebEngineView
    HAS_WEBENGINE = True
except ImportError:
    HAS_WEBENGINE = False

from frontend.plugins.doom.engine import (
    DoomEngine, Entity, LEVELS, SCREEN_H, SCREEN_W,
    WALL_BLUE_DOOR, WALL_BRONZE, WALL_COMPUTER, WALL_DOOR_OPEN,
    WALL_EMPTY, WALL_EXIT, WALL_HAZARD, WALL_RED_DOOR, WALL_SLIME,
    WALL_TECH, WEAPONS
)

if TYPE_CHECKING:
    from frontend.plugins.context import PluginContext


# ── Retro Palette & Material Shading ─────────────────────────────────────────
WALL_COLORS = {
    WALL_TECH: (QColor("#4a5568"), QColor("#2d3748")),
    WALL_BRONZE: (QColor("#9c4221"), QColor("#652b14")),
    WALL_HAZARD: (QColor("#d69e2e"), QColor("#744210")),
    WALL_COMPUTER: (QColor("#2b6cb0"), QColor("#1a365d")),
    WALL_SLIME: (QColor("#38a169"), QColor("#1c4532")),
    WALL_RED_DOOR: (QColor("#e53e3e"), QColor("#742a2a")),
    WALL_BLUE_DOOR: (QColor("#3182ce"), QColor("#1a365d")),
    WALL_EXIT: (QColor("#dd6b20"), QColor("#652b14")),
    WALL_DOOR_OPEN: (QColor("#2d3748"), QColor("#1a202c")),
}


class DoomCanvas(QWidget):
    """Real-time 3D Raycaster Viewport and authentic DOOM HUD."""

    def __init__(self, engine: DoomEngine, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.engine = engine
        self.setFocusPolicy(Qt.FocusPolicy.StrongFocus)
        self.setMinimumSize(480, 360)
        self.setSizePolicy(QSizePolicy.Policy.Expanding, QSizePolicy.Policy.Expanding)

        # Active key tracker for smooth keyboard movement
        self.active_keys: Set[int] = set()

        # FPS calculation
        self.last_frame_time = time.perf_counter()
        self.fps: float = 60.0

    def keyPressEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        self.active_keys.add(key)

        # Discrete key actions
        if key in (Qt.Key.Key_Space, Qt.Key.Key_Control, Qt.Key.Key_F):
            self.engine.fire_weapon()
        elif key == Qt.Key.Key_E:
            self.engine.interact_use()
        elif key == Qt.Key.Key_1:
            self.engine.select_weapon("fist")
        elif key == Qt.Key.Key_2:
            self.engine.select_weapon("pistol")
        elif key == Qt.Key.Key_3:
            self.engine.select_weapon("shotgun")
        elif key == Qt.Key.Key_4:
            self.engine.select_weapon("chaingun")
        elif key == Qt.Key.Key_5:
            self.engine.select_weapon("bfg")

        event.accept()

    def keyReleaseEvent(self, event: QKeyEvent) -> None:
        key = event.key()
        self.active_keys.discard(key)
        event.accept()

    def process_keyboard_input(self, dt: float) -> None:
        """Poll active keys for smooth translation and rotation."""
        forward = 0.0
        strafe = 0.0
        turn = 0.0

        if Qt.Key.Key_W in self.active_keys or Qt.Key.Key_Up in self.active_keys:
            forward += 1.0
        if Qt.Key.Key_S in self.active_keys or Qt.Key.Key_Down in self.active_keys:
            forward -= 1.0
        if Qt.Key.Key_A in self.active_keys:
            strafe -= 1.0
        if Qt.Key.Key_D in self.active_keys:
            strafe += 1.0
        if Qt.Key.Key_Left in self.active_keys or Qt.Key.Key_Q in self.active_keys:
            turn -= 1.0
        if Qt.Key.Key_Right in self.active_keys or Qt.Key.Key_E in self.active_keys:
            turn += 1.0

        if turn != 0.0:
            self.engine.turn_player(turn * 2.8 * dt)
        if forward != 0.0 or strafe != 0.0:
            self.engine.move_player(forward, strafe, dt)

    def paintEvent(self, event: QPaintEvent) -> None:
        now = time.perf_counter()
        dt = max(0.0001, now - self.last_frame_time)
        self.last_frame_time = now
        self.fps = 0.9 * self.fps + 0.1 * (1.0 / dt)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing, False)

        w = self.width()
        h = self.height()

        # Allocate 80% to 3D Viewport, 20% to DOOM HUD Status Bar
        hud_h = max(68, int(h * 0.18))
        view_h = h - hud_h

        # ── 1. Render Sky & Floor ────────────────────────────────────────────
        # Dark tech ceiling
        painter.fillRect(0, 0, w, view_h // 2, QColor("#12141a"))
        # Metallic floor
        painter.fillRect(0, view_h // 2, w, view_h // 2, QColor("#1e2029"))

        # Toxic acid sludge patch in center
        slime_brush = QBrush(QColor("#0d2e1a"))
        painter.fillRect(0, int(view_h * 0.72), w, int(view_h * 0.28), slime_brush)

        # ── 2. DDA Raycast Walls ─────────────────────────────────────────────
        z_buffer, wall_hits = self.engine.render_raycast()
        col_w = w / float(SCREEN_W)

        for hit in wall_hits:
            col = hit["col"]
            dist = hit["dist"]
            tile = hit["tile"]
            side = hit["side"]
            wall_h_screen = int((SCREEN_H / dist) * (view_h / float(SCREEN_H)))

            top_y = (view_h - wall_h_screen) // 2
            x = int(col * col_w)
            slice_w = max(1, int(col_w + 1.0))

            base_col, dark_col = WALL_COLORS.get(tile, (QColor("#4a5568"), QColor("#2d3748")))
            color = dark_col if side == 1 else base_col

            # Distance depth fog
            fog_factor = min(1.0, dist / 12.0)
            fogged_r = int(color.red() * (1.0 - fog_factor * 0.75))
            fogged_g = int(color.green() * (1.0 - fog_factor * 0.75))
            fogged_b = int(color.blue() * (1.0 - fog_factor * 0.75))

            # Hazard stripe pattern
            if tile == WALL_HAZARD and ((col // 4) % 2 == 0):
                fogged_r = int(220 * (1.0 - fog_factor * 0.7))
                fogged_g = int(180 * (1.0 - fog_factor * 0.7))
                fogged_b = 20

            # Computer terminal blinkers
            if tile == WALL_COMPUTER and ((col // 3) % 3 == 0) and (int(now * 3) % 2 == 0):
                fogged_r, fogged_g, fogged_b = 30, 240, 180

            painter.fillRect(x, max(0, top_y), slice_w, min(view_h, wall_h_screen), QColor(fogged_r, fogged_g, fogged_b))

        # ── 3. Render Sprites (Demons, Pickups, Barrels) ─────────────────────
        sprites = self.engine.get_sorted_sprites(z_buffer)
        for s in sprites:
            e: Entity = s["entity"]
            dist = s["dist"]
            screen_x_norm = s["screen_x"] / float(SCREEN_W)
            center_x = int(screen_x_norm * w)
            sprite_size = int((SCREEN_H / dist) * (view_h / float(SCREEN_H)) * 0.85)

            if sprite_size <= 2:
                continue

            sx = center_x - sprite_size // 2
            sy = (view_h - sprite_size) // 2 + int(sprite_size * 0.15)

            # Check if occluded by walls
            col_idx = max(0, min(SCREEN_W - 1, int(screen_x_norm * SCREEN_W)))
            if z_buffer[col_idx] < dist:
                continue

            self._draw_entity_sprite(painter, e, sx, sy, sprite_size)

        # ── 4. Crosshair ─────────────────────────────────────────────────────
        cx = w // 2
        cy = view_h // 2
        painter.setPen(QPen(QColor(255, 50, 50, 180), 2))
        painter.drawLine(cx - 6, cy, cx + 6, cy)
        painter.drawLine(cx, cy - 6, cx, cy + 6)

        # ── 5. Weapon in Hand (First Person) ─────────────────────────────────
        self._draw_weapon(painter, w, view_h)

        # ── 6. Screen Flash Overlay ──────────────────────────────────────────
        if self.engine.screen_flash:
            r, g, b, alpha = self.engine.screen_flash
            flash_col = QColor(r, g, b, int(alpha * 255))
            painter.fillRect(0, 0, w, view_h, flash_col)

        # ── 7. DOOM HUD Status Bar ───────────────────────────────────────────
        self._draw_doom_hud(painter, 0, view_h, w, hud_h)

    def _draw_entity_sprite(self, painter: QPainter, e: Entity, sx: int, sy: int, size: int) -> None:
        """Render demons, pickups, and barrels with authentic silhouettes."""
        if e.state == "dead":
            # Splattered gibs / puddle
            painter.setBrush(QBrush(QColor("#742a2a")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(sx, sy + int(size * 0.7), size, int(size * 0.25))
            return

        if e.is_enemy:
            # Color by demon type
            if e.kind == "imp":
                body_col = QColor("#8c5836")
                eye_col = QColor("#ff2222")
            elif e.kind == "pinky":
                body_col = QColor("#d53f8c")
                eye_col = QColor("#ffffff")
            elif e.kind == "cacodemon":
                body_col = QColor("#c53030")
                eye_col = QColor("#38a169")
            else:  # Baron of Hell
                body_col = QColor("#4a154b")
                eye_col = QColor("#48bb78")

            if e.pain_timer > 0.0:
                body_col = QColor("#ffffff")  # Hurt flash

            # Torso & Head
            painter.setBrush(QBrush(body_col))
            painter.setPen(QPen(QColor("#000000"), 2))
            painter.drawEllipse(sx + size // 4, sy, size // 2, int(size * 0.65))

            # Horns / spikes
            if e.kind in ("imp", "baron", "cacodemon"):
                horn_poly = QPolygon([
                    QPoint(sx + size // 4, sy + size // 6),
                    QPoint(sx + size // 6, sy - size // 8),
                    QPoint(sx + size // 3, sy + size // 8),
                ])
                painter.drawPolygon(horn_poly)

            # Glowing demonic eyes
            painter.setBrush(QBrush(eye_col))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(sx + size // 3, sy + size // 5, max(2, size // 10), max(2, size // 10))
            painter.drawEllipse(sx + int(size * 0.55), sy + size // 5, max(2, size // 10), max(2, size // 10))

            # Health bar above demon
            if e.health < e.max_health:
                hb_w = max(16, size // 2)
                hb_x = sx + (size - hb_w) // 2
                hb_y = sy - 8
                painter.fillRect(hb_x, hb_y, hb_w, 4, QColor("#333333"))
                cur_w = int(hb_w * (e.health / float(e.max_health)))
                painter.fillRect(hb_x, hb_y, cur_w, 4, QColor("#e53e3e"))

        elif e.is_pickup:
            # Pickup item boxes
            if e.kind == "medikit":
                painter.fillRect(sx + size // 4, sy + size // 4, size // 2, size // 2, QColor("#f7fafc"))
                painter.setPen(QPen(QColor("#e53e3e"), max(2, size // 8)))
                cx = sx + size // 2
                cy = sy + size // 2
                d = size // 6
                painter.drawLine(cx - d, cy, cx + d, cy)
                painter.drawLine(cx, cy - d, cx, cy + d)
            elif e.kind == "stimpack":
                painter.fillRect(sx + size // 3, sy + size // 4, size // 3, size // 2, QColor("#edf2f7"))
                painter.fillRect(sx + int(size * 0.42), sy + size // 8, size // 6, size // 4, QColor("#3182ce"))
            elif e.kind == "armor":
                painter.setBrush(QBrush(QColor("#38a169")))
                painter.setPen(QPen(QColor("#1c4532"), 2))
                painter.drawRoundedRect(sx + size // 4, sy + size // 4, size // 2, size // 2, 4, 4)
            elif e.kind in ("blue_key", "red_key"):
                col = QColor("#3182ce") if e.kind == "blue_key" else QColor("#e53e3e")
                painter.setBrush(QBrush(col))
                painter.setPen(QPen(QColor("#ffffff"), 1))
                painter.drawRect(sx + size // 3, sy + size // 3, size // 3, size // 2)
            else:  # Ammo boxes
                painter.fillRect(sx + size // 4, sy + size // 3, size // 2, size // 3, QColor("#d69e2e"))

        elif e.is_barrel:
            # Toxic radioactive hazard barrel
            painter.setBrush(QBrush(QColor("#276749")))
            painter.setPen(QPen(QColor("#1c4532"), 2))
            painter.drawRoundedRect(sx + size // 4, sy + size // 5, size // 2, int(size * 0.7), 4, 4)
            # Hazard radiation symbol
            painter.setBrush(QBrush(QColor("#ecc94b")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(sx + int(size * 0.42), sy + int(size * 0.45), size // 6, size // 6)

    def _draw_weapon(self, painter: QPainter, w: int, h: int) -> None:
        """Render weapon in hand with recoil kickback and muzzle flash."""
        weapon = self.engine.current_weapon
        anim = self.engine.weapon_anim  # >0 when firing

        recoil_y = int(anim * 40.0)
        base_x = w // 2
        base_y = h - 10 + recoil_y

        if weapon == "fist":
            # Clenched boxing knuckle glove
            painter.setBrush(QBrush(QColor("#b7791f")))
            painter.setPen(QPen(QColor("#000000"), 3))
            punch_offset = int(anim * 50)
            painter.drawRoundedRect(base_x - 35, base_y - 70 - punch_offset, 70, 80, 10, 10)
        elif weapon == "pistol":
            # Tactical handgun
            painter.setBrush(QBrush(QColor("#2d3748")))
            painter.setPen(QPen(QColor("#1a202c"), 2))
            painter.drawRect(base_x - 12, base_y - 85, 24, 85)
            # Barrel top
            painter.fillRect(base_x - 8, base_y - 110, 16, 25, QColor("#1a202c"))
            if anim > 0.08:
                # Muzzle flash
                painter.setBrush(QBrush(QColor("#ecc94b")))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(base_x - 18, base_y - 135, 36, 36)
        elif weapon == "shotgun":
            # Double barrel sawed-off shotgun
            painter.setBrush(QBrush(QColor("#4a5568")))
            painter.setPen(QPen(QColor("#1a202c"), 3))
            painter.drawRect(base_x - 30, base_y - 110, 26, 110)
            painter.drawRect(base_x + 4, base_y - 110, 26, 110)
            if anim > 0.08:
                # Enormous shotgun flash
                painter.setBrush(QBrush(QColor("#f6ad55")))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(base_x - 45, base_y - 155, 90, 50)
        elif weapon == "chaingun":
            # Triple rotating barrel
            painter.setBrush(QBrush(QColor("#2d3748")))
            painter.setPen(QPen(QColor("#000000"), 2))
            painter.drawRect(base_x - 24, base_y - 100, 48, 100)
            if anim > 0.02:
                painter.setBrush(QBrush(QColor("#f6e05e")))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawEllipse(base_x - 20, base_y - 125, 40, 30)
        elif weapon == "bfg":
            # BFG 9000 Green Energy Cannon
            painter.setBrush(QBrush(QColor("#1c4532")))
            painter.setPen(QPen(QColor("#22543d"), 3))
            painter.drawRoundedRect(base_x - 65, base_y - 120, 130, 120, 15, 15)
            # Glowing plasma core
            painter.setBrush(QBrush(QColor("#48bb78")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(base_x - 30, base_y - 100, 60, 60)
            if anim > 0.05:
                # Huge green plasma blast
                painter.setBrush(QBrush(QColor("#9ae6b4")))
                painter.drawEllipse(base_x - 80, base_y - 160, 160, 90)

    def _draw_doom_hud(self, painter: QPainter, x: int, y: int, w: int, h: int) -> None:
        """Render the classic DOOM Status Bar HUD."""
        # 1. Beveled background bar
        painter.fillRect(x, y, w, h, QColor("#1e2029"))
        painter.setPen(QPen(QColor("#4a5568"), 3))
        painter.drawLine(x, y, x + w, y)

        section_w = w // 5

        # Font setup
        font_num = QFont("Helvetica", max(13, int(h * 0.32)), QFont.Weight.Bold)
        font_lbl = QFont("Helvetica", max(8, int(h * 0.15)), QFont.Weight.Bold)

        # ── SLOT 1: AMMO ─────────────────────────────────────────────────────
        spec = WEAPONS.get(self.engine.current_weapon, WEAPONS["pistol"])
        ammo_type = spec["ammo_type"]
        cur_ammo = self.engine.ammo.get(ammo_type, 0) if ammo_type else "∞"

        painter.setFont(font_lbl)
        painter.setPen(QColor("#718096"))
        painter.drawText(x + 12, y + int(h * 0.32), "AMMO")
        painter.setFont(font_num)
        painter.setPen(QColor("#ecc94b"))
        painter.drawText(x + 12, y + int(h * 0.78), str(cur_ammo))

        # ── SLOT 2: HEALTH ───────────────────────────────────────────────────
        hp = self.engine.health
        painter.setFont(font_lbl)
        painter.setPen(QColor("#718096"))
        painter.drawText(x + section_w, y + int(h * 0.32), "HEALTH")
        painter.setFont(font_num)
        hp_col = QColor("#e53e3e") if hp <= 30 else QColor("#f7fafc")
        painter.setPen(hp_col)
        painter.drawText(x + section_w, y + int(h * 0.78), f"{hp}%")

        # ── SLOT 3: DOOMGUY FACE ─────────────────────────────────────────────
        face_cx = x + int(section_w * 2.5)
        face_box_size = int(h * 0.8)
        face_x = face_cx - face_box_size // 2
        face_y = y + int(h * 0.1)

        # Beveled inset
        painter.fillRect(face_x, face_y, face_box_size, face_box_size, QColor("#12141a"))
        painter.setPen(QPen(QColor("#4a5568"), 1))
        painter.drawRect(face_x, face_y, face_box_size, face_box_size)

        self._draw_doomguy_face(painter, face_x, face_y, face_box_size)

        # ── SLOT 4: ARMOR ────────────────────────────────────────────────────
        armor = self.engine.armor
        painter.setFont(font_lbl)
        painter.setPen(QColor("#718096"))
        painter.drawText(x + int(section_w * 3.3), y + int(h * 0.32), "ARMOR")
        painter.setFont(font_num)
        painter.setPen(QColor("#3182ce"))
        painter.drawText(x + int(section_w * 3.3), y + int(h * 0.78), f"{armor}%")

        # ── SLOT 5: ARMS & KEYS ──────────────────────────────────────────────
        right_x = x + int(section_w * 4.1)
        painter.setFont(font_lbl)
        painter.setPen(QColor("#718096"))
        painter.drawText(right_x, y + int(h * 0.32), "ARMS / KEYS")

        # Weapon slots [1][2][3][4][5]
        arms_text = ""
        for i, w_id in enumerate(["fist", "pistol", "shotgun", "chaingun", "bfg"], 1):
            if self.engine.owned_weapons.get(w_id):
                arms_text += f" {i}"
        painter.setFont(QFont("Monospace", max(9, int(h * 0.2)), QFont.Weight.Bold))
        painter.setPen(QColor("#e2e8f0"))
        painter.drawText(right_x, y + int(h * 0.58), f"W:{arms_text}")

        # Keycards
        keys_str = ""
        if "blue" in self.engine.keys_held:
            keys_str += "🟦 "
        if "red" in self.engine.keys_held:
            keys_str += "🟥 "
        if not keys_str:
            keys_str = "—"
        painter.drawText(right_x, y + int(h * 0.85), f"K: {keys_str}")

    def _draw_doomguy_face(self, painter: QPainter, x: int, y: int, size: int) -> None:
        """Render procedurally animated pixel-art Doom Marine face."""
        cx = x + size // 2
        cy = y + size // 2

        # 1. Hair & Head
        painter.setBrush(QBrush(QColor("#5c3a21")))  # Brown hair
        painter.setPen(Qt.PenStyle.NoPen)
        painter.drawRoundedRect(cx - size // 3, cy - int(size * 0.42), int(size * 0.66), int(size * 0.84), 4, 4)

        # 2. Skin tone
        skin_col = QColor("#d49b74")
        if self.engine.is_dead:
            skin_col = QColor("#8c7365")  # Pale dead skin

        painter.setBrush(QBrush(skin_col))
        painter.drawRect(cx - size // 4, cy - int(size * 0.28), size // 2, int(size * 0.65))

        # 3. Eyes
        eye_y = cy - int(size * 0.12)
        eye_size = max(2, size // 8)
        left_eye_x = cx - size // 5
        right_eye_x = cx + size // 10

        if self.engine.god_mode:
            # IDDQD: Radiant glowing golden eyes!
            painter.setBrush(QBrush(QColor("#ecc94b")))
            painter.drawRect(left_eye_x, eye_y, eye_size, eye_size)
            painter.drawRect(right_eye_x, eye_y, eye_size, eye_size)
        elif self.engine.is_dead:
            # X eyes
            painter.setPen(QPen(QColor("#742a2a"), 2))
            painter.drawLine(left_eye_x, eye_y, left_eye_x + eye_size, eye_y + eye_size)
            painter.drawLine(right_eye_x, eye_y, right_eye_x + eye_size, eye_y + eye_size)
        else:
            # White sclera
            painter.setBrush(QBrush(QColor("#ffffff")))
            painter.drawRect(left_eye_x, eye_y, eye_size, eye_size)
            painter.drawRect(right_eye_x, eye_y, eye_size, eye_size)

            # Pupils tracking
            pupil_offset = 0
            if self.engine.face_state == "look_left":
                pupil_offset = -1
            elif self.engine.face_state == "look_right":
                pupil_offset = 1

            painter.setBrush(QBrush(QColor("#1a202c")))
            painter.drawRect(left_eye_x + 1 + pupil_offset, eye_y + 1, eye_size // 2, eye_size // 2)
            painter.drawRect(right_eye_x + 1 + pupil_offset, eye_y + 1, eye_size // 2, eye_size // 2)

        # 4. Mouth / Expressions
        mouth_y = cy + int(size * 0.16)
        if self.engine.face_state == "grin":
            # Evil grin!
            painter.setPen(QPen(QColor("#1a202c"), 2))
            painter.drawLine(cx - size // 6, mouth_y, cx + size // 6, mouth_y)
            painter.drawLine(cx - size // 6, mouth_y - 2, cx - size // 6, mouth_y)
            painter.drawLine(cx + size // 6, mouth_y - 2, cx + size // 6, mouth_y)
        elif self.engine.face_state == "hurt" or self.engine.health < 40:
            # Bloody battered grimace / ouch
            painter.setBrush(QBrush(QColor("#742a2a")))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRect(cx - size // 8, mouth_y, size // 4, size // 6)
            # Nose bleed
            painter.fillRect(cx - 1, cy, 3, size // 5, QColor("#e53e3e"))
        else:
            # Determined grimace
            painter.setPen(QPen(QColor("#1a202c"), 2))
            painter.drawLine(cx - size // 7, mouth_y, cx + size // 7, mouth_y)


# ── The Main DOOM Plugin Panel ───────────────────────────────────────────────
class DoomPanel(QWidget):
    """Integrated DOOM Simulation, NeSyRL Bot Testbed, and Benchmark Panel."""

    def __init__(self, context: PluginContext, parent: Optional[QWidget] = None):
        super().__init__(parent)
        self.context = context
        self.engine = DoomEngine("E1M1: Hangar")
        self.is_ai_bot_active = False

        self._build_ui()

        # Main game simulation timer: 40 FPS
        self.timer = QTimer(self)
        self.timer.setInterval(25)  # ~40 FPS
        self.timer.timeout.connect(self._game_tick)
        self.timer.start()

    def _build_ui(self) -> None:
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(14, 12, 14, 12)
        main_layout.setSpacing(10)

        # ── 1. Top Header & Control Toolbar ──────────────────────────────────
        top_bar = QHBoxLayout()
        top_bar.setSpacing(8)

        # Title & eyebrow
        title_box = QVBoxLayout()
        title_box.setSpacing(2)
        eyebrow = QLabel("CAN IT RUN DOOM? — SIMULATOR")
        eyebrow.setStyleSheet("font-size: 11px; font-weight: 700; letter-spacing: 1px; color: #f38ba8;")
        self.lbl_title = QLabel("DOOM Sim & NeSyRL Agent Benchmark")
        self.lbl_title.setStyleSheet("font-size: 14px; font-weight: 600;")
        title_box.addWidget(eyebrow)
        title_box.addWidget(self.lbl_title)
        top_bar.addLayout(title_box)

        top_bar.addStretch()

        # Level Selector
        self.combo_level = QComboBox()
        for lvl in LEVELS.keys():
            self.combo_level.addItem(lvl)
        self.combo_level.currentTextChanged.connect(self._change_level)
        top_bar.addWidget(self.combo_level)

        # Mode Selector: Manual vs AI Bot
        self.btn_mode = QPushButton("🎮 Manual Play")
        self.btn_mode.setToolTip("Toggle between Keyboard Control and Autonomous AI Bot Simulation")
        self.btn_mode.clicked.connect(self._toggle_mode)
        top_bar.addWidget(self.btn_mode)

        # Benchmark Button
        self.btn_benchmark = QPushButton("⚡ Benchmark FPS")
        self.btn_benchmark.setStyleSheet("font-weight: 600; color: #a6e3a1;")
        self.btn_benchmark.setToolTip("Stress-test raycasting performance to answer: Can Theta run DOOM?")
        self.btn_benchmark.clicked.connect(self.run_benchmark_test)
        top_bar.addWidget(self.btn_benchmark)

        # Cheats Menu
        self.btn_cheats = QPushButton("💀 Cheats")
        cheats_menu = QMenu(self)
        cheats_menu.addAction("IDDQD: Toggle God Mode", self._cheat_god_mode)
        cheats_menu.addAction("IDKFA: All Weapons & Keys", self._cheat_ammo)
        cheats_menu.addAction("IDCLIP: Toggle No-Clip", self._cheat_noclip)
        self.btn_cheats.setMenu(cheats_menu)
        top_bar.addWidget(self.btn_cheats)

        # Export Trajectory Dataset
        self.btn_export = QPushButton("💾 Export Dataset")
        self.btn_export.setToolTip("Export recorded RL transitions (s, a, r, s', done) to Theta dataset")
        self.btn_export.clicked.connect(self._export_dataset)
        top_bar.addWidget(self.btn_export)

        # Restart
        self.btn_restart = QPushButton("🔄 Restart")
        self.btn_restart.clicked.connect(self._restart_level)
        top_bar.addWidget(self.btn_restart)

        # Mode Switcher: RL Simulator vs Playable DOOM (WASM)
        self.btn_view_rl = QPushButton("🤖 RL Simulator")
        self.btn_view_rl.setCheckable(True)
        self.btn_view_rl.setChecked(True)
        self.btn_view_rl.setStyleSheet("font-weight: 600;")
        self.btn_view_rl.clicked.connect(lambda: self._switch_view(0))
        top_bar.addWidget(self.btn_view_rl)

        self.btn_view_wasm = QPushButton("🕹️ Play Real DOOM")
        self.btn_view_wasm.setCheckable(True)
        self.btn_view_wasm.setToolTip("Play the authentic 1993 DOOM engine in embedded WebEngine")
        self.btn_view_wasm.clicked.connect(lambda: self._switch_view(1))
        top_bar.addWidget(self.btn_view_wasm)

        main_layout.addLayout(top_bar)

        # ── 2. Stacked Views (RL Simulator vs Real DOOM) ─────────────────────
        self.stacked = QStackedWidget(self)

        # View 0: 3D Viewport & Telemetry Splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: 3D Canvas
        self.canvas = DoomCanvas(self.engine, parent=self)
        splitter.addWidget(self.canvas)

        # Right: Telemetry Sidebar
        sidebar_widget = QWidget()
        sidebar_layout = QVBoxLayout(sidebar_widget)
        sidebar_layout.setContentsMargins(8, 0, 0, 0)
        sidebar_layout.setSpacing(10)

        # Card 1: Benchmark Card
        bench_card = QGroupBox("CAN IT RUN DOOM?")
        bench_card.setStyleSheet("QGroupBox { font-weight: bold; color: #f38ba8; }")
        bench_layout = QVBoxLayout(bench_card)
        self.lbl_bench_verdict = QLabel("🔥 VERDICT: IT RUNS DOOM!")
        self.lbl_bench_verdict.setStyleSheet("font-size: 13px; font-weight: bold; color: #a6e3a1;")
        self.lbl_fps_live = QLabel("Live Framerate: 60.0 FPS")
        self.lbl_bench_detail = QLabel("Raycast DDA: 160 columns\nTheta Performance Certified")
        self.lbl_bench_detail.setStyleSheet("color: #89b4fa; font-size: 11px;")
        bench_layout.addWidget(self.lbl_bench_verdict)
        bench_layout.addWidget(self.lbl_fps_live)
        bench_layout.addWidget(self.lbl_bench_detail)
        sidebar_layout.addWidget(bench_card)

        # Card 2: Mission Telemetry
        stats_card = QGroupBox("MISSION TELEMETRY")
        stats_layout = QGridLayout(stats_card)
        stats_layout.addWidget(QLabel("Frags / Slain:"), 0, 0)
        self.lbl_frags = QLabel("0")
        stats_layout.addWidget(self.lbl_frags, 0, 1)

        stats_layout.addWidget(QLabel("Score:"), 1, 0)
        self.lbl_score = QLabel("0")
        stats_layout.addWidget(self.lbl_score, 1, 1)

        stats_layout.addWidget(QLabel("RL Reward:"), 2, 0)
        self.lbl_reward = QLabel("0.0")
        stats_layout.addWidget(self.lbl_reward, 2, 1)

        stats_layout.addWidget(QLabel("Bot Action:"), 3, 0)
        self.lbl_action = QLabel("IDLE")
        stats_layout.addWidget(self.lbl_action, 3, 1)

        stats_layout.addWidget(QLabel("Transitions:"), 4, 0)
        self.lbl_steps = QLabel("0")
        stats_layout.addWidget(self.lbl_steps, 4, 1)

        sidebar_layout.addWidget(stats_card)

        # Card 3: Mini Radar / Map
        radar_card = QGroupBox("TACTICAL RADAR (MINIMAP)")
        radar_layout = QVBoxLayout(radar_card)
        self.radar_label = QLabel()
        self.radar_label.setFixedSize(160, 160)
        self.radar_label.setStyleSheet("background-color: #11111b; border: 1px solid #45475a; border-radius: 4px;")
        radar_layout.addWidget(self.radar_label, alignment=Qt.AlignmentFlag.AlignCenter)
        sidebar_layout.addWidget(radar_card)

        sidebar_layout.addStretch()

        # Keyboard instructions
        help_lbl = QLabel(
            "<b>Controls:</b><br>"
            "• <b>W / S</b>: Forward / Back<br>"
            "• <b>A / D</b>: Strafe Left / Right<br>"
            "• <b>← / →</b>: Turn View<br>"
            "• <b>Space / F</b>: Fire Weapon<br>"
            "• <b>E</b>: Open Door / Switch<br>"
            "• <b>1 - 5</b>: Switch Weapon"
        )
        help_lbl.setStyleSheet("font-size: 11px; color: #a6adc8; padding: 4px;")
        sidebar_layout.addWidget(help_lbl)

        splitter.addWidget(sidebar_widget)
        splitter.setStretchFactor(0, 4)
        splitter.setStretchFactor(1, 1)
        self.stacked.addWidget(splitter)

        # View 1: Real Playable DOOM WebEngine View
        if HAS_WEBENGINE:
            self.web_view = QWebEngineView(self)
            doom_html = Path(__file__).parent / "doom.html"
            if doom_html.exists():
                self.web_view.setUrl(QUrl.fromLocalFile(str(doom_html.resolve())))
            else:
                self.web_view.setUrl(QUrl("https://dos.zone/player/?bundleUrl=https%3A%2F%2Fcdn.dos.zone%2Fcustom%2Fdos%2Fdoom.jsdos"))
            self.stacked.addWidget(self.web_view)
        else:
            self.web_view = None
            fallback_label = QLabel(
                "🎮 <b>Playable DOOM (WASM)</b><br><br>"
                "To play the authentic 1993 DOOM engine, install <code>PyQt6-WebEngine</code>.<br>"
                "Switch back to <b>RL Simulator</b> to run the local 3D raycasting engine!"
            )
            fallback_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
            fallback_label.setStyleSheet("font-size: 14px; color: #cdd6f4; padding: 40px;")
            self.stacked.addWidget(fallback_label)

        main_layout.addWidget(self.stacked)

    # ── Simulation Tick ──────────────────────────────────────────────────────
    def _game_tick(self) -> None:
        dt = 0.025  # ~40 FPS

        if self.is_ai_bot_active:
            res = self.engine.step_ai_bot(dt)
            self.lbl_action.setText(res.get("action", "IDLE"))
        else:
            self.canvas.process_keyboard_input(dt)
            self.engine.update(dt)

        # Update telemetry labels
        self.lbl_fps_live.setText(f"Live Framerate: {self.canvas.fps:.1f} FPS")
        self.lbl_frags.setText(str(self.engine.frags))
        self.lbl_score.setText(str(self.engine.score))
        self.lbl_reward.setText(f"{self.engine.episode_reward:.1f}")
        self.lbl_steps.setText(f"{self.engine.total_steps} transitions")

        # Repaint canvas & radar
        self.canvas.update()
        self._update_radar()

    def _update_radar(self) -> None:
        """Render 2D top-down minimap pixmap."""
        pixmap = QPixmap(160, 160)
        pixmap.fill(QColor("#11111b"))
        painter = QPainter(pixmap)

        mw = self.engine.map_w
        mh = self.engine.map_h
        cell_w = 160.0 / mw
        cell_h = 160.0 / mh

        # Draw walls
        for y in range(mh):
            for x in range(mw):
                tile = self.engine.map_grid[y][x]
                if tile > 0 and tile != WALL_DOOR_OPEN:
                    painter.fillRect(int(x * cell_w), int(y * cell_h), int(cell_w), int(cell_h), QColor("#45475a"))

        # Draw pickups
        painter.setBrush(QBrush(QColor("#f9e2af")))
        painter.setPen(Qt.PenStyle.NoPen)
        for e in self.engine.entities:
            if e.is_pickup and e.state != "dead":
                painter.drawEllipse(int(e.x * cell_w) - 2, int(e.y * cell_h) - 2, 4, 4)

        # Draw enemies (red dots)
        painter.setBrush(QBrush(QColor("#f38ba8")))
        for e in self.engine.entities:
            if e.is_enemy and e.state != "dead":
                painter.drawEllipse(int(e.x * cell_w) - 3, int(e.y * cell_h) - 3, 6, 6)

        # Draw player & view angle arrow
        px = int(self.engine.player_x * cell_w)
        py = int(self.engine.player_y * cell_h)
        painter.setBrush(QBrush(QColor("#a6e3a1")))
        painter.drawEllipse(px - 3, py - 3, 6, 6)

        # View direction
        p_angle = self.engine.player_angle
        ax = int(px + math.cos(p_angle) * 10)
        ay = int(py + math.sin(p_angle) * 10)
        painter.setPen(QPen(QColor("#a6e3a1"), 2))
        painter.drawLine(px, py, ax, ay)

        painter.end()
        self.radar_label.setPixmap(pixmap)

    # ── Level & Mode Management ──────────────────────────────────────────────
    def _change_level(self, level_name: str) -> None:
        self.engine = DoomEngine(level_name)
        self.canvas.engine = self.engine
        self.context.log(f"DOOM Sim: Loaded level {level_name}")

    def _restart_level(self) -> None:
        lvl = self.combo_level.currentText()
        self._change_level(lvl)
        self.context.show_status_message(f"Level {lvl} restarted")

    def _toggle_mode(self) -> None:
        self.is_ai_bot_active = not self.is_ai_bot_active
        if self.is_ai_bot_active:
            self.btn_mode.setText("🤖 AI Bot Sim")
            self.btn_mode.setStyleSheet("font-weight: bold; color: #89b4fa;")
            self.context.show_status_message("DOOM Sim: AI Autonomous RL Bot Active")
        else:
            self.btn_mode.setText("🎮 Manual Play")
            self.btn_mode.setStyleSheet("")
            self.lbl_action.setText("MANUAL")
            self.context.show_status_message("DOOM Sim: Manual Keyboard Mode Active")

    # ── Benchmark Test: Can It Run DOOM? ─────────────────────────────────────
    def run_benchmark_test(self) -> Dict[str, Any]:
        """Execute unthrottled raycaster stress-test and report results."""
        res = self.engine.run_benchmark(num_frames=150)
        self.lbl_bench_verdict.setText(f"🔥 {res['verdict']}")
        self.lbl_bench_detail.setText(
            f"Benchmark FPS: {res['fps']:.1f} FPS\n"
            f"Latency: {res['avg_ms']} ms/frame\n"
            f"Rays Cast: {res['total_rays']:,}\n"
            f"Rating: {res['rating']}"
        )
        self.context.log(f"DOOM Benchmark Results: {res}")
        self.context.show_status_message(f"DOOM Benchmark: {res['fps']} FPS — {res['rating']}")
        return res

    # ── Cheats ───────────────────────────────────────────────────────────────
    def _cheat_god_mode(self) -> None:
        msg = self.engine.cheat_iddqd()
        self.context.show_status_message(msg)

    def _cheat_ammo(self) -> None:
        msg = self.engine.cheat_idkfa()
        self.context.show_status_message(msg)

    def _cheat_noclip(self) -> None:
        msg = self.engine.cheat_idclip()
        self.context.show_status_message(msg)

    # ── Dataset Export (NeSyRL Integration) ──────────────────────────────────
    def _export_dataset(self) -> None:
        """Export recorded RL transitions to JSON for offline RL training."""
        if not self.engine.trajectory:
            QMessageBox.information(self, "Export Dataset", "No transitions recorded yet. Play or run AI Bot to record transitions.")
            return

        save_dir = self.context.get_data_dir() / "datasets" / "doom"
        save_dir.mkdir(parents=True, exist_ok=True)
        file_path = save_dir / f"doom_transitions_{int(time.time())}.json"

        data = {
            "environment": "doom_raycaster_sim",
            "level": self.engine.level_name,
            "total_transitions": len(self.engine.trajectory),
            "episode_reward": self.engine.episode_reward,
            "transitions": [
                {
                    "obs": t[0],
                    "action": t[1],
                    "reward": t[2],
                    "next_obs": t[3],
                    "done": t[4],
                }
                for t in self.engine.trajectory[:5000]
            ],
        }

        try:
            file_path.write_text(json.dumps(data, indent=2), encoding="utf-8")
            self.context.log(f"Exported {len(self.engine.trajectory)} transitions to {file_path}")
            self.context.show_status_message(f"Saved RL dataset: {file_path.name}")
            QMessageBox.information(self, "Dataset Exported", f"Successfully exported {len(self.engine.trajectory)} transitions to:\n{file_path}")
        except Exception as exc:
            self.context.log(f"Failed to export dataset: {exc}")
            QMessageBox.warning(self, "Export Failed", f"Error writing dataset: {exc}")

    def _switch_view(self, idx: int) -> None:
        """Switch between RL Simulator and Playable WebEngine DOOM."""
        self.stacked.setCurrentIndex(idx)
        self.btn_view_rl.setChecked(idx == 0)
        self.btn_view_wasm.setChecked(idx == 1)
        if idx == 0:
            if not self.timer.isActive():
                self.timer.start()
        else:
            if self.timer.isActive():
                self.timer.stop()

    # ── Cleanup ──────────────────────────────────────────────────────────────
    def cleanup(self) -> None:
        """Stop animation timer and free resources."""
        if self.timer.isActive():
            self.timer.stop()
        if hasattr(self, "web_view") and self.web_view:
            self.web_view.stop()
