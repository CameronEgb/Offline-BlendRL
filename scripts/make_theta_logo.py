"""Generate the ThetaIDE logo SVGs: frontend/icons/theta_logo.svg (mark) and theta_app_icon.svg (on a tile).

Run from anywhere: python scripts/make_theta_logo.py
"""
import random
from pathlib import Path

OUT = Path(__file__).resolve().parents[1] / "frontend" / "icons"

CX, CY = 256, 250
ORX, ORY = 148, 188          # outer ellipse
IRX, IRY = 96, 146           # inner ellipse: thicker sides than top/bottom, like a typographic theta
BAR_H = 38
BAR_X0, BAR_X1 = CX - ORX + 26, CX + ORX - 26
LAYERS, DX, DY = 16, 1.15, 1.45   # extrusion: 16 slices toward the lower right
SKEW = -11                   # forward lean (degrees)

# Binary engraved into the gold face
DIGIT_FONT, DIGIT_W, DIGIT_H = 15, 9, 11   # font size and the glyph cell used for fitting
DIGIT_STEP_X, DIGIT_STEP_Y = 14, 16
DIGIT_MARGIN = 5                            # keep digits this far inside the face's edges


def ellipse(rx, ry, sweep=0):
    return (f"M {CX - rx} {CY} A {rx} {ry} 0 1 {sweep} {CX + rx} {CY} "
            f"A {rx} {ry} 0 1 {sweep} {CX - rx} {CY} Z")


def lerp_hex(a, b, t):
    a = [int(a[i:i + 2], 16) for i in (1, 3, 5)]
    b = [int(b[i:i + 2], 16) for i in (1, 3, 5)]
    return "#" + "".join(f"{round(x + (y - x) * t):02x}" for x, y in zip(a, b))


def on_face(x, y):
    """Whether (x, y) lies on the theta's front face, at least DIGIT_MARGIN inside its edges."""
    m = DIGIT_MARGIN

    def in_ellipse(rx, ry):
        return ((x - CX) / rx) ** 2 + ((y - CY) / ry) ** 2 <= 1

    in_ring = in_ellipse(ORX - m, ORY - m) and not in_ellipse(IRX + m, IRY + m)
    in_bar = BAR_X0 + m <= x <= BAR_X1 - m and abs(y - CY) <= BAR_H / 2 - m
    return in_ring or in_bar


def face_digits():
    """One <text> per digit, kept only where its whole glyph cell sits on the gold face.

    Placing digits geometrically (instead of clipping) renders identically in browsers and in
    Qt's SVG renderer, which ignores clipPath, and never leaves half-cut glyphs at the edges.
    Each digit is an engraving: a dark glyph with a faint light copy just below it.
    """
    rng = random.Random(0x7E7A)
    cuts, lights = [], []
    for row, y in enumerate(range(CY - ORY + 16, CY + ORY, DIGIT_STEP_Y)):
        for x in range(CX - ORX + (row % 2) * (DIGIT_STEP_X // 2), CX + ORX, DIGIT_STEP_X):
            cell = [(x, y - DIGIT_H), (x + DIGIT_W, y - DIGIT_H), (x, y + 1), (x + DIGIT_W, y + 1)]
            if not all(on_face(px, py) for px, py in cell):
                continue
            digit, opacity = rng.choice("01"), round(rng.uniform(0.22, 0.5), 2)
            cuts.append(f'<text x="{x}" y="{y}" fill-opacity="{opacity}">{digit}</text>')
            lights.append(f'<text x="{x}" y="{y + 1}" fill-opacity="{round(opacity * 0.6, 2)}">{digit}</text>')
    return "\n          ".join(lights), "\n          ".join(cuts)


def mark(tile):
    lights, cuts = face_digits()
    extrusion = "\n      ".join(
        f'<use href="#theta" transform="translate({i * DX:.2f} {i * DY:.2f})" '
        f'fill="{lerp_hex("#7a4a0c", "#b8791a", (LAYERS - i) / LAYERS)}"/>'
        for i in range(LAYERS, 0, -1))
    background = ""
    if tile:
        background = """
  <rect x="8" y="8" width="496" height="496" rx="108" fill="url(#tile)"/>
  <rect x="8.75" y="8.75" width="494.5" height="494.5" rx="107.25" fill="none" stroke="#3c3836" stroke-width="1.5"/>
  <rect x="8" y="8" width="496" height="248" rx="108" fill="url(#tileSheen)"/>"""
    scale = ' transform="translate(256 262) scale(0.8) translate(-256 -262)"' if tile else ""
    return f"""<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 512 512" width="512" height="512">
  <title>ThetaIDE</title>
  <desc>A forward-leaning, extruded gold theta with binary digits engraved into its face.</desc>
  <defs>
    <g id="theta">
      <path fill-rule="evenodd" d="{ellipse(ORX, ORY)} {ellipse(IRX, IRY, 1)}"/>
      <rect x="{BAR_X0}" y="{CY - BAR_H / 2}" width="{BAR_X1 - BAR_X0}" height="{BAR_H}" rx="5"/>
    </g>
    <linearGradient id="face" gradientUnits="userSpaceOnUse" x1="{CX - ORX}" y1="{CY - ORY}" x2="{CX + 0.35 * ORX}" y2="{CY + ORY}">
      <stop offset="0" stop-color="#ffe08a"/>
      <stop offset="0.45" stop-color="#fabd2f"/>
      <stop offset="1" stop-color="#e8871a"/>
    </linearGradient>
    <linearGradient id="gloss" gradientUnits="userSpaceOnUse" x1="0" y1="{CY - ORY}" x2="0" y2="{CY + ORY}">
      <stop offset="0" stop-color="#ffffff" stop-opacity="0.55"/>
      <stop offset="0.42" stop-color="#ffffff" stop-opacity="0.08"/>
      <stop offset="0.5" stop-color="#ffffff" stop-opacity="0"/>
    </linearGradient>
    <linearGradient id="tile" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#32302f"/>
      <stop offset="1" stop-color="#1d2021"/>
    </linearGradient>
    <linearGradient id="tileSheen" x1="0" y1="0" x2="0" y2="1">
      <stop offset="0" stop-color="#ffffff" stop-opacity="0.06"/>
      <stop offset="1" stop-color="#ffffff" stop-opacity="0"/>
    </linearGradient>
  </defs>
{background}
  <g{scale}>
    <g transform="translate({CX} {CY}) skewX({SKEW}) translate({-CX} {-CY})">
      <!-- Extrusion, back to front -->
      {extrusion}
      <!-- Front face -->
      <use href="#theta" fill="url(#face)"/>
      <!-- Binary engraved into the face: light catch below, dark cut on top -->
      <g font-family="Consolas, 'Cascadia Code', 'DejaVu Sans Mono', monospace" font-size="{DIGIT_FONT}" font-weight="700">
        <g fill="#fff3c4">
          {lights}
        </g>
        <g fill="#7a4a0c">
          {cuts}
        </g>
      </g>
      <!-- Gloss over everything, then the rim light -->
      <use href="#theta" fill="url(#gloss)"/>
      <path fill="none" stroke="#fff3c4" stroke-opacity="0.55" stroke-width="2"
            d="{ellipse(ORX - 1, ORY - 1)}"/>
    </g>
  </g>
</svg>
"""


if __name__ == "__main__":
    (OUT / "theta_logo.svg").write_text(mark(tile=False), encoding="utf-8")
    (OUT / "theta_app_icon.svg").write_text(mark(tile=True), encoding="utf-8")
    print(f"Wrote {OUT / 'theta_logo.svg'} and {OUT / 'theta_app_icon.svg'}")
