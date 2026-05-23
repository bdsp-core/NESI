"""
Build a self-contained HTML file with a smoothing-sigma slider for the
PaCMAP IIIC-colored map.

Reads NESI_pacmap_iiic_data.npz (XY + per-point RGB + labels), renders the
Gaussian-splatted smoothed map at a range of sigma values, and bakes them
all (plus a legend PNG) as base64 into one HTML page.
"""

from __future__ import annotations

import base64
import io
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.colors import LinearSegmentedColormap, to_rgb
from scipy.ndimage import gaussian_filter

SCRIPT_DIR = Path(__file__).resolve().parent
NPZ = SCRIPT_DIR / "NESI_pacmap_iiic_data.npz"
OUT_HTML = SCRIPT_DIR / "NESI_pacmap_explorer.html"

# Must match the palette used in nesi_pacmap_main.py
IIIC_PALETTE = {
    'Seizure': '#E13238',
    'LPD':     '#F08C2A',
    'GPD':     '#F2D549',
    'LRDA':    '#AABF45',
    'GRDA':    '#7AC8E3',
    'Other':   '#5C3A87',
    'Burst':   '#000000',
}
IIIC_LEGEND_ORDER = ['Burst', 'Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']

SIGMA_VALUES = [1.0, 1.5, 2.0, 2.5, 3.0, 3.5, 4.0, 5.0, 6.5, 8.0, 10.0, 13.0, 17.0]
DEFAULT_SIGMA_INDEX = 3   # 2.5 — light smoothing
N_PIX = 700
DENSITY_FLOOR_REL = 0.05  # mask pixels with smoothed density < 5% of peak


def _light_tint(hex_color, mix=0.85):
    rgb = np.array(to_rgb(hex_color))
    return tuple(mix * 1.0 + (1.0 - mix) * rgb)


def make_category_cmap(cat):
    base = IIIC_PALETTE[cat]
    if cat == 'Other':
        return LinearSegmentedColormap.from_list(
            f'cat_{cat}', [base, _light_tint(base, mix=0.85)])
    return LinearSegmentedColormap.from_list(
        f'cat_{cat}', [_light_tint(base, mix=0.80), base])


def smooth_map(XY, rgb, sigma_px, n_pix=N_PIX, floor_rel=DENSITY_FLOOR_REL):
    xmin, xmax = float(XY[:, 0].min()), float(XY[:, 0].max())
    ymin, ymax = float(XY[:, 1].min()), float(XY[:, 1].max())
    pad = 0.04 * max(xmax - xmin, ymax - ymin)
    xmin -= pad; xmax += pad; ymin -= pad; ymax += pad

    xpix = ((XY[:, 0] - xmin) / (xmax - xmin) * (n_pix - 1)).astype(int)
    ypix = ((XY[:, 1] - ymin) / (ymax - ymin) * (n_pix - 1)).astype(int)
    valid = (xpix >= 0) & (xpix < n_pix) & (ypix >= 0) & (ypix < n_pix)

    density = np.zeros((n_pix, n_pix), dtype=np.float64)
    color = np.zeros((n_pix, n_pix, 3), dtype=np.float64)
    np.add.at(density, (ypix[valid], xpix[valid]), 1.0)
    for c in range(3):
        np.add.at(color[..., c], (ypix[valid], xpix[valid]), rgb[valid, c])

    if sigma_px > 0:
        density = gaussian_filter(density, sigma=sigma_px)
        for c in range(3):
            color[..., c] = gaussian_filter(color[..., c], sigma=sigma_px)

    mean_color = np.ones_like(color)
    nonzero = density > 0
    for c in range(3):
        mean_color[..., c] = np.where(
            nonzero, color[..., c] / np.maximum(density, 1e-12), 1.0)

    if nonzero.any():
        peak = float(density.max())
        floor = peak * floor_rel
        mean_color[density < floor] = 1.0

    return mean_color, (xmin, xmax, ymin, ymax)


def fig_to_b64(fig, dpi=140):
    buf = io.BytesIO()
    fig.savefig(buf, format='png', dpi=dpi, bbox_inches='tight',
                facecolor='white')
    plt.close(fig)
    return base64.b64encode(buf.getvalue()).decode('ascii')


def render_map_image(XY, rgb, sigma_px):
    mean_color, extent = smooth_map(XY, rgb, sigma_px)
    fig, ax = plt.subplots(figsize=(7.5, 7.5), dpi=140)
    ax.imshow(mean_color, origin='lower', extent=extent,
              interpolation='bilinear')
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(f"PaCMAP — Gaussian-splat smoothing, σ = {sigma_px:g} px",
                 fontsize=11, fontweight='bold', pad=8)
    return fig_to_b64(fig)


def render_legend_image(counts):
    fig, ax = plt.subplots(figsize=(2.4, 5.0), dpi=140)
    ax.set_axis_off()
    n_cat = len(IIIC_LEGEND_ORDER)
    row_h = 1.0 / n_cat
    for i, cat in enumerate(IIIC_LEGEND_ORDER):
        n = int(counts.get(cat, 0))
        cmap = make_category_cmap(cat)
        gradient = np.linspace(1, 0, 64).reshape(-1, 1)
        y_lo = 1.0 - (i + 1) * row_h + 0.04 * row_h
        y_hi = 1.0 - i * row_h - 0.04 * row_h
        ax.imshow(gradient, cmap=cmap, aspect='auto',
                  extent=[0.0, 0.18, y_lo, y_hi],
                  transform=ax.transAxes, zorder=2)
        suffix = "← sleep | wake →" if cat == 'Other' else "← low | high →"
        ax.text(0.22, (y_lo + y_hi) / 2,
                f"{cat}  (n={n})\n  intensity: {suffix}",
                ha='left', va='center', fontsize=9,
                transform=ax.transAxes)
    ax.set_xlim(0, 1); ax.set_ylim(0, 1)
    return fig_to_b64(fig)


HTML_TEMPLATE = """<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>PaCMAP smoothness explorer</title>
  <style>
    body {{
      font-family: -apple-system, system-ui, sans-serif;
      margin: 24px;
      color: #222;
    }}
    h1 {{ font-size: 18px; margin-bottom: 4px; }}
    .subtitle {{ color: #666; font-size: 13px; margin-bottom: 20px; }}
    .row {{ display: flex; gap: 24px; align-items: flex-start; }}
    .map {{ flex: 0 0 auto; }}
    .map img {{ display: block; max-width: 720px; height: auto;
                border: 1px solid #ddd; }}
    .legend img {{ display: block; max-height: 720px;
                   border: 1px solid #eee; }}
    .controls {{ margin: 16px 0; max-width: 720px; }}
    .slider-row {{ display: flex; gap: 12px; align-items: center;
                   font-size: 14px; }}
    .slider-row input[type=range] {{ flex: 1; }}
    .value-box {{ display: inline-block; min-width: 60px;
                  font-variant-numeric: tabular-nums; font-weight: 600;
                  font-size: 14px; }}
    .ticks {{ display: flex; justify-content: space-between;
              font-size: 11px; color: #888; margin-top: 2px;
              padding: 0 6px; }}
  </style>
</head>
<body>
  <h1>PaCMAP IIIC-colored map &mdash; smoothing explorer</h1>
  <div class="subtitle">
    Gaussian splatting of n={n_points} window-vectors. Drag the slider to
    vary the smoothing kernel; pixels whose smoothed density falls below
    5% of the local peak are hidden.
  </div>

  <div class="controls">
    <div class="slider-row">
      <label for="slider">Smoothing &sigma;:</label>
      <input type="range" id="slider" min="0" max="{n_max}" value="{default_idx}" step="1" />
      <span class="value-box"><span id="value">{default_val}</span> px</span>
    </div>
    <div class="ticks">
      <span>{min_label} (sharp)</span>
      <span>{max_label} (very smooth)</span>
    </div>
  </div>

  <div class="row">
    <div class="map">
      <img id="img" alt="PaCMAP smoothed map" />
    </div>
    <div class="legend">
      <img src="data:image/png;base64,{legend_b64}" alt="legend" />
    </div>
  </div>

  <script>
    const SIGMAS = {sigmas_json};
    const IMAGES = {images_json};
    const slider = document.getElementById('slider');
    const valEl = document.getElementById('value');
    const imgEl = document.getElementById('img');
    function update() {{
      const i = parseInt(slider.value, 10);
      imgEl.src = 'data:image/png;base64,' + IMAGES[i];
      valEl.textContent = SIGMAS[i];
    }}
    slider.addEventListener('input', update);
    update();
  </script>
</body>
</html>
"""


def main():
    import json
    data = np.load(NPZ, allow_pickle=True)
    XY = data['XY']
    rgb = data['rgb']
    labels = data['labels']
    print(f"Loaded {len(XY)} points from {NPZ}")

    # counts for legend
    import pandas as pd
    counts = pd.Series(labels).value_counts()

    print(f"Rendering legend...")
    legend_b64 = render_legend_image(counts)

    print(f"Rendering {len(SIGMA_VALUES)} smoothed maps "
          f"(sigmas: {SIGMA_VALUES})")
    images_b64 = []
    for s in SIGMA_VALUES:
        print(f"  sigma = {s}", flush=True)
        images_b64.append(render_map_image(XY, rgb, s))

    html = HTML_TEMPLATE.format(
        n_points=len(XY),
        n_max=len(SIGMA_VALUES) - 1,
        default_idx=DEFAULT_SIGMA_INDEX,
        default_val=f"{SIGMA_VALUES[DEFAULT_SIGMA_INDEX]:g}",
        min_label=f"{SIGMA_VALUES[0]:g}",
        max_label=f"{SIGMA_VALUES[-1]:g}",
        legend_b64=legend_b64,
        sigmas_json=json.dumps([f"{s:g}" for s in SIGMA_VALUES]),
        images_json=json.dumps(images_b64),
    )
    OUT_HTML.write_text(html)
    size_mb = OUT_HTML.stat().st_size / 1024**2
    print(f"Wrote {OUT_HTML}  ({size_mb:.1f} MB)")


if __name__ == '__main__':
    main()
