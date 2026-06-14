"""
Generate explorer.html — interactive PaCMAP scatter with click-to-EEG
and a color-mode selector (NESI, dominant IIIC pattern, or any single
PaCMAP feature).

Click semantics: clicking any point snaps to the nearest point for which
an EEG PNG is actually present on disk (in --png-dir), and a red ring
marks the snapped location on the map.

Per-point data is embedded inline so the page works from file:// without
a local server.

  Usage:
    build_explorer.py                          # uses eeg_pngs/, writes explorer.html
    build_explorer.py --png-dir deploy/eeg_pngs --out deploy/explorer.html
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
from matplotlib.colors import LinearSegmentedColormap, to_hex, to_rgb

SCRIPT_DIR = Path(__file__).resolve().parent
COORDS_CSV = (SCRIPT_DIR.parent / "MorgothFeatureEmbedding"
              / "NESI_pacmap_coords.csv")

CLICKABLE_DATASETS = {"CAMS", "ICANS", "RASS"}

# PaCMAP input features, in display order for the dropdown.
# Each is a probability in [0, 1].
FEATURE_DISPLAY = [
    ('IIIC_Seizure',       'Seizure',       False),
    ('IIIC_GPD',           'GPD',           False),
    ('IIIC_GRDA',          'GRDA',          False),
    ('IIIC_LPD',           'LPD',           False),
    ('IIIC_LRDA',          'LRDA',          False),
    ('Burst_vs_NoBurst',   'Suppression',   False),
    ('GenSlowing',         'Gen slowing',   False),
    ('FocalSlowing',       'Focal slowing', False),
    ('Normal_vs_Abnormal', 'Normal',        True),   # invert for display
    ('Awake',              'Awake',         False),
    ('N1',                 'N1',            False),
    ('N2',                 'N2',            False),
]

# ─── IIIC dominant-pattern coloring (lifted from nesi_pacmap_main.py) ───
IIIC_PALETTE = {
    'Seizure': '#E13238', 'LPD':     '#F08C2A', 'GPD':     '#F2D549',
    'LRDA':    '#AABF45', 'GRDA':    '#7AC8E3', 'Other':   '#5C3A87',
    'Burst':   '#000000',
}
IIIC_LEGEND_ORDER = ['Burst', 'Seizure', 'LPD', 'GPD',
                     'LRDA', 'GRDA', 'Other']
CAT_PROB_COL = {
    'Burst':   'Burst_vs_NoBurst', 'Seizure': 'IIIC_Seizure',
    'LPD':     'IIIC_LPD',         'GPD':     'IIIC_GPD',
    'LRDA':    'IIIC_LRDA',        'GRDA':    'IIIC_GRDA',
    'Other':   'Awake',
}
BURST_THRESHOLD = 0.5


def _light_tint(hex_color, mix=0.80):
    rgb = np.array(to_rgb(hex_color))
    return tuple(mix * 1.0 + (1.0 - mix) * rgb)


def _category_cmap(cat):
    base = IIIC_PALETTE[cat]
    if cat == 'Other':
        # high p_Awake = light (awake), low = dark (sleep)
        return LinearSegmentedColormap.from_list(
            f'cat_{cat}', [base, _light_tint(base, mix=0.85)],
        )
    return LinearSegmentedColormap.from_list(
        f'cat_{cat}', [_light_tint(base, mix=0.80), base],
    )


def assign_iiic_labels(df):
    iiic_cols = ['IIIC_Seizure', 'IIIC_LPD', 'IIIC_GPD',
                 'IIIC_LRDA', 'IIIC_GRDA', 'IIIC_Other']
    iiic_names = ['Seizure', 'LPD', 'GPD', 'LRDA', 'GRDA', 'Other']
    M = df[iiic_cols].to_numpy()
    labels = np.array(iiic_names)[np.argmax(M, axis=1)]
    labels[df['Burst_vs_NoBurst'].to_numpy() >= BURST_THRESHOLD] = 'Burst'
    return labels


def build_iiic_hex_colors(df):
    """Per-point hex color for the dominant IIIC pattern (with within-
    category intensity gradient driven by the dominant probability)."""
    labels = assign_iiic_labels(df)
    cmaps = {cat: _category_cmap(cat) for cat in IIIC_PALETTE}
    rgb = np.zeros((len(df), 3))
    for cat in IIIC_PALETTE:
        m = labels == cat
        if not m.any():
            continue
        probs = df.loc[m, CAT_PROB_COL[cat]].to_numpy()
        if len(probs) > 20:
            lo, hi = np.quantile(probs, [0.05, 0.95])
        else:
            lo, hi = float(probs.min()), float(probs.max())
        if hi - lo < 1e-6:
            hi = lo + 1e-6
        norm_p = np.clip((probs - lo) / (hi - lo), 0, 1)
        rgb[m] = cmaps[cat](norm_p)[:, :3]
    return [to_hex(c) for c in rgb], labels.tolist()


def scan_available_indices(df: pd.DataFrame, png_dir: Path) -> list[int]:
    """Return the list of row indices in df whose PNG exists under png_dir
    (laid out as png_dir/<Dataset>/<stem>.png)."""
    if not png_dir.exists():
        return []
    available = []
    for i, row in enumerate(df.itertuples(index=False)):
        png_path = png_dir / row.Dataset / (row.stem + ".png")
        if png_path.exists():
            available.append(i)
    return available


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--png-dir", type=Path,
                    default=SCRIPT_DIR / "eeg_pngs",
                    help="directory of rendered PNGs (laid out per Dataset)")
    ap.add_argument("--out", type=Path,
                    default=SCRIPT_DIR / "explorer.html",
                    help="output HTML path")
    ap.add_argument("--rel-png-dir", type=str, default="eeg_pngs",
                    help="path the HTML uses to fetch PNGs (relative to "
                         "the HTML file)")
    args = ap.parse_args()

    df = pd.read_csv(COORDS_CSV)
    must_have = {"pacmap_x_nesi", "pacmap_y_nesi", "NESI",
                 "Dataset", "MorgothOutputFilename",
                 "Burst_vs_NoBurst", "IIIC_Seizure", "IIIC_LPD",
                 "IIIC_GPD", "IIIC_LRDA", "IIIC_GRDA", "IIIC_Other"}
    missing = must_have - set(df.columns)
    if missing:
        raise SystemExit(
            f"coords.csv missing columns: {missing}.  "
            "Re-run nesi_pacmap_main.py first."
        )

    df = df.assign(stem=df.MorgothOutputFilename.str.replace(r"\.csv$",
                                                              "", regex=True))

    available_idx = scan_available_indices(df, args.png_dir)
    n_unique = (df.iloc[available_idx][['Dataset', 'stem']]
                  .drop_duplicates().shape[0])
    print(f"PNGs available: {n_unique} unique segments  "
          f"({len(available_idx)} window rows / {len(df)} total) "
          f"(scanned {args.png_dir})")

    iiic_hex, iiic_labels = build_iiic_hex_colors(df)

    feature_arrays = {fname: df[fname].round(4).tolist()
                       for (fname, _, _) in FEATURE_DISPLAY}

    payload = {
        "x":          df.pacmap_x_nesi.tolist(),
        "y":          df.pacmap_y_nesi.tolist(),
        "nesi":       df.NESI.round(3).tolist(),
        "dataset":    df.Dataset.tolist(),
        "stem":       df.stem.tolist(),
        "iiic_hex":   iiic_hex,
        "iiic_label": iiic_labels,
        "features":   feature_arrays,
        "feature_display": [
            {"key": k, "label": lbl, "invert": inv}
            for (k, lbl, inv) in FEATURE_DISPLAY
        ],
        "iiic_palette":     IIIC_PALETTE,
        "iiic_legend_order": IIIC_LEGEND_ORDER,
        "available_idx":    available_idx,
        "png_base":         args.rel_png_dir.rstrip("/"),
    }

    html = HTML_TEMPLATE
    html = html.replace("__PAYLOAD__", json.dumps(payload, separators=(",", ":")))
    html = html.replace("__N_TOTAL__", str(len(df)))
    html = html.replace("__N_AVAILABLE__", str(n_unique))
    html = html.replace("__N_CLICKABLE__",
                        str(int(df.Dataset.isin(CLICKABLE_DATASETS).sum())))
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(html)
    print(f"Wrote {args.out}  ({args.out.stat().st_size / 1e6:.1f} MB)")


HTML_TEMPLATE = r"""<!DOCTYPE html>
<html lang="en">
<head>
  <meta charset="utf-8" />
  <title>NESI PaCMAP Explorer</title>
  <script src="https://cdn.plot.ly/plotly-2.35.2.min.js"></script>
  <style>
    html, body { margin: 0; padding: 0; height: 100%;
                  box-sizing: border-box; }
    body { font-family: -apple-system, system-ui, sans-serif;
            color: #222; padding: 6px 8px;
            display: flex; flex-direction: column; }
    .header { display: flex; align-items: center; gap: 14px;
               margin-bottom: 6px; flex-wrap: wrap; font-size: 12px;
               flex: 0 0 auto; }
    .header h1 { font-size: 15px; margin: 0; }
    .header .meta { color: #666; }
    .header .hint { color: #888; font-style: italic; }
    .mode-radios { display: flex; flex-wrap: wrap; gap: 4px;
                    margin-bottom: 6px; flex: 0 0 auto; }
    .mode-radios label { display: inline-flex; align-items: center;
                          padding: 3px 8px; border: 1px solid #ccc;
                          border-radius: 3px; cursor: pointer;
                          font-size: 12px; background: #f6f6f6;
                          user-select: none; }
    .mode-radios label:hover { background: #ececec; }
    .mode-radios input { display: none; }
    .mode-radios label.checked { background: #2b8cbe; color: white;
                                  border-color: #2b8cbe; font-weight: 600; }
    .mode-radios .sep { width: 1px; background: #ddd; margin: 0 2px; }
    .row { flex: 1 1 auto; min-height: 0;
            display: flex; gap: 10px; width: 100%; }
    .map-col { flex: 0 0 33%; max-width: 33%; min-width: 0;
                display: flex; flex-direction: column; }
    #map { flex: 1 1 auto; min-height: 0;
            border: 1px solid #ddd; border-radius: 3px; background: white; }
    #legend-iiic { display: none; margin-top: 4px;
                    font-size: 11px; line-height: 1.5; flex: 0 0 auto; }
    #legend-iiic .sw { display: inline-block; width: 14px; height: 11px;
                       vertical-align: middle; margin-right: 4px;
                       border: 1px solid #ccc; }
    .eeg-col { flex: 1 1 67%; min-width: 0; min-height: 0;
                display: flex; flex-direction: column; }
    #eeg-error { color: #b30000; display: none; padding: 6px 10px;
                  background: #fff5f5; border: 1px solid #f0c0c0;
                  border-radius: 3px; font-size: 12px; margin-bottom: 6px;
                  flex: 0 0 auto; }
    .eeg-view { flex: 1 1 auto; min-height: 0;
                 border: 1px solid #ddd; border-radius: 3px;
                 background: #fafafa; overflow: hidden;
                 display: flex; align-items: center; justify-content: center; }
    #eeg-img { display: none; max-width: 100%; max-height: 100%;
                object-fit: contain; }
    #eeg-placeholder { color: #888; font-style: italic; font-size: 13px;
                        padding: 20px; text-align: center; }
  </style>
</head>
<body>
  <div class="header">
    <h1>NESI PaCMAP Explorer</h1>
    <span class="meta">
      n=__N_TOTAL__  •  __N_AVAILABLE__ EEGs available  •
      click anywhere → snaps to nearest available (marked
      <span style="color:#d4071e; font-weight:bold;">●</span>)
    </span>
    <span class="hint">← → to cycle overlays</span>
  </div>
  <div class="mode-radios" id="mode-radios"></div>

  <div class="row">
    <div class="map-col">
      <div id="map"></div>
      <div id="legend-iiic"></div>
    </div>

    <div class="eeg-col">
      <div id="eeg-error"></div>
      <div class="eeg-view">
        <div id="eeg-placeholder">Click a point on the map →
          EEG + multitaper spectrogram will appear here.</div>
        <img id="eeg-img" alt="EEG + spectrogram" />
      </div>
    </div>
  </div>

  <script id="payload" type="application/json">__PAYLOAD__</script>
  <script>
    const D = JSON.parse(document.getElementById('payload').textContent);
    const CLICKABLE = new Set(['CAMS', 'ICANS', 'RASS']);

    // Indices split by clickability (preserve original order).
    const idxClickable = [], idxGrey = [];
    for (let i = 0; i < D.x.length; i++) {
      (CLICKABLE.has(D.dataset[i]) ? idxClickable : idxGrey).push(i);
    }
    const pick = (arr, idx) => idx.map(i => arr[i]);

    // Gray -> dark-blue sequential gradient (matches fig2 PROB_CMAP).
    const GRAY_BLUE = [
      [0.0,  '#eeeeee'],
      [0.33, '#bcdff1'],
      [0.67, '#2b8cbe'],
      [1.0,  '#08306b'],
    ];
    const GREY = '#cccccc';

    // ── Build mode list: NESI, dominant pattern, then each feature ──
    const MODES = [
      { value: 'nesi',          label: 'NESI' },
      { value: 'iiic_dominant', label: 'Dominant pattern' },
    ];
    D.feature_display.forEach(f => MODES.push({
      value: 'feat:' + f.key, label: f.label,
    }));

    const radiosEl = document.getElementById('mode-radios');
    MODES.forEach((m, i) => {
      const wrap = document.createElement('label');
      wrap.className = i === 0 ? 'checked' : '';
      wrap.innerHTML = '<input type="radio" name="mode" value="' +
                        m.value + '"' + (i === 0 ? ' checked' : '') +
                        '> ' + m.label;
      radiosEl.appendChild(wrap);
      if (i === 1) {                                // separator after dominant
        const sep = document.createElement('span');
        sep.className = 'sep';
        radiosEl.appendChild(sep);
      }
    });

    // ── Build initial traces (color mode applied after Plotly.newPlot) ──
    const traceClick = {
      type: 'scattergl', mode: 'markers',
      x: pick(D.x, idxClickable), y: pick(D.y, idxClickable),
      customdata: idxClickable,
      marker: {
        size: 5, opacity: 0.7,
        color: pick(D.nesi, idxClickable),
        colorscale: GRAY_BLUE,
        cmin: -3, cmax: 3,
        showscale: false,
      },
      // Pre-format hover text so the NESI value stays correct even when
      // the marker.color array is swapped to a different overlay.
      text: idxClickable.map(i =>
        `Dataset: ${D.dataset[i]}<br>NESI: ${D.nesi[i].toFixed(2)}`),
      hovertemplate: '%{text}<extra>click for EEG</extra>',
      name: 'CAMS / ICANS / RASS',
    };

    const traceGrey = {
      type: 'scattergl', mode: 'markers',
      x: pick(D.x, idxGrey), y: pick(D.y, idxGrey),
      customdata: idxGrey,
      marker: { size: 3, color: GREY, opacity: 0.35 },
      hoverinfo: 'skip',
      name: 'GCS (no EEG)',
    };

    // Single-point highlight trace. Uses scattergl (same renderer as the
    // map points) so it composites ON TOP of them — an SVG scatter would
    // render behind the WebGL canvas and get obscured.
    const traceHighlight = {
      type: 'scattergl', mode: 'markers',
      x: [], y: [],
      marker: {
        size: 16, symbol: 'circle',
        color: '#d4071e',
        line: { width: 1.5, color: '#ffffff' },
      },
      hoverinfo: 'skip', showlegend: false, name: 'selected',
    };
    const HL_TRACE = 2;

    const layout = {
      margin: { l: 6, r: 6, t: 6, b: 6 },
      xaxis: { showgrid: false, zeroline: false, showticklabels: false,
                fixedrange: true },
      yaxis: { showgrid: false, zeroline: false, showticklabels: false,
                scaleanchor: 'x', scaleratio: 1, fixedrange: true },
      hovermode: 'closest',
      dragmode: false,
      legend: { x: 0.01, y: 0.99, bgcolor: 'rgba(255,255,255,0.85)',
                 bordercolor: '#ddd', borderwidth: 1, font: { size: 11 } },
      plot_bgcolor: '#fafafa',
      paper_bgcolor: 'white',
    };

    const config = { responsive: true, displaylogo: false,
                      scrollZoom: false, displayModeBar: false,
                      staticPlot: false };

    Plotly.newPlot('map', [traceGrey, traceClick, traceHighlight], layout, config);

    // ── IIIC categorical legend (shown only in iiic_dominant mode) ──
    const legendEl = document.getElementById('legend-iiic');
    function buildIiicLegend() {
      const counts = {};
      for (const lab of D.iiic_label) counts[lab] = (counts[lab] || 0) + 1;
      const parts = D.iiic_legend_order.map(cat => {
        const color = D.iiic_palette[cat];
        const n = counts[cat] || 0;
        return `<span class="sw" style="background:${color}"></span>` +
               `${cat} (n=${n})`;
      });
      legendEl.innerHTML = parts.join('  &nbsp; ');
    }
    buildIiicLegend();

    // ── Apply selected color mode ──
    function applyMode(mode) {
      const featMatch = mode.startsWith('feat:') ? mode.slice(5) : null;
      const isCategorical = (mode === 'iiic_dominant');

      let colors_click;
      let cmin, cmax;
      if (mode === 'nesi') {
        colors_click = pick(D.nesi, idxClickable);
        cmin = -3; cmax = 3;
      } else if (isCategorical) {
        colors_click = pick(D.iiic_hex, idxClickable);
      } else if (featMatch) {
        const fd = D.feature_display.find(f => f.key === featMatch);
        let vals = D.features[featMatch];
        if (fd.invert) vals = vals.map(v => 1 - v);
        colors_click = pick(vals, idxClickable);
        cmin = 0; cmax = 1;
      }

      if (isCategorical) {
        Plotly.restyle('map', {
          'marker.color':     [colors_click],
          'marker.showscale': false,
        }, [1]);
        legendEl.style.display = 'block';
      } else {
        Plotly.restyle('map', {
          'marker.color':      [colors_click],
          'marker.colorscale': [GRAY_BLUE],
          'marker.cmin':       cmin,
          'marker.cmax':       cmax,
          'marker.showscale':  false,
        }, [1]);
        legendEl.style.display = 'none';
      }
    }

    // Radio change + visual highlight
    function selectMode(value) {
      const radios = [...document.querySelectorAll('input[name="mode"]')];
      radios.forEach(r => {
        r.checked = (r.value === value);
        r.parentElement.classList.toggle('checked', r.checked);
      });
      applyMode(value);
    }
    document.querySelectorAll('input[name="mode"]').forEach(r => {
      r.addEventListener('change', () => selectMode(r.value));
    });

    // Left / right arrow keys cycle through modes
    document.addEventListener('keydown', (e) => {
      if (e.target.tagName === 'INPUT' && e.target.type !== 'radio') return;
      if (e.key !== 'ArrowLeft' && e.key !== 'ArrowRight') return;
      const radios = [...document.querySelectorAll('input[name="mode"]')];
      const cur = radios.findIndex(r => r.checked);
      const dir = (e.key === 'ArrowRight') ? 1 : -1;
      const nxt = (cur + dir + radios.length) % radios.length;
      selectMode(radios[nxt].value);
      e.preventDefault();
    });

    // ── Click handler: snap to nearest available point, load its PNG ──
    const elImg   = document.getElementById('eeg-img');
    const elPlace = document.getElementById('eeg-placeholder');
    const elError = document.getElementById('eeg-error');
    const PNG_BASE = D.png_base || 'eeg_pngs';
    const AVAIL = D.available_idx || [];

    function pngPath(dataset, stem) {
      return PNG_BASE + '/' + encodeURIComponent(dataset) + '/' +
              encodeURIComponent(stem + '.png');
    }

    function nearestAvailable(x, y) {
      let best = -1, bestD = Infinity;
      for (let k = 0; k < AVAIL.length; k++) {
        const i  = AVAIL[k];
        const dx = D.x[i] - x, dy = D.y[i] - y;
        const d  = dx*dx + dy*dy;
        if (d < bestD) { bestD = d; best = i; }
      }
      return best;
    }

    document.getElementById('map').on('plotly_click', (ev) => {
      if (!ev || !ev.points || !ev.points.length) return;
      const pt = ev.points[0];
      if (pt.curveNumber === HL_TRACE) return;       // ignore clicks on the ring

      const snap = nearestAvailable(pt.x, pt.y);
      if (snap < 0) {
        elImg.style.display   = 'none';
        elPlace.style.display = 'block';
        elPlace.textContent   = 'No EEG images available.';
        return;
      }
      const ds    = D.dataset[snap];
      const fname = D.stem[snap];

      // Move red ring onto the snapped point.
      Plotly.restyle('map',
        { x: [[D.x[snap]]], y: [[D.y[snap]]] }, [HL_TRACE]);

      elError.style.display = 'none';
      elPlace.style.display = 'none';
      const url = pngPath(ds, fname);
      elImg.onload  = () => { elImg.style.display = 'block'; };
      elImg.onerror = () => {
        elImg.style.display = 'none';
        elError.style.display = 'block';
        elError.textContent = 'PNG load failed at ' + url +
          '   NESI=' + D.nesi[snap].toFixed(2) +
          '   IIIC=' + D.iiic_label[snap];
      };
      elImg.src = url;
    });
  </script>
</body>
</html>
"""


if __name__ == "__main__":
    main()
