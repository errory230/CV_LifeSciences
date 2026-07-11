"""HTML builder (layer B): inject viz_data.json into the reusable viewer.

The viewer template (`viewer_template.html`) is molecule-agnostic — it contains
all the CSS/JS for the scatter, slider, selection panel and time-series chart and
a single `__VIZ_DATA__` placeholder. We substitute the molecule's JSON to produce
a fully self-contained `explore_<mol_id>.html` (no server, no CDN, images already
base64-embedded). Because the viewer never changes per molecule, adding molecules
never touches viewer code.
"""

from __future__ import annotations

import json
from pathlib import Path

_TEMPLATE = Path(__file__).with_name("viewer_template.html")
_PLACEHOLDER = "__VIZ_DATA__"


def build_html(viz_data: dict, out_path: Path) -> Path:
    """Write a self-contained explorer HTML for one molecule."""
    template = _TEMPLATE.read_text()
    payload = json.dumps(viz_data)
    # keep the inline <script> from being closed early by any "</" in the data
    payload = payload.replace("</", "<\\/")
    html = template.replace(_PLACEHOLDER, payload)
    out_path = Path(out_path)
    out_path.write_text(html)
    return out_path


def build_index(entries: list[dict], out_path: Path) -> Path:
    """Write an index.html linking each molecule's explorer.

    `entries`: list of {mol_id, href, n_frames, n_clusters}.
    """
    rows = "\n".join(
        f'<li><a href="{e["href"]}">{e["mol_id"]}</a>'
        f' <span style="color:#6b7480">· {e.get("n_frames","?")} frames'
        f' · {e.get("n_clusters","?")} clusters</span></li>'
        for e in entries
    )
    html = f"""<!doctype html><html><head><meta charset="utf-8">
<title>MDS-Ensemble explorers</title>
<style>body{{font:15px -apple-system,Segoe UI,Roboto,sans-serif;max-width:680px;
margin:40px auto;padding:0 16px;color:#1c2330}}h1{{font-size:19px}}
li{{margin:6px 0}}a{{color:#3457d5;text-decoration:none}}a:hover{{text-decoration:underline}}</style>
</head><body><h1>MDS-Ensemble QSAR — conformation explorers</h1>
<ul>{rows}</ul></body></html>"""
    Path(out_path).write_text(html)
    return Path(out_path)
