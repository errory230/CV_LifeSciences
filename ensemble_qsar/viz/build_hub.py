"""Catalog hub: one self-contained index.html over all molecule explorers.

Layer separation is preserved — the per-molecule `explore_<mol_id>.html` pages
(and the Stage-3 feature code) are untouched. This module only *collects* a
lightweight summary per molecule (from its `viz_data.json` + `run_manifest.json`,
never the full frame data) into `hub_index.json`, and renders a sortable/
filterable table into a single self-contained `index.html`: all summary data and
2D structure thumbnails are inlined (no fetch), links to explorers are relative,
so it opens by double-click over file://. Re-running is idempotent.

mol_class comes from run_manifest.json and MW is computed from SMILES with RDKit,
so no existing code changes.
"""

from __future__ import annotations

import json
from pathlib import Path

from rdkit import Chem
from rdkit.Chem import Descriptors
from rdkit.Chem.Draw import rdMolDraw2D

_METRICS = ["psa3d", "rg", "intra_hbond", "sasa", "rmsd"]
# markers that identify a directory as a (possibly incomplete) molecule run
_MD_MARKERS = ("run_manifest.json", "manifest.json", "analysis", "solvation")


def render_thumbnail(smiles: str, size: int = 84) -> str:
    """RDKit 2D depiction as an inline SVG string ("" on failure)."""
    mol = Chem.MolFromSmiles(smiles) if smiles else None
    if mol is None:
        return ""
    d = rdMolDraw2D.MolDraw2DSVG(size, size)
    d.drawOptions().padding = 0.08
    rdMolDraw2D.PrepareAndDrawMolecule(d, mol)
    d.FinishDrawing()
    svg = d.GetDrawingText()
    # drop any XML prolog so it embeds cleanly inside HTML
    i = svg.find("<svg")
    return svg[i:] if i >= 0 else svg


def _mol_class(md_dir: Path) -> str:
    for name in ("run_manifest.json", "manifest.json"):
        p = md_dir / name
        if p.exists():
            data = json.loads(p.read_text())
            prep = data.get("prep_manifest", data)
            if prep.get("mol_class"):
                return prep["mol_class"]
    return "unknown"


def _is_molecule_dir(d: Path) -> bool:
    return d.is_dir() and not d.name.startswith("_") \
        and any((d / m).exists() for m in _MD_MARKERS)


def collect_hub_index(md_root: Path) -> dict:
    md_root = Path(md_root)
    molecules, pending = [], []
    dist: dict[str, int] = {}

    for d in sorted(p for p in md_root.iterdir() if _is_molecule_dir(p)):
        viz_path = d / "viz_data.json"
        if not viz_path.exists():
            pending.append({"mol_id": d.name,
                            "reason": "no viz_data.json (stage 2/3 incomplete)"})
            continue
        viz = json.loads(viz_path.read_text())
        meta, summ = viz["meta"], viz["summary"]
        smiles = meta.get("smiles", "")
        mol = Chem.MolFromSmiles(smiles) if smiles else None
        mol_class = _mol_class(d)
        dist[mol_class] = dist.get(mol_class, 0) + 1
        molecules.append({
            "mol_id": meta["mol_id"],
            "smiles": smiles,
            "mol_class": mol_class,
            "mw": round(Descriptors.MolWt(mol), 2) if mol else None,
            "n_frames": meta.get("n_frames"),
            "equilibration_frame": meta.get("equilibration_frame", 0),
            "n_clusters": len(viz.get("clusters", [])),
            "explore_href": f"{d.name}/explore_{meta['mol_id']}.html",
            "thumbnail_svg": render_thumbnail(smiles),
            "metrics": {m: {"mean": summ[m]["mean"], "std": summ[m]["std"]}
                        for m in _METRICS if m in summ},
            "pca_explained_variance": summ.get("pca_explained_variance", []),
        })

    return {
        "n_molecules": len(molecules),
        "n_pending": len(pending),
        "class_distribution": dist,
        "molecules": molecules,
        "pending": pending,
    }


def build_hub(md_root: Path) -> tuple[Path, Path]:
    """Write hub_index.json + a self-contained index.html; return both paths."""
    md_root = Path(md_root)
    hub = collect_hub_index(md_root)
    json_path = md_root / "hub_index.json"
    json_path.write_text(json.dumps(hub))
    html = _HUB_TEMPLATE.replace("__HUB_DATA__", json.dumps(hub).replace("</", "<\\/"))
    html_path = md_root / "index.html"
    html_path.write_text(html)
    return html_path, json_path


_HUB_TEMPLATE = r"""<!doctype html>
<html lang="en"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1">
<title>MDS-Ensemble QSAR — hub</title>
<style>
  :root{--bg:#f7f8fa;--card:#fff;--ink:#1c2330;--muted:#6b7480;--line:#e4e8ee;--accent:#3457d5;}
  *{box-sizing:border-box;} body{margin:0;font:14px/1.45 -apple-system,Segoe UI,Roboto,Arial,sans-serif;background:var(--bg);color:var(--ink);}
  header{padding:16px 22px;background:var(--card);border-bottom:1px solid var(--line);}
  h1{margin:0 0 6px;font-size:18px;} .sub{color:var(--muted);font-size:13px;}
  .controls{display:flex;flex-wrap:wrap;gap:10px;align-items:center;padding:12px 22px;background:var(--card);border-bottom:1px solid var(--line);}
  .controls input,.controls select{font:13px inherit;padding:5px 8px;border:1px solid var(--line);border-radius:6px;background:#fff;}
  .controls label{font-size:12px;color:var(--muted);display:inline-flex;gap:4px;align-items:center;}
  .wrap{padding:16px 22px;overflow-x:auto;}
  table{border-collapse:collapse;width:100%;background:var(--card);border:1px solid var(--line);border-radius:10px;overflow:hidden;}
  th,td{padding:8px 10px;text-align:left;border-bottom:1px solid var(--line);white-space:nowrap;}
  th{background:#fbfcfe;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.03em;cursor:pointer;user-select:none;}
  th.num,td.num{text-align:right;font-variant-numeric:tabular-nums;}
  tbody tr{cursor:pointer;} tbody tr:hover{background:#f2f6ff;}
  tr.pending{opacity:.5;cursor:default;} tr.pending:hover{background:none;}
  .thumb{width:52px;height:52px;} .thumb svg{width:52px;height:52px;}
  .badge{display:inline-block;padding:1px 8px;border-radius:10px;font-size:11px;background:#eef1f6;color:#42506a;}
  .badge.peptide{background:#e7f0ff;color:#2451b8;} .badge.small_molecule{background:#eef6ec;color:#2f7d3a;}
  .muted{color:var(--muted);} .arrow{font-size:10px;color:var(--accent);}
</style></head><body>
<header>
  <h1>MDS-Ensemble QSAR — conformation hub</h1>
  <div class="sub" id="overview"></div>
</header>
<div class="controls">
  <input id="q" placeholder="search mol_id / SMILES" size="22">
  <label>class <select id="fclass"><option value="">all</option></select></label>
  <label>3D-PSA μ <input id="psaMin" type="number" size="5" style="width:70px"> – <input id="psaMax" type="number" style="width:70px"></label>
  <label>IMHB μ <input id="hbMin" type="number" style="width:60px"> – <input id="hbMax" type="number" style="width:60px"></label>
  <span class="muted" id="count"></span>
</div>
<div class="wrap"><table id="tbl"><thead><tr>
  <th></th>
  <th data-key="mol_id">mol_id</th>
  <th data-key="mol_class">class</th>
  <th class="num" data-key="mw">MW</th>
  <th class="num" data-key="psa3d_mean">3D-PSA μ±σ</th>
  <th class="num" data-key="psa3d_std">PSA σ</th>
  <th class="num" data-key="intra_hbond_mean">IMHB μ</th>
  <th class="num" data-key="rg_mean">Rg μ</th>
  <th class="num" data-key="rmsd_mean">RMSD μ±σ</th>
  <th class="num" data-key="n_clusters">clusters</th>
  <th class="num" data-key="n_frames">frames</th>
  <th data-key="status">status</th>
</tr></thead><tbody id="tb"></tbody></table></div>
<script>
const HUB = __HUB_DATA__;
let sortKey="mol_id", asc=true;
const g=(m,k,s)=> (m.metrics && m.metrics[k]) ? m.metrics[k][s] : null;
const num=x=> (x==null?"":(Math.abs(x)>=100?(+x).toFixed(0):(+x).toFixed(2)));

function overview(){
  const d=HUB.class_distribution||{}; const cls=Object.entries(d).map(([k,v])=>`${k}: ${v}`).join(" · ");
  document.getElementById("overview").textContent =
    `${HUB.n_molecules} molecules · ${cls||"—"} · done ${HUB.n_molecules} / pending ${HUB.n_pending}`;
  const sel=document.getElementById("fclass");
  Object.keys(d).forEach(c=>{const o=document.createElement("option");o.value=c;o.textContent=c;sel.appendChild(o);});
}
function sortVal(m){
  if(sortKey==="mol_id") return m.mol_id.toLowerCase();
  if(sortKey==="mol_class") return m.mol_class;
  if(sortKey==="mw") return m.mw??-1;
  if(sortKey==="n_clusters") return m.n_clusters??-1;
  if(sortKey==="n_frames") return m.n_frames??-1;
  const [k,s]=sortKey.split(/_(mean|std)$/); return g(m,k,sortKey.endsWith("std")?"std":"mean")??-1;
}
function filtered(){
  const q=document.getElementById("q").value.toLowerCase();
  const fc=document.getElementById("fclass").value;
  const pmin=parseFloat(document.getElementById("psaMin").value), pmax=parseFloat(document.getElementById("psaMax").value);
  const hmin=parseFloat(document.getElementById("hbMin").value), hmax=parseFloat(document.getElementById("hbMax").value);
  let rows=HUB.molecules.filter(m=>{
    if(q && !(m.mol_id.toLowerCase().includes(q)||(m.smiles||"").toLowerCase().includes(q))) return false;
    if(fc && m.mol_class!==fc) return false;
    const psa=g(m,"psa3d","mean"), hb=g(m,"intra_hbond","mean");
    if(!isNaN(pmin)&&(psa==null||psa<pmin))return false; if(!isNaN(pmax)&&(psa==null||psa>pmax))return false;
    if(!isNaN(hmin)&&(hb==null||hb<hmin))return false; if(!isNaN(hmax)&&(hb==null||hb>hmax))return false;
    return true;
  });
  rows.sort((a,b)=>{const x=sortVal(a),y=sortVal(b); return (x<y?-1:x>y?1:0)*(asc?1:-1);});
  return rows;
}
function render(){
  const tb=document.getElementById("tb"); tb.innerHTML="";
  const rows=filtered();
  rows.forEach(m=>{
    const tr=document.createElement("tr");
    tr.innerHTML=`<td class="thumb">${m.thumbnail_svg||""}</td>
      <td>${m.mol_id}</td>
      <td><span class="badge ${m.mol_class}">${m.mol_class}</span></td>
      <td class="num">${num(m.mw)}</td>
      <td class="num">${num(g(m,"psa3d","mean"))} <span class="muted">±${num(g(m,"psa3d","std"))}</span></td>
      <td class="num">${num(g(m,"psa3d","std"))}</td>
      <td class="num">${num(g(m,"intra_hbond","mean"))}</td>
      <td class="num">${num(g(m,"rg","mean"))}</td>
      <td class="num">${num(g(m,"rmsd","mean"))} <span class="muted">±${num(g(m,"rmsd","std"))}</span></td>
      <td class="num">${m.n_clusters??""}</td>
      <td class="num">${m.n_frames??""}</td>
      <td><span class="badge">done</span></td>`;
    tr.onclick=()=>window.open(m.explore_href,"_blank");
    tb.appendChild(tr);
  });
  (HUB.pending||[]).forEach(p=>{
    const tr=document.createElement("tr"); tr.className="pending";
    tr.innerHTML=`<td></td><td>${p.mol_id}</td><td colspan="9" class="muted">${p.reason}</td><td><span class="badge">pending</span></td>`;
    tb.appendChild(tr);
  });
  document.getElementById("count").textContent=`${rows.length} shown`;
}
document.querySelectorAll("th[data-key]").forEach(th=>th.onclick=()=>{
  const k=th.dataset.key; if(k==="status")return;
  if(sortKey===k)asc=!asc; else {sortKey=k;asc=true;}
  document.querySelectorAll("th .arrow").forEach(a=>a.remove());
  const s=document.createElement("span");s.className="arrow";s.textContent=asc?" ▲":" ▼";th.appendChild(s);
  render();
});
["q","fclass","psaMin","psaMax","hbMin","hbMax"].forEach(id=>document.getElementById(id).oninput=render);
overview(); render();
</script></body></html>
"""
