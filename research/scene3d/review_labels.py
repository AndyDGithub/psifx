"""Page to check and correct the identity labels of the talk-show people, by eye.

Writes one self-contained HTML file (crops embedded, no server, nothing sent anywhere): each annotated person is
shown with some context around its box, with the current label. Clicking a card cycles the label (or use the keys
1-4 on the hovered card); the filter shows one label at a time. "Exporter" saves a labels.json in the same format,
to replace the old one (then rerun merge_people.py --labels).

    python research/scene3d/review_labels.py .e2e_outputs/extrait_talkshow/identity_labels/labels.json \\
        --images .e2e_outputs/extrait_talkshow/pose_annotation/images
"""

from __future__ import annotations

import argparse
import base64
import json
import sys
from pathlib import Path

import cv2

LABELS = ["man", "woman", "incrustation", "unknown"]

PAGE = """<!doctype html><html lang="fr"><head><meta charset="utf-8">
<meta name="viewport" content="width=device-width, initial-scale=1"><title>Identity labels</title><style>
:root{--bg:#f6f6f4;--card:#fff;--ink:#222;--muted:#666;--man:#2a6fdb;--woman:#d2386c;--incrustation:#c98a00;
--unknown:#888}
@media (prefers-color-scheme:dark){:root{--bg:#1b1b1d;--card:#2a2a2d;--ink:#eee;--muted:#aaa}}
body{background:var(--bg);color:var(--ink);font:14px system-ui,sans-serif;margin:0;padding:12px 16px}
header{position:sticky;top:0;background:var(--bg);padding:8px 0;display:flex;flex-wrap:wrap;gap:8px;
align-items:center;z-index:1}
button{font:inherit;padding:4px 10px;border-radius:6px;border:1px solid var(--muted);background:var(--card);
color:var(--ink);cursor:pointer}button.on{outline:2px solid var(--ink)}
#grid{display:grid;grid-template-columns:repeat(auto-fill,minmax(150px,1fr));gap:8px}
.c{background:var(--card);border-radius:8px;padding:4px;border:4px solid var(--unknown);cursor:pointer}
.c img{width:100%;display:block;border-radius:4px}.c div{font-size:12px;display:flex;justify-content:space-between}
.c.changed{box-shadow:0 0 0 2px var(--ink)}
.man{border-color:var(--man)}.woman{border-color:var(--woman)}.incrustation{border-color:var(--incrustation)}
</style></head><body><header><b>Étiquettes d'identité</b><span id="counts"></span>
<span>Filtre :</span><span id="filters"></span><button id="export">Exporter labels.json</button>
<span style="color:var(--muted)">clic = étiquette suivante · survol + touches 1-4</span></header>
<div id="grid"></div><script>
const LABELS=__LABELS__, DATA=__DATA__, META=__META__;
const orig=DATA.map(d=>d.person); let filter=null, hovered=null;
const grid=document.getElementById('grid');
function card(d,i){const c=document.createElement('div');c.className='c '+d.person;c.dataset.i=i;
 c.innerHTML=`<img loading="lazy" src="data:image/jpeg;base64,${d.crop}"><div><span>#${d.n} ${d.image.slice(-10,-4)}</span>
 <b class="l">${d.person}</b></div>`;
 c.onclick=()=>set(i,LABELS[(LABELS.indexOf(d.person)+1)%LABELS.length]);
 c.onmouseenter=()=>hovered=i;c.onmouseleave=()=>hovered=null;return c}
function set(i,l){DATA[i].person=l;const c=grid.children[i];c.className='c '+l+(l!==orig[i]?' changed':'');
 c.querySelector('.l').textContent=l;render()}
function render(){const n={};DATA.forEach(d=>n[d.person]=(n[d.person]||0)+1);
 const ch=DATA.filter((d,i)=>d.person!==orig[i]).length;
 document.getElementById('counts').textContent=LABELS.map(l=>`${l} ${n[l]||0}`).join(' · ')+` · modifiées ${ch}`;
 [...grid.children].forEach((c,i)=>c.style.display=(!filter||DATA[i].person===filter)?'':'none');
 document.querySelectorAll('#filters button').forEach(b=>b.classList.toggle('on',b.dataset.l===String(filter)))}
DATA.forEach((d,i)=>grid.appendChild(card(d,i)));
const f=document.getElementById('filters');
[null,...LABELS].forEach(l=>{const b=document.createElement('button');b.textContent=l||'tout';b.dataset.l=String(l);
 b.onclick=()=>{filter=l;render()};f.appendChild(b)});
document.addEventListener('keydown',e=>{const k=parseInt(e.key);if(hovered!==null&&k>=1&&k<=LABELS.length)
 set(hovered,LABELS[k-1])});
document.getElementById('export').onclick=()=>{const out={labeled_by:META.labeled_by+'; checked by the author',
 annotations:DATA.map(({crop,...d})=>d)};const a=document.createElement('a');
 a.href=URL.createObjectURL(new Blob([JSON.stringify(out,null,1)],{type:'application/json'}));
 a.download='labels.json';a.click()};
render();
</script></body></html>"""


def crop(image, bbox, margin=0.35, height=220):
    """The box with some context around it, the box drawn, resized to a fixed height."""
    h, w = image.shape[:2]
    x1, y1, bw, bh = bbox  # COCO box: [x, y, width, height]
    x2, y2 = x1 + bw, y1 + bh
    mx, my = (x2 - x1) * margin + 20, (y2 - y1) * margin + 20
    X1, Y1, X2, Y2 = int(max(0, x1 - mx)), int(max(0, y1 - my)), int(min(w, x2 + mx)), int(min(h, y2 + my))
    part = image[Y1:Y2, X1:X2].copy()
    cv2.rectangle(part, (int(x1 - X1), int(y1 - Y1)), (int(x2 - X1), int(y2 - Y1)), (0, 255, 0), 2)
    scale = height / max(1, part.shape[0])
    part = cv2.resize(part, (max(1, int(part.shape[1] * scale)), height))
    ok, data = cv2.imencode(".jpg", part, [cv2.IMWRITE_JPEG_QUALITY, 80])
    if not ok:
        raise RuntimeError("JPEG encoding failed")
    return base64.b64encode(data.tobytes()).decode()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("labels", type=Path, help="labels.json to check")
    parser.add_argument("--images", type=Path, required=True, help="folder with the annotated frames")
    parser.add_argument("--output", type=Path, help="default: review.html next to labels.json")
    args = parser.parse_args()

    if not args.labels.is_file() or not args.images.is_dir():
        sys.exit(f"error: missing {args.labels} or {args.images}")
    meta = json.loads(args.labels.read_text())
    data, cache = [], {}
    for a in meta["annotations"]:
        if a.get("person") not in LABELS:
            sys.exit(f"error: unknown label {a.get('person')!r} for annotation {a.get('n')}")
        if a["image"] not in cache:
            image = cv2.imread(str(args.images / a["image"]))
            if image is None:
                sys.exit(f"error: cannot read {args.images / a['image']}")
            cache = {a["image"]: image}  # Annotations come sorted by frame: one frame in memory is enough.
        data.append({**a, "crop": crop(cache[a["image"]], a["bbox"])})
    page = (PAGE.replace("__LABELS__", json.dumps(LABELS)).replace("__DATA__", json.dumps(data))
            .replace("__META__", json.dumps({"labeled_by": meta.get("labeled_by", "")})))
    output = args.output or args.labels.parent / "review.html"
    output.write_text(page)
    print(f"{len(data)} people -> {output}")


if __name__ == "__main__":
    main()
