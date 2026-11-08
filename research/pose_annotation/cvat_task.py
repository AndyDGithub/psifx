"""Creates the CVAT task to correct the pre-annotations of ``prepare.py``.

The task has one skeleton label, ``person``, with the 17 COCO keypoints and edges of the archive. The
images are uploaded at JPEG quality 95 and the pre-annotations imported in the COCO Keypoints format.
Stops if a task of that name exists, so a second run never doubles (or hides) the corrections.

Runs with ``cvat-sdk`` of the server's version, outside the psifx environment:

    cvat-sdk-venv/bin/python -I research/pose_annotation/cvat_task.py \\
        .e2e_outputs/extrait_talkshow/pose_annotation --login cvat_login.txt
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from cvat_sdk import make_client
from cvat_sdk.core.proxies.tasks import ResourceType

# Where each keypoint is drawn in the label editor (100 x 100 box, person facing the viewer).
TEMPLATE = {
    "nose": (50, 10), "left_eye": (54, 7), "right_eye": (46, 7), "left_ear": (58, 9), "right_ear": (42, 9),
    "left_shoulder": (62, 22), "right_shoulder": (38, 22), "left_elbow": (68, 36), "right_elbow": (32, 36),
    "left_wrist": (72, 50), "right_wrist": (28, 50), "left_hip": (57, 52), "right_hip": (43, 52),
    "left_knee": (58, 72), "right_knee": (42, 72), "left_ankle": (59, 92), "right_ankle": (41, 92),
}


def skeleton_label(category: dict) -> dict:
    """CVAT skeleton label: one ``points`` sublabel per keypoint, edges and nodes as SVG (1-based ids)."""
    names, edges = category["keypoints"], category["skeleton"]
    svg = []
    for a, b in edges:
        (x1, y1), (x2, y2) = TEMPLATE[names[a - 1]], TEMPLATE[names[b - 1]]
        svg.append(f'<line x1="{x1}" y1="{y1}" x2="{x2}" y2="{y2}" stroke="black" data-type="edge" '
                   f'data-node-from="{a}" stroke-width="0.5" data-node-to="{b}"></line>')
    for i, name in enumerate(names, 1):
        x, y = TEMPLATE[name]
        svg.append(f'<circle r="1.5" stroke="black" fill="#b3b3b3" cx="{x}" cy="{y}" stroke-width="0.1" '
                   f'data-type="element node" data-element-id="{i}" data-node-id="{i}" '
                   f'data-label-name="{name}"></circle>')
    return {"name": category["name"], "type": "skeleton", "attributes": [], "svg": "".join(svg),
            "sublabels": [{"name": n, "type": "points", "attributes": []} for n in names]}


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="folder written by prepare.py")
    parser.add_argument("--login", type=Path, required=True,
                        help="file with 'utilisateur: ...' and 'mot de passe: ...' lines")
    parser.add_argument("--name", default="ExtraitTalkShow poses")
    parser.add_argument("--url", default="http://localhost:8080")
    args = parser.parse_args()

    coco_path = args.out_dir / "annotations" / "person_keypoints_default.json"
    archive = args.out_dir / "preannotations_coco.zip"
    for path in (coco_path, archive, args.login):
        if not path.is_file():
            sys.exit(f"error: missing {path}")
    login = dict(line.split(": ", 1) for line in args.login.read_text().splitlines() if ": " in line)
    coco = json.loads(coco_path.read_text())
    images = [args.out_dir / "images" / im["file_name"] for im in coco["images"]]
    if missing := [p for p in images if not p.is_file()]:
        sys.exit(f"error: {len(missing)} images missing, first {missing[0]}")

    with make_client(args.url, credentials=(login["utilisateur"], login["mot de passe"])) as client:
        if any(t.name == args.name for t in client.tasks.list()):
            sys.exit(f"error: a task named {args.name!r} exists in CVAT, it may hold corrections: "
                     f"delete it by hand or choose another --name")
        task = client.tasks.create_from_data(
            spec={"name": args.name, "labels": [skeleton_label(coco["categories"][0])]},
            resource_type=ResourceType.LOCAL,
            resources=[str(p) for p in images],
            data_params={"image_quality": 95, "sorting_method": "natural"},
        )
        task.import_annotations(format_name="COCO Keypoints 1.0", filename=str(archive))
        shapes = task.get_annotations().shapes
        skeletons = [s for s in shapes if s.type.value == "skeleton"]
        print(f"task {task.id}: {task.size} images, {len(skeletons)} skeletons imported "
              f"(archive: {len(coco['annotations'])}), {args.url}/tasks/{task.id}")
        if len(skeletons) != len(coco["annotations"]):
            sys.exit("error: not every pre-annotation was imported, check the task before annotating")


if __name__ == "__main__":
    main()
