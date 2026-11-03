"""Downloads the corrected annotations of the CVAT task made by ``cvat_task.py``.

Exports the task in the COCO Keypoints format (no images: they are in ``out_dir/images``) into
``out_dir/ground_truth.zip`` and extracts its JSON to ``out_dir/ground_truth.json``. Stops if they
exist, unless ``--overwrite``: a later export could hold fewer corrections (an unsaved job).

Runs with ``cvat-sdk``, like ``cvat_task.py``:

    cvat-sdk-venv/bin/python -I research/pose_annotation/fetch_annotations.py \\
        .e2e_outputs/extrait_talkshow/pose_annotation --login cvat_login.txt
"""

from __future__ import annotations

import argparse
import json
import sys
import zipfile
from pathlib import Path

from cvat_sdk import make_client


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("out_dir", type=Path, help="folder written by prepare.py")
    parser.add_argument("--login", type=Path, required=True,
                        help="file with 'utilisateur: ...' and 'mot de passe: ...' lines")
    parser.add_argument("--name", default="ExtraitTalkShow poses")
    parser.add_argument("--url", default="http://localhost:8080")
    parser.add_argument("--overwrite", action="store_true")
    args = parser.parse_args()

    archive, json_path = args.out_dir / "ground_truth.zip", args.out_dir / "ground_truth.json"
    if not args.login.is_file():
        sys.exit(f"error: missing {args.login}")
    if (archive.exists() or json_path.exists()) and not args.overwrite:
        sys.exit(f"error: {archive} or {json_path} exists, pass --overwrite to replace them")
    login = dict(line.split(": ", 1) for line in args.login.read_text().splitlines() if ": " in line)

    with make_client(args.url, credentials=(login["utilisateur"], login["mot de passe"])) as client:
        tasks = [t for t in client.tasks.list() if t.name == args.name]
        if len(tasks) != 1:
            sys.exit(f"error: {len(tasks)} tasks named {args.name!r} in CVAT, expected 1")
        task = tasks[0]
        for job in task.get_jobs():
            print(f"job {job.id}: state {job.state}, stage {job.stage}, updated {job.updated_date}")
        task.export_dataset(format_name="COCO Keypoints 1.0", filename=str(archive), include_images=False)

    with zipfile.ZipFile(archive) as z:
        names = [n for n in z.namelist() if n.endswith(".json")]
        if len(names) != 1:
            sys.exit(f"error: expected one JSON in {archive}, found {names}")
        coco = json.loads(z.read(names[0]))
    json_path.write_text(json.dumps(coco))
    print(f"{len(coco['images'])} images, {len(coco['annotations'])} people -> {json_path}")


if __name__ == "__main__":
    main()
