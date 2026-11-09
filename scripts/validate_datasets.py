#!/usr/bin/env python
"""Validate downloaded datasets; write JSON reports + a Markdown summary and update manifests.

    python scripts/validate_datasets.py
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

from face_ai.config import MANIFEST_DIR, RAW_DIR, REPORTS_DIR, ensure_dirs
from face_ai.data.manifests import update_manifest
from face_ai.data.validate import validate_lfw, validate_wider
from face_ai.logging_config import setup_logging

VALIDATORS = {"wider_face": validate_wider, "lfw": validate_lfw}


def _md(name: str, r: dict) -> str:
    out = [f"## {name}: {r['status']}", ""]
    if name == "wider_face":
        out += ["| split | annotated images | on disk | missing | corrupt | duplicates | faces | degenerate | invalid flag | out-of-bounds | <4px |",
                "|---|---|---|---|---|---|---|---|---|---|---|"]
        for s, v in r["splits"].items():
            if v.get("status") == "missing":
                out.append(f"| {s} | missing | | | | | | | | | |")
                continue
            out.append(f"| {s} | {v['annotated_images']} | {v['images_on_disk']} | {len(v['missing_images'])} | "
                       f"{len(v['corrupt_images'])} | {len(v['duplicate_groups'])} | {v['faces']} | {v['degenerate_boxes']} | "
                       f"{v['invalid_flag_boxes']} | {v['out_of_bounds_boxes']} | {v['tiny_boxes_lt4px']} |")
        if "eval_ground_truth" in r:
            g = r["eval_ground_truth"]
            out += ["", f"Evaluation ground truth: {', '.join(g['files'])}; val faces .mat={g['val_faces_in_mat']} txt={g['val_faces_in_txt']}"]
        out.append(f"\nByte-identical images shared by train and val: {r.get('train_val_identical_files')}")
    else:
        out += [f"- images: {r['images']}, identities: {r['identities']} (>=2 imgs: {r['identities_with_ge2_images']}, "
                f">=3: {r['identities_with_ge3_images']}, >=5: {r['identities_with_ge5_images']})",
                f"- image sizes: {r['image_sizes']}", f"- corrupt: {len(r['corrupt_images'])}, misnamed: {len(r['misnamed_files'])}, "
                f"duplicate groups: {len(r['duplicate_groups'])}"]
        if "pairs" in r:
            p = r["pairs"]
            out.append(f"- pairs.txt: {p['folds']} folds, {p['pairs']} pairs ({p['positive']} same / {p['negative']} different), "
                       f"missing referenced images: {len(p['missing_referenced_images'])}")
    for p in r.get("problems", []):
        out.append(f"- PROBLEM: {p}")
    for n in r.get("notes", []):
        out.append(f"- note: {n}")
    return "\n".join(out) + "\n"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--only", nargs="*")
    ap.add_argument("--workers", type=int, default=8)
    args = ap.parse_args()
    log = setup_logging()
    ensure_dirs()
    md = ["# Dataset validation report", ""]
    rc = 0
    for name in args.only or list(VALIDATORS):
        ds_dir = RAW_DIR / name
        if not ds_dir.exists():
            log.error("%s not downloaded (expected %s); run scripts/download_datasets.py", name, ds_dir)
            rc = 1
            continue
        r = VALIDATORS[name](ds_dir, args.workers)
        out = MANIFEST_DIR / f"{name}_validation.json"
        out.write_text(json.dumps(r, indent=2))
        update_manifest(name, {"validation_status": r["status"], "validation_report": str(out),
                               "validation_problems": r["problems"]})
        log.info("%s: %s (%d problems) -> %s", name, r["status"], len(r["problems"]), out)
        md.append(_md(name, r))
    (REPORTS_DIR / "dataset_validation.md").write_text("\n".join(md))
    return rc


if __name__ == "__main__":
    sys.exit(main())
