#!/usr/bin/env python
"""Manage the private enrollment gallery (consent required; stored under gallery/, git-ignored).

    python scripts/manage_gallery.py enroll --person-id alice --images ~/consented/alice/ \
        --consent-granted-by "Alice Example" --purpose "home door unlock demo" [--retention-days 365]
    python scripts/manage_gallery.py list
    python scripts/manage_gallery.py remove --person-id alice
    python scripts/manage_gallery.py purge            # delete the whole gallery

Only enroll people who have agreed to it. Do not enroll images from public datasets.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

from face_ai.api.service import FaceAIService
from face_ai.config import GALLERY_DIR, RAW_DIR
from face_ai.logging_config import setup_logging
from face_ai.recognition.gallery import ConsentRecord, Gallery
from face_ai.utils.image_io import is_image, list_images


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("command", choices=["enroll", "list", "remove", "purge"])
    ap.add_argument("--gallery", default=str(GALLERY_DIR))
    ap.add_argument("--person-id")
    ap.add_argument("--images", nargs="*", help="image files and/or directories")
    ap.add_argument("--consent-granted-by")
    ap.add_argument("--purpose")
    ap.add_argument("--retention-days", type=int)
    ap.add_argument("--embedder", choices=["sface", "arcface_r50"])
    args = ap.parse_args()
    log = setup_logging()
    gdir = Path(args.gallery)

    if args.command == "purge":
        Gallery.delete(gdir)
        log.info("gallery at %s deleted", gdir)
        return 0

    svc = FaceAIService.from_configs(embedder_backend=args.embedder)
    if (gdir / "gallery.json").exists():
        svc.load_gallery(gdir)

    if args.command == "list":
        for pid in svc.gallery.people():
            r = svc.gallery.get(pid)
            c = r.consent
            print(f"{pid}\trefs={r.num_references}\tconsent={'yes' if c and c.granted else 'NO'}"
                  f"\tdate={c.date if c else '-'}\tretention_days={c.retention_days if c else '-'}\tupdated={r.updated_at}")
        print(f"{len(svc.gallery)} people, model={svc.gallery.model}")
        return 0

    if not args.person_id:
        ap.error("--person-id is required")

    if args.command == "remove":
        if svc.gallery.remove(args.person_id):
            svc.save_gallery(gdir)
            log.info("removed %s", args.person_id)
            return 0
        log.error("%s not enrolled", args.person_id)
        return 1

    if not (args.consent_granted_by and args.purpose):
        ap.error("enroll requires --consent-granted-by and --purpose (record of the person's consent)")
    files: list[Path] = []
    for p in map(Path, args.images or []):
        if p.resolve().is_relative_to(RAW_DIR.resolve()):
            ap.error(f"{p} is inside the public dataset directory; enrollment is for consenting individuals only")
        files += list_images(p) if p.is_dir() else ([p] if is_image(p) else [])
    if not files:
        ap.error("no images given")
    consent = ConsentRecord(True, args.consent_granted_by, args.purpose, retention_days=args.retention_days)
    report = svc.enroll_person(args.person_id, files, consent)
    for ref, why in report.rejected:
        log.warning("rejected %s: %s", ref, why)
    for w in report.warnings:
        log.warning(w)
    if not report.ok:
        log.error("no usable reference images; nothing enrolled")
        return 1
    svc.save_gallery(gdir)
    log.info("enrolled %s with %d reference image(s) -> %s", args.person_id, report.accepted, gdir)
    return 0


if __name__ == "__main__":
    sys.exit(main())
